from src.GNN.helper import *
from src.GNN.CONSTANTS import *
from src.utils import *
from src.GNN.oldmodels import *
torch.manual_seed(1)

# Contains the action prediction task models. Will be released in a future publication.

def action2vec(action, num_objects, num_states):
    actionArray = torch.zeros(len(possibleActions))
    actionArray[possibleActions.index(action['name'])] = 1
    predicate1 = torch.zeros(num_objects+1)
    #predicate 2 and 3 will be predicted together
    predicate2 = torch.zeros(num_objects+1)
    predicate3 = torch.zeros(num_states)
    if len(action['args']) == 0:
        predicate1[-1] = 1
        predicate2[-1] = 1
    elif len(action['args']) == 1:
        predicate1[object2idx[action['args'][0]]] = 1
        predicate2[-1] = 1
    else:
        # action['args'][1] can be a state or an object
        if action['args'][1] in object2idx:
            predicate1[object2idx[action['args'][0]]] = 1
            predicate2[object2idx[action['args'][1]]] = 1
        else:
            predicate1[object2idx[action['args'][0]]] = 1
            predicate3[possibleStates.index(action['args'][1])] = 1
    return torch.cat((actionArray, predicate1, predicate2, predicate3), 0)

def action2ids(action, num_objects, num_states):
    actionID = possibleActions.index(action['name'])
    predicate1, predicate2 = 0, 0
    if len(action['args']) == 0:
        predicate1 = num_objects+1
        predicate2 = num_objects+1
    elif len(action['args']) == 1:
        predicate1 = object2idx[action['args'][0]]
        predicate2 = num_objects+1
    else:
        # action['args'][1] can be a state or an object
        if action['args'][1] in object2idx:
            predicate1 = object2idx[action['args'][0]]
            predicate2 = object2idx[action['args'][1]]
        else:
            predicate1 = object2idx[action['args'][0]]
            predicate2 = num_objects + 1 + possibleStates.index(action['args'][1])
    return actionID, predicate1, predicate2

def vec2action(vec, num_objects, num_states, idx2object):
    ret_action = {}
    action_array = list(vec[:len(possibleActions)])
    ret_action["name"] = possibleActions[action_array.index(max(action_array))]
    ret_action["args"] = []
    object1_array = list(vec[len(possibleActions):len(possibleActions)+num_objects+1])
    object1_ind = object1_array.index(max(object1_array))
    if object1_ind == len(object1_array) - 1:
        return ret_action
    else:
        ret_action["args"].append(idx2object[object1_ind])
    object2_or_state_array = list(vec[len(possibleActions)+num_objects+1:])
    object2_or_state_ind = object2_or_state_array.index(max(object2_or_state_array))
    if (object2_or_state_ind < num_objects):
        ret_action["args"].append(idx2object[object2_or_state_ind])
    elif (object2_or_state_ind == num_objects):
        pass
    else:
        ret_action["args"].append(possibleStates[object2_or_state_ind - num_objects - 1])
    return ret_action

def vec2action_grammatical(vec, num_objects, num_states, idx2object):
    ret_action = {}
    action_array = list(vec[:len(possibleActions)])
    ret_action["name"] = possibleActions[action_array.index(max(action_array))]
    ret_action["args"] = []
    object1_array = list(vec[len(possibleActions):len(possibleActions)+num_objects+1])
    object1_ind = object1_array.index(max(object1_array))
    if object1_ind == len(object1_array) - 1:
        # return ret_action
        # Removing the case in which zero objects are predicted
        object1_array = list(vec[len(possibleActions):len(possibleActions)+num_objects])
        object1_ind = object1_array.index(max(object1_array))
        ret_action["args"].append(idx2object[object1_ind])
    else:
        ret_action["args"].append(idx2object[object1_ind])
    if ret_action["name"] in ["moveTo", "pick", "climbUp", "climbDown", "clean"]:
        return ret_action
    object2_array = list(vec[len(possibleActions)+num_objects+1:len(possibleActions)+num_objects+1+num_objects])
    state_array = list(vec[len(possibleActions)+num_objects+1+num_objects+1:])
    assert len(state_array) == len(possibleStates)
    if ret_action["name"] == "changeState":
        ret_action["args"].append(possibleStates[state_array.index(max(state_array))])
    else:
        ret_action["args"].append(idx2object[object2_array.index(max(object2_array))])
    return ret_action
    # object2_or_state_array = list(vec[len(possibleActions)+num_objects+1:])
    # object2_or_state_ind = object2_or_state_array.index(max(object2_or_state_array))
    # if (object2_or_state_ind < num_objects):
    #     ret_action["args"].append(idx2object[object2_or_state_ind])
    # elif (object2_or_state_ind == num_objects):
    #     pass
    # else:
    #     ret_action["args"].append(possibleStates[object2_or_state_ind - num_objects - 1])
    return ret_action

def tool2object_likelihoods(num_objects, tool_likelihoods):
    object_likelihoods = torch.zeros(num_objects)
    for i, tool in enumerate(TOOLS2):
        object_likelihoods[object2idx[tool]] = tool_likelihoods[i]
    return object_likelihoods

#############################################################################

class DGL_AGCN_Action(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_states,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(DGL_AGCN_Action, self).__init__()
        self.name = "GatedHeteroRGCN_Attention_Action_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.attention = nn.Linear(n_hidden + n_hidden, 1)
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, len(possibleActions))
        self.p1  = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.p2  = nn.Linear(n_hidden, n_hidden)
        self.p3  = nn.Linear(n_hidden, n_objects+1)
        self.q1  = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.q2  = nn.Linear(n_hidden, n_hidden)
        self.q3  = nn.Linear(n_hidden, n_objects+1+n_states)
        self.activation = nn.LeakyReLU()

    def forward(self, g, goalVec, goalObjectsVec):
        h = g.ndata['feat']
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
        goalObjectsVec = self.activation(self.embed(torch.Tensor(goalObjectsVec)))
        attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1)], 1)
        attn_weights = F.softmax(self.attention(attn_embedding), dim=0)
        # print(attn_weights)
        scene_embedding = torch.mm(attn_weights.t(), h)
        goal_embed = self.activation(self.embed(torch.Tensor(goalVec.reshape(1, -1))))
        final_to_decode = torch.cat([scene_embedding, goal_embed], 1)
        action = self.activation(self.fc1(final_to_decode))
        action = self.activation(self.fc2(action))
        action = self.activation(self.fc3(action))
        action = F.softmax(action, dim=1)
        pred1 = self.activation(self.p1(final_to_decode))
        pred1 = self.activation(self.p2(pred1))
        pred1 = F.softmax(self.activation(self.p3(pred1)), dim=1)
        pred2 = self.activation(self.q1(final_to_decode))
        pred2 = self.activation(self.q2(pred2))
        pred2 = F.softmax(self.activation(self.q3(pred2)), dim=1)
        return torch.cat((action, pred1, pred2), 1).flatten()

class GGCN_Action(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_states,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(GGCN_Action, self).__init__()
        self.name = "GGCN_Action_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.attention = nn.Linear(n_hidden + n_hidden, 1)
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, len(possibleActions))
        self.p1  = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.p2  = nn.Linear(n_hidden, n_hidden)
        self.p3  = nn.Linear(n_hidden, n_objects+1)
        self.q1  = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.q2  = nn.Linear(n_hidden, n_hidden)
        self.q3  = nn.Linear(n_hidden, n_objects+1+n_states)
        self.activation = nn.LeakyReLU()

    def forward(self, g, goalVec, goalObjectsVec):
        h = g.ndata['feat']
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
        goalObjectsVec = self.activation(self.embed(torch.Tensor(goalObjectsVec)))
        scene_embedding = torch.sum(h, dim=0).view(1,-1)
        goal_embed = self.activation(self.embed(torch.Tensor(goalVec.reshape(1, -1))))
        final_to_decode = torch.cat([scene_embedding, goal_embed], 1)
        action = self.activation(self.fc1(final_to_decode))
        action = self.activation(self.fc2(action))
        action = self.activation(self.fc3(action))
        action = F.softmax(action, dim=1)
        pred1 = self.activation(self.p1(final_to_decode))
        pred1 = self.activation(self.p2(pred1))
        pred1 = F.softmax(self.activation(self.p3(pred1)), dim=1)
        pred2 = self.activation(self.q1(final_to_decode))
        pred2 = self.activation(self.q2(pred2))
        pred2 = F.softmax(self.activation(self.q3(pred2)), dim=1)
        return torch.cat((action, pred1, pred2), 1).flatten()

class GGCN_metric_Action(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_states,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(GGCN_metric_Action, self).__init__()
        self.name = "GGCN_metric_Action_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.attention = nn.Linear(n_hidden + n_hidden, 1)
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_hidden + n_hidden + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, len(possibleActions))
        self.p1  = nn.Linear(n_hidden + n_hidden + n_hidden, n_hidden)
        self.p2  = nn.Linear(n_hidden, n_hidden)
        self.p3  = nn.Linear(n_hidden, n_objects+1)
        self.q1  = nn.Linear(n_hidden + n_hidden + n_hidden, n_hidden)
        self.q2  = nn.Linear(n_hidden, n_hidden)
        self.q3  = nn.Linear(n_hidden, n_objects+1+n_states)
        self.activation = nn.LeakyReLU()
        self.metric1 = nn.Linear(in_feats, n_hidden)
        self.metric2 = nn.Linear(n_hidden, n_hidden)

    def forward(self, g, goalVec, goalObjectsVec):
        h = g.ndata['feat']
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
        metric_part = g.ndata['feat']
        metric_part = self.activation(self.metric1(metric_part))
        metric_part = self.activation(self.metric2(metric_part))
        h = torch.cat([h, metric_part], dim = 1)
        goalObjectsVec = self.activation(self.embed(torch.Tensor(goalObjectsVec)))
        scene_embedding = torch.sum(h, dim=0).view(1,-1)
        goal_embed = self.activation(self.embed(torch.Tensor(goalVec.reshape(1, -1))))
        final_to_decode = torch.cat([scene_embedding, goal_embed], 1)
        action = self.activation(self.fc1(final_to_decode))
        action = self.activation(self.fc2(action))
        action = self.activation(self.fc3(action))
        action = F.softmax(action, dim=1)
        pred1 = self.activation(self.p1(final_to_decode))
        pred1 = self.activation(self.p2(pred1))
        pred1 = F.softmax(self.activation(self.p3(pred1)), dim=1)
        pred2 = self.activation(self.q1(final_to_decode))
        pred2 = self.activation(self.q2(pred2))
        pred2 = F.softmax(self.activation(self.q3(pred2)), dim=1)
        return torch.cat((action, pred1, pred2), 1).flatten()

class GGCN_metric_att_Action(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_states,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(GGCN_metric_att_Action, self).__init__()
        self.name = "GGCN_metric_att_Action_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.attention = nn.Linear(n_hidden + n_hidden + n_hidden, 1)
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_hidden + n_hidden + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, len(possibleActions))
        self.p1  = nn.Linear(n_hidden + n_hidden + n_hidden, n_hidden)
        self.p2  = nn.Linear(n_hidden, n_hidden)
        self.p3  = nn.Linear(n_hidden, n_objects+1)
        self.q1  = nn.Linear(n_hidden + n_hidden + n_hidden, n_hidden)
        self.q2  = nn.Linear(n_hidden, n_hidden)
        self.q3  = nn.Linear(n_hidden, n_objects+1+n_states)
        self.activation = nn.LeakyReLU()
        self.metric1 = nn.Linear(in_feats, n_hidden)
        self.metric2 = nn.Linear(n_hidden, n_hidden)

    def forward(self, g, goalVec, goalObjectsVec):
        h = g.ndata['feat']
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
        metric_part = g.ndata['feat']
        metric_part = self.activation(self.metric1(metric_part))
        metric_part = self.activation(self.metric2(metric_part))
        h = torch.cat([h, metric_part], dim = 1)
        goalObjectsVec = self.activation(self.embed(torch.Tensor(goalObjectsVec)))
        attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1)], 1)
        attn_weights = F.softmax(self.attention(attn_embedding), dim=0)
        scene_embedding = torch.mm(attn_weights.t(), h)
        goal_embed = self.activation(self.embed(torch.Tensor(goalVec.reshape(1, -1))))
        final_to_decode = torch.cat([scene_embedding, goal_embed], 1)
        action = self.activation(self.fc1(final_to_decode))
        action = self.activation(self.fc2(action))
        action = self.activation(self.fc3(action))
        action = F.softmax(action, dim=1)
        pred1 = self.activation(self.p1(final_to_decode))
        pred1 = self.activation(self.p2(pred1))
        pred1 = F.softmax(self.activation(self.p3(pred1)), dim=1)
        pred2 = self.activation(self.q1(final_to_decode))
        pred2 = self.activation(self.q2(pred2))
        pred2 = F.softmax(self.activation(self.q3(pred2)), dim=1)
        return torch.cat((action, pred1, pred2), 1).flatten()

class GGCN_metric_att_aseq_Action(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_states,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(GGCN_metric_att_aseq_Action, self).__init__()
        self.name = "GGCN_metric_att_aseq_Action_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.attention = nn.Sequential(nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden, n_hidden), nn.Linear(n_hidden, 1))
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, len(possibleActions))
        self.p1  = nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden, n_hidden)
        self.p2  = nn.Linear(n_hidden, n_hidden)
        self.p3  = nn.Linear(n_hidden, n_objects+1)
        self.q1  = nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden, n_hidden)
        self.q2  = nn.Linear(n_hidden, n_hidden)
        self.q3  = nn.Linear(n_hidden, n_objects+1+n_states)
        self.activation = nn.LeakyReLU()
        self.metric1 = nn.Linear(in_feats, n_hidden)
        self.metric2 = nn.Linear(n_hidden, n_hidden)
        self.action_lstm = nn.LSTM(len(possibleActions) + n_objects + 1 + n_objects + 1 + n_states, n_hidden)
        self.n_hidden = n_hidden
        self.n_objects = n_objects
        self.n_states = n_states

    def forward(self, g_list, goalVec, goalObjectsVec, a_list):
        a_list = [action2vec(i, self.n_objects, self.n_states) for i in a_list]
        predicted_actions = []
        lstm_hidden = (torch.randn(1, 1, self.n_hidden),torch.randn(1, 1, self.n_hidden))
        goalObjectsVec = self.activation(self.embed(torch.Tensor(goalObjectsVec)))
        goal_embed = self.activation(self.embed(torch.Tensor(goalVec.reshape(1, -1))))
        for ind,g in enumerate(g_list):
            h = g.ndata['feat']
            for i, layer in enumerate(self.layers):
                h = layer(g, h)
            metric_part = g.ndata['feat']
            metric_part = self.activation(self.metric1(metric_part))
            metric_part = self.activation(self.metric2(metric_part))
            h = torch.cat([h, metric_part], dim = 1)
            if (ind != 0):
                lstm_out, lstm_hidden = self.action_lstm(a_list[ind-1].view(1,1,-1), lstm_hidden)
            else:
                lstm_out = torch.zeros(1, 1, self.n_hidden)
            lstm_out = lstm_out.view(-1)
            attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1), lstm_out.repeat(h.size(0)).view(h.size(0), -1)], 1)
            attn_weights = F.softmax(self.attention(attn_embedding), dim=0)
            scene_embedding = torch.mm(attn_weights.t(), h)
            final_to_decode = torch.cat([scene_embedding, goal_embed, lstm_out.view(1,-1)], 1)
            action = self.activation(self.fc1(final_to_decode))
            action = self.activation(self.fc2(action))
            action = self.activation(self.fc3(action))
            action = F.softmax(action, dim=1)
            pred1 = self.activation(self.p1(final_to_decode))
            pred1 = self.activation(self.p2(pred1))
            pred1 = F.softmax(self.activation(self.p3(pred1)), dim=1)
            pred2 = self.activation(self.q1(final_to_decode))
            pred2 = self.activation(self.q2(pred2))
            pred2 = F.softmax(self.activation(self.q3(pred2)), dim=1)
            predicted_actions.append(torch.cat((action, pred1, pred2), 1).flatten())
        return predicted_actions

# class GGCN_metric_att_aseq_auto_Action(nn.Module):
#     def __init__(self,
#                  in_feats,
#                  n_objects,
#                  n_hidden,
#                  n_states,
#                  n_layers,
#                  etypes,
#                  activation,
#                  dropout):
#         super(GGCN_metric_att_aseq_auto_Action, self).__init__()
#         self.name = "GGCN_metric_att_aseq_auto_Action_" + str(n_hidden) + "_" + str(n_layers)
#         self.layers = nn.ModuleList()
#         self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
#         for i in range(n_layers - 1):
#             self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
#         self.attention = nn.Sequential(nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden, n_hidden), nn.Linear(n_hidden, 1))
#         self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
#         self.fc1 = nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden, n_hidden)
#         self.fc2 = nn.Linear(n_hidden, n_hidden)
#         self.fc3 = nn.Linear(n_hidden, len(possibleActions))
#         self.p1  = nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden + len(possibleActions), n_hidden + n_hidden)
#         self.p2  = nn.Linear(n_hidden + n_hidden, n_hidden)
#         self.p3  = nn.Linear(n_hidden, n_objects+1)
#         self.q1  = nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden + len(possibleActions), n_hidden + n_hidden)
#         self.q2  = nn.Linear(n_hidden + n_hidden, n_hidden)
#         self.q3  = nn.Linear(n_hidden, n_objects+1+n_states)
#         self.activation = nn.LeakyReLU()
#         self.metric1 = nn.Linear(in_feats, n_hidden)
#         self.metric2 = nn.Linear(n_hidden, n_hidden)
#         self.action_lstm = nn.LSTM(len(possibleActions) + n_objects + 1 + n_objects + 1 + n_states, n_hidden)
#         self.n_hidden = n_hidden
#         self.n_objects = n_objects
#         self.n_states = n_states

#     def forward(self, g_list, goalVec, goalObjectsVec, a_list):
#         a_list = [action2vec(i, self.n_objects, self.n_states) for i in a_list]
#         predicted_actions = []
#         lstm_hidden = (torch.randn(1, 1, self.n_hidden),torch.randn(1, 1, self.n_hidden))
#         goalObjectsVec = self.activation(self.embed(torch.Tensor(goalObjectsVec)))
#         goal_embed = self.activation(self.embed(torch.Tensor(goalVec.reshape(1, -1))))
#         for ind,g in enumerate(g_list):
#             h = g.ndata['feat']
#             for i, layer in enumerate(self.layers):
#                 h = layer(g, h)
#             metric_part = g.ndata['feat']
#             metric_part = self.activation(self.metric1(metric_part))
#             metric_part = self.activation(self.metric2(metric_part))
#             h = torch.cat([h, metric_part], dim = 1)
#             if (ind != 0):
#                 lstm_out, lstm_hidden = self.action_lstm(a_list[ind-1].view(1,1,-1), lstm_hidden)
#             else:
#                 lstm_out = torch.zeros(1, 1, self.n_hidden)
#             lstm_out = lstm_out.view(-1)
#             attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1), lstm_out.repeat(h.size(0)).view(h.size(0), -1)], 1)
#             attn_weights = F.softmax(self.attention(attn_embedding), dim=0)
#             scene_embedding = torch.mm(attn_weights.t(), h)
#             final_to_decode = torch.cat([scene_embedding, goal_embed, lstm_out.view(1,-1)], 1)
#             action = self.activation(self.fc1(final_to_decode))
#             action = self.activation(self.fc2(action))
#             action = self.activation(self.fc3(action))
#             action = F.softmax(action, dim=1)
#             pred1 = self.activation(self.p1(torch.cat([final_to_decode, action], 1)))
#             pred1 = self.activation(self.p2(pred1))
#             pred1 = F.softmax(self.activation(self.p3(pred1)), dim=1)
#             pred2 = self.activation(self.q1(torch.cat([final_to_decode, action], 1)))
#             pred2 = self.activation(self.q2(pred2))
#             pred2 = F.softmax(self.activation(self.q3(pred2)), dim=1)
#             predicted_actions.append(torch.cat((action, pred1, pred2), 1).flatten())
#         return predicted_actions

class GGCN_metric_att_aseq_auto_Action(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_states,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(GGCN_metric_att_aseq_auto_Action, self).__init__()
        self.name = "GGCN_metric_att_aseq_auto_Action_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.attention = nn.Sequential(nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden, n_hidden), nn.Linear(n_hidden, 1))
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, len(possibleActions))
        self.p1  = nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden + len(possibleActions), n_hidden + n_hidden)
        self.p2  = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.p3  = nn.Linear(n_hidden, n_objects+1)
        self.q1  = nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden + len(possibleActions), n_hidden + n_hidden)
        self.q2  = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.q3  = nn.Linear(n_hidden, n_objects+1+n_states)
        self.activation = nn.LeakyReLU()
        self.metric1 = nn.Linear(in_feats, n_hidden)
        self.metric2 = nn.Linear(n_hidden, n_hidden)
        self.action_lstm = nn.LSTM(len(possibleActions) + n_objects + 1 + n_objects + 1 + n_states, n_hidden)
        self.n_hidden = n_hidden
        self.n_objects = n_objects
        self.n_states = n_states

    def forward(self, g_list, goalVec, goalObjectsVec, a_list):
        a_list = [action2vec(i, self.n_objects, self.n_states) for i in a_list]
        predicted_actions = []
        lstm_hidden = (torch.randn(1, 1, self.n_hidden),torch.randn(1, 1, self.n_hidden))
        goalObjectsVec = self.activation(self.embed(torch.Tensor(goalObjectsVec)))
        goal_embed = self.activation(self.embed(torch.Tensor(goalVec.reshape(1, -1))))
        for ind,g in enumerate(g_list):
            h = g.ndata['feat']
            for i, layer in enumerate(self.layers):
                h = layer(g, h)
            metric_part = g.ndata['feat']
            metric_part = self.activation(self.metric1(metric_part))
            metric_part = self.activation(self.metric2(metric_part))
            h = torch.cat([h, metric_part], dim = 1)
            if (ind != 0):
                lstm_out, lstm_hidden = self.action_lstm(a_list[ind-1].view(1,1,-1), lstm_hidden)
            else:
                lstm_out = torch.zeros(1, 1, self.n_hidden)
            lstm_out = lstm_out.view(-1)
            attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1), lstm_out.repeat(h.size(0)).view(h.size(0), -1)], 1)
            attn_weights = F.softmax(self.attention(attn_embedding), dim=0)
            scene_embedding = torch.mm(attn_weights.t(), h)
            final_to_decode = torch.cat([scene_embedding, goal_embed, lstm_out.view(1,-1)], 1)
            action = self.activation(self.fc1(final_to_decode))
            action = self.activation(self.fc2(action))
            action = self.activation(self.fc3(action))
            action = F.softmax(action, dim=1)
            pred_action_values = list(action[0])
            ind_max_action = pred_action_values.index(max(pred_action_values))
            one_hot_action = [0 for i in range(len(pred_action_values))]; one_hot_action[ind_max_action] = 1
            one_hot_action = torch.Tensor(one_hot_action).view(1,-1)
            pred1 = self.activation(self.p1(torch.cat([final_to_decode, one_hot_action], 1)))
            pred1 = self.activation(self.p2(pred1))
            pred1 = F.softmax(self.activation(self.p3(pred1)), dim=1)
            pred2 = self.activation(self.q1(torch.cat([final_to_decode, one_hot_action], 1)))
            pred2 = self.activation(self.q2(pred2))
            pred2 = F.softmax(self.activation(self.q3(pred2)), dim=1)
            predicted_actions.append(torch.cat((action, pred1, pred2), 1).flatten())
        return predicted_actions

class GGCN_metric_att_aseq_L_Action(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_states,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(GGCN_metric_att_aseq_L_Action, self).__init__()
        self.name = "GGCN_metric_att_aseq_L_Action_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.attention = nn.Sequential(nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden, n_hidden), nn.Linear(n_hidden, 1))
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_hidden*4, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, len(possibleActions))
        self.p1_object  = nn.Linear(n_hidden*5, n_hidden)
        self.p2_object  = nn.Linear(n_hidden, n_hidden)
        self.p3_object  = nn.Linear(n_hidden, 1)
        self.p1_no_object  = nn.Linear(n_hidden*4, n_hidden)
        self.p2_no_object  = nn.Linear(n_hidden, n_hidden)
        self.p3_no_object  = nn.Linear(n_hidden, 1)
        self.q1_object  = nn.Linear(n_hidden*5, n_hidden)
        self.q2_object  = nn.Linear(n_hidden, n_hidden)
        self.q3_object  = nn.Linear(n_hidden, 1)
        self.q1_no_object  = nn.Linear(n_hidden*4, n_hidden)
        self.q2_no_object  = nn.Linear(n_hidden, n_hidden)
        self.q3_no_object  = nn.Linear(n_hidden, 1)
        self.q1_state  = nn.Linear(n_hidden*4, n_hidden)
        self.q2_state  = nn.Linear(n_hidden, n_hidden)
        self.q3_state  = nn.Linear(n_hidden, n_states)
        self.activation = nn.LeakyReLU()
        self.metric1 = nn.Linear(in_feats, n_hidden)
        self.metric2 = nn.Linear(n_hidden, n_hidden)
        self.action_lstm = nn.LSTM(len(possibleActions) + n_objects + 1 + n_objects + 1 + n_states, n_hidden)
        self.n_hidden = n_hidden
        self.n_objects = n_objects
        self.n_states = n_states
        l = []
        for i in range(n_objects):
            l.append(object2vec[idx2object[i]])
        self.object_vec = torch.Tensor(l)

    def forward(self, g_list, goalVec, goalObjectsVec, a_list):
        a_list = [action2vec(i, self.n_objects, self.n_states) for i in a_list]
        predicted_actions = []
        lstm_hidden = (torch.randn(1, 1, self.n_hidden),torch.randn(1, 1, self.n_hidden))
        goalObjectsVec = self.activation(self.embed(torch.Tensor(goalObjectsVec)))
        goal_embed = self.activation(self.embed(torch.Tensor(goalVec.reshape(1, -1))))
        for ind,g in enumerate(g_list):
            h = g.ndata['feat']
            for i, layer in enumerate(self.layers):
                h = layer(g, h)
            metric_part = g.ndata['feat']
            metric_part = self.activation(self.metric1(metric_part))
            metric_part = self.activation(self.metric2(metric_part))
            h = torch.cat([h, metric_part], dim = 1)
            if (ind != 0):
                lstm_out, lstm_hidden = self.action_lstm(a_list[ind-1].view(1,1,-1), lstm_hidden)
            else:
                lstm_out = torch.zeros(1, 1, self.n_hidden)
            lstm_out = lstm_out.view(-1)
            attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1), lstm_out.repeat(h.size(0)).view(h.size(0), -1)], 1)
            attn_weights = F.softmax(self.attention(attn_embedding), dim=0)
            scene_embedding = torch.mm(attn_weights.t(), h)
            final_to_decode = torch.cat([scene_embedding, goal_embed, lstm_out.view(1,-1)], 1)
            action = self.activation(self.fc1(final_to_decode))
            action = self.activation(self.fc2(action))
            action = self.activation(self.fc3(action))
            action = F.softmax(action, dim=1)

            #Predicting the first argument of the action
            pred1_object = self.activation(self.p1_object(
                        torch.cat([final_to_decode.view(-1).repeat(self.n_objects).view(self.n_objects, -1), self.activation(self.embed(self.object_vec))], 1)))
            pred1_object = self.activation(self.p2_object(pred1_object))
            pred1_object = torch.sigmoid(self.p3_object(pred1_object))

            pred1_no_object = self.activation(self.p1_no_object(final_to_decode))
            pred1_no_object = self.activation(self.p2_no_object(pred1_no_object))
            pred1_no_object = torch.sigmoid(self.p3_no_object(pred1_no_object))

            # Predicting the second argument of the action
            pred2_object = self.activation(self.q1_object(
                        torch.cat([final_to_decode.view(-1).repeat(self.n_objects).view(self.n_objects, -1), self.activation(self.embed(self.object_vec))], 1)))
            pred2_object = self.activation(self.q2_object(pred2_object))
            pred2_object = torch.sigmoid(self.q3_object(pred2_object))

            pred2_no_object = self.activation(self.q1_no_object(final_to_decode))
            pred2_no_object = self.activation(self.q2_no_object(pred2_no_object))
            pred2_no_object = torch.sigmoid(self.q3_no_object(pred2_no_object))

            pred2_state = self.activation(self.q1_state(final_to_decode))
            pred2_state = self.activation(self.q2_state(pred2_state))
            pred2_state = F.softmax(self.q3_state(pred2_state), dim = 1)
            predicted_actions.append(torch.cat((action, pred1_object.view(1,-1), pred1_no_object, pred2_object.view(1,-1), pred2_no_object, pred2_state), 1).flatten())
        return predicted_actions

class GGCN_metric_att_aseq_tool_Action(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_states,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(GGCN_metric_att_aseq_tool_Action, self).__init__()
        self.name = "GGCN_metric_att_aseq_tool_Action_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.attention = nn.Sequential(nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden + 1, n_hidden), nn.Linear(n_hidden, 1))
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, len(possibleActions))
        self.p1  = nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden, n_hidden)
        self.p2  = nn.Linear(n_hidden, n_hidden)
        self.p3  = nn.Linear(n_hidden, n_objects+1)
        self.q1  = nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden, n_hidden)
        self.q2  = nn.Linear(n_hidden, n_hidden)
        self.q3  = nn.Linear(n_hidden, n_objects+1+n_states)
        self.activation = nn.LeakyReLU()
        self.metric1 = nn.Linear(in_feats, n_hidden)
        self.metric2 = nn.Linear(n_hidden, n_hidden)
        self.action_lstm = nn.LSTM(len(possibleActions) + n_objects + 1 + n_objects + 1 + n_states, n_hidden)
        self.n_hidden = n_hidden
        self.n_objects = n_objects
        self.n_states = n_states

    def forward(self, g_list, goalVec, goalObjectsVec, a_list, object_likelihoods):
        a_list = [action2vec(i, self.n_objects, self.n_states) for i in a_list]
        predicted_actions = []
        lstm_hidden = (torch.randn(1, 1, self.n_hidden),torch.randn(1, 1, self.n_hidden))
        goalObjectsVec = self.activation(self.embed(torch.Tensor(goalObjectsVec)))
        goal_embed = self.activation(self.embed(torch.Tensor(goalVec.reshape(1, -1))))
        for ind,g in enumerate(g_list):
            h = g.ndata['feat']
            for i, layer in enumerate(self.layers):
                h = layer(g, h)
            metric_part = g.ndata['feat']
            metric_part = self.activation(self.metric1(metric_part))
            metric_part = self.activation(self.metric2(metric_part))
            h = torch.cat([h, metric_part], dim = 1)
            if (ind != 0):
                lstm_out, lstm_hidden = self.action_lstm(a_list[ind-1].view(1,1,-1), lstm_hidden)
            else:
                lstm_out = torch.zeros(1, 1, self.n_hidden)
            lstm_out = lstm_out.view(-1)
            attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1), lstm_out.repeat(h.size(0)).view(h.size(0), -1), object_likelihoods[ind].view(h.size(0),-1)], 1)
            attn_weights = F.softmax(self.attention(attn_embedding), dim=0)
            scene_embedding = torch.mm(attn_weights.t(), h)
            final_to_decode = torch.cat([scene_embedding, goal_embed, lstm_out.view(1,-1)], 1)
            action = self.activation(self.fc1(final_to_decode))
            action = self.activation(self.fc2(action))
            action = self.activation(self.fc3(action))
            action = F.softmax(action, dim=1)
            pred1 = self.activation(self.p1(final_to_decode))
            pred1 = self.activation(self.p2(pred1))
            pred1 = F.softmax(self.activation(self.p3(pred1)), dim=1)
            pred2 = self.activation(self.q1(final_to_decode))
            pred2 = self.activation(self.q2(pred2))
            pred2 = F.softmax(self.activation(self.q3(pred2)), dim=1)
            predicted_actions.append(torch.cat((action, pred1, pred2), 1).flatten())
        return predicted_actions

class GGCN_metric_att_aseq_tool_auto_Action(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_states,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(GGCN_metric_att_aseq_tool_auto_Action, self).__init__()
        self.name = "GGCN_metric_att_aseq_tool_auto_Action_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.attention = nn.Sequential(nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden + 1, n_hidden), nn.Linear(n_hidden, 1))
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, len(possibleActions))
        self.p1  = nn.Linear(len(possibleActions) + n_hidden + n_hidden + n_hidden + n_hidden, n_hidden)
        self.p2  = nn.Linear(n_hidden, n_hidden)
        self.p3  = nn.Linear(n_hidden, n_objects+1)
        self.q1  = nn.Linear(len(possibleActions)+n_objects+1 + n_hidden + n_hidden + n_hidden + n_hidden, n_hidden)
        self.q2  = nn.Linear(n_hidden, n_hidden)
        self.q3  = nn.Linear(n_hidden, n_objects+1+n_states)
        self.activation = nn.LeakyReLU()
        self.metric1 = nn.Linear(in_feats, n_hidden)
        self.metric2 = nn.Linear(n_hidden, n_hidden)
        self.action_lstm = nn.LSTM(len(possibleActions) + n_objects + 1 + n_objects + 1 + n_states, n_hidden)
        self.n_hidden = n_hidden
        self.n_objects = n_objects
        self.n_states = n_states

    def forward(self, g_list, goalVec, goalObjectsVec, a_list, object_likelihoods):
        a_list = [action2vec(i, self.n_objects, self.n_states) if i else None for i in a_list]
        predicted_actions = []
        lstm_hidden = (torch.randn(1, 1, self.n_hidden),torch.randn(1, 1, self.n_hidden))
        goalObjectsVec = self.activation(self.embed(torch.Tensor(goalObjectsVec)))
        goal_embed = self.activation(self.embed(torch.Tensor(goalVec.reshape(1, -1))))
        for ind,g in enumerate(g_list):
            h = g.ndata['feat']
            for i, layer in enumerate(self.layers):
                h = layer(g, h)
            metric_part = g.ndata['feat']
            metric_part = self.activation(self.metric1(metric_part))
            metric_part = self.activation(self.metric2(metric_part))
            h = torch.cat([h, metric_part], dim = 1)
            if (ind != 0):
                lstm_out, lstm_hidden = self.action_lstm(a_list[ind-1].view(1,1,-1), lstm_hidden)
            else:
                lstm_out = torch.zeros(1, 1, self.n_hidden)
            lstm_out = lstm_out.view(-1)
            attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1), lstm_out.repeat(h.size(0)).view(h.size(0), -1), object_likelihoods[ind].view(h.size(0),-1)], 1)
            attn_weights = F.softmax(self.attention(attn_embedding), dim=0)
            scene_embedding = torch.mm(attn_weights.t(), h)
            final_to_decode = torch.cat([scene_embedding, goal_embed, lstm_out.view(1,-1)], 1)
            action = self.activation(self.fc1(final_to_decode))
            action = self.activation(self.fc2(action))
            action = self.activation(self.fc3(action))
            action = F.softmax(action, dim=1)
            pred_action_values = list(action[0])
            ind_max_action = pred_action_values.index(max(pred_action_values))
            one_hot_action = [0] * len(pred_action_values); one_hot_action[ind_max_action] = 1
            one_hot_action = torch.Tensor(one_hot_action).view(1,-1)
            pred1 = self.activation(self.p1(torch.cat([one_hot_action, final_to_decode], dim=1)))
            pred1 = self.activation(self.p2(pred1))
            pred1 = F.softmax(self.activation(self.p3(pred1)), dim=1)
            pred2 = self.activation(self.q1(torch.cat([action, pred1, final_to_decode], dim=1)))
            pred2 = self.activation(self.q2(pred2))
            pred2 = F.softmax(self.activation(self.q3(pred2)), dim=1)
            predicted_actions.append(torch.cat((action, pred1, pred2), 1).flatten())
        return predicted_actions

class DGL_AGCN_Action_List(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_states,
                 n_layers,
                 etypes,
                 activation,
                 dropout,
                 num_states_in_list):
        super(DGL_AGCN_Action_List, self).__init__()
        self.name = "GatedHeteroRGCN_Attention_Action_List_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.attention = nn.Linear(n_hidden + n_hidden, 1)
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_hidden * num_states_in_list + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, len(possibleActions))
        self.p1  = nn.Linear(n_hidden * num_states_in_list + n_hidden, n_hidden)
        self.p2  = nn.Linear(n_hidden, n_hidden)
        self.p3  = nn.Linear(n_hidden, n_objects+1)
        self.q1  = nn.Linear(n_hidden * num_states_in_list + n_hidden, n_hidden)
        self.q2  = nn.Linear(n_hidden, n_hidden)
        self.q3  = nn.Linear(n_hidden, n_objects+1+n_states)
        self.n_hidden = n_hidden
        self.activation = nn.LeakyReLU()
        self.num_states_in_list = num_states_in_list

    def forward(self, g_list, goalVec, goalObjectsVec):
        goalObjectsVec = self.activation(self.embed(torch.Tensor(goalObjectsVec)))
        goal_embed = self.activation(self.embed(torch.Tensor(goalVec.reshape(1, -1))))
        scene_embedding_list = [torch.zeros(1,self.n_hidden) for i in range(self.num_states_in_list - len(g_list))]
        for g in g_list:    
            h = g.ndata['feat']
            for i, layer in enumerate(self.layers):
                h = layer(g, h)
        
            attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1)], 1)
            attn_weights = F.softmax(self.attention(attn_embedding), dim=0)
            # print(attn_weights)
            scene_embedding = torch.mm(attn_weights.t(), h)
            scene_embedding_list.append(scene_embedding)
        scene_embedding = torch.cat(scene_embedding_list,1)
        final_to_decode = torch.cat([scene_embedding, goal_embed], 1)
        action = self.activation(self.fc1(final_to_decode))
        action = self.activation(self.fc2(action))
        action = self.activation(self.fc3(action))
        action = F.softmax(action, dim=1)
        pred1 = self.activation(self.p1(final_to_decode))
        pred1 = self.activation(self.p2(pred1))
        pred1 = F.softmax(self.activation(self.p3(pred1)), dim=1)
        pred2 = self.activation(self.q1(final_to_decode))
        pred2 = self.activation(self.q2(pred2))
        pred2 = F.softmax(self.activation(self.q3(pred2)), dim=1)
        return torch.cat((action, pred1, pred2), 1).flatten()

#############################################################################


class Metric_Action(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_states,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(Metric_Action, self).__init__()
        self.etypes = etypes
        self.name = "Metric_Action_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_feats + n_objects*4, n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        self.attention = nn.Linear(n_hidden + n_hidden, 1)
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, len(possibleActions))
        self.p1  = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.p2  = nn.Linear(n_hidden, n_hidden)
        self.p3  = nn.Linear(n_hidden, n_objects+1)
        self.q1  = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.q2  = nn.Linear(n_hidden, n_hidden)
        self.q3  = nn.Linear(n_hidden, n_objects+1+n_states)
        self.activation = nn.LeakyReLU()

    def forward(self, g, goalVec, goalObjectsVec):
        h = g.ndata['feat']
        edgeMatrices = [g.adjacency_matrix(etype=t) for t in self.etypes]
        edges = torch.cat(edgeMatrices, 1).to_dense()
        h = torch.cat((h, edges), 1)
        for i, layer in enumerate(self.layers):
            h = self.activation(layer(h))
        scene_embedding = torch.sum(h, dim=0).view(1,-1)
        goal_embed = self.activation(self.embed(torch.Tensor(goalVec.reshape(1, -1))))
        final_to_decode = torch.cat([scene_embedding, goal_embed], 1)
        action = self.activation(self.fc1(final_to_decode))
        action = self.activation(self.fc2(action))
        action = self.activation(self.fc3(action))
        action = F.softmax(action, dim=1)
        pred1 = self.activation(self.p1(final_to_decode))
        pred1 = self.activation(self.p2(pred1))
        pred1 = F.softmax(self.activation(self.p3(pred1)), dim=1)
        pred2 = self.activation(self.q1(final_to_decode))
        pred2 = self.activation(self.q2(pred2))
        pred2 = F.softmax(self.activation(self.q3(pred2)), dim=1)
        return torch.cat((action, pred1, pred2), 1).flatten()

class Metric_att_Action(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_states,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(Metric_att_Action, self).__init__()
        self.etypes = etypes
        self.name = "Metric_att_Action_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_feats + n_objects*4, n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        self.attention = nn.Linear(n_hidden + n_hidden, 1)
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, len(possibleActions))
        self.p1  = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.p2  = nn.Linear(n_hidden, n_hidden)
        self.p3  = nn.Linear(n_hidden, n_objects+1)
        self.q1  = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.q2  = nn.Linear(n_hidden, n_hidden)
        self.q3  = nn.Linear(n_hidden, n_objects+1+n_states)
        self.activation = nn.LeakyReLU()

    def forward(self, g, goalVec, goalObjectsVec):
        h = g.ndata['feat']
        edgeMatrices = [g.adjacency_matrix(etype=t) for t in self.etypes]
        edges = torch.cat(edgeMatrices, 1).to_dense()
        h = torch.cat((h, edges), 1)
        for i, layer in enumerate(self.layers):
            h = self.activation(layer(h))
        goalObjectsVec = self.activation(self.embed(torch.Tensor(goalObjectsVec)))
        attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1)], 1)
        attn_weights = F.softmax(self.attention(attn_embedding), dim=0)
        scene_embedding = torch.mm(attn_weights.t(), h)
        goal_embed = self.activation(self.embed(torch.Tensor(goalVec.reshape(1, -1))))
        final_to_decode = torch.cat([scene_embedding, goal_embed], 1)
        action = self.activation(self.fc1(final_to_decode))
        action = self.activation(self.fc2(action))
        action = self.activation(self.fc3(action))
        action = F.softmax(action, dim=1)
        pred1 = self.activation(self.p1(final_to_decode))
        pred1 = self.activation(self.p2(pred1))
        pred1 = F.softmax(self.activation(self.p3(pred1)), dim=1)
        pred2 = self.activation(self.q1(final_to_decode))
        pred2 = self.activation(self.q2(pred2))
        pred2 = F.softmax(self.activation(self.q3(pred2)), dim=1)
        return torch.cat((action, pred1, pred2), 1).flatten()