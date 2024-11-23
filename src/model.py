import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.utils import k_hop_subgraph
import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from utils import try_gpu

EPS = 1e-15

class GCNLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, dropout=0.):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        input = F.dropout(x, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class VGAE(nn.Module):
    """
    The self-supervised module of DeepDSI
    """
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(VGAE, self).__init__()
        #self.gc1 = GCNLayer(input_feat_dim, hidden_dim1, dropout)   # F.relu
        #self.gc2 = GCNLayer(hidden_dim1, hidden_dim2, dropout)    # lambda x: x
        #$self.gc3 = GCNLayer(hidden_dim1, hidden_dim2, dropout)
        self.gc1= GraphAttentionLayer(input_feat_dim,hidden_dim1, dropout)
        self.gc2= GraphAttentionLayer(hidden_dim1,hidden_dim2, dropout)
        self.gc3= GraphAttentionLayer(hidden_dim1,hidden_dim2, dropout)
        self.act1 = nn.ReLU()

    def encode(self, x, adj):
        hidden1 = self.act1(self.gc1(x, adj))
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        adj_hat = torch.mm(z, z.t())
        return adj_hat

    def forward(self, x, adj, sigmoid: bool = True):
        mu, logstd = self.encode(x, adj)
        z = self.reparameterize(mu, logstd)
        return (torch.sigmoid(self.decode(z)), z, mu, logstd) if sigmoid else (self.decode(z), z, mu, logstd)


class DSIPredictor(nn.Module):
    """
    The semi-supervised module of DeepDSI
    """
    def __init__(self, in_features, out_features):
        super(DSIPredictor, self).__init__()

       # self.gc1 = GCNLayer(343, 343, 0.1)
        # self.gc2 = GCNLayer(343, 343, 0.1)
        self.gc1=GraphAttentionLayer(343,343,0.2)
        #self.gc2=GraphAttentionLayer(343,343,0.2)

        self.fc1 = nn.Linear(in_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, out_features)
        self.dropout = nn.Dropout(p=0.4)
        self.act1 = nn.ReLU()
        self.act2 = nn.Mish()

    def forward(self, x, adj, pro1_index, pro2_index, sigmoid: bool = True):
        x1 = self.act1(self.gc1(x, adj))
        #x2 = self.gc2(x1, adj)

        pro1 = x1[pro1_index]
        pro2 = x1[pro2_index]

        dsi = torch.cat([pro1, pro2], dim = 1)

        h1 = self.dropout(self.bn1(self.act2(self.fc1(dsi))))
        h2 = self.dropout(self.bn2(self.act2(self.fc2(h1))))
        h3 = self.dropout(self.bn3(self.act2(self.fc3(h2))))
        h4 = self.fc4(h3)

        return torch.sigmoid(h4) if sigmoid else h4


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Linear(20398, 343)
        self.decoder = nn.Linear(343, 20398)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Classifier(nn.Module):
    # __constants__ = ['in_features', 'out_features']
    def __init__(self, in_features, out_features):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(in_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, out_features)
        self.dropout = nn.Dropout(p=0.4)
        self.act = nn.Mish()

    def forward(self, x, pro1_index, pro2_index):
        pro1 = x[pro1_index]
        pro2 = x[pro2_index]

        x = torch.cat([pro1, pro2], dim = 1)

        x = self.dropout(self.bn1(self.act(self.fc1(x))))
        x = self.dropout(self.bn2(self.act(self.fc2(x))))
        x = self.dropout(self.bn3(self.act(self.fc3(x))))
        x = torch.sigmoid(self.fc4(x))

        return x


class PairExplainer(torch.nn.Module):
    """The explainable module of DeepDSI

    Args:
        model (torch.nn.Module): The module to importance.
        lr (float, optional): The learning rate to apply.
        num_hops (int, optional): The number of hops the 'model' is aggregating information from.
        feat_mask_obj (str, optional): Denotes the object of feature mask that will be learned.
        log (bool, optional): Choose whether to log learning progress.
        **kwargs (optional): Additional hyper-parameters to override default settings of the 'coeffs'.
    """

    coeffs = {
        'feat_size': 1.0,
        'feat_reduction': 'mean',
        'feat_ent': 0.5,   # 0.1
    }

    def __init__(self, model, lr: float = 0.01,
                 num_hops: int = 2,
                 feat_mask_obj: str = 'dsi',
                 log: bool = True, **kwargs):
        super().__init__()
        assert feat_mask_obj in ['dub', 'sub', 'dsi']
        self.model = model
        self.lr = lr
        self.num_hops = num_hops
        self.feat_mask_obj = feat_mask_obj
        self.log = log
        self.coeffs.update(kwargs)
        self.device = next(model.parameters()).device

    def __set_masks__(self, num_feat):
        std = 0.1
        if self.feat_mask_obj == 'dsi':
            self.feat_mask = torch.nn.Parameter(torch.randn(2, num_feat) * std)
        else:
            self.feat_mask = torch.nn.Parameter(torch.randn(1, num_feat) * std)

    def __clear_masks__(self):
        self.feat_masks = None

    def __subgraph__(self, node_idx, x, edge_index):
        # covert the type of edge_index
        edge_index = edge_index.cpu().detach().to_dense()
        tmp_coo = sp.coo_matrix(edge_index)
        edge_index = np.vstack((tmp_coo.row, tmp_coo.col))
        edge_index = torch.LongTensor(edge_index)

        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, self.num_hops, edge_index, relabel_nodes=True,
            num_nodes=x.size(0), flow='source_to_target')

        return subset

    def __loss__(self, log_logits, pred_label):
        loss1 = torch.cdist(log_logits, pred_label)
        m = self.feat_mask.sigmoid()
        node_feat_reduce = getattr(torch, self.coeffs['feat_reduction'])
        loss2 = self.coeffs['feat_size'] * node_feat_reduce(m)
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss3 =  self.coeffs['feat_ent'] * ent.mean()

        return loss1 + loss2 + loss3

    def explain(self, x, adj, pro1_index, pro2_index, epochs: int = 100):
        """Learn and return a feature mask that explains the importance of each dimension of the feature

        Args:
            x (Tensor): The node feature matrix.
            adj (Tensor): The adjacency matrix.
            pro1_index (int): The protein1 to importance.
            pro2_index (int): The protein2 to importance.
            epochs (int, optional): The number of epochs to train.

        rtype: (Tensor)
        """

        self.model.eval()
        self.__clear_masks__()

        # 1. Get the subgraphs.
        subset1 = self.__subgraph__(pro1_index, x, adj)
        subset2 = self.__subgraph__(pro2_index, x, adj)

        # 2. Get the initial prediction.
        with torch.no_grad():
            pred_label = self.model(x, adj, [pro1_index], [pro2_index])

        # 3. Initialize the weight
        self.__set_masks__(x.size(1))
        self.to(self.device)


        parameters = [self.feat_mask]
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        if self.log:
            pbar = tqdm(total=epochs)
            pbar.set_description(f'importance this pair of DSI')

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()

            # 4. Use the weight
            h = x.clone()
            if self.feat_mask_obj == 'dub':
                h[subset1] = h[subset1].clone() * self.feat_mask.sigmoid()
            if self.feat_mask_obj == 'sub':
                h[subset2] = h[subset2].clone() * self.feat_mask.sigmoid()
            if self.feat_mask_obj == 'dsi':
                h[subset1] = h[subset1].clone() * self.feat_mask[0].sigmoid()
                h[subset2] = h[subset2].clone() * self.feat_mask[1].sigmoid()

            log_logits = self.model(h, adj, [pro1_index], [pro2_index])

            loss = self.__loss__(log_logits, pred_label)
            loss.backward()
            optimizer.step()

            if self.log:
                pbar.update(1)

        if self.log:
            pbar.close()

        feat_mask = self.feat_mask.detach().sigmoid().cpu()
        self.__clear_masks__()

        return feat_mask



class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features,dropout=0.1, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.weight = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.weight.data)
        print(self.weight)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        #mps_device = torch.device("mps")
        Wh = torch.mm(h, self.weight.to(device=try_gpu())) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        #print(adj.size())
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        #mps_device = torch.device("mps")
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :].to(device=try_gpu()))
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :].to(device=try_gpu()))
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
      
# class GATLayer(torch.nn.Module):
#     """
#     Base class for all implementations as there is much code that would otherwise be copy/pasted.

#     """

#     head_dim = 1

#     def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
#                  dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

#         super().__init__()

#         # Saving these as we'll need them in forward propagation in children layers (imp1/2/3)
#         self.num_of_heads = num_of_heads
#         self.num_out_features = num_out_features
#         self.concat = concat  # whether we should concatenate or average the attention heads
#         self.add_skip_connection = add_skip_connection

#         #
#         # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
#         # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
#         #

        
       
#             # You can treat this one matrix as num_of_heads independent W matrices
#         self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

#         # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
#         # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.

#         # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
#         # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
#         self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
#         self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

#         # if layer_type == layer_type.IMP1:  # simple reshape in the case of implementation 1
#         #     self.scoring_fn_target = nn.Parameter(self.scoring_fn_target.reshape(num_of_heads, num_out_features, 1))
#         #     self.scoring_fn_source = nn.Parameter(self.scoring_fn_source.reshape(num_of_heads, num_out_features, 1))

#         # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
#         if bias and concat:
#             self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
#         elif bias and not concat:
#             self.bias = nn.Parameter(torch.Tensor(num_out_features))
#         else:
#             self.register_parameter('bias', None)

#         if add_skip_connection:
#             self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
#         else:
#             self.register_parameter('skip_proj', None)

#         #
#         # End of trainable weights
#         #

#         self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
#         self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the log-softmax along the last dimension
#         self.activation = activation
#         # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
#         # and for attention coefficients. Functionality-wise it's the same as using independent modules.
#         self.dropout = nn.Dropout(p=dropout_prob)

#         self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
#         self.attention_weights = None  # for later visualization purposes, I cache the weights here

#         #self.init_params(layer_type)

#     def init_params(self, layer_type):
#         """
#         The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
#             https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

#         The original repo was developed in TensorFlow (TF) and they used the default initialization.
#         Feel free to experiment - there may be better initializations depending on your problem.

#         """
#         nn.init.xavier_uniform_(self.proj_param if layer_type == layer_type.IMP1 else self.linear_proj.weight)
#         nn.init.xavier_uniform_(self.scoring_fn_target)
#         nn.init.xavier_uniform_(self.scoring_fn_source)

#         if self.bias is not None:
#             torch.nn.init.zeros_(self.bias)

#     def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
#         if self.log_attention_weights:  # potentially log for later visualization in playground.py
#             self.attention_weights = attention_coefficients

#         # if the tensor is not contiguously stored in memory we'll get an error after we try to do certain ops like view
#         # only imp1 will enter this one
#         if not out_nodes_features.is_contiguous():
#             out_nodes_features = out_nodes_features.contiguous()

#         if self.add_skip_connection:  # add skip or residual connection
#             if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
#                 # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
#                 # thus we're basically copying input vectors NH times and adding to processed vectors
#                 out_nodes_features += in_nodes_features.unsqueeze(1)
#             else:
#                 # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
#                 # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
#                 out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

#         if self.concat:
#             # shape = (N, NH, FOUT) -> (N, NH*FOUT)
#             out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
#         else:
#             # shape = (N, NH, FOUT) -> (N, FOUT)
#             out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

#         if self.bias is not None:
#             out_nodes_features += self.bias

#         return out_nodes_features if self.activation is None else self.activation(out_nodes_features)
# class GATLayerImp3(GATLayer):
#     """
#     Implementation #3 was inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric

#     But, it's hopefully much more readable! (and of similar performance)

#     It's suitable for both transductive and inductive settings. In the inductive setting we just merge the graphs
#     into a single graph with multiple components and this layer is agnostic to that fact! <3

#     """

#     src_nodes_dim = 0  # position of source nodes in edge index
#     trg_nodes_dim = 1  # position of target nodes in edge index

#     nodes_dim = 0      # node dimension/axis
#     head_dim = 1       # attention head dimension/axis

#     def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
#                  dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

#         # Delegate initialization to the base class
#         super().__init__(num_in_features, num_out_features, num_of_heads, concat, activation, dropout_prob,
#                       add_skip_connection, bias, log_attention_weights)

#     def forward(self, data):
#         #
#         # Step 1: Linear Projection + regularization
#         #

#         in_nodes_features, edge_index = data  # unpack data
#         num_of_nodes = in_nodes_features.shape[self.nodes_dim]
#         assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

#         # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
#         # We apply the dropout to all of the input node features (as mentioned in the paper)
#         # Note: for Cora features are already super sparse so it's questionable how much this actually helps
#         in_nodes_features = self.dropout(in_nodes_features)

#         # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
#         # We project the input node features into NH independent output features (one for each attention head)
#         nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

#         nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

#         #
#         # Step 2: Edge attention calculation
#         #

#         # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
#         # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
#         # Optimization note: torch.sum() is as performant as .sum() in my experiments
#         scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
#         scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

#         # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
#         # the possible combinations of scores we just prepare those that will actually be used and those are defined
#         # by the edge index.
#         # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
#         scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
#         scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

#         # shape = (E, NH, 1)
#         attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
#         # Add stochasticity to neighborhood aggregation
#         attentions_per_edge = self.dropout(attentions_per_edge)

#         #
#         # Step 3: Neighborhood aggregation
#         #

#         # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
#         # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
#         nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

#         # This part sums up weighted and projected neighborhood feature vectors for every target node
#         # shape = (N, NH, FOUT)
#         out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)

#         #
#         # Step 4: Residual/skip connections, concat and bias
#         #

#         out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
#         return (out_nodes_features, edge_index)

#     #
#     # Helper functions (without comments there is very little code so don't be scared!)
#     #

#     def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
#         """
#         As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
#         Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
#         into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
#         in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
#         (where 1-3 is overloaded notation it represents the edge 1-3 and it's (exp) score) and similarly for 2-3 and 3-3
#          i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.

#         Note:
#         Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
#         and it's a fairly common "trick" used in pretty much every deep learning framework.
#         Check out this link for more details:

#         https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning

#         """
#         # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
#         scores_per_edge = scores_per_edge - scores_per_edge.max()
#         exp_scores_per_edge = scores_per_edge.exp()  # softmax

#         # Calculate the denominator. shape = (E, NH)
#         neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

#         # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
#         # possibility of the computer rounding a very small number all the way to 0.
#         attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

#         # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
#         return attentions_per_edge.unsqueeze(-1)

#     def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
#         # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
#         trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

#         # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
#         size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
#         size[self.nodes_dim] = num_of_nodes
#         neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

#         # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
#         # target index)
#         neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

#         # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
#         # all the locations where the source nodes pointed to i (as dictated by the target index)
#         # shape = (N, NH) -> (E, NH)
#         return neighborhood_sums.index_select(self.nodes_dim, trg_index)

#     def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
#         size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
#         size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
#         out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

#         # shape = (E) -> (E, NH, FOUT)
#         trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
#         # aggregation step - we accumulate projected, weighted node features for all the attention heads
#         # shape = (E, NH, FOUT) -> (N, NH, FOUT)
#         out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

#         return out_nodes_features

#     def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
#         """
#         Lifts i.e. duplicates certain vectors depending on the edge index.
#         One of the tensor dims goes from N -> E (that's where the "lift" comes from).

#         """
#         src_nodes_index = edge_index[self.src_nodes_dim]
#         trg_nodes_index = edge_index[self.trg_nodes_dim]

#         # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
#         scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
#         scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
#         nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

#         return scores_source, scores_target, nodes_features_matrix_proj_lifted

#     def explicit_broadcast(self, this, other):
#         # Append singleton dimensions until this.dim() == other.dim()
#         for _ in range(this.dim(), other.dim()):
#             this = this.unsqueeze(-1)

#         # Explicitly expand so that shapes are the same
#         return this.expand_as(other)