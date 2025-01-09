import numpy as np
import pandas as pd
import os
import pickle
import dgl
from dgl import function as fn
from dgl.nn.functional import edge_softmax
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.utils import expand_as_pair
from operator import itemgetter
from . import BaseModel, register_model
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

'''
model
'''


@register_model('MAGNN')
class MAGNN(BaseModel):
    r"""
    This is the main method of model MAGNN

    Parameters
    ----------
    ntypes: list
        the nodes' types of the dataset
    h_feats: int
        hidden dimension
    inter_attn_feats: int
        the dimension of attention vector in inter-metapath aggregation
    num_heads: int
        the number of heads in intra metapath attention
    num_classes: int
        the number of output classes
    num_layers: int
        the number of hidden layers
    metapath_list: list
        the list of metapaths, e.g ['M-D-M', 'M-A-M', ...],
    edge_type_list: list
        the list of edge types, e.g ['M-A', 'A-M', 'M-D', 'D-M'],
    dropout_rate: float
        the dropout rate of feat dropout and attention dropout
    mp_instances : dict
        the metapath instances indices dict. e.g mp_instances['MAM'] stores MAM instances indices.
    encoder_type: str
        the type of encoder, e.g ['RotateE', 'Average', 'Linear']
    activation: callable activation function
        the activation function used in MAGNN.  default: F.elu

    Notes
    -----
    Please make sure that the please make sure that all the metapath is symmetric, e.g ['MDM', 'MAM' ...] are symmetric,
    while ['MAD', 'DAM', ...] are not symmetric.

    please make sure that the edge_type_list meets the following form:
    [edge_type_1, edge_type_1_reverse, edge_type_2, edge_type_2_reverse, ...], like the example above.

    All the activation in MAGNN are the same according to the codes of author.

    """
    @classmethod
    def build_model_from_args(cls, args, hg):
        ntypes = hg.ntypes
        dataset_list = ['imdb4MAGNN','dblp4MAGNN']
        if args.dataset in dataset_list:
            edge_type_list = hg.etypes
            metapath_list = []
            metapath_list = metapath_list + metapath_generator(ntypes,edge_type_list,5,2)
            metapath_list = metapath_list + metapath_generator(ntypes,edge_type_list,3,2)
            metapath_idx_dict = mp_instance_sampler(hg, metapath_list, 'imdb4MAGNN')
        # if args.dataset == 'imdb4MAGNN':
        #     # build model
        #     metapath_list = ['M-D-M', 'M-A-M', 'D-M-D', 'D-M-A-M-D', 'A-M-A', 'A-M-D-M-A']
        #     edge_type_list = hg.etypes
        #     # in_feats: {'n1type': n1_dim, 'n2type', n2_dim, ...}
        #     in_feats = {'M': 3066, 'D': 2081, 'A': 5257}
        #     metapath_idx_dict = mp_instance_sampler(hg, metapath_list, 'imdb4MAGNN')
        #
        # elif args.dataset == 'dblp4MAGNN':
        #     # build model
        #     metapath_list = ['A-P-A', 'A-P-T-P-A', 'A-P-V-P-A']
        #     edge_type_list = ['A-P', 'P-A', 'P-T', 'T-P', 'P-V', 'V-P']
        #     # in_feats: {'n1type': n1_dim, 'n2type', n2_dim, ...}
        #     in_feats = {'A': 334, 'P': 14328, 'T': 7723, 'V': 20}
        #     metapath_idx_dict = mp_instance_sampler(hg, metapath_list, 'dblp4MAGNN')

        else:
            raise NotImplementedError("MAGNN on dataset {} has not been implemented".format(args.dataset))

        return cls(ntypes=ntypes,
                   h_feats=args.hidden_dim // args.num_heads,
                   inter_attn_feats=args.inter_attn_feats,
                   num_heads=args.num_heads,
                   num_classes=args.out_dim,
                   num_layers=args.num_layers,
                   metapath_list=metapath_list,
                   edge_type_list=edge_type_list,
                   dropout_rate=args.dropout,
                   encoder_type=args.encoder_type,
                   metapath_idx_dict=metapath_idx_dict)

    def __init__(self, ntypes, h_feats, inter_attn_feats, num_heads, num_classes, num_layers,
                 metapath_list, edge_type_list, dropout_rate, metapath_idx_dict, encoder_type='RotateE',
                 activation=F.elu):
        super(MAGNN, self).__init__()

        self.encoder_type = encoder_type
        self.ntypes = ntypes
        self.h_feats = h_feats
        self.inter_attn_feats = inter_attn_feats
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.metapath_list = metapath_list
        self.edge_type_list = edge_type_list
        self.activation = activation
        self.backup = {}
        self.is_backup = False

        # input projection
        # self.ntypes = in_feats.keys()
        # self.input_projection = nn.ModuleDict()
        # for ntype in self.ntypes:
        #     self.input_projection[ntype] = nn.Linear(in_features=in_feats[ntype], out_features=h_feats * num_heads)

        # for layer in self.input_projection.values():
        #     nn.init.xavier_normal_(layer.weight, gain=1.414)

        # dropout
        self.feat_drop = nn.Dropout(p=dropout_rate)

        # extract ntypes that have corresponding metapath
        # If there're only metapaths like ['M-A-M', 'M-D-M'], 'A' and 'D' have no metapath, so that 'A' and 'D' shouldn't
        # be considered as nodes that need to aggregate information from metapath.
        self.dst_ntypes = set([metapath.split('-')[0] for metapath in metapath_list])

        # hidden layers
        self.layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.layers.append(
                MAGNN_layer(in_feats=h_feats, inter_attn_feats=inter_attn_feats, out_feats=h_feats, num_heads=num_heads,
                            metapath_list=metapath_list, ntypes=self.ntypes, edge_type_list=edge_type_list,
                            dst_ntypes=self.dst_ntypes, encoder_type=encoder_type, last_layer=False))

        # output layer
        self.layers.append(
            MAGNN_layer(in_feats=h_feats, inter_attn_feats=inter_attn_feats, out_feats=num_classes, num_heads=num_heads,
                        metapath_list=metapath_list, ntypes=self.ntypes, edge_type_list=edge_type_list,
                        dst_ntypes=self.dst_ntypes, encoder_type=encoder_type, last_layer=True))

        self.metapath_idx_dict = metapath_idx_dict

    def mini_reset_params(self, new_metapth_idx_dict):
        '''
        This method is utilized for reset some parameters including metapath_idx_dict, metapath_list, dst_ntypes...
        Other Parameters like weight matrix don't need to be updated.

        '''
        if not self.is_backup:  # the params of the original graph has not been stored
            self.backup['metapath_idx_dict'] = self.metapath_idx_dict
            self.backup['metapath_list'] = self.metapath_list
            self.backup['dst_ntypes'] = self.dst_ntypes
            self.is_backup = True

        self.metapath_idx_dict = new_metapth_idx_dict
        self.metapath_list = list(new_metapth_idx_dict.keys())
        self.dst_ntypes = set([meta[0] for meta in self.metapath_list])

        for layer in self.layers:
            layer.metapath_list = self.metapath_list
            layer.dst_ntypes = self.dst_ntypes

    def restore_params(self):
        assert self.backup, 'The model.backup is empty'
        self.metapath_idx_dict = self.backup['metapath_idx_dict']
        self.metapath_list = self.backup['metapath_list']
        self.dst_ntypes = self.backup['dst_ntypes']

        for layer in self.layers:
            layer.metapath_list = self.metapath_list
            layer.dst_ntypes = self.dst_ntypes

    def forward(self, g, feat_dict=None):
        r"""
        The forward part of MAGNN

        Parameters
        ----------
        g : object
            the dgl heterogeneous graph
        feat_dict : dict
            the feature matrix dict of different node types, e.g {'M':feat_of_M, 'D':feat_of_D, ...}

        -------
        Returns
        dict
            The predicted logit after the output projection. e.g For the predicted node type, such as M(movie),
            dict['M'] contains the probability that each node is classified as each class. For other node types, such as
            D(director), dict['D'] contains the result after the output projection.

        dict
            The embeddings before the output projection. e.g dict['M'] contains embeddings of every node of M type.
        """

        # hidden layer
        for i in range(self.num_layers - 1):
            h, _ = self.layers[i](feat_dict, self.metapath_idx_dict)
            for key in h.keys():
                h[key] = self.activation(h[key])

        # output layer
        h_output, embedding = self.layers[-1](feat_dict, self.metapath_idx_dict)

        # return h_output, embedding
        return h_output


class MAGNN_layer(nn.Module):
    def __init__(self, in_feats, inter_attn_feats, out_feats, num_heads, metapath_list,
                 ntypes, edge_type_list, dst_ntypes, encoder_type='RotateE', last_layer=False):
        super(MAGNN_layer, self).__init__()
        self.in_feats = in_feats
        self.inter_attn_feats = inter_attn_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.metapath_list = metapath_list  # ['M-D-M', 'M-A-M', ...]
        self.ntypes = ntypes  # ['M', 'D', 'A']
        self.edge_type_list = edge_type_list  # ['M-A', 'A-M', ...]
        self.dst_ntypes = dst_ntypes
        self.encoder_type = encoder_type
        self.last_layer = last_layer

        # in_feats_dst_meta = (feature dimension of dst nodes,
        #                      feature dimension of metapath instances after encoding)
        in_feats_dst_meta = tuple((in_feats, in_feats))

        self.intra_attn_layers = nn.ModuleDict()
        for metapath in self.metapath_list:
            self.intra_attn_layers[metapath] = \
                MAGNN_attn_intra(in_feats=in_feats_dst_meta, out_feats=in_feats, num_heads=num_heads)

        # The linear transformation at the beginning of inter metapath aggregation, including all metapath
        # The attention mechanism in inter metapath aggregation
        self.inter_linear = nn.ModuleDict()
        self.inter_attn_vec = nn.ModuleDict()
        for ntype in dst_ntypes:
            self.inter_linear[ntype] = \
                nn.Linear(in_features=in_feats * num_heads, out_features=inter_attn_feats, bias=True)
            self.inter_attn_vec[ntype] = nn.Linear(in_features=inter_attn_feats, out_features=1, bias=False)
            nn.init.xavier_normal_(self.inter_linear[ntype].weight, gain=1.414)
            nn.init.xavier_normal_(self.inter_attn_vec[ntype].weight, gain=1.414)

        # Some initialization related to encoder
        if encoder_type == 'RotateE':
            # r_vec: [r1, r1_inverse, r2, r2_inverse, ...., rn, rn_inverse]
            r_vec_ = nn.Parameter(th.empty(size=(len(edge_type_list) // 2, in_feats * num_heads // 2, 2)))
            nn.init.xavier_normal_(r_vec_.data, gain=1.414)
            self.r_vec = F.normalize(r_vec_, p=2, dim=2)
            self.r_vec = th.stack([self.r_vec, self.r_vec], dim=1)
            self.r_vec[:, 1, :, 1] = -self.r_vec[:, 1, :, 1]
            self.r_vec = self.r_vec.reshape(r_vec_.shape[0] * 2, r_vec_.shape[1], 2)
            self.r_vec_dict = nn.ParameterDict()
            for i, edge_type in zip(range(len(edge_type_list)), edge_type_list):
                self.r_vec_dict[edge_type] = nn.Parameter(self.r_vec[i])

        # The dimension here does not change because multi-heads conversion has been done before
        # This part is a little different from the original author's codes.
        elif encoder_type == 'Linear':
            self.encoder_linear = \
                nn.Linear(in_features=in_feats * num_heads, out_features=in_feats * num_heads)

        # output layer
        if last_layer:
            self._output_projection = nn.Linear(in_features=num_heads * in_feats, out_features=out_feats)
        else:
            self._output_projection = nn.Linear(in_features=num_heads * in_feats, out_features=num_heads * out_feats)
        nn.init.xavier_normal_(self._output_projection.weight, gain=1.414)

    def forward(self, feat_dict, metapath_idx_dict):
        # Intra-metapath latent transformation
        feat_intra = {}
        for _metapath in self.metapath_list:
            feat_intra[_metapath] = \
                self.intra_metapath_trans(feat_dict, metapath=_metapath, metapath_idx_dict=metapath_idx_dict)

        # Inter-metapath latent transformation
        feat_inter = \
            self.inter_metapath_trans(feat_dict=feat_dict, feat_intra=feat_intra, metapath_list=self.metapath_list)

        # output projection
        feat_final = self.output_projection(feat_inter=feat_inter)

        # return final features after output projection (without nonlinear activation) and embedding
        # nonlinear activation will be added in MAGNN
        return feat_final, feat_inter

    def intra_metapath_trans(self, feat_dict, metapath, metapath_idx_dict):

        metapath_idx = metapath_idx_dict[metapath]

        # encoder metapath instances
        # intra_metapath_feat: feature matrix of every metapath instance of param metapath
        intra_metapath_feat = self.encoder(feat_dict, metapath, metapath_idx)

        # aggregate metapath instances into metapath using ATTENTION
        feat_intra = \
            self.intra_attn_layers[metapath]([intra_metapath_feat, feat_dict[metapath.split('-')[0]]],
                                             metapath, metapath_idx)
        return feat_intra

    def inter_metapath_trans(self, feat_dict, feat_intra, metapath_list):
        meta_s = {}
        feat_inter = {}
        # construct spi, where pi = ['M-A-M', 'M-D-M', ...]
        for metapath in metapath_list:
            _metapath = metapath.split('-')
            meta_feat = feat_intra[metapath]
            meta_feat = th.tanh(self.inter_linear[_metapath[0]](meta_feat)).mean(dim=0)  # s_pi
            meta_s[metapath] = self.inter_attn_vec[_metapath[0]](meta_feat)  # e_pi

        for ntype in self.ntypes:
            if ntype in self.dst_ntypes:
                # extract the metapath with the dst node type of ntype to construct a tensor
                # in order to compute softmax
                # metapaths: e.g if ntype is M, then ['M-A-M', 'M-D-M']
                metapaths = np.array(metapath_list)[[meta.split('-')[0] == ntype for meta in metapath_list]]
                # extract the e_pi of metapaths
                # e.g the e_pi of ['M-A-M', 'M-D-M'] if ntype is M
                meta_b = th.tensor(itemgetter(*metapaths)(meta_s))
                # compute softmax, obtain b_pi, which is attention score of metapaths
                # e.g the b_pi of ['M-A-M', 'M-D-M'] if ntype is M
                meta_b = F.softmax(meta_b, dim=0)
                # extract corresbonding features of metapath
                # e.g ['MDM_feat_attn', 'MAM_feat_attn'] if ntype is M
                meta_feat = itemgetter(*metapaths)(feat_intra)
                # compute the embedding feature of nodes
                feat_inter[ntype] = th.stack([meta_b[i] * meta_feat[i] for i in range(len(meta_b))], dim=0).sum(dim=0)
            else:
                feat_inter[ntype] = feat_dict[ntype]
        return feat_inter

    def encoder(self, feat_dict, metapath, metapath_idx):
        _metapath = metapath.split('-')
        device = feat_dict[_metapath[0]].device
        feat = th.zeros((len(_metapath), metapath_idx.shape[0], feat_dict[_metapath[0]].shape[1]), device=device)
        for i, ntype in zip(range(len(_metapath)), _metapath):
            feat[i] = feat_dict[ntype][metapath_idx[:, i]]
        feat = feat.reshape(feat.shape[0], feat.shape[1], feat.shape[2] // 2, 2)

        if self.encoder_type == 'RotateE':
            temp_r_vec = th.zeros((len(_metapath), feat.shape[-2], 2), device=device)
            temp_r_vec[0, :, 0] = 1

            for i in range(1, len(_metapath), 1):
                edge_type = '{}-{}'.format(_metapath[i - 1], _metapath[i])
                temp_r_vec[i] = self.complex_hada(temp_r_vec[i - 1], self.r_vec_dict[edge_type])
                feat[i] = self.complex_hada(feat[i], temp_r_vec[i], opt='feat')

            feat = feat.reshape(feat.shape[0], feat.shape[1], -1)
            return th.mean(feat, dim=0)

        elif self.encoder_type == 'Linear':
            feat = feat.reshape(feat.shape[0], feat.shape[1], -1)
            feat = self.encoder_linear(th.mean(feat, dim=0))
            return feat

        elif self.encoder_type == 'Average':
            feat = feat.reshape(feat.shape[0], feat.shape[1], -1)
            return th.mean(feat, dim=0)

        else:
            raise ValueError("The encoder type {} has not been implemented yet.".format(self.encoder_type))

    @staticmethod
    def complex_hada(h, v, opt='r_vec'):
        if opt == 'r_vec':
            h_h, l_h = h[:, 0].clone(), h[:, 1].clone()
        else:
            h_h, l_h = h[:, :, 0].clone(), h[:, :, 1].clone()
        h_v, l_v = v[:, 0].clone(), v[:, 1].clone()
        res = th.zeros_like(h)

        if opt == 'r_vec':
            res[:, 0] = h_h * h_v - l_h * l_v
            res[:, 1] = h_h * l_v + l_h * h_v
        else:
            res[:, :, 0] = h_h * h_v - l_h * l_v
            res[:, :, 1] = h_h * l_v + l_h * h_v
        return res

    # def output_projection(self, g):
    def output_projection(self, feat_inter):
        feat_final = {}
        for ntype in self.ntypes:
            feat_final[ntype] = self._output_projection(feat_inter[ntype])
        return feat_final


class MAGNN_attn_intra(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, feat_drop=0.5, attn_drop=0.5, negative_slope=0.01,
                 activation=F.elu):
        super(MAGNN_attn_intra, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats

        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        nn.init.xavier_normal_(self.attn_r, gain=1.414)

    def forward(self, feat, metapath, metapath_idx):
        _metapath = metapath.split('-')
        device = feat[0].device
        h_meta = self.feat_drop(feat[0]).view(-1, self._num_heads,
                                              self._out_feats)  # feature matrix of metapath instances

        # metapath(right) part of attention
        er = (h_meta * self.attn_r).sum(dim=-1).unsqueeze(-1)

        graph_data = {
            ('meta_inst', 'meta2{}'.format(_metapath[0]), _metapath[0]): (th.arange(0, metapath_idx.shape[0]),
                                                                          th.tensor(metapath_idx[:, 0]),)
        }
        num_nodes_dict = {'meta_inst': metapath_idx.shape[0], _metapath[0]: feat[1].shape[0]}

        g_meta = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict).to(device)

        # feature vector of metapath instances and nodes
        g_meta.nodes['meta_inst'].data.update({'feat_src': h_meta, 'er': er})
        # g_meta.nodes[metapath[0]].data.update({'feat':feat[1]})

        # compute attention without concat with hv
        g_meta.apply_edges(func=fn.copy_u('er', 'e'), etype='meta2{}'.format(_metapath[0]))

        e = self.leaky_relu(g_meta.edata.pop('e'))
        g_meta.edata['a'] = self.attn_drop(edge_softmax(g_meta, e))

        # message passing, there's only one edge type
        # by default DGL would fill nodes without in-degree with zero
        g_meta.update_all(message_func=fn.u_mul_e('feat_src', 'a', 'm'), reduce_func=fn.sum('m', 'feat'))

        feat = self.activation(g_meta.dstdata['feat'])

        # return dst nodes' features after attention
        return feat.flatten(1)


'''
methods
'''


def mp_instance_sampler(g, metapath_list, dataset):
    """
    Sampling the indices of all metapath instances in g according to the metapath list

    Parameters
    ----------
    g : object
        the dgl heterogeneous graph
    metapath_list : list
        the list of metapaths in g, e.g. ['M-A-M', M-D-M', ...]
    dataset : str
        the name of dataset, e.g. 'imdb4MAGNN'

    Returns
    -------
    dict
        the indices of all metapath instances. e.g dict['MAM'] contains the indices of all MAM instances

    Notes
    -----
    Please make sure that the metapath in metapath_list are all symmetric

    We'd store the metapath instances in the disk after one metapath instances sampling and next time the
    metapath instances will be extracted directly from the disk if they exists.

    """

    file_dir = 'openhgnn/output/MAGNN/'
    file_addr = file_dir + '{}'.format(dataset) + '_mp_inst.pkl'
    test = True  # TODO

    if os.path.exists(file_addr) and test is False:  # TODO
        with open(file_addr, 'rb') as file:
            res = pickle.load(file)
    else:
        etype_idx_dict = {}
        for etype in g.etypes:
            edges_idx_i = g.edges(etype=etype)[0].cpu().numpy()
            edges_idx_j = g.edges(etype=etype)[1].cpu().numpy()
            etype_idx_dict[etype] = pd.DataFrame([edges_idx_i, edges_idx_j]).T
            _etype = etype.split('-')
            etype_idx_dict[etype].columns = [_etype[0], _etype[1]]

        res = {}
        for metapath in metapath_list:
            res[metapath] = None
            _metapath = metapath.split('-')
            for i in range(1, len(_metapath) - 1):
                if i == 1:
                    res[metapath] = etype_idx_dict['-'.join(_metapath[:i + 1])]
                feat_j = etype_idx_dict['-'.join(_metapath[i:i + 2])]
                col_i = res[metapath].columns[-1]
                col_j = feat_j.columns[0]
                res[metapath] = pd.merge(res[metapath], feat_j,
                                         left_on=col_i,
                                         right_on=col_j,
                                         how='inner')
                if col_i != col_j:
                    res[metapath].drop(columns=col_j, inplace=True)
            res[metapath] = res[metapath].values

        with open(file_addr, 'wb') as file:
            pickle.dump(res, file)

    return res


def mini_mp_instance_sampler(seed_nodes, mp_instances, num_samples):
    '''
    Sampling metapath instances with seed_nodes as dst nodes. This method is exclusive to mini batch train/validate/test
    which need to sample subsets of metapath instances of the whole graph.

    Parameters
    ----------
    seed_nodes : dict
        sampling metapath instances based on seed_nodes. e.g. {'A':[0, 1, 2], 'M':[0, 1, 2], ...}, then we'll sample
        metapath instances with 0 or 1 or 2 as dst_nodes of type 'A' and type 'B'.
    mp_instances : list
        the sampled metapath instances of the whole graph. It should be the return value of method
        ``mp_instance_sampler(g, metapath_list, dataset)``
    num_samples : int
        the maximal number of sampled metapath instances of each metapath type.

    Returns
    -------
    dict
        sampled metapath instances

    '''
    mini_mp_inst = {}
    metapath_list = list(mp_instances.keys())

    for ntype in seed_nodes.keys():
        target_mp_types = np.array(metapath_list)[[meta.split('-')[0] == ntype for meta in metapath_list]]
        for metapath in target_mp_types:  # the metapath instances of the certain metapath
            for node in seed_nodes[ntype]:
                _mp_inst = mp_instances[metapath][mp_instances[metapath][:, 0] == node]
                dst_nodes, dst_counts = np.unique(_mp_inst[:, -1], return_counts=True)

                # the method of computing sampling probabilities originates from author's codes
                p = np.repeat((dst_counts ** (3 / 4)) / dst_counts, dst_counts)
                p = p / p.sum()

                _num_samples = min(num_samples, len(p))
                mp_choice = np.random.choice(len(p), _num_samples, replace=False, p=p)
                if metapath not in mini_mp_inst.keys():
                    mini_mp_inst[metapath] = _mp_inst[mp_choice]
                else:
                    mini_mp_inst[metapath] = np.concatenate((mini_mp_inst[metapath], _mp_inst[mp_choice]),
                                                            axis=0)

    return mini_mp_inst


def svm_test(X, y, test_sizes=(0.2, 0.4, 0.6, 0.8), repeat=10):
    # This method is implemented by author
    random_states = [182318 + i for i in range(repeat)]
    result_macro_f1_list = []
    result_micro_f1_list = []
    for test_size in test_sizes:
        macro_f1_list = []
        micro_f1_list = []
        for i in range(repeat):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=random_states[i])
            svm = LinearSVC(dual=False)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            macro_f1_list.append(macro_f1)
            micro_f1_list.append(micro_f1)
        result_macro_f1_list.append((np.mean(macro_f1_list), np.std(macro_f1_list)))
        result_micro_f1_list.append((np.mean(micro_f1_list), np.std(micro_f1_list)))
    return result_macro_f1_list, result_micro_f1_list


def metapath_generator(ntypes,etypes,length,num):
    ntypes_num = len(ntypes)
    metapath_list = []
    for i in range(num):
        temp = ""
        random_index = random.randrange(ntypes_num)
        temp = temp + ntypes[random_index]
        for j in range(length-1):
            temp = temp + '-'
            random_index = random.randrange(ntypes_num)
            temp_e = temp[-2] + '-' + ntypes[random_index]
            while temp_e not in etypes:
                random_index = random.randrange(ntypes_num)
                temp_e = temp[-2] + '-' + ntypes[random_index]
            temp = temp + ntypes[random_index]
        if temp not in metapath_list:
            metapath_list.append(temp)
    return metapath_list
