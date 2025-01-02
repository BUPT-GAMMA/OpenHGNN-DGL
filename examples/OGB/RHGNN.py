from openhgnn import Experiment
from openhgnn import NodeClassificationDataset
import openhgnn.utils as utils
import os
import argparse
import numpy
import torch
import dgl
import dgl.function as fn
from openhgnn.dataset import AsNodeClassificationDataset, generate_random_hg
from dgl import transforms as T
from dgl import DGLHeteroGraph
from dgl.data import DGLDataset
from ogb.nodeproppred import DglNodePropPredDataset

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class MyNCDataset(DGLDataset):
    def __init__(self, dataset_name, *args, **kwargs):
        super().__init__(dataset_name)
        dataset = DglNodePropPredDataset(name='ogbn-mag')
        self.category = 'paper'  # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
        self._meta_paths_dict = {'metapath':[
            ('author', 'writes', 'paper'),
            ('paper', 'has_topic', 'field_of_study'),
            ('field_of_study', 'has_topic_inv', 'paper'),
            ('paper', 'cites_inv', 'paper'),
            ('paper', 'writes_inv', 'author'),
            ('author', 'affiliated_with', 'institution'),
            ('institution', 'affiliated_with_inv', 'author'),
            ('author', 'writes', 'paper'),
            ('paper', 'cites', 'paper'),
            ('paper', 'writes_inv', 'author')
        ]}
        feature = {}
        split_idx = dataset.get_idx_split()
        self.num_classes = dataset.num_classes
        self.train_idx, self.valid_idx, self.test_idx = split_idx["train"][self.category], split_idx["valid"][
            self.category], split_idx["test"][self.category]
        
        self._g, self.label_dict = dataset[0]
        self._g = self.mag4HGT(self._g)
        self.label = self.label_dict[self.category].squeeze(dim=-1)
        num_nodes = self._g.num_nodes(self.category)
        train_mask = torch.zeros(num_nodes).bool()
        val_mask = torch.zeros(num_nodes).bool()
        test_mask = torch.zeros(num_nodes).bool()
        train_mask[self.train_idx] = True
        val_mask[self.valid_idx] = True
        test_mask[self.test_idx] = True
        self._g.nodes[self.category].data['train_mask'] = train_mask
        self._g.nodes[self.category].data['val_mask'] = val_mask
        self._g.nodes[self.category].data['test_mask'] = test_mask
        self._g.nodes[self.category].data['label'] = self.label
        feat = self._g.nodes[self.category].data['h']
        
        # 2-dim label
        self.in_dim = self._g.ndata['h'][self.category].shape[1]
        self.has_feature = True
        # emb = numpy.load('./openhgnn/output/Metapath2vec/ogbn-mag_mp2vec_embeddings.npy')
        # emb = torch.tensor(emb)
        # start = 0
        # for ntype in self._g.ntypes:
        #     end = start + self._g.num_nodes(ntype)
        #     if ntype == self.category:
        #         feature[ntype] = feat
        #     else:
        #         feature[ntype] = emb[start:end]
        #     start = end

        for ntype in self._g.ntypes:
            if ntype == self.category:
                feature[ntype] = feat
            else:
                emb = torch.load("./dataset/ogbn_mag/preprocess/{}.pt".format(ntype), map_location=torch.device('cpu')).float()
                feature[ntype] = emb
            
        self._g.ndata['h'] = feature

        # train_mask = utils.generate_mask_tensor(utils.idx2mask(self.train_idx, n))
        # val_mask = utils.generate_mask_tensor(utils.idx2mask(self.val_idx, n))
        # test_mask = utils.generate_mask_tensor(utils.idx2mask(self.test_idx, n))
        # pass

    def get_split(self, validation=True):
        return self.train_idx, self.valid_idx, self.test_idx

    def get_labels(self):
        return self.label

    def mag4HGT(self, hg):
        # Add reverse edge types

        edges = {etype: hg.edges(etype=etype) for etype in hg.canonical_etypes}
        edges.update({(v, e + '_inv', u): (dst, src) for (u, e, v), (src, dst) in edges.items()})
        hg2 = dgl.heterograph(edges)
        hg2 = dgl.to_simple(hg2)

        # Initialize year
        hg2.nodes['paper'].data['timestamp'] = hg.nodes['paper'].data['year'].squeeze()
        for ntype in hg.ntypes:
            if ntype != 'paper':
                hg2.nodes[ntype].data['timestamp'] = torch.zeros(hg2.num_nodes(ntype), dtype=torch.int64)

        # Aggregate bag-of-paper features
        hg2.nodes['paper'].data['h'] = hg.nodes['paper'].data['feat']
        hg2.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'), etype='has_topic')  # field_of_study
        hg2.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'), etype='writes_inv')  # author
        hg2.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'), etype='affiliated_with')  # institution

        # Attach log-degree to feature of each node type
        for ntype in hg2.ntypes:
            hg2.nodes[ntype].data['deg'] = torch.zeros(hg2.num_nodes(ntype))
        for utype, etype, vtype in hg2.canonical_etypes:
            hg2.nodes[vtype].data['deg'] += hg2.in_degrees(etype=etype)
        for ntype in hg2.ntypes:
            hg2.nodes[ntype].data['h'] = torch.cat([
                hg2.nodes[ntype].data['h'],
                torch.log10(hg2.nodes[ntype].data['deg'][:, None])], 1)
            del hg2.nodes[ntype].data['deg']

        return hg2
    # Some models require meta paths, you can set meta path dict for this dataset.
    @property
    def meta_paths_dict(self):
        return self._meta_paths_dict
    
    @property
    def name(self):
        return "ogbn-mag"

    def __getitem__(self, idx):
        return self._g

    def __len__(self):
        return 1

def search_space(trial):
    return {
        "lr": trial.suggest_categorical("lr", [1e-3, 5e-3, 1e-2]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [16, 32, 64]),
        "out_dim": trial.suggest_categorical("out_dim", [8, 16]),
        "weight_decay": trial.suggest_categorical("weight_decay", [1e-3, 1e-4, 1e-2, 1e-5]),
        "dropout": trial.suggest_uniform("dropout", 0.0, 0.5),
        "n_heads": trial.suggest_categorical("n_heads", [4, 8]),
        # 'n_layers': trial.suggest_int('n_layers', 2, 3)
        "negative_slope": trial.suggest_uniform("negative_slope", 0.0, 0.5),
        
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mini_batch_flag', action='store_false')
    parser.add_argument('--max_epoch', default='2000', type=int)
    parser.add_argument('--gpu', '-g', default='2', type=int, help='-1 means cpu')
    parser.add_argument('--batch_size', default='2000', type=int)
    parser.add_argument('--hidden_dim', default='64', type=int)
    parser.add_argument('--out_dim', default='16', type=int)
    parser.add_argument('--num_layers', default='2', type=int)
    parser.add_argument('--n_heads', default='8', type=int)
    parser.add_argument('--lr', default='0.0001', type=float)
    parser.add_argument('--weight_decay', default='0.00001', type=float)
    parser.add_argument('--dropout', default='0.02', type=float)
    parser.add_argument('--negative_slope', default='0.02', type=float)
    parser.add_argument('--fanout', default='20', type=int)
    parser.add_argument('--residual', action='store_false')
    parser.add_argument('--norm', action='store_false')
    parser.add_argument('--relation_hidden_units', default='8', type=int)
    parser.add_argument('--node_neighbors_min_num', default='10', type=int)
    args = parser.parse_args()

    myNCDataset = AsNodeClassificationDataset(MyNCDataset(dataset_name = "mag"), target_ntype='paper',
                                            force_reload=True)
    
    experiment = Experiment( model='RHGNN', dataset=myNCDataset,
                            task='node_classification', gpu = args.gpu, mini_batch_flag = args.mini_batch_flag, 
                            max_epoch=args.max_epoch, batch_size=args.batch_size, out_dim = args.out_dim, n_heads = args.n_heads,
                            hidden_dim=args.hidden_dim, n_layers=args.num_layers, lr=args.lr,
                            weight_decay=args.weight_decay, dropout=args.dropout, fanout=args.fanout,
                            residual = args.residual, norm = args.norm, relation_hidden_units = args.relation_hidden_units,
                            node_neighbors_min_num = args.node_neighbors_min_num
                            )
    experiment.run()
