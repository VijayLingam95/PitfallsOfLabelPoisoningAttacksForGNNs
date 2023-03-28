import pickle as pkl
import os.path as osp

import numpy as np
import scipy.sparse as sp

import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


class Dataset(): 

    def __init__(self, name, split=0, setting='cv', adj_type='sym_adj', transform=None, pre_transform=None, random_train_val=True):
        self.name = name.lower()
        self.split = split
        self.pre_transform = pre_transform
        
        if random_train_val:
            self.data_dir = "./datasets/small_val_random" 
        else:
            self.data_dir = "./datasets"

        self.dataset_data = pkl.load(open(osp.join(self.data_dir, self.name + ".pkl"), 'rb'))
        self.adj = self.dataset_data[adj_type]

        self.edges = torch.tensor(self.adj.nonzero(), dtype=torch.long)

        self.features = torch.tensor(self.dataset_data['X'], dtype=torch.float)
        self.labels = torch.tensor(self.dataset_data['labels'].argmax(1), dtype=torch.long)
        self.data = Data(x=self.features, edge_index=self.edges, y=self.labels)

        self.num_features = self.dataset_data['X'].shape[1]
        self.num_classes = self.dataset_data['labels'].shape[1]

        self.data = self.data if self.pre_transform is None else self.pre_transform(self.data)
        self.val_ids = self.dataset_data['split_'+str(split)]['val_ids']
        self.test_ids = self.dataset_data['split_'+str(split)]['test_ids']

        if setting == 'cv':
            self.train_ids = np.append(self.dataset_data['split_'+str(split)]['train_ids'],
                            self.dataset_data['split_'+str(split)]['val_ids']) 
        
        elif setting == 'small_val':
            self.train_ids = self.dataset_data['split_'+str(split)]['train_ids']
        
        elif setting == 'large_val':
            _data = pkl.load( open(osp.join("./datasets/large_val/", self.name + ".pkl"), 'rb') )
            self.train_ids = _data['split_'+str(split)]['train_ids']
            self.val_ids = _data['split_'+str(split)]['val_ids']
            self.test_ids = _data['split_'+str(split)]['test_ids']

        self.data.train_mask = index_to_mask(self.train_ids, size=self.dataset_data['X'].shape[0])
        self.data.val_mask   = index_to_mask(self.val_ids, size=self.dataset_data['X'].shape[0])
        self.data.test_mask  = index_to_mask(self.test_ids, size=self.dataset_data['X'].shape[0])

        self.data.adj = sparse_mx_to_torch_sparse_tensor( normalize_adj(sp.csr_matrix(self.dataset_data['sym_adj'])) )

        self.data.y_train_onehot = torch.tensor(self.dataset_data['labels'][self.data.train_mask], dtype=torch.long).cuda()

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def DataLoader(name, split=0, setting='cv', adj_type='sym_adj', attack_type=None, random_train_val=True):

    dataset = Dataset(name=name, \
                      split=split, \
                      setting=setting, \
                      adj_type=adj_type, \
                      transform=T.NormalizeFeatures(), \
                      random_train_val=random_train_val,
                    )
    
    return dataset 

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1 / 2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx
