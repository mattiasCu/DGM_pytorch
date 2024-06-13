import torch
import pickle
import numpy as np
import os.path as osp
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


class UKBBAgeDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, fold=0, train=True, samples_per_epoch=100, device='cpu'):
        with open('data/UKBB.pickle', 'rb') as f:
            X_,y_,train_mask_,test_mask_, weight_ = pickle.load(f) # Load the data

        self.X = torch.from_numpy(X_[:,:,fold]).float().to(device)
        self.y = torch.from_numpy(y_[:,:,fold]).float().to(device)
        self.weight = torch.from_numpy(np.squeeze(weight_[:1,fold])).float().to(device)
        if train:
            self.mask = torch.from_numpy(train_mask_[:,fold]).to(device)
        else:
            self.mask = torch.from_numpy(test_mask_[:,fold]).to(device)
            
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        return self.X,self.y,self.mask
    
    
    
class TadpoleDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, fold=0, train=True, samples_per_epoch=100, device='cpu',full=False):
        with open('data/tadpole_data.pickle', 'rb') as f:
            X_,y_,train_mask_,test_mask_, weight_ = pickle.load(f) # Load the data
        
        if not full:
            X_ = X_[...,:30,:] # For DGM we use modality 1 (M1) for both node representation and graph learning.

        
        self.n_features = X_.shape[-2]
        self.num_classes = y_.shape[-2]
        
        self.X = torch.from_numpy(X_[:,:,fold]).float().to(device)
        self.y = torch.from_numpy(y_[:,:,fold]).float().to(device)
        self.weight = torch.from_numpy(np.squeeze(weight_[:1,fold])).float().to(device)
        if train:
            self.mask = torch.from_numpy(train_mask_[:,fold]).to(device)
        else:
            self.mask = torch.from_numpy(test_mask_[:,fold]).to(device)
            
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        return self.X,self.y,self.mask, [[]]
 
    
# class TadpoleDataset(torch.utils.data.Dataset):
#     """Face Landmarks dataset."""

#     def __init__(self, fold=0, split='train', samples_per_epoch=100, device='cpu'):
       
#         with open('data/train_data.pickle', 'rb') as f:
#             X_,y_,train_mask_,test_mask_, weight_ = pickle.load(f) # Load the data
        
#         X_ = X_[...,:30,:] # For DGM we use modality 1 (M1) for both node representation and graph learning.

#         self.X = torch.from_numpy(X_[:,:,fold]).float().to(device)
#         self.y = torch.from_numpy(y_[:,:,fold]).float().to(device)
#         self.weight = torch.from_numpy(np.squeeze(weight_[:1,fold])).float().to(device)

#         # split train set in train/val
#         train_mask = train_mask_[:,fold]
#         nval = int(train_mask.sum()*0.2)
#         val_idxs = np.random.RandomState(1).choice(np.nonzero(train_mask.flatten())[0],(nval,),replace=False)
#         train_mask[val_idxs] = 0;
#         val_mask = train_mask*0
#         val_mask[val_idxs] = 1
                          
#         print('DATA STATS: train: %d val: %d' % (train_mask.sum(),val_mask.sum()))
            
#         if split=='train':
#             self.mask = torch.from_numpy(train_mask).to(device)
#         if split=='val':
#             self.mask = torch.from_numpy(val_mask).to(device)
#         if split=='test':
#             self.mask = torch.from_numpy(test_mask_[:,fold]).to(device)
            
#         self.samples_per_epoch = samples_per_epoch

#     def __len__(self):
#         return self.samples_per_epoch

#     def __getitem__(self, idx):
#         return self.X,self.y,self.mask
    

# 处理数据集
def get_planetoid_dataset(name, normalize_features=True, transform=None, split="complete"):
    
    """
        join的工作原理是接受一个或多个路径组件作为输入，然后将它们连接起来，形成一个单一的路径字符串
        在给定的代码片段中，osp.join函数被用来将当前目录（表示为.），
        子目录'data'，以及一个名为name的变量所代表的文件或目录名连接起来。
        这里的.代表当前工作目录，'data'是一个子目录的名称，而name是一个变量，其值可能在函数调用之前已经被定义
    """
    path = osp.join('.', 'data', name)          # 生成路径 './data/Cora'
    
    
    """
        数据结构如下：
        Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
            -x: 节点特征矩阵，形状为 [2708, 1433]，表示图中有 2708 个节点，每个节点有 1433 维特征
            -edge_index: 边索引矩阵，形状为 [2, 10556]，表示图中有 10556 条边
                            第一列表示边的起始节点，第二列表示边的终止节点
            -y: 节点标签，形状为 [2708]，表示图中有 2708 个节点，每个节点有一个标签
            -train_mask: 训练集掩码，用于指示哪些节点属于训练集
            -val_mask: 验证集掩码，用于指示哪些节点属于验证集
            -test_mask: 测试集掩码，用于指示哪些节点属于测试集

    """
    if split == 'complete':
        dataset = Planetoid(path, name)
        dataset[0].train_mask.fill_(False)
        dataset[0].train_mask[:dataset[0].num_nodes - 1000] = 1                             #将前 num_nodes - 1000 个节点的掩码设置为 True，表示这些节点用于训练
        dataset[0].val_mask.fill_(False)
        dataset[0].val_mask[dataset[0].num_nodes - 1000:dataset[0].num_nodes - 500] = 1     #将接下来的 500 个节点的掩码设置为 True，表示这些节点用于验证
        dataset[0].test_mask.fill_(False)
        dataset[0].test_mask[dataset[0].num_nodes - 500:] = 1                               #将剩下的节点的掩码设置为 True，表示这些节点用于测试
    else:
        dataset = Planetoid(path, name, split=split)
    
    #为数据集设置特定的变换操作，具体地说，是在加载数据集时对其进行特定的预处理，如特征归一化等
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])                   #将特征归一化和 transform 两个操作组合起来
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    return dataset


# 进行one-hot编码
def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes) 
    return y[labels] 


class PlanetoidDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', samples_per_epoch=100, name='Cora', device='cpu'):
        dataset = get_planetoid_dataset(name)
        self.X = dataset[0].x.float().to(device)                                            #torch.Size([2708, 1433])
        self.y = one_hot_embedding(dataset[0].y,dataset.num_classes).float().to(device)     #dataset[0].y是节点标签, g共2708个
        self.edge_index = dataset[0].edge_index.to(device)                                  #torch.Size([2, 10556])
        self.n_features = dataset[0].num_node_features                                      #int 1433
        self.num_classes = dataset.num_classes                                              #int 7                                                
        
        if split=='train':
            self.mask = dataset[0].train_mask.to(device)
        if split=='val':
            self.mask = dataset[0].val_mask.to(device)
        if split=='test':
            self.mask = dataset[0].test_mask.to(device)
         
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        return self.X,self.y,self.mask,self.edge_index