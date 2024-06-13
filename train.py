import sys
#sys.path.insert(0,'./keops')

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["USE_KEOPS"] = "True"

import pickle
import numpy as np

import torch
from torch.utils.data import DataLoader
from datasets import PlanetoidDataset, TadpoleDataset
import pytorch_lightning as pl
from DGMlib.model_dDGM import DGM_Model

from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

def run_training_process(run_params):
    
    #数据集加载=======================================================================================================
    train_data = None
    test_data = None
    
    if run_params.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        train_data = PlanetoidDataset(split='train', name=run_params.dataset, device='cuda')
        val_data = PlanetoidDataset(split='val', name=run_params.dataset, samples_per_epoch=1)
        test_data = PlanetoidDataset(split='test', name=run_params.dataset, samples_per_epoch=1)
        
    if run_params.dataset == 'tadpole':
        train_data = TadpoleDataset(fold=run_params.fold,train=True, device='cuda')
        val_data = test_data = TadpoleDataset(fold=run_params.fold, train=False,samples_per_epoch=1)
                                   
    if train_data is None:
        raise Exception("Dataset %s not supported" % run_params.dataset)
        
    train_loader = DataLoader(train_data, batch_size=1,num_workers=0)
    val_loader = DataLoader(val_data, batch_size=1)
    test_loader = DataLoader(test_data, batch_size=1)

    class MyDataModule(pl.LightningDataModule):
        def setup(self,stage=None):                 #it 会调用 setup 方法来准备数据，包括训练和验证数据
            pass                #fit调用后，setup->config_optimizer->val_dataloader->获取samples_per_epoch->获取self.X,self.y,self.mask,self.edge_index
        def train_dataloader(self):
            return train_loader
        def val_dataloader(self):
            return val_loader
        def test_dataloader(self):
            return test_loader
    
    
    #配置输入特征大小
    if run_params.pre_fc is None or len(run_params.pre_fc)==0: 
        if len(run_params.dgm_layers[0])>0:
            run_params.dgm_layers[0][0]=train_data.n_features
        run_params.conv_layers[0][0]=train_data.n_features
    else:
        run_params.pre_fc[0]=train_data.n_features
    run_params.fc_layers[-1] = train_data.num_classes
    #================================================================================================================
    
    #模型训练=========================================================================================================
    model = DGM_Model(run_params)       # 初始化模型

    #保存模型
    checkpoint_callback = ModelCheckpoint(
        save_last=True,         #如果为 True，则会在每个训练结束时保存最后一个 epoch 的模型
        save_top_k=1,           #保存最好的 k 个模型
        verbose=True,           #如果为 True，则会在模型保存时打印信息  
        monitor='val_loss',     #监视的指标，这里是验证集上的损失，如果模型在验证集上的损失最小，则保存模型
        mode='min'              #模式，min 表示监视的指标越小越好，max 表示监视的指标越大越好
    )
    
    #提前停止
    early_stop_callback = EarlyStopping(
        monitor='val_loss',     #监视的指标，这里是验证集上的损失，如果模型在验证集上的损失最小，则保存模型
        min_delta=0.00,         #最小变化，如果两次连续的验证损失的变化小于这个值，则认为没有改善。设置为 0.00 表示任何微小的改善都被认为是有效的
        patience=20,            #耐心，如果验证损失在 patience 轮内没有改善，则停止训练
        verbose=False,          #如果为 True，则会在停止训练时打印信息
        mode='min')
    callbacks = [checkpoint_callback,early_stop_callback]
    
    if val_data==test_data:
        callbacks = None
        
    logger = TensorBoardLogger("logs/")
    
    #使用 PL 的 from_argparse_args 方法从命令行参数中初始化 Trainer 实例
    trainer = pl.Trainer.from_argparse_args(run_params,logger=logger,
                                            callbacks=callbacks)
    
    trainer.fit(model, datamodule=MyDataModule())
    trainer.test()
    
if __name__ == "__main__":

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    params = parser.parse_args(['--gpus','1',                         
                              '--log_every_n_steps','100',                          
                              '--max_epochs','100',
                              '--progress_bar_refresh_rate','10',                         
                              '--check_val_every_n_epoch','1'])
    parser.add_argument("--num_gpus", default=10, type=int)
    
    parser.add_argument("--dataset", default='Cora')
    parser.add_argument("--fold", default='0', type=int) #Used for k-fold cross validation in tadpole/ukbb
    
    
    parser.add_argument("--conv_layers", default=[[32,32],[32,16],[16,8]], type=lambda x :eval(x))
    parser.add_argument("--dgm_layers", default= [[32,16,4],[],[]], type=lambda x :eval(x))
    parser.add_argument("--fc_layers", default=[8,8,3], type=lambda x :eval(x))
    parser.add_argument("--pre_fc", default=[-1,32], type=lambda x :eval(x))

    parser.add_argument("--gfun", default='gcn')
    parser.add_argument("--ffun", default='gcn')
    parser.add_argument("--k", default=5, type=int) 
    parser.add_argument("--pooling", default='add')
    parser.add_argument("--distance", default='euclidean')

    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--test_eval", default=10, type=int)

    parser.set_defaults(default_root_path='./log')
    params = parser.parse_args(namespace=params)

    run_training_process(params)
