import torch
import pykeops
from pykeops.torch import LazyTensor
from torch.nn import Module, ModuleList, Sequential
from torch import nn


#欧氏距离
def pairwise_euclidean_distances(x, dim=-1):
    dist = torch.cdist(x,x)**2
    return dist, x

#Poincarè disk 距离 r=1 (双曲距离)
def pairwise_poincare_distances(x, dim=-1):
    x_norm = (x**2).sum(dim,keepdim=True)
    x_norm = (x_norm.sqrt()-1).relu() + 1 
    x = x/(x_norm*(1+1e-2))
    x_norm = (x**2).sum(dim,keepdim=True)
    
    pq = torch.cdist(x,x)**2
    dist = torch.arccosh(1e-6+1+2*pq/((1-x_norm)*(1-x_norm.transpose(-1,-2))))**2
    return dist, x

# 生成size维的稀疏单位矩阵
def sparse_eye(size):
    """
        1. 这行代码首先使用torch.arange(0, size)创建一个从0到size（不包括size）的一维张量。
            然后，.long()方法将张量的数据类型转换为长整型。
            接着，.unsqueeze(0)方法在第0维（即最前面）增加一个维度，使其成为二维张量。
            最后，.expand(2, size)方法将这个二维张量扩展为一个形状为(2, size)的张量，其中每一行都是从0到size-1的序列。
        2. 这行代码创建一个值为1.0的张量，然后使用.float()方法确保这个张量的数据类型是浮点型。
            .expand(size)方法将这个单一值扩展为一个长度为size的一维张量，其中每个元素都是1.0。
        3. 这行代码首先通过values.type()获取values张量的数据类型（返回一个字符串，如torch.FloatTensor），
            然后使用.split(".")[-1]获取这个类型名称的最后一个部分（例如，从torch.FloatTensor中获取FloatTensor）。
            getattr(torch.sparse, ...)根据这个类型名称从torch.sparse模块中获取相应的稀疏张量类。
        4. 最后，这行代码使用这个稀疏张量类的构造函数创建一个稀疏单位矩阵。
    """
    indices = torch.arange(0, size).long().unsqueeze(0).expand(2, size)
    values = torch.tensor(1.0).float().expand(size)
    cls = getattr(torch.sparse, values.type().split(".")[-1])
    return cls(indices, values, torch.Size([size, size])) 


class DGM_d(nn.Module):
    
    """
        构造函数为嵌入函数embed_f、采样数量k、距离度量distance和是否稀疏sparse
    """
    def __init__(self, embed_f, k=5, distance="euclidean", sparse=True):
        super(DGM_d, self).__init__()
        
        self.sparse=sparse
        
        self.temperature = nn.Parameter(torch.tensor(1. if distance=="hyperbolic" else 4.).float())         #只有一个元素的张量
        self.embed_f = embed_f
        self.centroid=None
        self.scale=None
        self.k = k
        self.distance = distance                #str
        
        self.debug=False
        
    def forward(self, x, A, not_used=None, fixedges=None):
        if x.shape[0]==1:
            x = x[0]
        x = self.embed_f(x,A)                   # 嵌入后的x
        if x.dim()==2:
            x = x[None,...]                     # 如果 x 是二维的，将其扩展为三维
    
        if self.training:
            if fixedges is not None:                
                return x, fixedges, torch.zeros(fixedges.shape[0],fixedges.shape[-1]//self.k,self.k,dtype=torch.float,device=x.device)
            #sampling here
            edges_hat, logprobs = self.sample_without_replacement(x)
                
        else:
            with torch.no_grad():
                if fixedges is not None:                
                    return x, fixedges, torch.zeros(fixedges.shape[0],fixedges.shape[-1]//self.k,self.k,dtype=torch.float,device=x.device)
                #sampling here
                edges_hat, logprobs = self.sample_without_replacement(x)
              
        if self.debug:
            if self.distance=="euclidean":
                D, _x = pairwise_euclidean_distances(x)
            if self.distance=="hyperbolic":
                D, _x = pairwise_poincare_distances(x)
                
            self.D = (D * torch.exp(torch.clamp(self.temperature,-5,5))).detach().cpu()
            self.edges_hat=edges_hat.detach().cpu()
            self.logprobs=logprobs.detach().cpu()
#             self.x=x

        return x, edges_hat, logprobs
    

    def sample_without_replacement(self, x):
        
        b,n,_ = x.shape
        
        if self.distance=="euclidean":
            G_i = LazyTensor(x[:, :, None, :])      # 在第三个维度上增加一个新维度，数量是1 [b, n, 1, m]
            X_j = LazyTensor(x[:, None, :, :])      # 在第二个维度上增加一个新维度，数量为1 [b, 1, n, m]
            """
                1.计算每个元素在最后一个维度上的差值，结果的形状为 [b, n, n, m]
                2. 对最后一个维度上的差值进行平方，然后在倒数第一个维度上求和，结果的形状为 [b, n, n]
            """
            mD = ((G_i - X_j) ** 2).sum(-1)         # 计算距离平方和 [b ,n ,n]

            #argKmin already add gumbel noise
            """
                1.torch.clamp 是 PyTorch 中用于限制张量元素值的函数，它可以将张量中的元素限制在指定的范围内。
                    具体来说，可以将张量中的每个元素值限制在一个最小值和最大值之间，
                    如果元素值小于最小值，则将其设为最小值；如果元素值大于最大值，则将其设为最大值
                2.使用 argKmin 找到每个 xi 到其他xj最近的 K 个点的索引。
                    参数 dim=1 指定沿第二个维度查找最近的 K 个点
            """
            lq = mD * torch.exp(torch.clamp(self.temperature,-5,5)) 
            indices = lq.argKmin(self.k, dim=1)     # [b, n ,k]

            """
                维度变化：[b, n, k] -> [b, n*k] -> [b, n*k, 1] -> [b, n*k, m]
                1.gather 函数的作用是根据索引从输入张量中取出对应的元素，然后根据索引重新排列成输出张量，input 和 index 的形状必须是相同的。
                    例如，对于输入张量 input[i][j]，如果索引是 index[i][j]，那么输出张量 output[i][j] = input[index[i][j]][j]
                    
                2.view 函数的作用是将输入张量重塑为指定形状的输出张量，但是输出张量的元素数目必须与输入张量的元素数目相同。
                
                3.[...,None] 表示在最后一个维度上增加一个新维度，数量是1
                
                4.repeat 函数的作用是将输入张量沿指定维度复制指定次数，然后将结果张量返回。
                
            """
            x1 = torch.gather(x, -2, indices.view(indices.shape[0],-1)[...,None].repeat(1,1,x.shape[-1]))       #[b, n*k, m]
            """
                [b, n, m] -> [b, n, 1, m] -> [b, n, k, m] -> [b, n*k, m]
            """
            x2 = x[:,:,None,:].repeat(1,1,self.k,1).view(x.shape[0],-1,x.shape[-1])                             #[b, n*k, m]
            
            #边的概率公式计算
            logprobs = (-(x1-x2).pow(2).sum(-1) * torch.exp(torch.clamp(self.temperature,-5,5))).reshape(x.shape[0],-1,self.k)

        if self.distance=="hyperbolic":
            pass
            x_norm = (x**2).sum(-1,keepdim=True)
            x_norm = (x_norm.sqrt()-1).relu() + 1 
            x = x/(x_norm*(1+1e-2)) #safe distance to the margin
            x_norm = (x**2).sum(-1,keepdim=True)
                
            G_i = LazyTensor(x[:, :, None, :])    # (M**2, 1, 2)
            X_j = LazyTensor(x[:, None, :, :])    # (1, N, 2)

            G_i2 = LazyTensor(1-x_norm[:, :, None, :])    # (M**2, 1, 2)
            X_j2 = LazyTensor(1-x_norm[:, None, :, :])    # (1, N, 2)

            pq = ((G_i - X_j) ** 2).sum(-1)
            N = (G_i2*X_j2)
            XX = (1e-6+1+2*pq/N)
            mD = (XX+(XX**2-1).sqrt()).log()**2

            lq = mD * torch.exp(torch.clamp(self.temperature,-5,5))
            indices = lq.argKmin(self.k, dim=1)

            x1 = torch.gather(x, -2, indices.view(indices.shape[0],-1)[...,None].repeat(1,1,x.shape[-1]))
            x2 = x[:,:,None,:].repeat(1,1,self.k,1).view(x.shape[0],-1,x.shape[-1])

            x1_n = torch.gather(x_norm, -2, indices.view(indices.shape[0],-1)[...,None].repeat(1,1,x_norm.shape[-1]))
            x2_n = x_norm[:,:,None,:].repeat(1,1,self.k,1).view(x.shape[0],-1,x_norm.shape[-1])

            pq = (x1-x2).pow(2).sum(-1)
            pqn = ((1-x1_n)*(1-x2_n)).sum(-1)
            XX = 1e-6+1+2*pq/pqn
            dist = torch.log(XX+(XX**2-1).sqrt())**2
            logprobs = (-dist * torch.exp(torch.clamp(self.temperature,-5,5))).reshape(x.shape[0],-1,self.k)

            if self.debug:
                self._x=x.detach().cpu()+0

        """
            rows: [n]->[1,n,1]->[b,n,k]
                arange用于生成等间隔序列，
                torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
            stack: [b, n, k], 
                stack用于沿着新维度将一组张量堆叠在一起       

        """
        rows = torch.arange(n).view(1,n,1).to(x.device).repeat(b,1,self.k)
        edges = torch.stack((indices.view(b,-1),rows.view(b,-1)),-2)

        if self.sparse:
            return (edges+(torch.arange(b).to(x.device)*n)[:,None,None]).transpose(0,1).reshape(2,-1), logprobs
        return edges, logprobs
    
class DGM_c(nn.Module):
    input_dim = 4
    debug=False
    
    def __init__(self, embed_f, k=None, distance="euclidean"):
        super(DGM_c, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(1).float())
        self.threshold = nn.Parameter(torch.tensor(0.5).float())
        self.embed_f = embed_f
        self.centroid=None
        self.scale=None
        self.distance = distance
        
        self.scale = nn.Parameter(torch.tensor(-1).float(),requires_grad=False)
        self.centroid = nn.Parameter(torch.zeros((1,1,DGM_c.input_dim)).float(),requires_grad=False)
        
        
    def forward(self, x, A, not_used=None, fixedges=None):
        
        x = self.embed_f(x,A)  
        
        # estimate normalization parameters
        if self.scale <0:            
            self.centroid.data = x.mean(-2,keepdim=True).detach()
            self.scale.data = (0.9/(x-self.centroid).abs().max()).detach()
        
        if self.distance=="hyperbolic":
            D, _x = pairwise_poincare_distances((x-self.centroid)*self.scale)
        else:
            D, _x = pairwise_euclidean_distances((x-self.centroid)*self.scale)
            
        A = torch.sigmoid(self.temperature*(self.threshold.abs()-D))
        
        if DGM_c.debug:
            self.A = A.data.cpu()
            self._x = _x.data.cpu()
            
#         self.A=A
#         A = A/A.sum(-1,keepdim=True)
        return x, A, None
 
 
class MLP(nn.Module): 
    def __init__(self, layers_size,final_activation=False, dropout=0):
        super(MLP, self).__init__()
        layers = []
        for li in range(1,len(layers_size)):
            if dropout>0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(layers_size[li-1],layers_size[li]))
            if li==len(layers_size)-1 and not final_activation:
                continue
            layers.append(nn.LeakyReLU(0.1))
            
            
        self.MLP = nn.Sequential(*layers)
        
    def forward(self, x, e=None):
        x = self.MLP(x)
        return x
    
class Identity(nn.Module):
    def __init__(self,retparam=None):
        self.retparam=retparam
        super(Identity, self).__init__()
        
    def forward(self, *params):
        if self.retparam is not None:
            return params[self.retparam]
        return params
    