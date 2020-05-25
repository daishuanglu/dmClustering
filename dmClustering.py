import numpy as np
import torch
import torch.nn as nn
import time
from sklearn.cluster import AffinityPropagation,MiniBatchKMeans,KMeans
from sklearn.metrics import normalized_mutual_info_score
from collections import Counter


def pwise_Euclid(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    dist[dist != dist] = 0
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return dist

def similarity_edge(pdist1,Nconn):
    e=torch.zeros(len(pdist1))
    _,ind=torch.topk(pdist1, Nconn,largest=False)
    e[ind]=1
    return e

def group_label(y,pred):
    N=len(pred)
    y=np.asarray(y)
    grp=np.array([0]*N)
    upred=np.unique(pred)
    pairs=[('predicted_to','true')]
    for p in upred:
        id= (pred==p)
        ctr=Counter(y[id])
        keys=[k for k in ctr.keys()]
        idmax=np.argmax([v for v in ctr.values()])
        grp[id]=keys[idmax]
        pairs.append((p,keys[idmax]))
    return grp,pairs

def group_label_acc(y,pred):
    N=len(pred)
    y=np.asarray(y)
    grp=np.array([0]*N)
    upred=np.unique(pred)
    for p in upred:
        id= (pred==p)
        ctr=Counter(y[id])
        keys=[k for k in ctr.keys()]
        grp[id]=keys[np.argmax([v for v in ctr.values()])]
    return sum(grp==y)/N

class metricLoss(nn.Module):
    def __init__(self,batchsize = 10):
        super(metricLoss, self).__init__()
        self.loss_fn=nn.SoftMarginLoss()
        self.bsize=batchsize

    def forward(self, fx,S):
        d=pwise_Euclid(fx)
        N=fx.size(0)
        if self.bsize>0:
            rows = torch.Tensor([range(N)]).t().expand(N, self.bsize) * N
            id1 = (torch.from_numpy(np.random.choice(N, self.bsize*N))).long() + rows.view(-1).long()
            l = d.view(-1)[id1] * S.view(-1)[id1].detach()
            return l.sum() / (N * self.bsize), d
        else:
            return (d.view(-1) * S.view(-1).detach()).sum()/ (N ** 2), d

class dmClustering():
    def __init__(self, net,Nfactor=0,Nclusters=3):
        self.net = net
        #Minimum Nd=floor(ln(N))+1 encoder bits used for N clusters
        if Nfactor==0: self.Nd=int(np.floor(np.log2(Nclusters)))+1
        else: self.Nd = Nfactor
        self.Nc=Nclusters

    def unSupervisedLearner(self,inputs,sparsity=0,Nepochs=20,Ninner=1,bsize=50,lamda=1):
        dims=inputs.size()
        if len(dims)==3:
            N, C, L = dims
            dw = pwise_Euclid(inputs.reshape(N, C * L))
        else:
            N, C, H, W = dims
            dw = pwise_Euclid(inputs.reshape(N, C * H*W))
        print('learning ... ')
        start_time = time.time()
        rows = torch.Tensor([range(N)]).t().expand(N, N - sparsity).long()
        criterion = metricLoss(batchsize = bsize)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        for epoch in range(Nepochs):
            if sparsity==0:
                gamma=-.5*dw.sum(1)
                yida=1/N*(1-1/gamma)
                ss=-(1/(2*gamma)).expand(N,N)*dw+yida.expand(N,N)
            else:
                sorted_dw,ind=dw.sort()
                gamma=.5*(sparsity* sorted_dw[:,sparsity]-sorted_dw[:,:sparsity].sum(1))
                yida=1/sparsity+1/(sparsity*gamma)*sorted_dw[:,:sparsity].sum(1)
                #ss = -(dw / (2 * gamma)).expand(N, N) + yida.expand(N, N)
                ss= (-(dw/(2*gamma).expand(N,N))+yida.expand(N,N)).view(-1)
                linear_id0=(ind[:,sparsity:]+rows*N).view(-1)
                ss[linear_id0.long()]=0
                ss=ss.reshape(N,N)
            S=torch.max(ss,torch.zeros(N,N))

            ss=.5 *(S+S.t())
            Ds=torch.eye(N)
            Ds[Ds==1]=ss.sum(1)
            Ls=Ds-ss
            _,v=Ls.symeig(eigenvectors=True)
            F=v[:,:self.Nd]
            for ii in range(Ninner):
                fx = self.net(inputs)
                loss,dw = criterion(fx,S)
                l1 = loss.data.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch % 2 == 0:
                print('epoch {:d}, loss: {:f}, {:d}s'.format(epoch,l1, int(time.time() - start_time)))
            dw += lamda*pwise_Euclid(F).detach()

        kmeans = MiniBatchKMeans(n_clusters=self.Nc, random_state=0).fit(F.detach().numpy())
        self.sim = S.detach()
        self.dm=dw.detach()
        return F, kmeans.labels_

    def semiSupervisedLearner(self, dfx, sparsity=0,num_iter=20, lamda=1):
        N=dfx.size(0)
        dw=dfx
        rows = torch.Tensor([range(N)]).t().expand(N, N - sparsity).long()
        print('Clustering on semi-supervised metric ... ')
        start_time = time.time()
        for iter in range(num_iter):
            if sparsity == 0:
                gamma = -.5 * dw.sum(1)
                yida = 1 / N * (1 - 1 / gamma)
                ss = -(1 / (2 * gamma)).expand(N, N) * dw + yida.expand(N, N)
            else:
                sorted_dw, ind = dw.sort()
                gamma = .5 * (sparsity * sorted_dw[:, sparsity] - sorted_dw[:, :sparsity].sum(1))
                yida = 1 / sparsity + 1 / (sparsity * gamma) * sorted_dw[:, :sparsity].sum(1)
                ss = (-(dw / (2 * gamma.expand(N, N)) ) + yida.expand(N, N)).view(-1)
                linear_id0 = (ind[:, sparsity:] + rows * N).view(-1)
                ss[linear_id0.long()] = 0
                ss = ss.reshape(N, N)
            S = torch.max(ss, torch.zeros(N, N))
            ss = .5 * (S + S.t())
            Ds = torch.eye(N)
            Ds[Ds == 1] = ss.sum(1)
            Ls = Ds - ss
            _, v = Ls.symeig(eigenvectors=True)
            F = v[:, :self.Nc]
            dw =  dfx+lamda * pwise_Euclid(F)
            if iter % 2 == 0:
                obj=(dw*S + gamma.mean()*(S**2)).view(-1).sum()
                print('epoch {:d}, objective: {:f}, {:d}s'.format(iter, obj.item(), int(time.time() - start_time)))

        #kmeans = MiniBatchKMeans(n_clusters=self.Nc, random_state=0).fit(F.detach().numpy())
        kmeans=KMeans(n_clusters=self.Nc, random_state=0).fit(F.detach().numpy())
        # ap =  AffinityPropagation().fit(F.detach().numpy())
        return F, kmeans.labels_

    def metricLearner(self,inputs,targets,Nepochs=101,bsize=50,tau=10.0):
        dims=inputs.size()
        if len(dims)==3:
            N, C, L =dims
        else:
            N, C, H, W = dims
        print('learning distance metric ... ')
        start_time = time.time()

        sim = (targets.view(-1,N) == targets.view(-1,N).t())
        #idsim= sim.nonzero().t()[0]
        #iddis= (1-sim).nonzero().t()[0]
        #tau=100000.0
        idsim=[]
        iddis = []
        for ii in range(N):
            idsim.append( sim[ii,:].nonzero().t()[0].numpy())
            iddis.append( (1-sim[ii,:].long()).nonzero().t()[0].numpy())
        id1=torch.Tensor(N*2*bsize).long()
        criterion=nn.MSELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
        for i in range(Nepochs):
            fx = self.net(inputs)
            for ii in range(N):
                id1[(2*bsize*ii):(2*bsize*(ii+1))]= torch.cat((torch.from_numpy(np.random.choice(idsim[ii], bsize))+N*ii ,
                                        torch.from_numpy(np.random.choice(iddis[ii], bsize))+N*ii )).long()
            d = (tau-pwise_Euclid(fx)).view(-1)
            #d[d!=d]=0
            loss= criterion(d[id1],sim.view(-1)[id1].float()*tau)
            l = loss.data.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('epoch {:d}, loss: {:f}, {:d}s'.format(i, l, int(time.time() - start_time)))
        fx=self.net(inputs)
        d= tau-pwise_Euclid(fx.detach())
        #sim=(targets.view(-1,N) == targets.view(-1,N).t())
        print('Similarity TPR: {:.2f}%'.format( ((d>tau/2)+sim==2).sum().item()/(sim.sum().item())*100 ) )
        return d,fx

class dmNN(nn.Module):
    def __init__(self,C = 3, H=64, W=64):
        super(dmNN, self).__init__()
        # Conv2d, MaxPool2d H_out=floor[ (H_in+2xpadding-dilationx(kH-1) )/stride +1]
        self.convFeature = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=7, stride=4, padding=2),  # H=(64+4-7-1)/4+1=16
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # H=(16-2-1)/2+1=7
            nn.Conv2d(32,64, kernel_size=3, padding=1),  # H=7
            nn.Tanh(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # H=7
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=3, stride=2),     # H=(7-2-1)/2+1=3
        )
        self.linFeature=nn.Sequential(
            #nn.Dropout(),
            nn.Linear(64 * 3 * 3, 256),
            nn.Tanh(),
            nn.Linear(256, 256)
        )
        BIAS_INIT = -1
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if isinstance(m, nn.Conv2d) and m.kernel_size == 1:
                m.bias.data.fill_(BIAS_INIT)

    def forward(self,x1):
        x1= self.convFeature(x1)
        x1 = x1.view(x1.size(0), 64 * 3 * 3)
        x1= self.linFeature(x1)
        return x1
class dmNN1d(nn.Module):
    def __init__(self,C = 1, L=128):
        super(dmNN1d, self).__init__()
        # Conv2d, MaxPool2d H_out=floor[ (H_in+2xpadding-dilationx(kH-1) )/stride +1]
        self.convFeature = nn.Sequential(
            nn.Conv1d(C, 32, kernel_size=11, stride=4, padding=2),  # H=(128+4-11-1)/4+1=31
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=3, stride=2),  # H=(31-2-1)/2+1=15
            nn.Conv1d(32,64, kernel_size=5, stride=2, padding=2),  # H=(15+4-4-1)/2+1=8
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=4, stride=2),  # H=(8-3-1)/2+1=3
        )
        self.linFeature=nn.Sequential(
            nn.Linear(64*3, 512),
            nn.Tanh(),
            #nn.Linear(512,512),
            #nn.Tanh(),
            nn.Linear(512,256)
        )
        BIAS_INIT = -1
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if isinstance(m, nn.Conv1d) and m.kernel_size == 1:
                m.bias.data.fill_(BIAS_INIT)

    def forward(self,x1):
        x1= self.convFeature(x1)
        x1 = x1.view(x1.size(0), 64*3)
        x1= self.linFeature(x1)
        return x1

if __name__=="__main__":
    pass