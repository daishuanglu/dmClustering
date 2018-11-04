from dmClustering import pwise_Euclid, dmClustering, group_label_acc, group_label
from dataReader import dataReader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans, SpectralClustering
from sklearn import datasets
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
import time
#from sklearn.model_selection import train_test_split
#from pathlib import Path

def avgPred(ntimes,loader,testcase):
    target, inputs, nfeat, nsamples = loader.spiral()
    preds = torch.Tensor(ntimes, nsamples)
    for i in range(ntimes):
        print('predicting {:d}/{:d} round'.format(i+1,ntimes))
        if type(inputs) is np.ndarray:
            preds[i, :] =torch.from_numpy(testcase(loader,True))
        else:
            preds[i, :] = testcase(loader, True)
    print('max Voting')
    pred,_=preds.mode(0)
    nmi = normalized_mutual_info_score(target, pred.numpy())
    acc = group_label_acc(target, pred.numpy())
    return acc,nmi
def metricCluster(dfx,ratio=0.0):
    N = dfx.size(0)
    sim = dfx.detach() > ratio
    clusters = np.arange(N)
    flag = torch.zeros(N)
    i = 0
    while flag.sum().item() < N:
        label = clusters[i]
        clusters[sim[i, :].nonzero().squeeze()] = label
        flag[sim[i, :].nonzero().squeeze()] = 1
        if flag.sum().item() == N:
            break
        i = flag.eq(0).nonzero()[0].long()
        #print('searching {:d} '.format(i.numpy()[0]))
    return clusters

class shapeNet(nn.Module):
    def __init__(self,dim=2):
        super(shapeNet, self).__init__()
        self.dim=dim
        self.linFeature=nn.Sequential(
            nn.Linear(dim, 10),
        )
    def forward(self,x1):
        x1 = x1.view(x1.size(0), self.dim)
        x1= self.linFeature(x1)
        return x1
def testSpiral(loader,averaging=False):
    target, inputs, nfeat, nsamples = loader.spiral()
    if type(inputs) is np.ndarray:
        inputs=torch.from_numpy(inputs).unsqueeze(1).float()
        ut=np.unique(target)
    else: ut=target.unique()
    print(nsamples,' samples',nfeat,' dimensions ,targets:',Counter(target))
    Nclusters = len(ut)
    Nsparse = 3
    batchsize = 10
    model = dmClustering(shapeNet(), Nfactor=0, Nclusters=Nclusters)
    F, labels = model.unSupervisedLearner(inputs, Nepochs=20, sparsity=Nsparse, bsize=batchsize, lamda=1)
    if not averaging:
        #print(F)
        print('clusters:', ''.join(str(l) for l in labels))
        print(Counter(labels))
        #print('targets:', ''.join(str(t) for t in target))
        nmi = normalized_mutual_info_score(target, labels)
        print('Normalized mutual information(NMI): {:.2f}%'.format(nmi * 100))
        acc = group_label_acc(target, labels)
        print('Group accuracy: {:.2f}%'.format(acc * 100))
        return acc,nmi
    else:
        return group_label(target, labels)
def testPath(loader,averaging=False):
    target, inputs, nfeat, nsamples = loader.path()
    inputs = torch.from_numpy(inputs).unsqueeze(1).float()
    ut = np.unique(target)
    print(nsamples, 'samples,', nfeat, 'dimensions,targets:', Counter(target))
    Nclusters = len(ut)
    Nsparse = 8
    batchsize = 10
    Nfactor = 5
    model = dmClustering(shapeNet(), Nfactor=Nfactor,Nclusters=Nclusters)
    F, labels = model.unSupervisedLearner(inputs, Nepochs=20, Ninner=1, sparsity=Nsparse, bsize=batchsize, lamda=1)
    if not averaging:
        print('clusters:', ''.join(str(l) for l in labels))
        print(Counter(labels))
        nmi = normalized_mutual_info_score(target, labels)
        print('Normalized mutual information(NMI): {:.2f}%'.format(nmi * 100))
        acc = group_label_acc(target, labels)
        print('Group accuracy: {:.2f}%'.format(acc * 100))
        return acc, nmi
    else:
        return group_label(target, labels)
def testCompound(loader,averaging=False):
    target, inputs, nfeat, nsamples = loader.compound()
    inputs = torch.from_numpy(inputs).unsqueeze(1).float()
    ut = np.unique(target)
    print(nsamples, 'samples,', nfeat, 'dimensions,targets:', Counter(target))
    Nclusters = len(ut)
    Nsparse = 22
    batchsize = 50
    Nfactor = 4
    model = dmClustering(shapeNet(), Nfactor=Nfactor, Nclusters=Nclusters)
    F, labels = model.unSupervisedLearner(inputs, Nepochs=14, Ninner=1, sparsity=Nsparse, bsize=batchsize, lamda=1)
    if not averaging:
        print('clusters:', ''.join(str(l) for l in labels))
        print(Counter(labels))
        # print('targets:', ''.join(str(t) for t in target))
        nmi = normalized_mutual_info_score(target, labels)
        print('Normalized mutual information(NMI): {:.2f}%'.format(nmi * 100))
        acc = group_label_acc(target, labels)
        print('Group accuracy: {:.2f}%'.format(acc * 100))
        return acc,nmi
    else:
        return group_label(target, labels)

class bioNet(nn.Module):
    def __init__(self,dim=8,winit=None):
        super(bioNet, self).__init__()
        self.dim=dim
        self.feature=nn.Sequential(
                nn.Linear(dim, 256),
            )
        if winit=='xavier':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight,gain=5)
    def forward(self,x1):
        x1 = x1.view(x1.size(0), self.dim)
        x1=self.feature(x1)
        return x1
def testYeast(loader, averaging=False):
    target, data, nfeat, nsamples = loader.yeast()
    inputs = torch.from_numpy(data).unsqueeze(1).float()
    ut = np.unique(target)
    print(nsamples, 'samples,', nfeat, 'dimensions,targets:', Counter(target))
    Nclusters = len(ut)
    Nsparse = 40
    batchsize = 40
    Nfactor = 20
    model = dmClustering(bioNet(dim=nfeat), Nfactor=Nfactor,Nclusters=Nclusters)
    F, labels = model.unSupervisedLearner(inputs, Nepochs=10, Ninner=1, sparsity=Nsparse, bsize=batchsize, lamda=1)
    if not averaging:
        print('clusters:', ''.join(str(l) for l in labels))
        print(Counter(labels))
        # print('targets:', ''.join(str(t) for t in target))
        nmi = normalized_mutual_info_score(target, labels)
        print('Normalized mutual information(NMI): {:.2f}%'.format(nmi * 100))
        acc = group_label_acc(target, labels)
        print('Group accuracy: {:.2f}%'.format(acc * 100))
        return acc,nmi
    else:
        return group_label_acc(target, labels)
def testGlass(loader,averaging=False):
    target, data, nfeat, nsamples = loader.glass()
    inputs = torch.from_numpy(data).unsqueeze(1).float()
    ut = np.unique(target)
    print(nsamples, 'samples,', nfeat, 'dimensions,targets:', Counter(target))
    Nclusters = len(ut)
    Nsparse = 40
    batchsize = 40
    Nfactor = 20
    model = dmClustering(bioNet(dim=nfeat), Nfactor=Nfactor,  Nclusters=Nclusters)
    F, labels = model.unSupervisedLearner(inputs, Nepochs=10, Ninner=1, sparsity=Nsparse, bsize=batchsize, lamda=1)
    if not averaging:
        print('clusters:', ''.join(str(l) for l in labels))
        print(Counter(labels))
        # print('targets:', ''.join(str(t) for t in target))
        nmi = normalized_mutual_info_score(target, labels)
        print('Normalized mutual information(NMI): {:.2f}%'.format(nmi * 100))
        acc = group_label_acc(target, labels)
        print('Group accuracy: {:.2f}%'.format(acc * 100))
        return acc,nmi
    else:
        return group_label_acc(target, labels)
def testEcoli(loader, averaging=False):
    target, data, nfeat, nsamples = loader.ecoli()
    inputs = torch.from_numpy(data).unsqueeze(1).float()
    ut = np.unique(target)
    print(nsamples, 'samples,', nfeat, 'dimensions,targets:', Counter(target))
    Nclusters = len(ut)
    Nsparse = 30
    batchsize = 50
    Nfactor = 10
    model = dmClustering(bioNet(dim=nfeat,winit='xavier'), Nfactor=Nfactor,  Nclusters=Nclusters)
    F, labels = model.unSupervisedLearner(inputs, Nepochs=10, Ninner=1, sparsity=Nsparse, bsize=batchsize, lamda=1)
    if not averaging:
        print('clusters:', ''.join(str(l) for l in labels))
        print(Counter(labels))
        print('targets:', ''.join(str(t) for t in target))
        nmi = normalized_mutual_info_score(target, labels)
        print('Normalized mutual information(NMI): {:.2f}%'.format(nmi * 100))
        acc = group_label_acc(target, labels)
        print('Group accuracy: {:.2f}%'.format(acc * 100))
        return acc,nmi
    else:
        return group_label(target, labels)
def testMove(loader,averaging=False):
    target, data, nfeat, nsamples = loader.movement()
    inputs = torch.from_numpy(data).unsqueeze(1).float()
    ut = np.unique(target)
    print(nsamples, 'samples,', nfeat, 'dimensions,targets:', Counter(target))
    Nclusters = len(ut)
    Nsparse = 20
    batchsize = 50
    Nfactor = 8
    model = dmClustering(bioNet(dim=nfeat), Nfactor=Nfactor, Nclusters=Nclusters)
    F, labels = model.unSupervisedLearner(inputs, Nepochs=10, Ninner=1, sparsity=Nsparse, bsize=batchsize, lamda=1)
    if not averaging:
        print('clusters:', ''.join(str(l) for l in labels))
        print(Counter(labels))
        nmi = normalized_mutual_info_score(target, labels)
        print('Normalized mutual information(NMI): {:.2f}%'.format(nmi * 100))
        acc = group_label_acc(target, labels)
        print('Group accuracy: {:.2f}%'.format(acc * 100))
        return acc,nmi
    else:
        return group_label(target, labels)
def testCoil(loader):
    target, data, nfeat, nsamples = loader.coil()
    inputs = torch.from_numpy(data).unsqueeze(1).float()
    ut = np.unique(target)
    print(nsamples, 'samples,', nfeat, 'dimensions,targets:', Counter(target))
    Nclusters = len(ut)
    Nsparse = 100
    batchsize = 50
    Nfactor = 0
    model = dmClustering(bioNet(dim=nfeat), Nfactor=Nfactor, Nclusters=Nclusters)
    F, labels = model.unSupervisedLearner(inputs, Nepochs=10, Ninner=1, sparsity=Nsparse, bsize=batchsize, lamda=1)
    print('clusters:', ''.join(str(l) for l in labels))
    print(Counter(labels))
    nmi = normalized_mutual_info_score(target, labels)
    print('Normalized mutual information(NMI): {:.2f}%'.format(nmi * 100))
    acc = group_label_acc(target, labels)
    print('Group accuracy: {:.2f}%'.format(acc * 100))
    return acc,nmi
def testStock(loader):
    target, data, nfeat, nsamples = loader.stock()
    print(nsamples, ' samples', nfeat, ' dimensions ,targets:', Counter(target))
    data = data[:60]
    target = target[:60]
    inputs = torch.from_numpy(data).unsqueeze(1).float()
    ut = np.unique(target)
    Nclusters = len(ut)
    Nsparse = 10
    batchsize = 50
    Nfactor = 5
    model = dmClustering(bioNet(dim=nfeat), Nfactor=Nfactor, Nclusters=Nclusters)
    F, labels = model.unSupervisedLearner(inputs, Nepochs=10, Ninner=1, sparsity=Nsparse, bsize=batchsize, lamda=1)

    ctr = Counter(labels)
    print('clusters:', ''.join(str(l) for l in labels))
    print(ctr)
    nmi = normalized_mutual_info_score(target, labels)
    print('Normalized mutual information(NMI): {:.2f}%'.format(nmi * 100))
    acc = group_label_acc(target, labels)
    print('Group accuracy: {:.2f}%'.format(acc * 100))
    print('Number of clusters: {:d} '.format(len(ctr)))
    return acc,nmi

class objectNet1d(nn.Module):
    def __init__(self,dim):
        super(objectNet1d, self).__init__()
        self.dim=dim
        self.feature=nn.Sequential(
            nn.Linear(dim, 1024),
            )
    def forward(self, x):
        x= x.view(-1,self.dim)
        x = self.feature(x)
        return x
def testCoil20(loader, pretrained=True):
    target, data, nfeat, nsamples = loader.coil20()
    inputs = torch.from_numpy(data).unsqueeze(1).float()
    ut = np.unique(target)
    print(nsamples, 'samples,', nfeat, 'dimensions,targets:', Counter(target))
    Nclusters = len(ut)
    modelFile = './coil20_metricNet.dat'
    net = objectNet1d(dim=nfeat)
    if pretrained:
        net.load_state_dict(torch.load(modelFile))
    model = dmClustering(net, Nclusters=Nclusters)
    tau = 100000.0
    dfx, _ = model.metricLearner(inputs, torch.from_numpy(target), Nepochs=1000, bsize=1000, tau=tau)
    torch.save(net.state_dict(), modelFile)
    # dfx=tau-pwise_Euclid(net(inputs))
    labels = metricCluster(dfx, ratio=tau / 2)
    ctr = Counter(labels)
    print('clusters:', ''.join(str(l) for l in labels))
    print(ctr)
    nmi = normalized_mutual_info_score(target, labels)
    print('Normalized mutual information(NMI): {:.2f}%'.format(nmi * 100))
    acc = group_label_acc(target, labels)
    print('Group accuracy: {:.2f}%'.format(acc * 100))
    print('Number of clusters: {:d} '.format(len(ctr)))
    return acc,nmi

class faceNet1d(nn.Module):
    def __init__(self,dim=256):
        super(faceNet1d, self).__init__()
        self.dim=dim
        self.feature=nn.Sequential(
            nn.Linear(dim, 512),
            #nn.Tanh(),
            #nn.Linear(1024,256)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m,nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
    def forward(self,x1):
        x1=x1.view(x1.size(0),self.dim)
        x1=self.feature(x1)
        return x1
def testUmist(loader):
    target, data, nfeat, nsamples = loader.umist()
    inputs = torch.from_numpy(data).unsqueeze(1).float()
    ut = np.unique(target)
    print(nsamples, 'samples,', nfeat, 'dimensions,targets:', Counter(target))
    Nclusters = len(ut)
    Nsparse = 10
    batchsize = 50
    Nfactor = 20
    model = dmClustering(faceNet1d(dim=nfeat), Nfactor=Nfactor, Nclusters=Nclusters)
    F, labels = model.unSupervisedLearner(inputs, Nepochs=10, Ninner=1, sparsity=Nsparse, bsize=batchsize, lamda=1)
    ctr = Counter(labels)
    print('clusters:', ''.join(str(l) for l in labels))
    print(ctr)
    nmi = normalized_mutual_info_score(target, labels)
    print('Normalized mutual information(NMI): {:.2f}%'.format(nmi * 100))
    acc = group_label_acc(target, labels)
    print('Group accuracy: {:.2f}%'.format(acc * 100))
    print('Number of clusters: {:d} '.format(len(ctr)))
    return acc,nmi
def testJaffe(loader,pretrained=True):
    target, data, nfeat, nsamples = loader.jaffe()
    inputs = torch.from_numpy(data).unsqueeze(1).float()
    ut = np.unique(target)
    print(nsamples, 'samples,', nfeat, 'dimensions,targets:', Counter(target))
    Nclusters = len(ut)
    modelFile = './jaffe_metricNet.dat'
    net = faceNet1d(dim=nfeat)
    if pretrained:
        net.load_state_dict(torch.load(modelFile))
    model = dmClustering(net, Nclusters=Nclusters)
    tau = 100000.0
    dfx, _ = model.metricLearner(inputs, torch.from_numpy(target), Nepochs=1500, bsize=1000, tau=tau)
    torch.save(net.state_dict(), modelFile)
    labels = metricCluster(dfx, ratio=tau / 2)
    ctr = Counter(labels)
    print('clusters:', ''.join(str(l) for l in labels))
    print(ctr)
    nmi = normalized_mutual_info_score(target, labels)
    print('Normalized mutual information(NMI): {:.2f}%'.format(nmi * 100))
    acc = group_label_acc(target, labels)
    print('Group accuracy: {:.2f}%'.format(acc * 100))
    print('Number of clusters: {:d} '.format(len(ctr)))
    return acc,nmi
def testUsps(loader,pretrained=True ):
    target, data, nfeat, nsamples = loader.usps()
    inputs = torch.from_numpy(data).unsqueeze(1).float()
    ut = np.unique(target)
    print(nsamples, 'samples,', nfeat, 'dimensions,targets:', Counter(target))
    Nclusters = len(ut)
    modelFile = './USPS_metricNet.dat'
    net = faceNet1d(dim=nfeat)
    if pretrained:
        net.load_state_dict(torch.load(modelFile))
    model = dmClustering(net, Nclusters=Nclusters)
    tau = 100000.0
    dfx, _ = model.metricLearner(inputs, torch.from_numpy(target), Nepochs=1500, bsize=1000, tau=tau)
    torch.save(net.state_dict(), modelFile)
    labels = metricCluster(dfx, ratio=tau / 2)
    ctr = Counter(labels)
    print('clusters:', ''.join(str(l) for l in labels))
    print(ctr)
    nmi = normalized_mutual_info_score(target, labels)
    print('Normalized mutual information(NMI): {:.2f}%'.format(nmi * 100))
    acc = group_label_acc(target, labels)
    print('Group accuracy: {:.2f}%'.format(acc * 100))
    print('Number of clusters: {:d} '.format(len(ctr)))
    return acc, nmi

def testMSRCV(loader):
    target, data, nfeat, nsamples = loader.msrcv()
    inputs = torch.from_numpy(data).unsqueeze(1).float()
    ut = np.unique(target)
    print(nsamples, 'samples,', nfeat, 'dimensions,targets:', Counter(target))
    Nclusters = len(ut)
    Nsparse = 10
    batchsize = 80
    Nfactor = 10
    model = dmClustering(objectNet1d(dim=nfeat), Nfactor=Nfactor, Nclusters=Nclusters)
    F, labels = model.unSupervisedLearner(inputs, Nepochs=10, Ninner=1, sparsity=Nsparse, bsize=batchsize, lamda=1)
    ctr = Counter(labels)
    print('clusters:', ''.join(str(l) for l in labels))
    print(ctr)
    nmi = normalized_mutual_info_score(target, labels)
    print('Normalized mutual information(NMI): {:.2f}%'.format(nmi * 100))
    acc = group_label_acc(target, labels)
    print('Group accuracy: {:.2f}%'.format(acc * 100))
    print('Number of clusters: {:d} '.format(len(ctr)))
    return acc, nmi
def testCal7(loader):
    target, data, nfeat, nsamples = loader.cal7()
    inputs = torch.from_numpy(data).unsqueeze(1).float()
    ut = np.unique(target)
    print(nsamples, 'samples,', nfeat, 'dimensions,targets:', Counter(target))
    Nclusters = len(ut)
    Nsparse = 20
    batchsize = 50
    Nfactor = 10
    model = dmClustering(objectNet1d(dim=nfeat), Nfactor=Nfactor, Nclusters=Nclusters)
    F, labels = model.unSupervisedLearner(inputs, Nepochs=10, Ninner=1, sparsity=Nsparse, bsize=batchsize, lamda=1)
    ctr = Counter(labels)
    print('clusters:', ''.join(str(l) for l in labels))
    print(ctr)
    nmi = normalized_mutual_info_score(target, labels)
    print('Normalized mutual information(NMI): {:.2f}%'.format(nmi * 100))
    acc = group_label_acc(target, labels)
    print('Group accuracy: {:.2f}%'.format(acc * 100))
    print('Number of clusters: {:d} '.format(len(ctr)))
    return acc, nmi
def testCal20(loader):
    target, data, nfeat, nsamples = loader.cal20()
    inputs = torch.from_numpy(data).unsqueeze(1).float()
    ut = np.unique(target)
    print(nsamples, 'samples,', nfeat, 'dimensions,targets:', Counter(target))
    Nclusters = len(ut)
    Nsparse = 20
    batchsize = 50
    Nfactor = 10
    model = dmClustering(objectNet1d(dim=nfeat), Nfactor=Nfactor, Nclusters=Nclusters)
    F, labels = model.unSupervisedLearner(inputs, Nepochs=10, Ninner=1, sparsity=Nsparse, bsize=batchsize, lamda=1)
    ctr = Counter(labels)
    print('clusters:', ''.join(str(l) for l in labels))
    print(ctr)
    nmi = normalized_mutual_info_score(target, labels)
    print('Normalized mutual information(NMI): {:.2f}%'.format(nmi * 100))
    acc = group_label_acc(target, labels)
    print('Group accuracy: {:.2f}%'.format(acc * 100))
    print('Number of clusters: {:d} '.format(len(ctr)))
    return acc, nmi
def testORL(loader):
    target, data, nfeat, nsamples = loader.orl()
    inputs = torch.from_numpy(data).unsqueeze(1).float()
    ut = np.unique(target)
    print(nsamples, 'samples,', nfeat, 'dimensions,targets:', Counter(target))
    Nclusters = len(ut)
    Nsparse = 10
    batchsize = 80
    Nfactor = 10
    model = dmClustering(objectNet1d(dim=nfeat), Nfactor=Nfactor, Nclusters=Nclusters)
    F, labels = model.unSupervisedLearner(inputs, Nepochs=10, Ninner=1, sparsity=Nsparse, bsize=batchsize, lamda=1)
    ctr = Counter(labels)
    print('clusters:', ''.join(str(l) for l in labels))
    print(ctr)
    nmi = normalized_mutual_info_score(target, labels)
    print('Normalized mutual information(NMI): {:.2f}%'.format(nmi * 100))
    acc = group_label_acc(target, labels)
    print('Group accuracy: {:.2f}%'.format(acc * 100))
    print('Number of clusters: {:d} '.format(len(ctr)))
    return acc, nmi
def testCol20(loader):
    target, data, nfeat, nsamples = loader.col20()
    inputs = torch.from_numpy(data).unsqueeze(1).float()
    ut = np.unique(target)
    print(nsamples, 'samples,', nfeat, 'dimensions,targets:', Counter(target))
    Nclusters = len(ut)
    Nsparse = 20
    batchsize = 50
    Nfactor = 10
    model = dmClustering(objectNet1d(dim=nfeat), Nfactor=Nfactor, Nclusters=Nclusters)
    F, labels = model.unSupervisedLearner(inputs, Nepochs=10, Ninner=1, sparsity=Nsparse, bsize=batchsize, lamda=1)
    ctr = Counter(labels)
    print('clusters:', ''.join(str(l) for l in labels))
    print(ctr)
    nmi = normalized_mutual_info_score(target, labels)
    print('Normalized mutual information(NMI): {:.2f}%'.format(nmi * 100))
    acc = group_label_acc(target, labels)
    print('Group accuracy: {:.2f}%'.format(acc * 100))
    print('Number of clusters: {:d} '.format(len(ctr)))
    return acc, nmi
def testEYB10(loader):
    target, data, nfeat, nsamples = loader.eyb(10)
    inputs = torch.from_numpy(data).unsqueeze(1).float()
    ut = np.unique(target)
    print(nsamples, 'samples,', nfeat, 'dimensions,targets:', Counter(target))
    Nclusters = len(ut)
    Nsparse = 80
    batchsize = 100
    Nfactor = 10
    model = dmClustering(objectNet1d(dim=nfeat), Nfactor=Nfactor, Nclusters=Nclusters)
    F, labels = model.unSupervisedLearner(inputs, Nepochs=10, Ninner=1, sparsity=Nsparse, bsize=batchsize, lamda=1)
    ctr = Counter(labels)
    print('clusters:', ''.join(str(l) for l in labels))
    print(ctr)
    nmi = normalized_mutual_info_score(target, labels)
    print('Normalized mutual information(NMI): {:.2f}%'.format(nmi * 100))
    acc = group_label_acc(target, labels)
    print('Group accuracy: {:.2f}%'.format(acc * 100))
    print('Number of clusters: {:d} '.format(len(ctr)))
    return acc, nmi
def testEYB20(loader):
    target, data, nfeat, nsamples = loader.eyb(20)
    inputs = torch.from_numpy(data).unsqueeze(1).float()
    ut = np.unique(target)
    print(nsamples, 'samples,', nfeat, 'dimensions,targets:', Counter(target))
    Nclusters = len(ut)
    Nsparse = 80
    batchsize = 100
    Nfactor = 15
    model = dmClustering(objectNet1d(dim=nfeat), Nfactor=Nfactor, Nclusters=Nclusters)
    F, labels = model.unSupervisedLearner(inputs, Nepochs=10, Ninner=1, sparsity=Nsparse, bsize=batchsize, lamda=1)
    ctr = Counter(labels)
    print('clusters:', ''.join(str(l) for l in labels))
    print(ctr)
    nmi = normalized_mutual_info_score(target, labels)
    print('Normalized mutual information(NMI): {:.2f}%'.format(nmi * 100))
    acc = group_label_acc(target, labels)
    print('Group accuracy: {:.2f}%'.format(acc * 100))
    print('Number of clusters: {:d} '.format(len(ctr)))
    return acc, nmi
def testEYB30(loader):
    target, data, nfeat, nsamples = loader.eyb(30)
    inputs = torch.from_numpy(data).unsqueeze(1).float()
    ut = np.unique(target)
    print(nsamples, 'samples,', nfeat, 'dimensions,targets:', Counter(target))
    Nclusters = len(ut)
    Nsparse = 50
    batchsize = 100
    Nfactor = 20
    model = dmClustering(objectNet1d(dim=nfeat), Nfactor=Nfactor, Nclusters=Nclusters)
    F, labels = model.unSupervisedLearner(inputs, Nepochs=10, Ninner=1, sparsity=Nsparse, bsize=batchsize, lamda=1)
    ctr = Counter(labels)
    print('clusters:', ''.join(str(l) for l in labels))
    print(ctr)
    nmi = normalized_mutual_info_score(target, labels)
    print('Normalized mutual information(NMI): {:.2f}%'.format(nmi * 100))
    acc = group_label_acc(target, labels)
    print('Group accuracy: {:.2f}%'.format(acc * 100))
    print('Number of clusters: {:d} '.format(len(ctr)))
    return acc, nmi
def testCol100(loader):
    target, data, nfeat, nsamples = loader.col100()
    inputs = torch.from_numpy(data).unsqueeze(1).float()
    ut = np.unique(target)
    print(nsamples, 'samples,', nfeat, 'dimensions,targets:', Counter(target))
    Nclusters = len(ut)
    Nsparse = 20
    batchsize = 50
    Nfactor = 10
    model = dmClustering(objectNet1d(dim=nfeat), Nfactor=Nfactor, Nclusters=Nclusters)
    F, labels = model.unSupervisedLearner(inputs, Nepochs=10, Ninner=1, sparsity=Nsparse, bsize=batchsize, lamda=1)
    ctr = Counter(labels)
    print('clusters:', ''.join(str(l) for l in labels))
    print(ctr)
    nmi = normalized_mutual_info_score(target, labels)
    print('Normalized mutual information(NMI): {:.2f}%'.format(nmi * 100))
    acc = group_label_acc(target, labels)
    print('Group accuracy: {:.2f}%'.format(acc * 100))
    print('Number of clusters: {:d} '.format(len(ctr)))
    return acc, nmi

def centering(inputs):
    N,dim=inputs.size()
    return inputs-inputs.mean(0).expand(N,dim)
def spcl(dfx,Nc,Nfactor):
    N = dfx.size(0)
    S = ((dfx + dfx.t()) / 2).float()
    degs = S.sum(1)
    Ds = torch.eye(N)
    Ds[Ds == 1] = degs
    Ls = Ds - S
    D = torch.eye(N)
    degs[degs == 0] = 0.000001
    D[D == 1] = 1 / (degs ** .5)
    Ls = D.mm(Ls).mm(D)
    _, v = Ls.symeig(eigenvectors=True)
    F = v[:, :Nfactor]
    #print(F[:10, :])
    return KMeans(n_clusters=Nc, random_state=0).fit(F.detach().numpy()).labels_,F.detach().numpy()

def testMoons(tosave=False):
    inputs, target = datasets.make_moons(n_samples=100, noise=.1)
    nsamples, nfeat = inputs.shape
    inputsTensor = torch.from_numpy(inputs).unsqueeze(1).float()
    ut = np.unique(target)
    print(nsamples, ' samples', nfeat, ' dimensions ,targets:', Counter(target))
    Nclusters = len(ut)
    Nsparse = 10
    Nfactor = 0
    batchsize = 10
    model = dmClustering(shapeNet(dim=nfeat), Nfactor=Nfactor, Nclusters=Nclusters)
    F, labels = model.unSupervisedLearner(inputsTensor, Nepochs=10, Ninner=1, sparsity=Nsparse, bsize=batchsize,
                                          lamda=1)
    conn = model.conn.numpy()
    print('clusters:', ''.join(str(l) for l in labels))
    print(Counter(labels))
    nmi = normalized_mutual_info_score(target, labels)
    print('Normalized mutual information(NMI): {:.2f}%'.format(nmi * 100))
    acc = group_label_acc(target, labels)
    print('Group accuracy: {:.2f}%'.format(acc * 100))

    # labels_kms=KMeans(n_clusters=2).fit(inputs).labels_
    model = SpectralClustering(n_clusters=2).fit(inputs)
    labels_spcl = model.labels_
    ss = torch.from_numpy(model.affinity_matrix_).float()
    N = ss.size(0)
    degs = ss.sum(1)
    Ds = torch.eye(N)
    Ds[Ds == 1] = degs
    Ls = Ds - ss
    _, v = Ls.symeig(eigenvectors=True)
    f_spcl = v[:, :2].numpy()
    fig, ax = plt.subplots(ncols=5, figsize=(20, 3))
    c = ['b', 'r', 'm']
    for i, l in enumerate(np.unique(target)):
        id = (l == target)
        ax[0].scatter(inputs[id, 0], inputs[id, 1], marker='.', color=c[i])
    ax[0].set_title('Two Moons')
    for i, l in enumerate(np.unique(labels_spcl)):
        id = (l == labels_spcl)
        ax[1].scatter(inputs[id, 0], inputs[id, 1], marker='.', color=c[i])
    ax[1].set_title('Spectral Clustering')
    for i, l in enumerate(np.unique(labels)):
        id = (l == labels)
        ax[2].scatter(f_spcl[id, 0], f_spcl[id, 1], marker='.', color=c[i])
    ax[2].set_title('Learnt Subspace')
    for i, l in enumerate(np.unique(labels)):
        id = (l == labels)
        ax[3].scatter(inputs[id, 0], inputs[id, 1], marker='.', color=c[i])
    ax[3].set_title('Proposed Clustering')
    for i, l in enumerate(np.unique(labels)):
        id = (l == labels)
        ax[4].scatter(F.detach().numpy()[id, 0], F.detach().numpy()[id, 1], marker='.', color=c[i])
    ax[4].set_title('Learnt Subspace')
    plt.tight_layout()
    if tosave:
        fig.savefig('ClusteringCompare-TwoMoons.png')
        return 1
    else:
        plt.show(block=False)
        time.sleep(5)
        return 1
def testThreeCircles(tosave=False):
    inputs1, target1 = datasets.make_circles(n_samples=200, noise=0.05, factor=.3)
    inputs2, target2 = datasets.make_circles(n_samples=200, noise=0.05, factor=.6)
    target2[target2 == 1] = 2
    inputs = np.vstack((inputs1, inputs2[target2 == 2, :]))
    target = np.hstack((target1, target2[target2 == 2]))

    nsamples, nfeat = inputs.shape
    inputsTensor = torch.from_numpy(inputs).unsqueeze(1).float()
    ut = np.unique(target)
    print(nsamples, ' samples', nfeat, ' dimensions ,targets:', Counter(target))
    Nclusters = len(ut)
    Nsparse = 10
    Nfactor = 0
    batchsize = 25
    model = dmClustering(shapeNet(dim=nfeat), Nfactor=Nfactor, Nclusters=Nclusters)
    F, labels = model.unSupervisedLearner(inputsTensor, Nepochs=8, Ninner=1, sparsity=Nsparse, bsize=batchsize, lamda=1)
    conn = model.conn.numpy()
    print('clusters:', ''.join(str(l) for l in labels))
    print(Counter(labels))
    nmi = normalized_mutual_info_score(target, labels)
    print('Normalized mutual information(NMI): {:.2f}%'.format(nmi * 100))
    acc = group_label_acc(target, labels)
    print('Group accuracy: {:.2f}%'.format(acc * 100))

    model = SpectralClustering(n_clusters=2).fit(inputs)
    labels_spcl = model.labels_
    ss = torch.from_numpy(model.affinity_matrix_).float()
    N = ss.size(0)
    degs = ss.sum(1)
    Ds = torch.eye(N)
    Ds[Ds == 1] = degs
    Ls = Ds - ss
    _, v = Ls.symeig(eigenvectors=True)
    f_spcl = v[:, :2].numpy()
    fig, ax = plt.subplots(ncols=5, figsize=(20, 3))
    c = ['b', 'r', 'm']
    for i, l in enumerate(np.unique(target)):
        id = (l == target)
        ax[0].scatter(inputs[id, 0], inputs[id, 1], marker='.', color=c[i])
    ax[0].set_title('Three Circles')
    for i, l in enumerate(np.unique(labels_spcl)):
        id = (l == labels_spcl)
        ax[1].scatter(inputs[id, 0], inputs[id, 1], marker='.', color=c[i])
    ax[1].set_title('Spectral Clustering')
    for i, l in enumerate(np.unique(labels)):
        id = (l == labels)
        ax[2].scatter(f_spcl[id, 0], f_spcl[id, 1], marker='.', color=c[i])
    ax[2].set_title('Learnt Subspace')
    for i, l in enumerate(np.unique(labels)):
        id = (l == labels)
        ax[3].scatter(inputs[id, 0], inputs[id, 1], marker='.', color=c[i])
    ax[3].set_title('Proposed Clustering')
    for i, l in enumerate(np.unique(labels)):
        id = (l == labels)
        ax[4].scatter(F.detach().numpy()[id, 0], F.detach().numpy()[id, 1], marker='.', color=c[i])
    ax[4].set_title('Learnt Subspace')
    plt.tight_layout()
    if tosave:
        fig.savefig('ClusteringCompare-ThreeCircles.png')
        return 1
    else:
        plt.show(block=False)
        time.sleep(5)
        return 1
def testBlobs(tosave=False):
    inputs, target = datasets.make_blobs(n_samples=600, cluster_std=[2.0, 1.5, 1.0])
    nsamples, nfeat = inputs.shape
    inputsTensor = torch.from_numpy(inputs).unsqueeze(1).float()
    ut = np.unique(target)
    print(nsamples, ' samples', nfeat, ' dimensions ,targets:', Counter(target))
    Nclusters = len(ut)
    Nsparse = 10
    Nfactor = 0
    batchsize = 25
    model = dmClustering(shapeNet(dim=nfeat), Nfactor=Nfactor, Nclusters=Nclusters)
    F, labels = model.unSupervisedLearner(inputsTensor, Nepochs=6, Ninner=1, sparsity=Nsparse, bsize=batchsize, lamda=1)
    conn = model.conn.numpy()
    print('clusters:', ''.join(str(l) for l in labels))
    print(Counter(labels))
    nmi = normalized_mutual_info_score(target, labels)
    print('Normalized mutual information(NMI): {:.2f}%'.format(nmi * 100))
    acc = group_label_acc(target, labels)
    print('Group accuracy: {:.2f}%'.format(acc * 100))

    model = SpectralClustering(n_clusters=2).fit(inputs)
    labels_spcl = model.labels_
    ss = torch.from_numpy(model.affinity_matrix_).float()
    N = ss.size(0)
    degs = ss.sum(1)
    Ds = torch.eye(N)
    Ds[Ds == 1] = degs
    Ls = Ds - ss
    _, v = Ls.symeig(eigenvectors=True)
    f_spcl = v[:, :2].numpy()
    fig, ax = plt.subplots(ncols=5, figsize=(20, 3))
    c = ['b', 'r', 'm']
    for i, l in enumerate(np.unique(target)):
        id = (l == target)
        ax[0].scatter(inputs[id, 0], inputs[id, 1], marker='.', color=c[i])
    ax[0].set_title('Three Blobs')
    for i, l in enumerate(np.unique(labels_spcl)):
        id = (l == labels_spcl)
        ax[1].scatter(inputs[id, 0], inputs[id, 1], marker='.', color=c[i])
    ax[1].set_title('Spectral Clustering')
    for i, l in enumerate(np.unique(labels)):
        id = (l == labels)
        ax[2].scatter(f_spcl[id, 0], f_spcl[id, 1], marker='.', color=c[i])
    ax[2].set_title('Learnt Subspace')
    for i, l in enumerate(np.unique(labels)):
        id = (l == labels)
        ax[3].scatter(inputs[id, 0], inputs[id, 1], marker='.', color=c[i])
    ax[3].set_title('Proposed Clustering')
    for i, l in enumerate(np.unique(labels)):
        id = (l == labels)
        ax[4].scatter(F.detach().numpy()[id, 0], F.detach().numpy()[id, 1], marker='.', color=c[i])
    ax[4].set_title('Learnt Subspace')
    plt.tight_layout()
    if tosave:
        fig.savefig('ClusteringCompare-ThreeBlobs.png')
        return 1
    else:
        plt.show(block=False)
        time.sleep(5)
        return 1

if __name__=="__main__":
    folder = "F:/datasets/UCIml/ClusteringTest"
    loader=dataReader(folder=folder)

    acc, nmi = testEYB30(loader)
    #acc, nmi = testUsps(loader,pretrained=True)
    #acc,nmi=avgPred(10, loader1d, testSpiral)
    #print('Max-vote Clustering accuracy: {:.2f}%, NMI: {:.2f}%'.format(acc * 100,nmi*100 ))


