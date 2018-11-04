from scipy.io import loadmat
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split

class dataReader():
    def __init__(self,folder):
        self._msrcv,self._cal7,self._cal20={},{},{}
        self._orl,self._col20,self._col100={},{},{}
        self._eyb={}
        self._eyb['10'],self._eyb['20'],self._eyb['30']= {},{},{}
        files= glob(folder+'\*')
        for f in files:
            if 'spiral' in f: self._spiral=f
            if 'path' in f: self._path=f
            if 'Compound' in f: self._comp = f
            if 'yeast.data' in f: self._yeast= f
            if 'glass' in f: self._glass=f
            if 'ecoli' in f: self._ecoli=f
            if 'movement' in f: self._move=f
            if 'coiltr' in f: self._coiltr=f
            if 'coilte' in f: self._coilte = f
            if 'stock' in f: self._stock=f
            if 'coil20_data' in f: self._coil20=f
            if 'umist_transformed' in f: self._umist=f
            if 'jaffe_data' in f: self._jaffe=f
            if 'wine' in f: self._wine=f
            if 'usps_data' in f: self._usps= f

            if 'caltech7' in f:
                if 'trainfeature' in f: self._cal7['fttr']=f
                if 'testfeature' in f: self._cal7['ftte'] = f
                if 'trainlabel' in f: self._cal7['labeltr']=f
                if 'testlabel' in f: self._cal7['labelte'] = f
            if 'caltech20' in f:
                if 'trainfeature' in f: self._cal20['fttr']=f
                if 'testfeature' in f: self._cal20['ftte'] = f
                if 'trainlabel' in f: self._cal20['labeltr']=f
                if 'testlabel' in f: self._cal20['labelte'] = f
            if 'MSRCV' in f:
                if 'trainfeature' in f: self._msrcv['fttr']=f
                if 'testfeature' in f: self._msrcv['ftte'] = f
                if 'trainlabel' in f: self._msrcv['labeltr']=f
                if 'testlabel' in f: self._msrcv['labelte'] = f
            if 'MSRCV' in f:
                if 'trainfeature' in f: self._msrcv['fttr']=f
                if 'testfeature' in f: self._msrcv['ftte'] = f
                if 'trainlabel' in f: self._msrcv['labeltr']=f
                if 'testlabel' in f: self._msrcv['labelte'] = f
            if 'orl' in f:
                if 'trainfeature' in f: self._orl['fttr']=f
                if 'testfeature' in f: self._orl['ftte'] = f
                if 'trainlabel' in f: self._orl['labeltr']=f
                if 'testlabel' in f: self._orl['labelte'] = f
            if 'coil20' in f and '.npy' in f:
                if 'trainfeature' in f: self._col20['fttr']=f
                if 'testfeature' in f: self._col20['ftte'] = f
                if 'trainlabel' in f: self._col20['labeltr']=f
                if 'testlabel' in f: self._col20['labelte'] = f
            if 'coil100' in f and '.npy' in f:
                if 'trainfeature' in f: self._col100['fttr']=f
                if 'testfeature' in f: self._col100['ftte'] = f
                if 'trainlabel' in f: self._col100['labeltr']=f
                if 'testlabel' in f: self._col100['labelte'] = f
            if 'EYB' in f and '.npy' in f:
                if '10trainfeature' in f: self._eyb['10']['fttr']=f
                if '10testfeature' in f: self._eyb['10']['ftte'] = f
                if '10trainlabel' in f: self._eyb['10']['labeltr']=f
                if '10testlabel' in f: self._eyb['10']['labelte'] = f
                if '20trainfeature' in f: self._eyb['20']['fttr']=f
                if '20testfeature' in f: self._eyb['20']['ftte'] = f
                if '20trainlabel' in f: self._eyb['20']['labeltr']=f
                if '20testlabel' in f: self._eyb['20']['labelte'] = f
                if '30trainfeature' in f: self._eyb['30']['fttr']=f
                if '30testfeature' in f: self._eyb['30']['ftte'] = f
                if '30trainlabel' in f: self._eyb['30']['labeltr']=f
                if '30testlabel' in f: self._eyb['30']['labelte'] = f

    def spiral(self):
        mat=np.loadtxt(self._spiral)
        nsamples,nfeat=mat.shape
        return mat[:,-1], mat[:,:-1], nfeat-1, nsamples
    def path(self):
        mat=np.loadtxt(self._path)
        nsamples,nfeat=mat.shape
        return mat[:,-1], mat[:,:-1], nfeat-1, nsamples
    def compound(self):
        mat = np.loadtxt(self._comp)
        nsamples, nfeat = mat.shape
        return mat[:, -1], mat[:, :-1], nfeat-1, nsamples
    def yeast(self):
        mat=np.genfromtxt(self._yeast,dtype='str')
        labels=list(np.unique(mat[:,-1]))
        target=np.array([labels.index(val) for val in mat[:,-1]])
        mat = np.genfromtxt(self._yeast)[:,1:-1]
        nsamples, nfeat= mat.shape
        return target, mat,  nfeat, nsamples
    def glass(self):
        mat = np.loadtxt(self._glass,delimiter=',')[1:]
        nsamples, nfeat= mat.shape
        return mat[:,-1], mat[:,:-1],  nfeat-1, nsamples
    def ecoli(self):
        mat = np.genfromtxt(self._ecoli, dtype='str')
        labels = list(np.unique(mat[:, -1]))
        target = np.array([labels.index(val) for val in mat[:, -1]])
        mat = np.genfromtxt(self._ecoli)[:,1:-1]
        nsamples, nfeat = mat.shape
        return target, mat, nfeat, nsamples
    def movement(self):
        mat = np.genfromtxt(self._move, delimiter=',')
        nsamples, nfeat = mat.shape
        return mat[:,-1], mat[:,:-1], nfeat-1, nsamples
    def coil(self):
        #mattr=loadmat(self._coiltr)
        mat= loadmat(self._coilte)
        nsamples,nfeat= mat['X'].shape
        return mat['t'].transpose()[0], mat['X'],  nfeat, nsamples
    def stock(self):
        mat= np.genfromtxt(self._stock,dtype='str',delimiter=',')
        labels = list(np.unique(mat[1:, 1]))
        target = np.array([labels.index(val) for val in mat[1:, 1]])
        data=[]
        for line in mat[1:,2:]:
            l=[]
            for s in line:
                if '$' in s: l.append(float(s[1:]))
                elif not s: l.append(0)
                elif '/' in s: l.append(float(s.split('/')[1]))
                else: l.append(float(s))
            data.append(l)
        data=np.array(data)
        #data[:,5]= data[:,7]-data[:,5]
        data=np.delete(data,[5,8],1)
        nsamples,nfeat =data.shape
        return target, data, nfeat, nsamples
    def coil20(self):
        mat = loadmat(self._coil20)
        nsamples,nfeat= mat['images'].shape
        return mat['targets'][0],mat['images'], nfeat, nsamples
    def wine(self):
        mat = np.loadtxt(self._wine, delimiter=',')
        nsamples, nfeat = mat.shape
        return mat[:, 0], mat[:, 1:], nfeat - 1, nsamples

    def umist(self):
        mat = loadmat(self._umist)
        nsamples,nfeat= mat['data'].shape
        return mat['target'][0],mat['data'], nfeat, nsamples
    def jaffe(self):
        mat= loadmat(self._jaffe)
        nsamples, nfeat = mat['faces'].shape
        return mat['targets'][0], mat['faces'], nfeat, nsamples
    def usps(self, size=.1):
        mat = loadmat(self._usps)
        _, X_test, _, y_test = train_test_split(mat['images'], mat['t'][0], test_size = size)
        nsamples, nfeat = X_test.shape
        return y_test, X_test, nfeat, nsamples

    def msrcv(self):
        tr,te,ltr,lte= np.load(self._msrcv['fttr']),np.load(self._msrcv['ftte']),\
                       np.load(self._msrcv['labeltr']),np.load(self._msrcv['labelte'])
        data= np.vstack((tr,te))
        target=np.hstack((ltr,lte))
        nsamples,nfeat= data.shape
        return target,data,nfeat,nsamples
    def cal7(self):
        tr,te,ltr,lte= np.load(self._cal7['fttr']),np.load(self._cal7['ftte']),\
                       np.load(self._cal7['labeltr']),np.load(self._cal7['labelte'])
        data= np.vstack((tr,te))
        target=np.hstack((ltr,lte))
        nsamples,nfeat= data.shape
        return target,data,nfeat,nsamples
    def cal20(self):
        tr,te,ltr,lte= np.load(self._cal20['fttr']),np.load(self._cal20['ftte']),\
                       np.load(self._cal20['labeltr']),np.load(self._cal20['labelte'])
        data= np.vstack((tr,te))
        target=np.hstack((ltr,lte))
        nsamples,nfeat= data.shape
        return target,data,nfeat,nsamples
    def orl(self):
        tr,te,ltr,lte= np.load(self._orl['fttr']),np.load(self._orl['ftte']),\
                       np.load(self._orl['labeltr']),np.load(self._orl['labelte'])
        data= np.vstack((tr,te))
        target=np.hstack((ltr,lte))
        nsamples,nfeat= data.shape
        return target,data,nfeat,nsamples
    def col20(self):
        tr,te,ltr,lte= np.load(self._col20['fttr']),np.load(self._col20['ftte']),\
                       np.load(self._col20['labeltr']),np.load(self._col20['labelte'])
        data= np.vstack((tr,te))
        target=np.hstack((ltr,lte))
        nsamples,nfeat= data.shape
        return target,data,nfeat,nsamples
    def col100(self):
        tr,te,ltr,lte= np.load(self._col100['fttr']),np.load(self._col100['ftte']),\
                       np.load(self._col100['labeltr']),np.load(self._col100['labelte'])
        data= np.vstack((tr,te))
        target=np.hstack((ltr,lte))
        nsamples,nfeat= data.shape
        return target,data,nfeat,nsamples

    def eyb(self,num=10):
        tr,te,ltr,lte= np.load(self._eyb[str(num)]['fttr']),np.load(self._eyb[str(num)]['ftte']),\
                       np.load(self._eyb[str(num)]['labeltr']),np.load(self._eyb[str(num)]['labelte'])
        data= np.vstack((tr,te))
        target=np.hstack((ltr,lte))
        nsamples,nfeat= data.shape
        return target,data,nfeat,nsamples

if __name__ == "__main__":
    folder='F:/datasets/UCIml/ClusteringTest'
    loader=dataReader(folder=folder)
    target, inputs, nfeat, nsamples = loader.spiral()
    print('spiral:',end=' ')
    print(nfeat, 'dimensions, ',nsamples,' samples')
    target, inputs, nfeat, nsamples = loader.path()
    print('pathbased:',end=' ')
    print(nfeat, 'dimensions, ', nsamples, ' samples')
    target, inputs, nfeat, nsamples = loader.compound()
    print('Compound:',end=' ')
    print(nfeat, 'dimensions, ', nsamples, ' samples')

    target, inputs, nfeat, nsamples = loader.yeast()
    print('yeast:',end=' ')
    print(nfeat, 'dimensions, ', nsamples, ' samples')
    target, inputs, nfeat, nsamples = loader.glass()
    print('glass:',end=' ')
    print(nfeat, 'dimensions, ', nsamples, ' samples')
    target, inputs, nfeat, nsamples = loader.ecoli()
    print('ecoli:',end=' ')
    print(nfeat, 'dimensions, ', nsamples, ' samples')
    target, inputs, nfeat, nsamples = loader.movement()
    print('movements:',end=' ')
    print(nfeat, 'dimensions, ', nsamples, ' samples')
    target, inputs, nfeat, nsamples = loader.coil()
    print('coil:',end=' ')
    print(nfeat, 'dimensions, ', nsamples, ' samples')
    target, inputs, nfeat, nsamples = loader.stock()
    print('stock:',end=' ')
    print(nfeat, 'dimensions, ', nsamples, ' samples')
    target, inputs, nfeat, nsamples = loader.coil20()
    print('coil20:',end=' ')
    print(nfeat, 'dimensions, ', nsamples, ' samples')
    target, inputs, nfeat, nsamples = loader.wine()
    print('wine:',end=' ')
    print(nfeat, 'dimensions, ', nsamples, ' samples')

    target, inputs, nfeat, nsamples = loader.umist()
    print('umist:',end=' ')
    print(nfeat, 'dimensions, ', nsamples, ' samples')
    target, inputs, nfeat, nsamples = loader.jaffe()
    print('jaffe:',end=' ')
    print(nfeat, 'dimensions, ', nsamples, ' samples')
    target, inputs, nfeat, nsamples = loader.usps(size=.1)
    print('usps:', end=' ')
    print(nfeat, 'dimensions, ', nsamples, ' samples')

    target, inputs, nfeat, nsamples = loader.eyb(10)
    print('eyb10:', end=' ')
    print(nfeat, 'dimensions, ', nsamples, ' samples')
    target, inputs, nfeat, nsamples = loader.eyb(20)
    print('eyb20:', end=' ')
    print(nfeat, 'dimensions, ', nsamples, ' samples')
    target, inputs, nfeat, nsamples = loader.eyb(30)
    print('eyb30:', end=' ')
    print(nfeat, 'dimensions, ', nsamples, ' samples')

