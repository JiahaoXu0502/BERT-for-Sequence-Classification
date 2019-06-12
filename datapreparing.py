import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

def dataPreparing(path,length,ratio,batch_size):
    langdic,dataset,_,line_count=readFile(path)
    data_mat,label_mat=np.zeros([line_count,length]),np.zeros([line_count,1])
    for row,(seq,label) in enumerate(dataset):
        tmp=[langdic.getWordId(c) for c in seq]
        tmp=np.array(normLen(tmp,length))
        data_mat[row]=tmp
        label_mat[row]=label
    trainLoader,testLoader=genLoader(data_mat,label_mat,ratio,batch_size)
    return langdic,trainLoader,testLoader

class LangDic():
    def __init__(self):
        self.word2id={}
        self.id2word={}
        self.word2count={}
        self.num_words=0

    def addWord(self,word):
        if word not in self.word2id:
            self.word2id[word]=self.num_words
            self.id2word[self.num_words]=word
            self.num_words+=1
            self.word2count[word]=1
        else:
            self.word2count[word]+=1

    def addSeq(self,seq):
        '''
        :param seq: a sequence list
        '''
        for c in seq:
            self.addWord(c)

    def getWordId(self,word):
        if word in self.word2id:
            return self.word2id[word]
        else:
            return 0

    def getSeqId(self,seq):
        '''
        :param seq: sequence list
        :return:
        '''
        l=np.zeros(len(seq))
        for i,w in enumerate(seq):
            l[i]=self.getWordId(w)
        return l


def readFile(path):
    '''
    :param path: file name, string
    :return: dic, dataset, line_count
    '''
    f=open(path,'r')
    langdic=LangDic()
    line_count=0
    dataset=[]
    seqset=[]
    for line in f:
        tmp=line.split()
        seq,label=tmp[:-1],int(tmp[-1])
        seqset.append((' '.join(seq),label))
        langdic.addSeq(seq)
        dataset.append((seq,label))
        line_count+=1
    return langdic,dataset,seqset,line_count

def normLen(seq,length):
    if len(seq)>=length:
        return seq[:length]
    else:
        return seq+[0]*(length-len(seq))

def genLoader(data_mat,label_mat,ratio,batch_size):
    num=int(len(data_mat)*ratio)
    trainset,testset=ImdbDataset(data_mat[:num],label_mat[:num]),\
                     ImdbDataset(data_mat[num:],label_mat[num:])
    trainLoader,testLoader=DataLoader(trainset,batch_size,shuffle=True),DataLoader(testset,batch_size,shuffle=True)
    return trainLoader,testLoader

class ImdbDataset(Dataset):
    def __init__(self,input_data,target_data):
        self.input_tensor=input_data
        self.target_tensor=target_data

    def __len__(self):
        return len(self.input_tensor)

    def __getitem__(self, item):
        return torch.tensor(self.input_tensor[item],dtype=torch.long),\
               torch.tensor(self.target_tensor[item],dtype=torch.long)

def test():
    path='imdb_labelled.txt'
    length=32
    ratio=0.7
    trainloader,testloader=dataPreparing(path,length,ratio,batch_size=4)
    print(trainloader,testloader)