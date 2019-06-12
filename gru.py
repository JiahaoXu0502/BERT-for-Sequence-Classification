import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUClassifier(nn.Module):
    def __init__(self,num_words,hidden_length,num_layers):
        super(GRUClassifier, self).__init__()
        self.embedding=nn.Embedding(num_words,hidden_length)
        self.gru=nn.GRU(hidden_length,hidden_length,num_layers,batch_first=True)
        self.linear=nn.Linear(hidden_length,2)

    def forward(self, input):
        out=self.embedding(input)
        out,hidden=self.gru(out)
        out=self.linear(out[:,-1,:])
        return out


class AttnGRU(nn.Module):
    def __init__(self,num_words,hidden_dim,num_layers,kqv_dim):
        super(AttnGRU, self).__init__()

        self.embedding=nn.Embedding(num_words,hidden_dim)
        self.gru=nn.GRU(hidden_dim,hidden_dim,num_layers,batch_first=True)
        self.attn=Attn(hidden_dim,kqv_dim)
        self.linear=nn.Linear(hidden_dim,2)

    def forward(self,input):
        out=self.embedding(input)
        out=self.attn(out)
        out,hidden=self.gru(out)
        out=self.linear(out[:,-1,:])
        return out


class MultiHead(nn.Module):
    def __init__(self,hidden_dim,kqv_dim,num_heads):
        super(MultiHead, self).__init__()


class Attn(nn.Module):
    def __init__(self,hidden_dim,kqv_dim):
        super(Attn, self).__init__()
        self.wk=nn.Linear(hidden_dim,kqv_dim)
        self.wq=nn.Linear(hidden_dim,kqv_dim)
        self.wv=nn.Linear(hidden_dim,kqv_dim)
        self.d=kqv_dim**0.5

    def forward(self, input):
        '''
        :param input: batch_size x seq_len x hidden_dim
        :return:
        '''
        k=self.wk(input)
        q=self.wq(input)
        v=self.wv(input)
        w=F.softmax(torch.bmm(q,k.transpose(-1,-2))/self.d,dim=-1)
        attn=torch.bmm(w,v)

        return attn
