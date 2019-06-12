from train import train
from gru import GRUClassifier,AttnGRU
from datapreparing import dataPreparing

import torch
import torch.optim as optim
import torch.nn as nn

#gru parameters:
hidden_len=128
gru_layers=6
kqv_dim=128
model_name='6 Layers Attn GRU Model'

#train parameters:
seq_len=32
path='imdb_labelled.txt'
ratio=0.7
batch_size=512
num_epoch=100
lr=0.01
warmup_proportion,
t_total=num_train_optimization_steps
use_cuda=True

if use_cuda:
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device=torch.device('cpu')
print(device)

langdic,trainloader,testloader=dataPreparing(path,seq_len,ratio,batch_size)

model=GRUClassifier(len(langdic.word2id),hidden_len,gru_layers).to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=lr,
                                 warmup=warmup_proportion,
                                 t_total=num_train_optimization_steps)
optimizer=optim.Adam(model.parameters(),lr)
criterion=nn.CrossEntropyLoss()



model=train(model,trainloader,testloader,num_epoch,optimizer,criterion,device,model_name)
