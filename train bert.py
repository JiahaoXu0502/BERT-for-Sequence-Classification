from datapreparing import readFile
from pytorch_pretrained_bert.tokenization import BertTokenizer
from bert import BertForSequenceClassification

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler,TensorDataset
import logging


logger = logging.getLogger(__name__)

#bert param
bert_model='bert-base-uncased'
do_lower_case=True
cache_dir=os.getcwd()
num_labels=2

#train param
use_cuda=False
max_seq_len=32
path='imdb_labelled.txt'
ratio=0.7
batch_size=32
num_epoch=10
lr=1e-5

if use_cuda:
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device=torch.device('cpu')
print(device)

_,_,seqset,line_count=readFile(path)
tokenizer=BertTokenizer.from_pretrained(bert_model,do_lower_case=do_lower_case)

def dataset2Loader(dataset,tokenizer,batch_size):

    all_input_ids,all_input_mask,all_segment_ids,all_label_ids=[],[],[],[]
    for i,item in enumerate(dataset):
        tokens_a=tokenizer.tokenize(item[0])
        tokens_b=None

        if len(tokens_a)>max_seq_len-2:
            tokens_a = tokens_a[:(max_seq_len - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_len - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

        label=item[1]

        if i < 5:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s " % (label))

        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_segment_ids.append(segment_ids)
        all_label_ids.append(label)
    all_input_ids=torch.tensor(all_input_ids,dtype=torch.long)
    all_input_mask=torch.tensor(all_input_mask,dtype=torch.long)
    all_segment_ids=torch.tensor(all_segment_ids,dtype=torch.long)
    all_label_ids=torch.tensor(all_label_ids,dtype=torch.long)
    set=TensorDataset(all_input_ids,all_input_mask,all_segment_ids,all_label_ids)
    sampler=RandomSampler(set)
    dataloader=DataLoader(set,sampler=sampler,batch_size=batch_size)
    return dataloader

def prepareBertDataLoader(dataset,tokenizer,batch_size,ratio):
    num_train_samples=int(len(dataset)*ratio)
    train_set,eval_set=dataset[:num_train_samples],dataset[num_train_samples:]
    train_dataloader=dataset2Loader(train_set,tokenizer,batch_size)
    eval_dataloader=dataset2Loader(eval_set,tokenizer,batch_size)
    return train_dataloader,eval_dataloader

import torch
import torch.nn as nn

from evaluation import eval,figPlot,evaluation


def trainStep(model,input_tensor,target_tensor,testLoader,optimizer,device,
              criterion=nn.CrossEntropyLoss()):
    '''
    :param model: no need to use device
    :param input_tensor:
    :param target_tensor:
    :param optimizer:
    :param criterion:
    :return:
    '''
    input_tensors, mask, segment_id=input_tensor
    output_tensor=model(input_tensors, segment_id, mask)

    # p,r,f1=eval(output_tensor,target_tensor)
    p,r,f1=evaluation(model,testLoader,device)

    loss=criterion(output_tensor,target_tensor.squeeze())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return model,loss,p,r,f1

def train(model,dataloader,testLoader,num_epoch,optimizer,criterion,
          device,model_name):

    trained_model=model
    L,P,R,F1=[],[],[],[]
    for epoch in range(num_epoch):
        for step,(input_tensors,mask,segment_id,target_tensors) \
                in enumerate(dataloader):
            input_tensors=input_tensors.to(device)
            mask=mask.to(device)
            segment_id=segment_id.to(device)
            target_tensors=target_tensors.to(device)

            trained_model,loss,p,r,f1=trainStep(trained_model,
                                                (input_tensors,mask,segment_id),
                                                target_tensors,testLoader,optimizer,device,criterion)
            L.append(loss)
            P.append(p)
            R.append(r)
            F1.append(f1)
            print('epoch:%d    step:%d    loss:%.2f    '
                  'p:%.2f    r:%.2f    f1:%.2f'
                  %(epoch,step,loss,p,r,f1))

    figPlot(L,P,R,F1,model_name)
    torch.save(model,model_name)
    return model

import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

eps=1e-3
def evaluation(model,testLoader,device):
    TP,TN,FP,FN=0,0,0,0
    for i,(input_tensors, mask, segment_id,target_tensors) \
            in enumerate(testLoader):
        input_tensors, mask, segment_id=input_tensors.to(device), \
                                        mask.to(device), segment_id.to(device)
        output_tensors=model(input_tensors,segment_id,mask)
        tp,tn,fp,fn=getTPTNFPFN(output_tensors,target_tensors)
        TP+=tp
        TN+=tn
        FP+=fp
        FN+=fn
    p, r = TP / (TP + FP+eps), TP / (TP + FN+eps)
    if p+r==0:
        f1=0
    else:
        f1 = 2 * p * r / (p + r)
    # print('precision:%.2f    recall:%.2f    f1:%.2f'%(p,r,f1))
    return p, r, f1

def getTPTNFPFN(predicted_tensor,target_tensor):
    TP,TN,FP,FN=0,0,0,0
    for p,t in zip(predicted_tensor,target_tensor):
        if t==1:
            if p[1]>=p[0]:
                TP+=1
            else:
                FN+=1
        elif t==0:
            if p[1]>=p[0]:
                FP+=1
            else:
                TN+=1
    return TP,TN,FP,FN

def eval(predicted_tensor,target_tensor):
    TP,TN,FP,FN=getTPTNFPFN(predicted_tensor,target_tensor)
    p,r=TP/(TP+FP+eps),TP/(TP+FN+eps)
    if p*r==0:
        f1=0
    else:
        f1=2*p*r/(p+r)
    return p,r,f1

def figPlot(loss,p,r,f1,title):
    index=np.arange(0,len(loss),1)

    fig=plt.figure()
    ax_loss=fig.add_subplot(111)
    ax_loss.plot(index,loss)
    ax_loss.set_ylabel('Loss')
    ax_loss.set_xlabel('Number of Steps')
    ax_loss.set_title('Training Loss of '+title)
    ax_eval=ax_loss.twinx()
    ax_eval.plot(index,f1,'r')
    ax_eval.set_ylabel('F1 Score')

    fig.savefig(title)
    plt.show()


def testFigPlot():
    loss=[0.9,0.4,0.1]
    p=[0,0.5,0.6]
    r=[0,0.6,0.5]
    f1=[0,0.7,0.4]
    figPlot(loss,p,r,f1,'gru')




train_loader,eval_loader=prepareBertDataLoader(seqset,tokenizer,batch_size,ratio)
model=BertForSequenceClassification.from_pretrained(
    bert_model,cache_dir=cache_dir,num_labels=num_labels).to(device)

optimizer=optim.Adam(model.parameters(),lr)
criterion=nn.CrossEntropyLoss()
model=train(model,train_loader,eval_loader,num_epoch,
            optimizer,criterion,device,model_name=bert_model)





