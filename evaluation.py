import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

eps=1e-3
def evaluation(model,testLoader,device):
    TP,TN,FP,FN=0,0,0,0
    for i,(input_tensors,target_tensors) in enumerate(testLoader):
        output_tensors=model(input_tensors.to(device))
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

