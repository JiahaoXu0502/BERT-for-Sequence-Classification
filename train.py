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

    output_tensor=model(input_tensor)

    # p,r,f1=eval(output_tensor,target_tensor)
    p,r,f1=evaluation(model,testLoader,device)

    loss=criterion(output_tensor,target_tensor.squeeze())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return model,loss,p,r,f1

def train(model,dataloader,testLoader,num_epoch,optimizer,criterion,device,model_name):

    trained_model=model
    L,P,R,F1=[],[],[],[]
    for epoch in range(num_epoch):
        for step,(input_tensors,target_tensors) in enumerate(dataloader):
            input_tensors=input_tensors.to(device)
            target_tensors=target_tensors.to(device)

            trained_model,loss,p,r,f1=trainStep(trained_model,input_tensors,
                                                target_tensors,testLoader,optimizer,device,criterion)
            L.append(loss)
            P.append(p)
            R.append(r)
            F1.append(f1)
            print('epoch:%d    step:%d    loss:%.2f    p:%.2f    r:%.2f    f1:%.2f'
                  %(epoch,step,loss,p,r,f1))

    figPlot(L,P,R,F1,model_name)
    torch.save(model,model_name)
    return model




