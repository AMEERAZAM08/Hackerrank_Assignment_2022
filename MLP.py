from symbol import eval_input
import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim


device  =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device,torch.cuda.get_device_name(0))
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#make trainloader using  datframe 
def  make_trainloader(df,train_size=0.8,batch_size=64):
    train_size=int(len(df)*train_size)
    train_dataset=df[:train_size]
    test_dataset=df[train_size:]
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    return  train_loader,test_loader




#MLP for  Binary classification using  Sigmoid  
class  MLP(nn.Module):
    def  __init__(self,input_dim,output_dim):
        super(MLP,self).__init__()
        self.fc1=nn.Linear(input_dim,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,output_dim)
    def  forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=torch.sigmoid(self.fc3(x))
        return  x

#train script 
def  train(model,device,train_loader,optimizer,epoch):
    model.train()
    for  batch_idx,(x,y) in  enumerate(train_loader):
        x=x.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        output=model(x)
        loss=F.binary_cross_entropy(output,y)
        loss.backward()
        optimizer.step()
        if  batch_idx % 100==0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,batch_idx*len(x),len(train_loader.dataset),
                100.*batch_idx/len(train_loader),loss.item()
            ))




#train
def main():
    #load data
    df=pd.read_csv("data.csv")
    train_loader,test_loader=make_trainloader(df)
    #model
    model=MLP(input_dim=2,output_dim=2).to(device)
    optimizer=optim.Adam(model.parameters(),lr=1e-3)
    #train
    for  epoch in range(1,10):
        train(model,device,train_loader,optimizer,epoch)
    #save model
    torch.save(model.state_dict(),"model.pth")


#list to tensor using torch  =>  torch.tensor(list)



#binary  classification  for  0,1
def train(model,device,train_loader,optimizer,epoch):
    model.train()
    for batch_idx,(x,y) in enumerate(train_loader):
        x=x.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        output=model(x)
        loss=F.binary_cross_entropy(output,y)
        loss.backward()
        optimizer.step()
        if batch_idx % 100==0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,batch_idx*len(x),len(train_loader.dataset),
                100.*batch_idx/len(train_loader),loss.item()
            ))
        #accuracy if output >0.5 => 1 else 0 in new pred list
        pred=output>0.5
        correct=pred.eq(y.view_as(pred)).sum().item()
        print("Train Accuracy: {}/{} ({:.0f}%)".format(
            correct,len(train_loader.dataset),
            100.*correct/len(train_loader.dataset)
        ))


#test
def test(model,device,test_loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for x,y in test_loader:
            x=x.to(device)
            y=y.to(device)
            output=model(x)
            test_loss+=F.binary_cross_entropy(output,y,reduction="sum").item()
            pred=output>0.5
            correct+=pred.eq(y.view_as(pred)).sum().item()
    test_loss/=len(eval_dataset)
    print("Test Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
        test_loss,correct,16,
        100.*correct/16
    ))

# todo type hinting TODO, FIXME , NOTE, NOTE ,ERROR , WARNING, 
# 








#make dataloader using torch.utils.data.DataLoader
def make_trainloader(df,train_size=0.8,batch_size=64):
    train_size=int(len(df)*train_size)
    train_dataset=df[:train_size]
    test_dataset=df[train_size:]
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    return train_loader,test_loader

