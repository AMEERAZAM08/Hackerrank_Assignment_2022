import  tqdm 
import  torch
import  torch.nn as nn
import  torch.optim as optim
import  torchvision
import  torch.nn.functional as F  
from  model import  VAE
from  torchvision import  transforms
from  torchvision.utils import  save_image
from  torch.utils.data import  DataLoader
from  torchvision.datasets import  MNIST



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM=784
Z_DIM=20
H_DIM=200
BATCH_SIZE=128
EPOCHS=10
LR_RATE=1e-3
# print(device,torch.cuda.get_device_name(0))

dataset=MNIST(root="dataset/",transform=transforms.ToTensor(),download=True)

train_loader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)
model=VAE(input_dim=INPUT_DIM,z_dim=Z_DIM,h_dim=H_DIM).to(device)
optimizer=optim.Adam(model.parameters(),lr=LR_RATE)


loss_fn=nn.BCELoss(reduction="sum")#y_i=1

#start Training
for epoch in range(EPOCHS):
    loop=tqdm.tqdm(enumerate(train_loader))
    for batch_idx,(data,_) in loop:
        #
        x=data.view(-1,INPUT_DIM).to(device)
        x_contructed,mean,sigma=model(x)
        #loss
        recon_loss=   loss_fn(x_contructed,x)
        kl_divergence=0.5*torch.sum(mean**2+sigma**2-torch.log(1e-8+sigma**2)-1)
        loss=recon_loss+kl_divergence
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print
        loop.set_description(f"Epoch[{epoch}/{EPOCHS}]")
        loop.set_postfix(loss=loss.item())
#save model
torch.save(model.state_dict(),"vae.pth")

#save image
with torch.no_grad():
    n=16
    sample=torch.randn(n,Z_DIM).to(device)
    sample=model.decode(sample).view(-1,1,28,28)
    save_image(sample,"sample.png",nrow=n)





# url="https://drive.google.com/file/d/1-9NKEJ4j6FDVicRSX_rPgi7Vk5i2ShJd/view?usp=sharing"
# url="https://drive.google.com/uc?export=download&id="+url.split('/')[-2]
# import  gdown  
# gdown.download(url,"vae.pth",quiet=False)


import os  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

