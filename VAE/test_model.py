import  torch
import torch.nn as nn  
import torch.nn.functional as F
import  torchvision
import  torchvision.transforms as transforms
import  torch.optim as optim
from  torch.utils.data import  DataLoader
from  torchvision.datasets import  MNIST
from  model import  VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#load model and data
INPUT_DIM=784
Z_DIM=20
H_DIM=200
BATCH_SIZE=128
model=VAE(input_dim=INPUT_DIM,z_dim=Z_DIM,h_dim=H_DIM).to(device)
model.load_state_dict(torch.load("vae.pth"))

#load data
dataset=MNIST(root="dataset/",transform=transforms.ToTensor(),download=True)
train_loader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)


#test
with torch.no_grad():
    n=20
    sample=torch.randn(n,Z_DIM).to(device)
    sample=model.decode(sample).view(-1,1,28,28)
    torchvision.utils.save_image(sample,"sample_test.png",nrow=n)

