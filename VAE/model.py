from statistics import mean
import  torch  
import   torch.nn.functional as F
import  torch.nn as nn
import  matplotlib.pylab as plt

#input image -> hiden layer -> mean , standard deviation -> decoder -> output image
class  VAE(nn.Module):
    def __init__(self,input_dim,h_dim=200,z_dim=20) -> None:
        super().__init__()

        #encoder
        self.img_2hid=nn.Linear(input_dim,h_dim)
        self.hid_2mean=nn.Linear(h_dim,z_dim)
        self.hid_2sigma=nn.Linear(h_dim,z_dim)
        
        #decoder
        self.z_2hid=nn.Linear(z_dim,h_dim)
        self.hid_2img=nn.Linear(h_dim,input_dim)
        
        
        self.relu=nn.ReLU()




    def encode(self, x):
        #q(z|x)
        h=self.relu(self.img_2hid(x))
        mean=self.hid_2mean(h)
        sigma=self.hid_2sigma(h)
        return mean,sigma


    def decode(self, z):
        #p(x|z)
        h=self.relu(self.z_2hid(z))
        img=self.hid_2img(h)
        return torch.sigmoid(img)

    def forward(self, x):
        #p(x|z)q(z|x)
        mean,sigma=self.encode(x)
        epsilon=torch.randn_like(sigma)
        z=mean+sigma*epsilon
        x_contruct=self.decode(z)
        return x_contruct,mean,sigma




if  __name__ == "__main__":
    vae = VAE(input_dim=784)
    x= torch.randn(4,28*28)
    x_constructed,mu,sigma=vae(x)
    print(x_constructed.shape)
    print(mu.shape)
    print(sigma.shape)
