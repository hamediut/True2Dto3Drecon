import torch
from torch import nn
import torch.nn.functional as F

class Generator(nn.Module):
    """I modified the architecture:
    added a conv3D before the last convTranspose layer to avoid checkerboard artifacts.
    another way is that the following 3 rules should be used in the combination of k,s, and p:
    1) k > s
    2) k mod s = 0
    3) p>= k-s
    """
    
    def __init__(self, num_layers, gf, gk, gs, gp):
        super(Generator, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.num_layers = num_layers
        self.gf = gf
        self.gk = gk
        self.gs = gs
        self.gp = gp
        for lay, (k,s,p) in enumerate(zip(self.gk, self.gs, self.gp)):
            if lay < self.num_layers -4:
                
                self.convs.append(nn.ConvTranspose3d(self.gf[lay], self.gf[lay + 1], k, s, p, bias = False))
                self.bns.append(nn.BatchNorm3d(self.gf[lay+1]))
            elif lay == self.num_layers -4 :
                
                self.convs.append(nn.Conv3d(self.gf[lay], self.gf[lay + 1], 4, 1, 1, bias = False) )
                self.bns.append(nn.BatchNorm3d(self.gf[lay+1]))
            else:# last layer
                self.convs.append(nn.ConvTranspose3d(self.gf[lay], self.gf[lay + 1], 4, 2, 2, bias = False))
                self.bns.append(nn.BatchNorm3d(self.gf[lay+1]))

    def forward(self, x):
        for conv,bn in zip(self.convs[:-1],self.bns[:-1]):
            x = F.relu_(bn(conv(x))) 
        
        # if image_type in ['grayscale', 'colour']:
        out = 0.5*(torch.tanh(self.convs[-1](x))+1)
        # else:
        #     out = torch.softmax(self.convs[-1](x),1)
        return out

class Discriminator(nn.Module):
    def __init__(self, num_layers, df, dk, ds, dp):
        super(Discriminator, self).__init__()
        self.convs = nn.ModuleList()
        self.num_layers = num_layers
        self.df = df
        self.dk = dk
        self.ds = ds
        self.dp = dp
        for lay, (k, s, p) in enumerate(zip(self.dk, self.ds, self.dp)):
            if lay< self.num_layers-4:
                
                
                self.convs.append(nn.Conv2d(self.df[lay], self.df[lay + 1], k, s, p, bias=False))
            else:
                self.convs.append(nn.Conv2d(self.df[lay], self.df[lay + 1], k-1, s+1, 0, bias=False))

    def forward(self, x):
        for conv in self.convs[:-1]:
            x = F.leaky_relu_(conv(x), negative_slope = 0.2)
        x = self.convs[-1](x)
        return x
    
# print(Generator())
# print(Discriminator())