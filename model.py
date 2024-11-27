import torch
import torch.nn as nn
import torch.nn.functional as F


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
    
    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2,3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return x * (self.gamma * Nx + 1) + self.beta

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, part, special=False):
        super(Block, self).__init__()
        assert part in ['encoder','decoder']
        
        self.conv_layers = None

        if special and part == "encoder":
            pool = nn.Identity()
        else:
            pool = nn.MaxPool2d(2)
        
        if part == "encoder":
            self.conv_layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                GRN(out_channels),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                GRN(out_channels),
                pool
               )
        else:
            if not special:
                self.conv_layers = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    GRN(out_channels),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    GRN(out_channels)
                   )
            else:
                self.conv_layers = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    GRN(out_channels),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    GRN(out_channels)
                   )
        
    def forward(self, x):    
        return self.conv_layers(x)

class EAF(nn.Module):
    def __init__(self, channel, dropout_rate=0.1):
        super(EAF, self).__init__()

        next_channel = channel * 2
        
        self.spatial_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.trans_conv = nn.ConvTranspose2d(next_channel, channel,
                                            kernel_size=3, 
                                            stride=2, 
                                            padding=1, 
                                            output_padding=1)

        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
           )

    def forward(self, x1, x2):
        x1_ = self.spatial_avgpool(x1) * torch.mean(x1, dim=1, keepdim=True)
        x2 = self.spatial_avgpool(x2) * torch.mean(x2, dim=1, keepdim=True)
        x2 = self.trans_conv(x2)
        x1_ = self.conv(x1_ + x2)
        x1_out = x1 * x1_
        return x1_out

class DAF(nn.Module):
    def __init__(self, in_channels):
        super(DAF, self).__init__()

        self.up_conv = nn.ConvTranspose2d(in_channels * 2, in_channels,
                                            kernel_size=3, 
                                            stride=2, 
                                            padding=1, 
                                            output_padding=1)
        
        # Local Details (LD) branch
        self.ld_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
        # Spatial Average (SA) branch
        self.sa_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.sa_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
        # Channel Average (CA) branch
        self.ca_conv = nn.Conv2d(1, in_channels, kernel_size=3, padding=1)
        
        # Sigmoid activation for combining branches
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, F_en, F_hae, F_de):
        F_de = self.up_conv(F_de)
        F_tsi = F_en + F_hae + F_de
        
        # Local Details (LD) branch
        ld = F.relu(self.ld_conv(F_tsi)) * F_tsi
        
        # Spatial Average (SA) branch
        sa = F.relu(self.sa_conv(self.sa_pool(F_tsi))) * F_tsi
        
        # Channel Average (CA) branch
        F_tsi = torch.mean(F_tsi, dim=1, keepdim=True)
        ca = F.relu(self.ca_conv(F_tsi)) * F_tsi
        
        # Combining the LD, SA, and CA branches
        W = self.sigmoid(ld + sa + ca)
        
        # Weighted sum of F_hae and F_de
        F_clsca = W * F_hae + (1 - W) * F_de
        
        return F_clsca

class AHF_U_Net(nn.Module):
    def __init__(self):
        super(AHF_U_Net, self).__init__()
        self.encoder_1 = Block(4, 64, "encoder", special=True)
        self.encoder_2 = Block(64, 128, "encoder")
        self.encoder_3 = Block(128, 256, "encoder")
        self.encoder_4 = Block(256, 512, "encoder")
        self.encoder_5 = Block(512, 1024, "encoder")

        self.eaf_1 = EAF(64)
        self.eaf_2 = EAF(128)
        self.eaf_3 = EAF(256)
        self.eaf_4 = EAF(512)

        self.decoder_1 = Block(512, 256, "decoder")
        self.decoder_2 = Block(256, 128, "decoder")
        self.decoder_3 = Block(128, 64, "decoder")
        self.decoder_4 = Block(64, 3, "decoder", special=True)

        self.daf_1 = DAF(64)
        self.daf_2 = DAF(128)
        self.daf_3 = DAF(256)
        self.daf_4 = DAF(512)

    def forward(self, x):
        x1 = self.encoder_1.forward(x)
        x2 = self.encoder_2.forward(x1)
        x3 = self.encoder_3.forward(x2)
        x4 = self.encoder_4.forward(x3)
        x5 = self.encoder_5.forward(x4)

        x1_ = self.eaf_1.forward(x1, x2)
        x2_ = self.eaf_2.forward(x2, x3)
        x3_ = self.eaf_3.forward(x3, x4)
        x4_ = self.eaf_4.forward(x4, x5)

        _x1 = self.daf_4.forward(x4, x4_, x5)
        _x1 = self.decoder_1.forward(_x1)
        
        _x2 = self.daf_3.forward(x3, _x1, x4)
        _x2 = self.decoder_2.forward(_x2)
        
        _x3 = self.daf_2.forward(x2, _x2, x3)
        _x3 = self.decoder_3.forward(_x3)
        
        _x4 = self.daf_1.forward(x1, _x3, x2)
        _x4 = self.decoder_4.forward(_x4)

        return _x4

