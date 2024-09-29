# U-Net Arch. Outputs binary segmented image
import torch
import torch.nn as nn
import math

class Decoder(nn.Module):
    
    def __init__(self, reduce_dim=128, cond_layer = None,
                 extract_layers=[8, 9, 10, 11], mha_heads=4):
        super(Decoder_UNET, self).__init__()
        
        self.cond_layer = cond_layer
        self.film_mul = nn.Linear(512, reduce_dim) 
        self.film_add = nn.Linear(512, reduce_dim) 
        self.depth = len(extract_layers)
        self.reduce_blocks = nn.ModuleList([nn.Linear(768, reduce_dim) for _ in range(self.depth)])
        self.mha_blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model=reduce_dim,
                                                              nhead=mha_heads) for _ in range(self.depth)])  
        self.trans_conv = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(reduce_dim, reduce_dim // 2, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(reduce_dim // 2, 1, kernel_size=8, stride=8)
        )
        
    def forward(self, encoder_out):
        batch_size = encoder_out[0].shape[0]
        
        prompt_cond, visual_cond, activations = encoder_out
        cond = torch.mul(prompt_cond, visual_cond)
        
        a = None
        for i, (skip, mha, reduce) in enumerate(zip(activations, self.mha_blocks, self.reduce_blocks)):
            if a is None:
                a = reduce(skip)
            else:
                a = a + reduce(skip)
            if (self.cond_layer==None or i==(self.cond_layer-1)):
                a = self.film_mul(cond)*a + self.film_add(cond)
            a = mha(a)

        a = a[:, 1:, :] #ignore CLS Token
        a = a.permute(0, 2, 1) #batch_size, features, tokens
        
        size = int(math.sqrt(a.shape[2]))
        a = a.view(batch_size, a.shape[1], size, size)

        a = self.trans_conv(a)
        return a