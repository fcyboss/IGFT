import string
import torch
import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
from colbert.parameters import DEVICE
import torch.nn.init as init




class LossNet(nn.Module):
    def __init__(self,dim = 128,n_heads=4):
        super(LossNet, self).__init__()
        self.cross_atten = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads).to(DEVICE)
        self.linear1 = nn.Linear(32*128, 32*32).to(DEVICE)
        self.linear2 = nn.Linear(32*32, 32*4).to(DEVICE)
        self.linear3 = nn.Linear(32*4, 32*1).to(DEVICE)
        self.linear4 = nn.Linear(32*1, 1).to(DEVICE)

        self.line_contrast = nn.Linear(2,1).to(DEVICE)


    def forward(self,Q,D):
        Q = Q.transpose(0, 1)
        D = D.transpose(0, 1)
        attn_output, _ = self.cross_atten(Q, D, D)
        attn_output = attn_output.view(attn_output.shape[1], -1)
        attn_output = self.linear1(attn_output)
        attn_output = self.linear2(attn_output)
        attn_output = self.linear3(attn_output)
        attn_output = self.linear4(attn_output)
        attn_output = torch.mean(attn_output)
        return attn_output

    def sample(self,Q,D):
        Q = Q.transpose(0, 1)
        D = D.transpose(0, 1)
        attn_output, _ = self.cross_atten(Q, D, D)
        attn_output = attn_output.view(attn_output.shape[1], -1)
        attn_output = self.linear1(attn_output)
        attn_output = self.linear2(attn_output)
        attn_output = self.linear3(attn_output)
        attn_output = self.linear4(attn_output)
        attn_output = torch.mean(attn_output)
        attn_output = torch.sigmoid(attn_output)
        return attn_output

