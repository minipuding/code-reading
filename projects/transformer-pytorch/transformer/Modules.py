import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"


# 用于实现注意力计算，原文称作Scaled Dot-Product Attention
# 公式如下：
# Attention(Q, K, V) = (QK'/sqrt(d_model))V
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # 是否使用mask
        # 通过将mask位置的值设置为很大的负数，如-1e9
        # 当计算softmax时，分子部分变成e^(-1e9)，几乎为0，从而实现mask功能
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        # 除了返回输出，还返回了注意力权重attn
        return output, attn
