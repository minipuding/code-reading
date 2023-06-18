''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        """
        args:
            n_head: 多头数量, 原文n_head=8
            d_model: 向量维度, 原文d_model=512
            d_k, d_v: key和value维度, 原文d_k, d_v = d_model // n_head = 64
            dropout: drop out概率, 原文取0.1
        """
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # 定义线性层，一个MultiHeadAttention层中存在4个线性层，分别是：
        # q, k, v投影层, 也就是这里的self.w_qs, self.w_ks, self.w_vs
        # 多头融合投射层，这里的self.fc
        # 其中"多头"直接在同一个线性层中合并了，等效于构建n_head个线性层
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        # 计算自注意力的层
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # dropout层，在self.fc中做dropout，kqv的dropout已经在`ScaledDotProductAttention`中实现了
        self.dropout = nn.Dropout(dropout)
        # layer norm层
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        # 由于我们把"多头"放在同一个线性层中一并做了，因此需要对输出做拆解，重新将维度变为batch x len x n_head x d
        # 其中batch就是batch_size, len表示文本序列长度，n_head表示多头数目，d表示q,k,v维度，即64。
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        # 把维度做一个翻转，从batch x len x n_head x d变为batch x n_head x len x d，因为后面计算注意力都是在最后两维做
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # 做完注意力后再把维度便会原来的batch x len x n_head x d，
        # 同时重新把n_head x d合并，这模拟原文中对多头输出进行的concat操作。
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # 对self.fc做dropout
        q = self.dropout(self.fc(q))

        # 残差连接再做layer norm
        # 注意原文图中写的是"Add & Norm"，所以先残差，后norm
        q += residual
        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        """
        前馈网络, 即两层MLP: Linear-ReLU-Linear
        d_in: 输入维度, 原文为512
        d_hid: 隐藏层维度, 一般为d_in的4倍, 即2048
        """
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)

        # 残差连接+LayerNorm
        x += residual
        x = self.layer_norm(x)

        return x
