''' Define the Layers '''
import torch.nn as nn
import torch
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward


__author__ = "Yu-Hsiang Huang"


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        """
        编码器层由两个子层组成, 即多头注意力层(self.slf_attn)和前馈层(self.pos_ffn)
        d_model: 向量维度, 512
        d_inner: 前馈网络中隐藏层维度, 4*512=2048
        n_head: 多头数目, 8
        d_k, d_v: 每个头key与value的维度, d_model // n_head = 64
        droupout: drop out 概率
        """
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        # 输入为encoder input, 一般仅有pad mask, 即为了将序列补全至指定长度需要加pad_token, 但这些pad_token只用来占位，不参与计算，因此需要mask掉。
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        
        # 经过注意力层后进入前馈层
        enc_output = self.pos_ffn(enc_output)
        
        # 输出encoder前向结果以及注意力权重
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        """
        解码器层由3个子层组成, 即待掩码的多头注意力层(self.slf_attn), 多头注意力层(self.enc_attn)和前馈层(self.pos_ffn)
        d_model: 向量维度, 512
        d_inner: 前馈网络中隐藏层维度, 4*512=2048
        n_head: 多头数目, 8
        d_k, d_v: 每个头key与value的维度, d_model // n_head = 64
        droupout: drop out 概率
        """
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        # 带掩码的自注意力，仅使用解码器输入作为k,q,v
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        
        # 另一个多头注意力，用于融合编码器结果与解码器(指self.slf_attn)结果
        # 其中quere是解码器的输出, key和value都是编码器输出
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        
        # 经过注意力层后进入前馈层
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn
