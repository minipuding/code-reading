''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from transformer.Layers import EncoderLayer, DecoderLayer


__author__ = "Yu-Hsiang Huang"


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    # 这里巧妙使用torch.triu方法构建上三角矩阵, 
    # 假设len_s=5, 那么torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)表示获取主对角线以上 diagonal 条对角线及以上的元素,
    # 即[[[0, 1, 1, 1, 1],
    #     [0, 0, 1, 1, 1],
    #     [0, 0, 0, 1, 1],
    #     [0, 0, 0, 0, 1],
    #     [0, 0, 0, 0, 0]]]
    #那么1减去以上就是subsequent_mask, 也就是：
    # [[[1, 0, 0, 0, 0],
    #   [1, 1, 0, 0, 0],
    #   [1, 1, 1, 0, 0],
    #   [1, 1, 1, 1, 0],
    #   [1, 1, 1, 1, 1]]]
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        """
        位置编码, 按照原论文公式完成, 即：
        PE(pos, 2i) = sin(pos / 10000 ^ (2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000 ^ (2i/d_model))
        d_hid: 即公式中的d_model
        n_position即公式中i的取值上限, 输入token限制的长度
        """
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        """
        forward过程就是查表, 在构建好的位置编码表中查到相应的编码, 
        再将查到的位置编码加到输入x中
        """
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):
        """
        Transformer完整的编码器由n_layers个(原文为6)EncoderLayers构成, 
        在正式输入EncoderLayers构成之前还有Embedding层以及需要添加位置编码,
        整个流程为：
        src_seq(inputs) -> Embedding -> PositionalEncoding -> EncoderLayer * n_layers -> ...
        Args:
            n_src_vocab: 词典大小, 用于构建Embedding
            d_word_vec: 向量长度, 和d_model一致, 都是512
            n_layers: EncoderLayer数量, 原文为6
            n_head: 多头注意力数目, 原文为8
            d_k, d_v: key和value维度, 原文d_k, d_v = d_model // n_head = 64
            d_model: 向量维度, 512
            d_inner: 前馈网络中隐藏层维度, 4*512=2048
            pad_idx: 词典中pad的编码id
            n_position: 位置编码长度, 可以认为是模型处理序列的长度上限
            scale_emb: bool类型, 决定是否将embedding向量进行放大
        """

        super().__init__()

        # Embedding层, 本质就是一个查找表, shape = n_src_vocab * d_word_vec
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)

        # PositionalEncoding位置编码层
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)

        # 用于dropout embedding层
        self.dropout = nn.Dropout(p=dropout)

        # 编码器主体, n_layers个编码器层
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        
        # 用于embedding之后的LayerNorm层
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        # 经过embedding层将每个token的id映射成d_word_vec(d_model)维的向量
        enc_output = self.src_word_emb(src_seq)

        # 原文将embedding输出乘上了sqrt(d_model), 有些解释认为保证embedding和位置编码在数值上保持同一个量级
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        
        # 位置编drop out + layer norm
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        # 经过n_layers层编码器层
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        # 返回编码器输出, 输出依旧是batch_size * len * d_model大小, 其中len表示序列长度
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False):
        """
        Transformer完整的解码器由n_layers个(原文为6)DecoderLayer, 
        在正式输入DecoderLayer构成之前还有Embedding层以及需要添加位置编码,
        整个流程为：
        trg_seq(inputs), enc_output -> Embedding -> PositionalEncoding -> DecoderLayer * n_layers -> ...
        Args:
            n_trg_vocab: 词典大小, 用于构建Embedding, 大小和编码器中的n_src_vocab一致
            d_word_vec: 向量长度, 和d_model一致, 都是512
            n_layers: EncoderLayer数量, 原文为6
            n_head: 多头注意力数目, 原文为8
            d_k, d_v: key和value维度, 原文d_k, d_v = d_model // n_head = 64
            d_model: 向量维度, 512
            d_inner: 前馈网络中隐藏层维度, 4*512=2048
            pad_idx: 词典中pad的编码id
            n_position: 位置编码长度, 可以认为是模型处理序列的长度上限
            scale_emb: bool类型, 决定是否将embedding向量进行放大
        """
        super().__init__()

        # Embedding层, 本质就是一个查找表, shape = n_src_vocab * d_word_vec
        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)

        # PositionalEncoding位置编码层
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)

        # 用于dropout embedding层
        self.dropout = nn.Dropout(p=dropout)

        # 解码器主体, n_layers个解码器层
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        
        # 用于embedding之后的LayerNorm层
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        # 经过embedding层将每个token的id映射成d_word_vec(d_model)维的向量
        dec_output = self.trg_word_emb(trg_seq)

        # 原文将embedding输出乘上了sqrt(d_model), 有些解释认为保证embedding和位置编码在数值上保持同一个量级
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5

        # 位置编drop out + layer norm
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        # 经过n_layers层解码器层
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        # 返回解码器输出, 输出依旧是batch_size * len * d_model大小, 其中len表示序列长度
        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj'):
        """
        梳理一下: Transformer由三部分组成, Encoder, Decoder以及Proj投射层
        其中Encoder由Embedding, 位置编码, 6个`EncoderLayer`组成;
            Decoder由Embedding, 位置编码, 6个`DecoderLayer`组成
        EncoderLayer由`MultiHeadAttention`以及`PositionwiseFeedForward`组成;
        DecoderLayer由`Masked-MultiHeadAttention`以及`PositionwiseFeedForward`组成;
        MultiHeadAttention由q, k, v三个投射层以及多头注意力投射层共4个线性层组成;
        PositionwiseFeedForward由两个线性层的MLP组成;
        
        Args:
            n_src_vocab, n_trg_vocab: 词典大小, 用于构建Embedding
            src_pad_idx, trg_pad_idx: 词典中pad的编码id
            d_word_vec, d_model: 向量维度, 512
            d_inner: 前馈网络中隐藏层维度, 4*512=2048
            n_layers: EncoderLayer数量, 原文为6
            n_head: 多头注意力数目, 原文为8
            d_k, d_v: key和value维度, 原文d_k, d_v = d_model // n_head = 64
            n_position: 位置编码长度, 可以认为是模型处理序列的长度上限
            trg_emb_prj_weight_sharing
        """

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        # 确定是否要对embedding 或者投射层参数进行缩放
        # 同时满足scale_emb_or_prj == 'emb'以及trg_emb_prj_weight_sharing==True时, 需要sacle, 否则不需要
        # (TODO: 为什么……？)
        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        # 是否将解码器的embedding参数和Transformer最后投射层的参数共享
        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

        # 是否将编码器embedding参数和解码器embedding参数共享
        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight


    def forward(self, src_seq, trg_seq):
        
        # 先获取mask, 其中编码器的输入只需要考虑pad_mask, 解码器的输入则还需要考虑序列mask
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        # 按照encoder -> decoder -> target_word_projector的顺序做前向传播
        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output)

        # 如果需要对投射层输出的logit做scaling就乘上sqrt(d_model)
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5

        return seq_logit.view(-1, seq_logit.size(2))
