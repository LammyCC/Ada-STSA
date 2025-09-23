# -*- coding: utf-8 -*-
# @Time : 2023/7/13 21:11
# @Author : Caisj

import math
import random

import torch
import torch.nn as nn
from model.Ada_STGL.Decoder import Decoder, DecoderLayer
from model.Ada_STGL.Encoder import ConvLayer, Encoder, EncoderLayer
from model.Ada_STGL.attention import Attention, AttentionLayer
from model.Ada_STGL.embed import DataEmbedding
from model.Ada_STGL.graph_learn import GraphLearn


class AdaSTGLStep(torch.nn.Module):
    """
        encoder_num: The number of Block.
        node_num: The number of nodes.
        time_step: input time step.
        conv_dim: hidden size of Conv2d.
        graph_dim: Graph embedding dimension.
        embed_size: Transformer embedding dimension.
        head: Transformer heads number
        dropout: dropout number
        forward_expansion: Magnification of the embedded layer in Transformer
    """

    def __init__(self, c_in, e_layers, d_layers, node_num, time_step, pre_num, graph_dim,
                 d_model, dropout, device, heads=4, d_ff=256, activation='gelu',
                 distil=True, mix=True, output_attention=False):
        super(AdaSTGLStep, self).__init__()
        self.device = device
        self.d_model = d_model
        self.node_num = node_num
        self.time_step = time_step
        self.pre_num = pre_num
        self.output_attention = output_attention
        
        # spatiol learn
        self.spatial_learn = GraphLearn(c_in, node_num, time_step, heads, dropout, 
                                        graph_dim, d_model, device)

        # Embedding
        self.enc_embedding = DataEmbedding(c_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(1, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attention(False, attention_dropout=dropout), d_model, heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            [
                ConvLayer(d_model) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attention(True, attention_dropout=dropout, output_attention=False),
                                   d_model, heads, mix=mix),
                    AttentionLayer(Attention(False, attention_dropout=dropout, output_attention=False),
                                   d_model, heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.conv_residual = nn.Sequential(
            nn.Conv1d(c_in, 32, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(32, d_model, kernel_size=3, padding=4, dilation=4)
        )
        self.attn = nn.Linear(d_model * 2, d_model)
        self.fusion_attn = nn.Linear(d_model * 2, d_model)
        self.mlp = nn.Sequential(
                    nn.Linear(d_model, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, traj_emb, adj_mx, enc_self_mask=None, dec_self_mask=None, cross_mask=None):
        #---------- spatiol relation  -----------
        # input:[b,t,n,c], output:[b,n,t,c]
        spatial_feature = self.spatial_learn(x_enc, traj_emb, adj_mx)
        
        #---------- temporal relation (PWT) -----------
        # change size：[b,t,n,c] --> [b,n,t,c]
        x_enc, x_mark_enc = x_enc.transpose(1, 2), x_mark_enc.transpose(1, 2)
        x_dec, x_mark_dec = x_dec.transpose(1, 2), x_mark_dec.transpose(1, 2)
        # PWT: embedding
        x_enc = self.enc_embedding(x_enc, x_mark_enc)
        x_dec = self.dec_embedding(x_dec, x_mark_dec)
        # PWT: Encode
        enc_out, attns = self.encoder(x_enc, attn_mask=enc_self_mask)
        # PWT: Decode
        dec_out = self.decoder(x_dec, enc_out, x_mask=dec_self_mask, cross_mask=cross_mask)
        temporal_feature = dec_out[:, :, -self.time_step:, :]
        
        #-------------- Fusion ---------------
        attn = torch.sigmoid(self.attn(torch.cat((spatial_feature, temporal_feature), dim=-1)))
        out = attn * spatial_feature + (1 - attn) * temporal_feature

        #------------ Residual & Prediction -------------
        out = self.mlp(out).squeeze(-1).transpose(1, 2) # [B,T,N]
        
        if self.output_attention:
            return out, attns
        else:
            return out  # [B, T, N]
