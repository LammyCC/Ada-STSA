# -*- coding: utf-8 -*-
# @Time : 2023/8/15 10:11
# @Author : Caisj
import math
import random
from model.Ada_STGL.Encoder import MultiConv
from model.Ada_STGL.GCN import GraphConv
from model.Ada_STGL.attention import Attention, AttentionLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import dense_to_sparse
from collections import Counter


class MacroLearn(torch.nn.Module):
    """
    Graph Learning Modoel for AdapGL.

    Args:
        num_nodes: The number of nodes.
        init_feature_num: The initial feature number (< num_nodes).
    """
    def __init__(self, num_nodes, init_feature_num):
        super(MacroLearn, self).__init__()
        self.epsilon = 1 / num_nodes * 0.5
        self.beta = torch.nn.Parameter(torch.ones(num_nodes, dtype=torch.float32), 
                                       requires_grad=True)
        self.w1 = torch.nn.Parameter(
            torch.zeros((num_nodes, init_feature_num), dtype=torch.float32),
            requires_grad=True)
        self.w2 = torch.nn.Parameter(
            torch.zeros((num_nodes, init_feature_num), dtype=torch.float32),
            requires_grad=True)

        self.attn = torch.nn.Conv2d(2, 1, kernel_size=1)

        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

    def forward(self, adj_mx):
        new_adj_mx = torch.mm(self.w1, self.w2.T) - torch.mm(self.w2, self.w1.T)
        new_adj_mx = torch.relu(new_adj_mx + torch.diag(self.beta))
        attn = torch.sigmoid(self.attn(torch.stack((new_adj_mx, adj_mx), dim=0).unsqueeze(dim=0)).squeeze())
        new_adj_mx = attn * new_adj_mx + (1. - attn) * adj_mx
        # d = new_adj_mx.sum(dim=1) ** (-0.5)
        # new_adj_mx = torch.relu(d.view(-1, 1) * new_adj_mx * d - self.epsilon)
        d = new_adj_mx.sum(dim=1) ** (-0.5)
        new_adj_mx = d.view(-1, 1) * new_adj_mx * d
        return new_adj_mx


class GraphLearn(torch.nn.Module):
    '''
    Multi-Step Train Strategy
    '''

    def __init__(self, c_in, num_nodes, time_step, heads, dropout, graph_emb, d_model, device):
        super(GraphLearn, self).__init__()
        self.num_nodes = num_nodes
        self.time_step = time_step
        self.device = device
        # 双向GCN
        self.conv1d = torch.nn.Conv1d(in_channels=c_in, out_channels=graph_emb, kernel_size=1)
        self.graph_conv_1 = GraphConv(graph_emb, d_model // 2, conv_type='gcn', activation=None, with_self=False)
        self.graph_conv_2 = GraphConv(graph_emb, d_model // 2, conv_type='gcn', activation=None, with_self=False)
        self.conv = MultiConv(in_channels=c_in, out_channels=d_model, kernel_size=3,
                                                    dilations=[1, 2, 4], dropout=dropout)
        # Fusion
        self.channel_align = nn.Conv2d(in_channels=100, out_channels=d_model, kernel_size=1)
        self.channel_fusion = SpatialChannelAttentionFusion(d_model, spatial_kernel_size=7, reduction_ratio=16)
        self.graph_attn = torch.nn.Conv2d(2*d_model, 1, kernel_size=1)
        self.fusion_gate = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        
        
    def spatial_conv(self, x, traj_emb, adj_mx):
        # x:[B,T,N,C]，adj_mx：[N,N]，traj_emb：[B,T,N,C]
        b, t, n, c = x.size()
        # Macro node attribution
        x = x.reshape(-1, n, c) # [b*t,n,c]
        # Bidirectional graph convolution
        x = self.conv1d(x.permute(0, 2, 1)).permute(0, 2, 1)
        g_1 = self.graph_conv_1(x=x, adj_mx=adj_mx)
        g_2 = self.graph_conv_2(x=x, adj_mx=adj_mx.T)
        macro = torch.relu(torch.cat((g_1, g_2), dim=-1).reshape(b, t, n, -1))
        macro_feature = macro.permute(0, 2, 1, 3) # [b,n,t,c]
        # Micro semantics feature
        traj_emb = traj_emb.unsqueeze(2) # [b,n,t,c]
        micro_feature = traj_emb.repeat(1, 1, 12, 1)
        micro_feature = self.channel_align(micro_feature.transpose(1, 3)).transpose(1, 3)
        # Channel fusion
        spatial_feature = self.channel_fusion(macro_feature, micro_feature)
        
        return spatial_feature
    
    def forward(self, x, traj_emb, adj_mx):
        # x:[B,T,N,C]
        # ---------------- Dynamic update of macro and micro spatial feature  ----------------
        # Bidirectional graph convolution
        spatial_feature = self.spatial_conv(x, traj_emb, adj_mx)

        return spatial_feature



class ChannelAttentionFusion(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttentionFusion, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False)
        )
        self.fusion_mlp = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(), 
            nn.Conv2d(channels // reduction_ratio, channels * 2, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, macro_feature, micro_feature):
        """
        Args:
            macro_feature: [B, N, T, C]
            micro_feature: [B, N, T, C]
        Returns:
            fused_feature: [B, N, T, C]
        """
        # [B, N, T, C] -> [B, C, N, T]
        macro_feat = macro_feature.permute(0, 3, 1, 2)  # [B, C, N, T]
        micro_feat = micro_feature.permute(0, 3, 1, 2)  # [B, C, N, T]
        macro_avg_out = self.mlp(self.avg_pool(macro_feat))  # [B, C, 1, 1]
        macro_max_out = self.mlp(self.max_pool(macro_feat))  # [B, C, 1, 1]
        macro_channel_attn = self.sigmoid(macro_avg_out + macro_max_out)  # [B, C, 1, 1]
        micro_avg_out = self.mlp(self.avg_pool(micro_feat))  # [B, C, 1, 1]
        micro_max_out = self.mlp(self.max_pool(micro_feat))  # [B, C, 1, 1]
        micro_channel_attn = self.sigmoid(micro_avg_out + micro_max_out)  # [B, C, 1, 1]
        macro_enhanced = macro_feat * macro_channel_attn  # [B, C, N, T]
        micro_enhanced = micro_feat * micro_channel_attn  # [B, C, N, T]
        combined_feat = torch.cat([macro_enhanced, micro_enhanced], dim=1)  # [B, 2C, N, T]
        combined_avg_out = self.fusion_mlp(self.avg_pool(combined_feat))  # [B, 2C, 1, 1]
        combined_max_out = self.fusion_mlp(self.max_pool(combined_feat))  # [B, 2C, 1, 1]
        fusion_weights = self.sigmoid(combined_avg_out + combined_max_out)  # [B, 2C, 1, 1]
        macro_weight = fusion_weights[:, :self.channels, :, :]  # [B, C, 1, 1]
        micro_weight = fusion_weights[:, self.channels:, :, :]  # [B, C, 1, 1]
        fused_feat = macro_enhanced * macro_weight + micro_enhanced * micro_weight  # [B, C, N, T]
        fused_feature = fused_feat.permute(0, 2, 3, 1)
        
        return fused_feature


class SpatialChannelAttentionFusion(nn.Module):
    def __init__(self, channels, spatial_kernel_size=7, reduction_ratio=16):
        super(SpatialChannelAttentionFusion, self).__init__()
        self.channels = channels
        self.channel_attention = ChannelAttentionFusion(channels, reduction_ratio)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel_size, 
                                     padding=spatial_kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, macro_feature, micro_feature):
        # channel fusion
        channel_fused = self.channel_attention(macro_feature, micro_feature)  # [B, N, T, C]
        # spatial attention
        # : [B, N, T, C] -> [B, C, N, T]
        channel_fused_2d = channel_fused.permute(0, 3, 1, 2)  # [B, C, N, T]
        avg_out = torch.mean(channel_fused_2d, dim=1, keepdim=True)  # [B, 1, N, T]
        max_out, _ = torch.max(channel_fused_2d, dim=1, keepdim=True)  # [B, 1, N, T]
        spatial_attn = self.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))  # [B, 1, N, T]
        spatially_attended = channel_fused_2d * spatial_attn  # [B, C, N, T]
        final_feature = spatially_attended.permute(0, 2, 3, 1)  # [B, N, T, C]
        
        return final_feature

    

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.init_weights()
    
    def init_weights(self):
        init_range = 0.5 / self.embedding_dim
        self.center_embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)
    
    def forward(self, center_words, context_words, negative_words):
        center_embeds = self.center_embeddings(center_words)  # [batch_size, embed_dim]
        context_embeds = self.context_embeddings(context_words)  # [batch_size, embed_dim]
        neg_embeds = self.context_embeddings(negative_words)  # [batch_size, neg_samples, embed_dim]
        pos_score = torch.sum(center_embeds * context_embeds, dim=1)  # [batch_size]
        pos_score = torch.sigmoid(pos_score)
        neg_score = torch.bmm(neg_embeds, center_embeds.unsqueeze(2)).squeeze(2)  # [batch_size, neg_samples]
        neg_score = torch.sigmoid(-neg_score)
        
        return pos_score, neg_score

class Word2VecTrainer:
    def __init__(self, vector_size=256, window=6, min_count=1, negative=10, epochs=50, lr=0.001, device='cuda'):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.negative = negative
        self.epochs = epochs
        self.lr = lr
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        self.vocab_size = 0
        self.model = None
        
    def build_vocab(self, sentences):
        for sentence in sentences:
            for word in sentence:
                self.word_counts[str(word)] += 1
        filtered_words = [word for word, count in self.word_counts.items() if count >= self.min_count]
        self.word2idx = {word: idx for idx, word in enumerate(filtered_words)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        self.create_negative_sampling_table()
    
    def create_negative_sampling_table(self):
        total_count = sum(self.word_counts.values())
        word_probs = []
        
        for word in self.idx2word.values():
            count = self.word_counts[word]
            prob = (count / total_count) ** 0.75
            word_probs.append(prob)
        
        # norm
        word_probs = np.array(word_probs)
        word_probs = word_probs / word_probs.sum()
        
        self.negative_sampling_probs = word_probs
    
    def generate_training_data(self, sentences):
        training_data = []
        
        for sentence in sentences:
            sentence_words = [str(word) for word in sentence if str(word) in self.word2idx]
            
            if len(sentence_words) < 2:
                continue
                
            for i, center_word in enumerate(sentence_words):
                start = max(0, i - self.window)
                end = min(len(sentence_words), i + self.window + 1)
                
                for j in range(start, end):
                    if i != j:
                        context_word = sentence_words[j]
                        training_data.append((center_word, context_word))
        
        return training_data
    
    def get_negative_samples(self, batch_size):
        negative_samples = np.random.choice(
            self.vocab_size, 
            size=(batch_size, self.negative), 
            p=self.negative_sampling_probs
        )
        return negative_samples
    
    def train(self, sentences):
        self.build_vocab(sentences)
        
        if self.vocab_size == 0:
            return

        self.model = SkipGramModel(self.vocab_size, self.vector_size).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        training_data = self.generate_training_data(sentences)
        
        if len(training_data) == 0:
            return
        
        # train
        batch_size = min(8192, len(training_data))
        
        for epoch in range(self.epochs):
            total_loss = 0
            random.shuffle(training_data)
            for i in range(0, len(training_data), batch_size):
                batch_data = training_data[i:i+batch_size]
                if len(batch_data) == 0:
                    continue
                center_words = [self.word2idx[pair[0]] for pair in batch_data]
                context_words = [self.word2idx[pair[1]] for pair in batch_data]
                negative_samples = self.get_negative_samples(len(batch_data))
                center_tensor = torch.LongTensor(center_words).to(self.device)
                context_tensor = torch.LongTensor(context_words).to(self.device)
                negative_tensor = torch.LongTensor(negative_samples).to(self.device)
                pos_score, neg_score = self.model(center_tensor, context_tensor, negative_tensor)
                pos_loss = -torch.log(pos_score + 1e-8).mean()
                neg_loss = -torch.log(neg_score + 1e-8).sum(dim=1).mean()
                loss = pos_loss + neg_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
    
    def get_vector(self, word):
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        word_str = str(word)
        if word_str not in self.word2idx:
            raise KeyError(f"Word '{word_str}' not in vocabulary")
        
        idx = self.word2idx[word_str]
        with torch.no_grad():
            embedding = self.model.center_embeddings.weight[idx].cpu().numpy()
        return embedding

def transfer_probability(traj_mx, nodes_grid, vector_size=100, window=5, min_count=1, 
                        negative=10, epochs=50, lr=0.001, device='cuda'):
    if not traj_mx or len(traj_mx) == 0:
        return np.eye(len(nodes_grid), dtype=np.float32)

    if isinstance(traj_mx, list) and len(traj_mx) > 0 and isinstance(traj_mx[0], str):
        traj_mx = [traj_mx]

    valid_trajs = [traj for traj in traj_mx if traj and len(traj) > 0]
    
    if not valid_trajs or len(valid_trajs) < 2:
        return np.eye(len(nodes_grid), dtype=np.float32)

    trainer = Word2VecTrainer(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        negative=negative,
        epochs=epochs,
        lr=lr,
        device=device
    )
    trainer.train(valid_trajs)
    
    node_embeddings = []
    for node in nodes_grid:
        if str(node) in trainer.word2idx:
            embedding = trainer.get_vector(str(node))
        else:
            embedding = np.zeros(vector_size, dtype=np.float32)
        node_embeddings.append(embedding)
    
    return np.array(node_embeddings, dtype=np.float32)