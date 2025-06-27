# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
#from msilib.schema import Shortcut
import random
from os.path import join as pjoin
#from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
#from models.mamba.mamba_ssm import Mamba
from models.vmamba import VSSBlock, CrossMambaFusionBlock, ConcatMambaFusionBlock

import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
#from torch_geometric.nn import GENConv, DeepGCNLayer
#from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention
from models.topk.svm import SmoothTop1SVM

from torch.nn import ReLU
import models.configs as configs
from .model_utils import *
from typing import Optional, Callable
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from functools import partial
from models.datten import DAttention
from models.mean_max import MeanMIL, MaxMIL

logger = logging.getLogger(__name__)
ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()

        self.patch_embeddings = nn.Sequential(*[nn.Linear(config.input_size, config.hidden_size), nn.ReLU(), nn.Dropout(0.2)]) #
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        ##self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x = x.unsqueeze(0)
        B = x.shape[0]
        #cls_tokens = self.cls_token.expand(B, -1, -1)
        x = self.patch_embeddings(x)
        #x = torch.cat((cls_tokens, x), dim=1)
        ##x = self.dropout(x)
        return x


class Embeddingsout(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddingsout, self).__init__()

        self.patch_embeddings = nn.Sequential(*[nn.Linear(config.hidden_size, config.input_size), nn.ReLU(), nn.Dropout(0.2)]) #
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        ##self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        #cls_tokens = self.cls_token.expand(B, -1, -1)
        x = self.patch_embeddings(x)
        #x = torch.cat((cls_tokens, x), dim=1)
        ##x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Part_GCN(torch.nn.Module):
    def __init__(self, config, edge_agg='spatial'):
        super(Part_GCN, self).__init__()
        self.edge_agg = edge_agg
        self.num_layers = 2

        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers+1):
            conv = GENConv(config.hidden_size, config.hidden_size, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=1, norm='layer')
            norm = LayerNorm(config.hidden_size, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', ckpt_grad=i % 2)
            self.layers.append(layer)
        #self.conv1 = nn.Conv2d(config.hidden_size*3, config.hidden_size, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(config.hidden_size*2, config.hidden_size, kernel_size=1, stride=1)

    def forward(self, data):

        if self.edge_agg == 'spatial':
            edge_index = data.edge_index
        elif self.edge_agg == 'latent':
            edge_index = data.edge_latent
        edge_index = edge_index.long()
        edge_attr = None
        batch = data.batch

        x = data.x
        x = self.layers[0].conv(x, edge_index, edge_attr)
        x_ = x
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            x_ = torch.cat([x_, x], axis=1)

        x = x_
        x = self.conv1(x.unsqueeze(-1).unsqueeze(-1))
        x = x.squeeze()

        return x, edge_index


class Block_graph(nn.Module):
    def __init__(self, config):
        super(Block_graph, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        #self.attn = Attention(config)
        self.gcn = Part_GCN(config)

    def forward(self, x, coord_s, threshold):
        """
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        """
        #token_graph = pt2graph(coord_s.squeeze(), x[:, 1:, :].squeeze(), threshold).to('cuda:0')
        token_graph = pt2graph(coord_s.squeeze(), x[:, :, :].squeeze(), threshold).to('cuda:0')
        graph_encoded, _ = self.gcn(token_graph)
        x[:, 0, :] = x[:, 0, :] + torch.mean(graph_encoded, dim=0).unsqueeze(0)

        return x


class ATTShort(nn.Module):
    def __init__(self, config):
        super(ATTShort, self).__init__()
        self.sa_layer = Block(config)
        self.cross_layer = MultiheadAttention(embed_dim=config.hidden_size, num_heads=1)
        self.cross_layer_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.position_embeddings = nn.Parameter(torch.randn(1, 512+1, config.hidden_size))

    def forward(self, hidden_states, coord_s, sequence_length):
        hidden_states, attention_map = self.sa_layer(hidden_states)
        weight = attention_map[:, :, 0, 1:]
        weight = torch.mean(weight, dim=1)
        _, max_inx = weight.topk(sequence_length)
        part_inx = torch.unique(max_inx.reshape(-1)).unsqueeze(0) + 1
        part_hidden_states = hidden_states[0, part_inx[0, :]].unsqueeze(0)

        pooling_hidden_states, _ = self.cross_layer(part_hidden_states.permute(1, 0, 2),
                                                    hidden_states[:, 1:, :], hidden_states[:, 1:, :])

        pooling_hidden_states = pooling_hidden_states.permute(1, 0, 2)
        pooling_hidden_states = pooling_hidden_states + part_hidden_states
        pooling_hidden_states = self.cross_layer_norm(pooling_hidden_states)
        pooling_hidden_states = torch.cat((hidden_states[:, 0, :].unsqueeze(1), pooling_hidden_states), dim=1)

        coord_s = coord_s[part_inx[0, :]-1]

        return pooling_hidden_states, part_inx, weight, coord_s

"""
# Knowledge-guided Graph Representation
# Graph Construction + Prototype-guided Graph Aggregation
"""
class WiKG(nn.Module):
    def __init__(self, dim_in=384, dim_hidden=512, topk=6, agg_type='bi-interaction', num_heads=8, dropout=0.25, qkv_bias=True):
        super().__init__()
        self._fc1 = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.LeakyReLU())
        self.W_head = nn.Linear(dim_hidden, dim_hidden)
        self.W_tail = nn.Linear(dim_hidden, dim_hidden)
        self.scale = dim_hidden ** -0.5
        self.topk = topk
        self.top = 1
        self.agg_type = agg_type
        head_dim = dim_hidden // num_heads
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.kv = nn.Linear(dim_hidden, head_dim * num_heads * 2, bias=qkv_bias)
        self.q = nn.Linear(dim_hidden, head_dim * num_heads, bias=qkv_bias)
        if self.agg_type == 'gcn':
            self.linear = nn.Linear(dim_hidden, dim_hidden)
        elif self.agg_type == 'sage':
            self.linear = nn.Linear(dim_hidden * 2, dim_hidden)
        elif self.agg_type == 'bi-interaction':
            #self.linear = nn.Linear(dim_hidden, dim_hidden)
            self.linear = nn.Linear(dim_hidden, dim_hidden)
            #self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        else:
            raise NotImplementedError
        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim_hidden)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(dropout)
        self.apply(initialize_weights)

    def forward(self, x):
        try:
            x = x["feature"]
        except:
            x = x
        
        """
        Graph Construction
        """
        x_input = x
        x = self._fc1(x)    # [B,N,C]
        # B, N, C = x.shape
        x = (x + x.mean(dim=1, keepdim=True)) * 0.5  # Base Embedding
        e_h = self.W_head(x)  # Head Embedding
        e_t = self.W_tail(x)  # Tail Embedding
        # construct neighbour
        attn_logit = (e_h * self.scale) @ e_t.transpose(-2, -1)  # 1
        topk_weight, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)
        topk_index_top = topk_index
        topk_index = topk_index.to(torch.long)
        topk_index_top = topk_index_top.to(torch.long)
        # expand topk_index dimensions to match e_t
        topk_index_expanded = topk_index.expand(e_t.size(0), -1, -1)  # shape: [1, 10000, 4]
        topk_index_top_expanded = topk_index_top.expand(e_t.size(0), -1, -1)  # shape: [1, 10000, 4]

        """
        Prototype-guided Graph Aggregation
        """
        topk_index_expanded_ = topk_index_expanded.view(1, -1)[0]
        unique_elements, counts = topk_index_expanded_.unique(return_counts=True) # Sort for the highest degree of node
        topk_index_expanded = unique_elements[counts.argsort(descending=True)[:self.top]]
        ins_top = x_input[:, topk_index_expanded, :] 
        embedding_topk = x[:, topk_index_expanded, :].unsqueeze(0) 
        embedding_topk = embedding_topk.expand(-1, e_t.size(1), -1, -1) # Prototype
        # create a RANGE tensor to help indexing
        batch_indices = torch.arange(e_t.size(0)).view(-1, 1, 1).to(topk_index.device)  # shape: [1, 1, 1]
        Nb_h = e_t[batch_indices, topk_index_top_expanded, :]  # shape: [1, 10000, 4, 512]
        ori_node = Nb_h
        q_node = embedding_topk.squeeze(0)
        kv_node = Nb_h.squeeze(0)
        B_, N, C = kv_node.shape
        N_q = 1
        q_ = self.q(q_node).reshape(B_, N_q, 1, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        kv = self.kv(kv_node).reshape(B_, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = q_[0]
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        Nb_h_attn = (attn @ v).transpose(1, 2).reshape(B_, N_q, self.num_heads*self.head_dim)
        Nb_h_attn = self.norm(Nb_h_attn)
        Nb_h_attn = Nb_h_attn.unsqueeze(0) 
        Nb_h_attn = Nb_h_attn.squeeze(-2) 
        e_Nh = Nb_h_attn
        sum_embedding = self.message_dropout(self.activation(self.linear(self.norm(e_h + e_Nh))))
        embedding = sum_embedding + x_input
        return embedding, ins_top

"""
# Knowledge-guided Graph Representation
"""
class GraphRepresentation(nn.Module):
    def __init__(self, config):
        super(GraphRepresentation, self).__init__()
        self.layer1 = WiKG(dim_in=96, dim_hidden=96, topk=9, agg_type='bi-interaction', dropout=0.2)

    def forward(self, hidden_states):
        attn_weights = []
        hidden_states, ins_top = self.layer1(hidden_states)

        return hidden_states, ins_top

"""
# Knowledge-guided Multi-scale Graph Mamba (KMG-Mamba)
"""
class CrossMamba(nn.Module):
    def __init__(self, config, img_size, num_classes, sequence_length_min):
        super(CrossMamba, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = GraphRepresentation(config)
        self.cross_mamba = CrossMambaFusionBlock(hidden_dim=96, mlp_ratio=1.0, d_state=4) 
        self.sequence_length_min = sequence_length_min
        self.repeat_num = sequence_length_min
        self.norm = nn.LayerNorm(config.hidden_size)
    def forward(self, input_ids, x_l):
        if(input_ids.shape[0] < self.sequence_length_min):
            num = self.repeat_num
            new_input_ids = input_ids
            for i in range(int(num)):
                new_input_ids = torch.cat((new_input_ids, input_ids), dim=0)
            input_ids = new_input_ids
            new_input_x_l = x_l
            for i in range(int(num)):
                new_input_x_l = torch.cat((new_input_x_l, x_l), dim=0)
            x_l = new_input_x_l
        """
        Knowledge-guided Graph Representation for low scale features
        Return F_LR and prototype
        """
        embedding_output = self.embeddings(input_ids)
        all_tokens, ins_top = self.encoder(embedding_output)
        """
        Knowledge-guided Graph Representation for high scale features
        Return F_HR and prototype
        """
        embedding_output_l = self.embeddings(x_l)
        all_tokens_l, ins_top_l = self.encoder(embedding_output_l)
        """
        Cross-scale Knowledge Interaction Mamba for F_LR and F_HR
        """
        cross_all_tokens, cross_all_tokens_l = self.cross_mamba(all_tokens, all_tokens_l)
        cross_all_tokens = cross_all_tokens + embedding_output
        cross_all_tokens_l = cross_all_tokens_l + embedding_output_l
        return cross_all_tokens, cross_all_tokens_l, ins_top, ins_top_l

"""
Train KMG-Mamba with aggregator 
"""
class MG_Mamba_main(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, sequence_length_min=0, smoothing_value=0, zero_head=False):
        super(MG_Mamba_main, self).__init__()
        self.transformer = CrossMamba(config, img_size, num_classes, sequence_length_min)
        self.pool_fn = DAttention(input_dim=96, act='relu', gated=False, bias=False, dropout=True)
        bag_classifiers = [nn.Linear(config.hidden_size, 1) for i in range(num_classes)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        self.instance_classifier = nn.Linear(config.hidden_size, 2)
        self.instance_loss_fn = SmoothTop1SVM(2).cuda()
        self.n_classes = num_classes
        self.subtyping = True
        self.loss_ce = nn.CrossEntropyLoss()
    def forward(self, x_s, x_l, label, attention_only=False, labels=None):
        all_tokens, all_tokens_l, ins_top, ins_top_l = self.transformer(x_s, x_l)
        fusion_tokens = torch.cat((all_tokens, all_tokens_l), dim=1)
        Z = self.pool_fn(fusion_tokens.squeeze(0)) 
        logits_ins = self.instance_classifier(ins_top.squeeze(0))
        logits_ins_l = self.instance_classifier(ins_top_l.squeeze(0))
        logits = torch.empty(1, self.n_classes).float().cuda()
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](Z[c])
        loss_ce = self.loss_ce(logits, label)
        loss_ins = self.instance_loss_fn(logits_ins, label)
        loss_ins_l = self.instance_loss_fn(logits_ins_l, label)
        loss = 0.5 * loss_ce + 0.5 * (loss_ins + loss_ins_l)
        Y_prob = F.softmax(logits, dim = 1)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        return Y_prob, Y_hat, loss

CONFIGS = {
    'MG_Trans': configs.get_config(),
}
