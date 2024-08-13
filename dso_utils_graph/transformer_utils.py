import torch
from torch import nn
from torch import functional as F
from math import log

nn.Transformer
class TransformerDSOEncoder(nn.Module):
    def __init__(self, vocabulary_size, scatter_max_degree, max_length, d_model=64, num_heads=8, d_ff=256, num_layers=6, structural_encoding=True):
        super().__init__()
        self.num_layers = num_layers
        self.structural_encoding = structural_encoding
        self.vocabulary_embedding = nn.Embedding(vocabulary_size+1, d_model) # begin token
        self.position_encoder = PositionalEncoding(d_model, max_length+2)

        if self.structural_encoding:
            self.scatter_degree_embedding = nn.Embedding(scatter_max_degree+1, d_model)
            self.relation_encoder = nn.Parameter(data=torch.rand(3), requires_grad=True)
            self.active_relation_encoder = nn.Embedding(2, d_model)

        self.self_attention = nn.ModuleList([MultiheadAttention(d_model, num_heads) for _ in range(num_layers)])
        self.feed_forward = nn.ModuleList([FeedForward(d_model, d_ff) for _ in range(num_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.logits = nn.Linear(d_model, vocabulary_size, bias=False)
        self._parent_sibling = torch.arange(2)

    def forward(self, raw_x, scatter_degree, parentchild_indices, parent_child_now, silbing_indices, silbing_now):

        x = self.vocabulary_embedding(raw_x) 
        x = self.position_encoder(x)

        if self.structural_encoding:
            x = x + self.scatter_degree_embedding(scatter_degree)
            active_parent_sibling_embedding = self.active_relation_encoder(self._parent_sibling)
            x[parent_child_now[:,0], parent_child_now[:,1]] += active_parent_sibling_embedding[0]
            x[silbing_now[:,0], silbing_now[:,1]] += active_parent_sibling_embedding[1]

            spatial = parentchild_indices, silbing_indices, self.relation_encoder

        else:
            spatial = None

        for i in range(self.num_layers):
            # Self-attention
            x = self.self_attention[i](x, x, x, spatial) + x
            x = self.layer_norms[i](x)
            
            # Feed forward
            x = self.feed_forward[i](x) + x
            x = self.layer_norms[i](x)
        
        x = self.logits(x[:,-1,...])
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[0, :x.size(1), :]
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiheadAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_per_head = self.d_model // self.num_heads
        self.norm_value = torch.sqrt(torch.tensor(self.d_per_head).float())
        
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

        self.activation = nn.Softmax(dim=-1)
        
    def forward(self, query, key, value, spatial):
        batch_size = query.size(0)
        
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)
        
        query = query.view(batch_size, -1, self.num_heads, self.d_per_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.d_per_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_per_head).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.norm_value

        if spatial is not None:
            parentchild_indices, silbing_indices, embedings = spatial

            scores[parentchild_indices[:,0],:,parentchild_indices[:,1],parentchild_indices[:,2]] += embedings[0]
            scores[parentchild_indices[:,0],:,parentchild_indices[:,2],parentchild_indices[:,1]] += embedings[1]
            scores[silbing_indices[:,0],:,silbing_indices[:,1],silbing_indices[:,2]] += embedings[2]
            scores[silbing_indices[:,0],:,silbing_indices[:,2],silbing_indices[:,1]] += embedings[2]

        attention = self.activation(scores)
        
        x = torch.matmul(attention, value)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        x = self.output_linear(x)
        
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
                
    def forward(self, x):        
        return self.model(x)