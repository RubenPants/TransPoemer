"""
shared.py

Shared building blocks that are used to create the Encoder and Decoder. Note, however, that one single building-block is
not shared between the encoder-decoder, only their architecture.
"""
import torch
import torch.nn as nn


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]
        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)
        
        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]
        query = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]
        energy = torch.matmul(query, key.permute(0, 1, 3, 2)) / self.scale
        
        # energy = [batch size, n heads, seq len, seq len]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        # attention = [batch size, n heads, query len, key len]
        attention = torch.softmax(energy, dim=-1)
        
        # x = [batch size, n heads, seq len, head dim]
        x = torch.matmul(self.dropout(attention), value)
        x = x.permute(0, 2, 1, 3).contiguous()
        
        # x = [batch size, seq len, hid dim]
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x = [batch size, seq len, hid dim]
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x
