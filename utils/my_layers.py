from math import sqrt

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Attention, self).__init__()
        self.dim_in = dim_in
        self.linear_q = nn.Linear(dim_in, dim_out, bias=True)
        self.linear_k = nn.Linear(dim_out, 1, bias=False)
        self._norm_fact = 1 / sqrt(dim_out)

    def forward(self, x):
        # x: batch, n, dim_in
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        q = self.linear_q(x)  # batch, n, dim_out

        dist = torch.squeeze(self.linear_k(q))  # batch, n, 1
        att = torch.softmax(dist, dim=-1)  # batch, n
        att = torch.unsqueeze(att, dim=-1)  # batch, n, 1
        ot = att * x  # batch, n, dim_in
        att = torch.sum(ot, dim=1)  # batch, dim_in
        return att


class SelfAttention(nn.Module):
    dim_in: int
    dim_out: int  # q, k ,v 设置为相同

    def __init__(self, dim_in, dim_out):
        super(SelfAttention, self).__init__()
        self.dim_in = dim_in
        self.linear_q = nn.Linear(dim_in, dim_out, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_out, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_out, bias=False)
        self._norm_fact = 1 / sqrt(dim_out)

    def forward(self, x):
        # x: batch, n, dim_in
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v

        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, n, n

        att = torch.bmm(dist, v)
        return att


class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att


class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d,self).__init__()
    def forward(self, x):
        x = torch.nn.MaxPool2d(kernel_size=(x.shape[-2],1))(x)
        x = torch.squeeze(x)
        return x


class GlobalAvgPool1d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool1d,self).__init__()
    def forward(self, x):
        x = torch.nn.AvgPool2d(kernel_size=(x.shape[-2],1))(x)
        x = torch.squeeze(x)
        return x
