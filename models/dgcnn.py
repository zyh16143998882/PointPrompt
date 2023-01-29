#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, Conv1d

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    """
    activation layer
    :param act:
    :param inplace:
    :param neg_slope:
    :param n_prelu:
    :return:
    """

    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

class Conv1dLayer(Seq):
    def __init__(self, channels, act='relu', norm=True, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(Conv1d(channels[i - 1], channels[i], 1, bias=bias))
            if norm:
                m.append(nn.BatchNorm1d(channels[i]))
            if act:
                m.append(act_layer(act))
        super(Conv1dLayer, self).__init__(*m)


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

class DGCNNColor(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNNColor, self).__init__()
        self.args = args
        self.use_avgpool = args.use_avgpool
        self.k = args.k
        self.leaky_relu = bool(args.leaky_relu)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        if self.leaky_relu:
            act_mod = nn.LeakyReLU
            act_mod_args = {'negative_slope': 0.2}
        else:
            act_mod = nn.ReLU
            act_mod_args = {}

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   act_mod(**act_mod_args))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   act_mod(**act_mod_args))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   act_mod(**act_mod_args))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   act_mod(**act_mod_args))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   act_mod(**act_mod_args))

        if self.use_avgpool:
            self.conv3s = Conv1dLayer([1024*3, 512, 64], act='relu', norm=True, bias=True)
        else:
            self.conv3s = Conv1dLayer([1024*2, 512, 64], act='relu', norm=True, bias=True)

        self.final = nn.Conv1d(64, 3, 1, bias=True)

    def forward(self, x):
        batch_size, num_points = x.size(0),x.size(2)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).repeat(1,1,num_points)
        if self.use_avgpool:
            x2 = F.adaptive_avg_pool1d(x, 1).repeat(1,1,num_points)
            x = torch.cat((x, x1, x2), 1)
        else:
            x = torch.cat((x, x1), 1)

        # 2.3.10 classification
        out = self.conv3s(x)
        out = torch.tanh(self.final(out)) / 2 # [-1,1]改为[0,1]之间
        return out,None


class DGCNNPrompt(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNNPrompt, self).__init__()
        self.args = args
        self.k = args.k
        self.leaky_relu = bool(args.leaky_relu)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        if self.leaky_relu:
            act_mod = nn.LeakyReLU
            act_mod_args = {'negative_slope': 0.2}
        else:
            act_mod = nn.ReLU
            act_mod_args = {}

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   act_mod(**act_mod_args))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   act_mod(**act_mod_args))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   act_mod(**act_mod_args))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   act_mod(**act_mod_args))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   act_mod(**act_mod_args))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        if self.leaky_relu:
            act = lambda y: F.leaky_relu(y, negative_slope=0.2)
        else:
            act = F.relu

        x = act(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = act(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x,None






class DGCNNView(nn.Module):
    def __init__(self, args, output_channels=40, dim=512):
        super(DGCNNView, self).__init__()
        self.args = args
        self.k = args.k
        self.leaky_relu = bool(args.leaky_relu)
        self.dim = dim
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm2d(self.dim)
        self.bn3 = nn.BatchNorm2d(self.dim)
        self.bn5 = nn.BatchNorm1d(self.dim)

        if self.leaky_relu:
            act_mod = nn.LeakyReLU
            act_mod_args = {'negative_slope': 0.2}
        else:
            act_mod = nn.ReLU
            act_mod_args = {}

        self.conv1 = nn.Sequential(nn.Conv2d(self.dim*2, self.dim, kernel_size=1, bias=False),
                                   self.bn1,
                                   act_mod(**act_mod_args))
        self.conv2 = nn.Sequential(nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
                                   self.bn2,
                                   act_mod(**act_mod_args))
        self.conv3 = nn.Sequential(nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
                                   self.bn3,
                                   act_mod(**act_mod_args))
        self.conv5 = nn.Sequential(nn.Conv1d(self.dim * 3, self.dim, kernel_size=1, bias=False),
                                   self.bn5,
                                   act_mod(**act_mod_args))




    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        # x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        # x = torch.cat((x1, x2), 1)

        return x1


# class DGCNNView(nn.Module):
#     def __init__(self, args, output_channels=40):
#         super(DGCNNView, self).__init__()
#         self.args = args
#         self.k = args.k
#         self.leaky_relu = bool(args.leaky_relu)
#
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.bn5 = nn.BatchNorm1d(args.emb_dims)
#
#         if self.leaky_relu:
#             act_mod = nn.LeakyReLU
#             act_mod_args = {'negative_slope': 0.2}
#         else:
#             act_mod = nn.ReLU
#             act_mod_args = {}
#
#         self.conv1 = nn.Sequential(nn.Conv2d(output_channels*2, 64, kernel_size=1, bias=False),
#                                    self.bn1,
#                                    act_mod(**act_mod_args))
#         self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
#                                    self.bn2,
#                                    act_mod(**act_mod_args))
#         self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
#                                    self.bn3,
#                                    act_mod(**act_mod_args))
#         self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
#                                    self.bn4,
#                                    act_mod(**act_mod_args))
#         self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
#                                    self.bn5,
#                                    act_mod(**act_mod_args))
#         self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
#         self.bn6 = nn.BatchNorm1d(512)
#         self.dp1 = nn.Dropout(p=args.dropout)
#         self.linear2 = nn.Linear(512, 256)
#         self.bn7 = nn.BatchNorm1d(256)
#         self.dp2 = nn.Dropout(p=args.dropout)
#         self.linear3 = nn.Linear(256, output_channels)
#
#     def forward(self, x):
#         batch_size = x.size(0)
#         x = get_graph_feature(x, k=self.k)
#         x = self.conv1(x)
#         x1 = x.max(dim=-1, keepdim=False)[0]
#
#         x = get_graph_feature(x1, k=self.k)
#         x = self.conv2(x)
#         x2 = x.max(dim=-1, keepdim=False)[0]
#
#         x = get_graph_feature(x2, k=self.k)
#         x = self.conv3(x)
#         x3 = x.max(dim=-1, keepdim=False)[0]
#
#         x = get_graph_feature(x3, k=self.k)
#         x = self.conv4(x)
#         x4 = x.max(dim=-1, keepdim=False)[0]
#
#         x = torch.cat((x1, x2, x3, x4), dim=1)
#
#         x = self.conv5(x)
#         x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
#         x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
#         x = torch.cat((x1, x2), 1)
#
#         if self.leaky_relu:
#             act = lambda y: F.leaky_relu(y, negative_slope=0.2)
#         else:
#             act = F.relu
#
#         x = act(self.bn6(self.linear1(x)))
#         x = self.dp1(x)
#         x = act(self.bn7(self.linear2(x)))
#         x = self.dp2(x)
#         x = self.linear3(x)
#         return x


class SELayer(nn.Module):
    """
    input:
        x:(b, c, m, n)

    output:
        out:(b, c, m', n')
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CAEView(nn.Module):
    def __init__(self, args, output_channels=40):
        super(CAEView, self).__init__()
        self.args = args
        self.k = args.k
        self.leaky_relu = bool(args.leaky_relu)
        self.use_SElayer = True

        self.conv1 = nn.Conv2d(512*2, 512, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(
            512*2, 512, kernel_size=1, bias=False
        )
        self.conv3 = nn.Conv2d(
            512*2, 1024, kernel_size=1, bias=False
        )
        self.conv4 = nn.Conv2d(
            1024*2, 1024, kernel_size=1, bias=False
        )
        self.conv5 = nn.Conv1d(
            1024*2, 2048, kernel_size=1, bias=False
        )

        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.relu4 = nn.LeakyReLU(negative_slope=0.2)
        self.relu5 = nn.LeakyReLU(negative_slope=0.2)

        if self.use_SElayer:
            self.se1 = SELayer(channel=512)
            self.se2 = SELayer(channel=512)
            self.se3 = SELayer(channel=1024)
            self.se4 = SELayer(channel=1024)

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(1024)
        self.bn4 = nn.BatchNorm2d(1024)
        self.bn5 = nn.BatchNorm1d(2048)

        # 再第2，3，4个CAE中的输入直接到输出的Linear层，也就是F3
        self.resconv1 = nn.Conv1d(
            512, 512, kernel_size=1, bias=False
        )
        self.resconv2 = nn.Conv1d(
            512, 1024, kernel_size=1, bias=False
        )
        self.resconv3 = nn.Conv1d(
            1024, 1024, kernel_size=1, bias=False
        )

        self.linear1 = nn.Linear(4096, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)  # batch_size = 2
        if self.use_SElayer:
            x = get_graph_feature(x,
                                  k=self.k)  # 计算边的特征 结果为：torch.Size([2, 6, 3000, 8])   [2, 3, 3000, 8]和[2, 3, 3000, 8]相cat，前面是qij-pi后面是pi
            x = self.relu1(self.se1(self.bn1(self.conv1(x))))  # conv1是CAE模块的第一个MLP，即F1，然后进入自注意力，然后外层relu
            x1 = x.max(dim=-1, keepdim=False)[0]  # 最后的最大池化

            x2_res = self.resconv1(x1)  # F3
            x = get_graph_feature(x1, k=self.k)  # KNN模块
            x = self.relu2(self.se2(self.bn2(self.conv2(x))))
            x2 = x.max(dim=-1, keepdim=False)[0]
            x2 = x2 + x2_res

            x3_res = self.resconv2(x2)
            x = get_graph_feature(x2, k=self.k)
            x = self.relu3(self.se3(self.bn3(self.conv3(x))))
            x3 = x.max(dim=-1, keepdim=False)[0]
            x3 = x3 + x3_res

            x4_res = self.resconv3(x3)  # 最后一个CAE的残差
            x = get_graph_feature(x3, k=self.k)  # knn
            x = self.relu4(self.se4(self.bn4(self.conv4(x))))
        else:
            x = get_graph_feature(x, k=self.k)
            x = self.relu1(self.bn1(self.conv1(x)))
            x1 = x.max(dim=-1, keepdim=False)[0]

            x2_res = self.resconv1(x1)
            x = get_graph_feature(x1, k=self.k)
            x = self.relu2(self.bn2(self.conv2(x)))
            x2 = x.max(dim=-1, keepdim=False)[0]
            x2 = x2 + x2_res

            x3_res = self.resconv2(x2)
            x = get_graph_feature(x2, k=self.k)
            x = self.relu3(self.bn3(self.conv3(x)))
            x3 = x.max(dim=-1, keepdim=False)[0]
            x3 = x3 + x3_res

            x4_res = self.resconv3(x3)
            x = get_graph_feature(x3, k=self.k)
            x = self.relu4(self.bn4(self.conv4(x)))

        x4 = x.max(dim=-1, keepdim=False)[0]  # 最后一层的最大池化
        x4 = x4 + x4_res  # 残差和特征加起来

        x = torch.cat((x1, x2, x3, x4),
                      dim=1)  # torch.Size([2, 256, 3000]) torch.Size([2, 256, 3000]) torch.Size([2, 512, 3000]) torch.Size([2, 1024, 3000]) 合起来为torch.Size([2, 2048, 3000])
        x = self.relu5(self.bn5(self.conv5(
            x)))  # conv5 bn5 relu5合起来构成encoder最终的MLP  torch.Size([2, 2048, 3000])用MLP转成torch.Size([2, 2048, 3000])

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # [bs, 2048]    最大池化，3000个点的特征合成一个点的特征
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # [bs, 2048]    平均池化
        x = torch.cat((x1, x2), 1)  # 两个[bs, 2048]转成[bs, 4096]

        if self.leaky_relu:
            act = lambda y: F.leaky_relu(y, negative_slope=0.2)
        else:
            act = F.relu

        x = act(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = act(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x