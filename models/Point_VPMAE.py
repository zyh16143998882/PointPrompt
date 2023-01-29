import os

import torch
import torch.nn as nn

from timm.models.layers import DropPath, trunc_normal_
import numpy as np

from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
from knn_cuda import KNN
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2, ChamferDistanceL2_all
from pointnet2_ops import pointnet2_utils as pn2_utils


from .image_models.color_pred import ColorPredict, PromptPredict
from .image_models.mvtn import MVTN
from .image_models.renderer import MVRenderer

import torchvision.models as models
from torchvision import utils as vutils

VISUALIZER_PRE = False
VIS_REAL_PATH_POINT = './output/gt/'
if VISUALIZER_PRE == True:
    if not os.path.isdir(VIS_REAL_PATH_POINT):
        os.makedirs(VIS_REAL_PATH_POINT)


class Encoder(nn.Module):  ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

class Group2(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center, fps_idx = misc.fps2(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center, fps_idx


## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x

class TransformerPromptEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.trans_dim = embed_dim
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])



    def forward(self, x, pos, prompt, num_group):
        for _, block in enumerate(self.blocks):
            # x = torch.cat((x[:,:1,:], prompt, x[:,-num_group:,:]), dim=1)
            x = torch.cat((x[:,:1,:], prompt[_].expand(x.size(0), -1, -1), x[:,-num_group:,:]), dim=1)
            x = block(x + pos)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x


# Pretrain model
class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[args] {config.transformer_config}', logger='Transformer')
        # embedding
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.mask_type = config.transformer_config.mask_type

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _compute_sqrdis_map(self, points_x, points_y):
        ## The shape of the input and output ##
        # points_x : batchsize * M * 3
        # points_y : batchsize * N * 3
        # output   : batchsize * M * N
        thisbatchsize = points_x.size()[0]
        pn_x = points_x.size()[1]
        pn_y = points_y.size()[1]
        x_sqr = torch.sum(torch.mul(points_x, points_x), dim=-1).view(thisbatchsize, pn_x, 1).expand(-1, -1, pn_y)
        y_sqr = torch.sum(torch.mul(points_y, points_y), dim=-1).view(thisbatchsize, 1, pn_y).expand(-1, pn_x, -1)
        inner = torch.bmm(points_x, points_y.transpose(1, 2))
        sqrdis = x_sqr + y_sqr - 2 * inner
        return sqrdis

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)        # mask多少个组

        overall_mask = np.zeros([B, G])                 # (128, 64)
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),            # 建一堆为0的数组和1的数组合并
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)                     # shuffle
            overall_mask[i, :] = mask                   # 将这一个的加到整个batch中
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G

    def _mask_center_rand_view(self, center, noaug=False, view_points = None):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_unmask = G - int(self.mask_ratio * G)        # 不mask多少个组

        # 生成随即视角索引
        views_index = torch.tensor(np.random.randint(0, 20, size=(B))).unsqueeze(dim=1).int().to(center.device)
        view = pn2_utils.gather_operation(view_points.permute(0, 2, 1).contiguous(), views_index)

        center2view_dismap = self._compute_sqrdis_map(center, view.permute(0, 2, 1))
        centers_id = torch.topk(center2view_dismap, k=self.num_unmask, dim=1, largest=False)[1].squeeze()       # largest表示最近的

        # 根据index修改mask中的值
        overall_mask = torch.ones(B, G).to(center.device)
        overall_mask.scatter_(1, centers_id, 0)

        return overall_mask.to(torch.bool)  # B G

    def _mask_recloss_max(self, center, noaug=False, recloss = None):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_unmask = G - int(self.mask_ratio * G)        # 不mask多少个组

        centers_id = torch.topk(recloss, k=self.num_unmask, dim=1, largest=True)[1].squeeze()

        # 根据index修改mask中的值
        overall_mask = torch.ones(B, G).to(center.device)
        overall_mask.scatter_(1, centers_id, 0)

        return overall_mask.to(torch.bool)  # B G

    def _mask_recloss_min(self, center, noaug=False, recloss = None):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_unmask = G - int(self.mask_ratio * G)        # 不mask多少个组

        centers_id = torch.topk(recloss, k=self.num_unmask, dim=1, largest=False)[1].squeeze()

        # 根据index修改mask中的值
        overall_mask = torch.ones(B, G).to(center.device)
        overall_mask.scatter_(1, centers_id, 0)

        return overall_mask.to(torch.bool)  # B G



    def forward(self, neighborhood, center, noaug=False, view_points=None):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)  # B G torch.Size([128, 64])个True或者False
        elif self.mask_type == 'rand_view':
            bool_masked_pos = self._mask_center_rand_view(center, noaug=noaug, view_points=view_points)  # B G
        elif self.mask_type == 'recloss_max':
            bool_masked_pos = self._mask_recloss_max(center, noaug=noaug, recloss=view_points)  # B G
        elif self.mask_type == 'recloss_min':
            bool_masked_pos = self._mask_recloss_min(center, noaug=noaug, recloss=view_points)  # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        group_input_tokens = self.encoder(neighborhood)  # B G C

        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos


class MaskTransformer_global(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[args] {config.transformer_config}', logger='Transformer')
        # embedding
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.mask_type = config.transformer_config.mask_type

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _compute_sqrdis_map(self, points_x, points_y):
        ## The shape of the input and output ##
        # points_x : batchsize * M * 3
        # points_y : batchsize * N * 3
        # output   : batchsize * M * N
        thisbatchsize = points_x.size()[0]
        pn_x = points_x.size()[1]
        pn_y = points_y.size()[1]
        x_sqr = torch.sum(torch.mul(points_x, points_x), dim=-1).view(thisbatchsize, pn_x, 1).expand(-1, -1, pn_y)
        y_sqr = torch.sum(torch.mul(points_y, points_y), dim=-1).view(thisbatchsize, 1, pn_y).expand(-1, pn_x, -1)
        inner = torch.bmm(points_x, points_y.transpose(1, 2))
        sqrdis = x_sqr + y_sqr - 2 * inner
        return sqrdis

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)        # mask多少个组

        overall_mask = np.zeros([B, G])                 # (128, 64)
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),            # 建一堆为0的数组和1的数组合并
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)                     # shuffle
            overall_mask[i, :] = mask                   # 将这一个的加到整个batch中
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G

    def _mask_center_rand_view(self, center, noaug=False, view_points = None):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_unmask = G - int(self.mask_ratio * G)        # 不mask多少个组

        # 生成随即视角索引
        views_index = torch.tensor(np.random.randint(0, 20, size=(B))).unsqueeze(dim=1).int().to(center.device)
        view = pn2_utils.gather_operation(view_points.permute(0, 2, 1).contiguous(), views_index)

        center2view_dismap = self._compute_sqrdis_map(center, view.permute(0, 2, 1))
        centers_id = torch.topk(center2view_dismap, k=self.num_unmask, dim=1, largest=False)[1].squeeze()       # largest表示最近的

        # 根据index修改mask中的值
        overall_mask = torch.ones(B, G).to(center.device)
        overall_mask.scatter_(1, centers_id, 0)

        return overall_mask.to(torch.bool)  # B G

    def _mask_recloss_max(self, center, noaug=False, recloss = None):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_unmask = G - int(self.mask_ratio * G)        # 不mask多少个组

        centers_id = torch.topk(recloss, k=self.num_unmask, dim=1, largest=True)[1].squeeze()

        # 根据index修改mask中的值
        overall_mask = torch.ones(B, G).to(center.device)
        overall_mask.scatter_(1, centers_id, 0)

        return overall_mask.to(torch.bool)  # B G

    def _mask_recloss_min(self, center, noaug=False, recloss = None):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_unmask = G - int(self.mask_ratio * G)        # 不mask多少个组

        centers_id = torch.topk(recloss, k=self.num_unmask, dim=1, largest=False)[1].squeeze()

        # 根据index修改mask中的值
        overall_mask = torch.ones(B, G).to(center.device)
        overall_mask.scatter_(1, centers_id, 0)

        return overall_mask.to(torch.bool)  # B G



    def forward(self, neighborhood, center, neighborhood_raw, center_raw, bool_masked_pos_raw, noaug=False, view_points=None):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)  # B G torch.Size([128, 64])个True或者False
        elif self.mask_type == 'rand_view':
            bool_masked_pos = self._mask_center_rand_view(center, noaug=noaug, view_points=view_points)  # B G
        elif self.mask_type == 'recloss_max':
            bool_masked_pos = self._mask_recloss_max(center, noaug=noaug, recloss=view_points)  # B G
        elif self.mask_type == 'recloss_min':
            bool_masked_pos = self._mask_recloss_min(center, noaug=noaug, recloss=view_points)  # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        group_input_tokens = self.encoder(neighborhood_raw)  # B G C

        batch_size, seq_len, C = group_input_tokens.size()

        neighborhood = neighborhood[bool_masked_pos].reshape(neighborhood.size(0), -1, neighborhood.size(2), neighborhood.size(3))

        tokens_partial_temp = group_input_tokens[~bool_masked_pos_raw].reshape(group_input_tokens.size(0), -1, group_input_tokens.size(2))
        tokens_last_temp = group_input_tokens[bool_masked_pos_raw].reshape(group_input_tokens.size(0), -1, group_input_tokens.size(2))

        x_vis = tokens_partial_temp[~bool_masked_pos].reshape(batch_size, -1, C)
        x_vis = torch.cat([tokens_last_temp,x_vis],dim=1)

        # add pos embedding
        # mask pos center

        center_partial_temp = center
        center_last_temp = center_raw[bool_masked_pos_raw].reshape(center_raw.size(0), -1,center_raw.size(2))

        unmasked_center = center_partial_temp[~bool_masked_pos].reshape(batch_size, -1, 3)
        masked_center = center_partial_temp[bool_masked_pos].reshape(batch_size, -1, 3)
        unmasked_center = torch.cat([center_last_temp,unmasked_center],dim=1)
        pos = self.pos_embed(unmasked_center)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, neighborhood, unmasked_center, masked_center





@MODELS.register_module()
class Point_LoMAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_VPMAE] ', logger='Point_VPMAE')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[Point_VPMAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='Point_VPMAE')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)





    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        elif loss_type == 'cdl2_all':
            self.loss_func = ChamferDistanceL2_all().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def _compute_sqrdis_map(self, points_x, points_y):
        ## The shape of the input and output ##
        # points_x : batchsize * M * 3
        # points_y : batchsize * N * 3
        # output   : batchsize * M * N
        thisbatchsize = points_x.size()[0]
        pn_x = points_x.size()[1]
        pn_y = points_y.size()[1]
        x_sqr = torch.sum(torch.mul(points_x, points_x), dim=-1).view(thisbatchsize, pn_x, 1).expand(-1, -1, pn_y)
        y_sqr = torch.sum(torch.mul(points_y, points_y), dim=-1).view(thisbatchsize, 1, pn_y).expand(-1, pn_x, -1)
        inner = torch.bmm(points_x, points_y.transpose(1, 2))
        sqrdis = x_sqr + y_sqr - 2 * inner
        return sqrdis

    def _mask_center_all_views(self, center, view_points=None):
        '''
            center : B G 3
            --------------
            mask : B V G (bool)
        '''
        B, G, _ = center.shape
        _, V, _ = view_points.shape


        # 计算中心点距离所有视点的距离
        center2view_dismap = self._compute_sqrdis_map(center, view_points)
        centers_id = torch.topk(center2view_dismap, k=10, dim=1, largest=False)[1]

        # 根据index修改mask中的值
        overall_mask = torch.ones(B, V, G).to(center.device)
        overall_mask.scatter_(2, centers_id.permute(0, 2, 1), 0)

        return overall_mask.to(torch.bool)  # B G

    def forward(self, pts, noaug=False,vis=False, **kwargs):
        view_points = pts[:,self.config.npoints:,:].float()
        pts = pts[:,:self.config.npoints,:].contiguous().float()
        neighborhood, center = self.group_divider(pts)

        bool_masked_pos_all = self._mask_center_all_views(center, view_points=view_points)

        neighborhood_all = []
        center_all = []
        for i in range(view_points.size(1)):
            bool_masked_pos = bool_masked_pos_all[:, i, :]
            neighborhood_temp = neighborhood[~bool_masked_pos].reshape(neighborhood.size(0), -1, neighborhood.size(2), neighborhood.size(3))
            center_temp = center[~bool_masked_pos].reshape(center.size(0), -1, center.size(2))
            neighborhood_all.append(neighborhood_temp.unsqueeze(1))
            center_all.append(center_temp.unsqueeze(1))
        neighborhood_all = torch.cat(neighborhood_all,dim=1)
        center_all = torch.cat(center_all, dim=1)

        neighborhood = torch.flatten(neighborhood_all,start_dim=0,end_dim=1)
        center = torch.flatten(center_all,start_dim=0,end_dim=1)

        x_vis, mask = self.MAE_encoder(neighborhood, center, noaug=noaug)
        if noaug == True:
            return x_vis.mean(1) + x_vis.max(1)[0]
        B, _, C = x_vis.shape  # B VIS C

        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)

        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, N)

        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

        gt_points = neighborhood[mask].reshape(B * M, -1, 3)
        loss1,_ = self.loss_func(rebuild_points, gt_points)

        if vis:  # visualization

            vis_points = neighborhood[~mask].reshape(B ,(10 - M), -1, 3)
            full_vis = vis_points + center[~mask].reshape(B ,(10 - M), 3).unsqueeze(2)
            full_rebuild = rebuild_points.reshape(B ,M, -1, 3) + center[mask].reshape(B ,M, 3).unsqueeze(2)
            full = torch.cat([full_vis, full_rebuild], dim=1)

            # full_points = torch.cat([rebuild_points,vis_points], dim=0)
            vis_center = center[~mask].reshape(B ,(10 - M), 3)
            center2 = center[mask].reshape(B ,M, 3)
            full_center = torch.cat([center2, vis_center], dim=1)
            # full = full_points + full_center.unsqueeze(1)
            # ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            # ret1 = full.reshape(-1, 3).unsqueeze(0)
            # return ret1, ret2

            ret2 = full_vis.reshape(B, -1, 3)
            ret1 = full.reshape(B, -1, 3)
            ret3 = neighborhood + center.unsqueeze(2)
            return ret1, ret2, ret3.reshape(B, -1, 3), full_center, vis_center, loss1
        else:
            return loss1












# # finetune model
# @MODELS.register_module()
# class PointTransformer(nn.Module):
#     def __init__(self, config, **kwargs):
#         super().__init__()
#         self.config = config
#
#         self.trans_dim = config.trans_dim
#         self.depth = config.depth
#         self.drop_path_rate = config.drop_path_rate
#         self.cls_dim = config.cls_dim
#         self.num_heads = config.num_heads
#
#         self.group_size = config.group_size
#         self.num_group = config.num_group
#         self.encoder_dims = config.encoder_dims
#
#         self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
#
#         self.encoder = Encoder(encoder_channel=self.encoder_dims)
#
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
#         self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
#
#         self.pos_embed = nn.Sequential(
#             nn.Linear(3, 128),
#             nn.GELU(),
#             nn.Linear(128, self.trans_dim)
#         )
#
#         dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
#         self.blocks = TransformerPromptEncoder(
#             embed_dim=self.trans_dim,
#             depth=self.depth,
#             drop_path_rate=dpr,
#             num_heads=self.num_heads,
#         )
#
#         self.norm = nn.LayerNorm(self.trans_dim)
#
#         self.cls_head_finetune = nn.Sequential(
#             nn.Linear(self.trans_dim * 2, 256),
#             # nn.Linear(self.trans_dim * 1, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(256, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(256, self.cls_dim)
#         )
#
#         self.build_loss_func()
#
#         trunc_normal_(self.cls_token, std=.02)
#         trunc_normal_(self.cls_pos, std=.02)
#
#         # 渲染器新建
#         self.nb_views = self.config.nb_views
#         self.object_color = self.config.color
#         self.mvtn = MVTN(nb_views=self.nb_views, views_config='gcn', canonical_distance=1.0)
#         self.img_color_predict = ColorPredict(shape_extractor="dgcnn", use_avgpool=False)
#         self.mvrenderer = MVRenderer(nb_views=self.nb_views, image_size=128,
#                                 object_color=self.object_color, background_color="black", points_radius=0.017,
#                                 points_per_pixel=5, cull_backfaces=True, depthmap="depth")
#
#         self.img_model = models.resnet101(pretrained=True)
#         self.img_model.fc = nn.Linear(2048, self.trans_dim)
#
#
#     def build_loss_func(self):
#         self.loss_ce = nn.CrossEntropyLoss()
#
#     def get_loss_acc(self, ret, gt):
#         loss = self.loss_ce(ret, gt.long())
#         pred = ret.argmax(-1)
#         acc = (pred == gt).sum() / float(gt.size(0))
#         return loss, acc * 100
#
#     def load_model_from_ckpt(self, bert_ckpt_path):
#         if bert_ckpt_path is not None:
#             ckpt = torch.load(bert_ckpt_path)
#             base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
#
#             for k in list(base_ckpt.keys()):
#                 if k.startswith('MAE_encoder'):
#                     base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
#                     del base_ckpt[k]
#                 elif k.startswith('base_model'):
#                     base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
#                     del base_ckpt[k]
#
#             incompatible = self.load_state_dict(base_ckpt, strict=False)
#
#             if incompatible.missing_keys:
#                 print_log('missing_keys', logger='Transformer')
#                 print_log(
#                     get_missing_parameters_message(incompatible.missing_keys),
#                     logger='Transformer'
#                 )
#             if incompatible.unexpected_keys:
#                 print_log('unexpected_keys', logger='Transformer')
#                 print_log(
#                     get_unexpected_parameters_message(incompatible.unexpected_keys),
#                     logger='Transformer'
#                 )
#
#             print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
#         else:
#             print_log('Training from scratch!!!', logger='Transformer')
#             self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv1d):
#             trunc_normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, pts):
#         # eyes_pos_list = view_points.repeat(pts.size(0),1,1)
#         # image field
#         azim, elev, dist = self.mvtn(pts, c_batch_size=pts.size(0))
#         color = None
#         if self.object_color == "custom":
#             color = self.img_color_predict(pts.permute(0, 2, 1))
#         rendered_images, _ = self.mvrenderer( None, pts, azim=azim, elev=elev, dist=dist, color=color, eyes_pos_list=azim)
#
#         if VISUALIZER_PRE == True:
#             for k in range(rendered_images.size(0)):
#                 for j in range(rendered_images.size(1)):
#                     img = rendered_images[k, j, :, :, :]
#                     vutils.save_image(img, VIS_REAL_PATH_POINT + '{}_{}.jpg'.format(str(k), str(j)), ormalize=True)
#
#         N, V, C, H, W = rendered_images.size()
#         rendered_images = rendered_images.contiguous().view(-1, C, H, W)
#
#
#         prompt = self.img_model(rendered_images).view(N, V, self.trans_dim)
#
#         neighborhood, center = self.group_divider(pts)
#         group_input_tokens = self.encoder(neighborhood)  # B G N
#
#         cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
#         cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
#
#         pos = self.pos_embed(center)
#         pos_v = self.pos_embed(azim.view(N,V,3).cuda())
#
#         x = torch.cat((cls_tokens, group_input_tokens), dim=1)
#         pos = torch.cat((cls_pos, pos_v, pos), dim=1)
#         # transformer
#         x = self.blocks(x, pos, prompt, self.num_group)
#         x = self.norm(x)            # torch.Size([32, 65, 384])
#         concat_f = torch.cat([x[:, 0], x[:, 1+V:].max(1)[0]], dim=-1)
#         # concat_f = torch.cat([x[:, 1:1+V].max(1)[0]], dim=-1)
#         ret = self.cls_head_finetune(concat_f)
#         return ret


@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group2(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            # nn.Linear(self.trans_dim * 1, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

        # 解码器
        self.decoder_depth = config.decoder_depth
        self.decoder_num_heads = config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )
        self.mask_pos = nn.Parameter(torch.zeros(1, 1, 3))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))


        # prompt
        self.prompt_num = config.prompt_num
        self.cls_offset = PromptPredict(shape_extractor="dgcnn", use_avgpool=False, nb_views=self.prompt_num)


    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):

        neighborhood, center, fps_idx = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        # x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        # pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(group_input_tokens, pos)
        x = self.norm(x)  # torch.Size([32, 65, 384])


        # decoder
        B, _, C = x.shape
        # 得到生成的基础点
        center_mask = torch.ones(B, pts.size(1)).to(center.device)
        center_mask = center_mask.scatter_(1, fps_idx.long(), 0).to(torch.bool)
        pos_emd_vis = self.decoder_pos_embed(center).reshape(B, -1, C)
        pts_un = pts[center_mask].reshape(B, -1, 3)

        # 随机采样prompt_num个基点
        center_dec = misc.fps(pts_un, self.num_group)
        # offset = self.cls_offset(center.permute(0, 2, 1))
        # center_dec = center_dec + offset / 4

        pos_emd_mask = self.decoder_pos_embed(center_dec).reshape(B, -1, C)

        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)

        x_full = torch.cat([x, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, N)

        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B, M, -1, 3)

        # prompt encoder
        group_input_tokens_prompt = self.encoder(rebuild_points)
        pos_prompt = self.pos_embed(center_dec)

        x = torch.cat((cls_tokens, group_input_tokens_prompt), dim=1)
        pos = torch.cat((cls_pos, pos_prompt), dim=1)

        x_prompt = self.blocks(x, pos)
        x_prompt = self.norm(x_prompt)  # torch.Size([32, 65, 384])


        concat_f = torch.cat([x_prompt[:, 0],x_prompt[:, 1:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret


@MODELS.register_module()
class PointTransformer2(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
            # nn.Linear(self.trans_dim * 2, 256),
            nn.Linear(self.trans_dim * 1, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

        # 渲染器新建
        self.mvtn = MVTN(nb_views=1, views_config='gcn', canonical_distance=1.0)
        self.cls_prompt = PromptPredict(shape_extractor="pointnet", use_avgpool=False, output_channels=self.trans_dim)



    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        # eyes_pos_list = view_points.repeat(pts.size(0),1,1)
        # image field
        azim, elev, dist = self.mvtn(pts, c_batch_size=pts.size(0))

        color = self.cls_prompt(pts.permute(0, 2, 1))


        prompt = color

        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        # cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        # cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)
        pos_v = self.pos_embed(azim.view(prompt.size(0),prompt.size(1),3).cuda())

        x = torch.cat((prompt, group_input_tokens), dim=1)
        pos = torch.cat((pos_v, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)            # torch.Size([32, 65, 384])
        # concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        concat_f = torch.cat([x[:, 0]], dim=-1)
        # concat_f = torch.cat([x[:, 1:1+V].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret


@MODELS.register_module()
class PointTransformer_img(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
            # nn.Linear(self.trans_dim * 2, 256),
            nn.Linear(self.trans_dim * 1, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

        # 渲染器新建
        self.nb_views = self.config.nb_views
        self.object_color = self.config.color
        self.mvtn = MVTN(nb_views=self.nb_views, views_config='gcn', canonical_distance=1.0)
        self.img_color_predict = ColorPredict(shape_extractor="pointnet", use_avgpool=False)
        self.mvrenderer = MVRenderer(nb_views=self.nb_views, image_size=224,
                                object_color=self.object_color, background_color="black", points_radius=0.017,
                                points_per_pixel=5, cull_backfaces=True, depthmap="depth")

        self.img_model = models.resnet101(pretrained=True)
        self.img_model.fc = nn.Linear(2048, self.trans_dim)


    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        # eyes_pos_list = view_points.repeat(pts.size(0),1,1)
        # image field
        azim, elev, dist = self.mvtn(pts, c_batch_size=pts.size(0))
        color = None
        if self.object_color == "custom":
            color = self.img_color_predict(pts.permute(0, 2, 1))
        rendered_images, _ = self.mvrenderer( None, pts, azim=azim, elev=elev, dist=dist, color=color, eyes_pos_list=azim)

        if VISUALIZER_PRE == True:
            for k in range(rendered_images.size(0)):
                for j in range(rendered_images.size(1)):
                    img = rendered_images[k, j, :, :, :]
                    vutils.save_image(img, VIS_REAL_PATH_POINT + '{}_{}.jpg'.format(str(k), str(j)), ormalize=True)

        N, V, C, H, W = rendered_images.size()
        rendered_images = rendered_images.contiguous().view(-1, C, H, W)


        prompt = self.img_model(rendered_images).view(N, V, self.trans_dim)

        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        # cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        # cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)
        pos_v = self.pos_embed(azim.view(N,V,3).cuda())

        x = torch.cat((prompt, group_input_tokens), dim=1)
        pos = torch.cat((pos_v, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)            # torch.Size([32, 65, 384])
        # concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        concat_f = torch.cat([x[:, 0]], dim=-1)
        # concat_f = torch.cat([x[:, 1:1+V].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret




@MODELS.register_module()
class PointTransformer_vis(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)



    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_func = ChamferDistanceL2().cuda()
        # self.loss_func = emd().cuda()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _compute_sqrdis_map(self, points_x, points_y):
        ## The shape of the input and output ##
        # points_x : batchsize * M * 3
        # points_y : batchsize * N * 3
        # output   : batchsize * M * N
        thisbatchsize = points_x.size()[0]
        pn_x = points_x.size()[1]
        pn_y = points_y.size()[1]
        x_sqr = torch.sum(torch.mul(points_x, points_x), dim=-1).view(thisbatchsize, pn_x, 1).expand(-1, -1, pn_y)
        y_sqr = torch.sum(torch.mul(points_y, points_y), dim=-1).view(thisbatchsize, 1, pn_y).expand(-1, pn_x, -1)
        inner = torch.bmm(points_x, points_y.transpose(1, 2))
        sqrdis = x_sqr + y_sqr - 2 * inner
        return sqrdis

    def _mask_center_all_views(self, center, view_points=None,k=10):
        '''
            center : B G 3
            --------------
            mask : B V G (bool)
        '''
        B, G, _ = center.shape
        _, V, _ = view_points.shape


        # 计算中心点距离所有视点的距离
        center2view_dismap = self._compute_sqrdis_map(center, view_points)
        centers_id = torch.topk(center2view_dismap, k=k, dim=1, largest=False)[1]      # n个视点 每个视点10个patch

        # 根据index修改mask中的值
        overall_mask = torch.ones(B, V, G).to(center.device)
        overall_mask.scatter_(2, centers_id.permute(0, 2, 1), 0)

        # overall_mask_ = torch.ones(B, V, k).to(center.device)
        # overall_mask_.scatter_(2, centers_id.permute(0, 2, 1), 0)


        return overall_mask.to(torch.bool), centers_id

    def forward(self, pts):
        view_points = pts[:, self.config.npoints:, :].float()
        pts = pts[:, :self.config.npoints, :].contiguous().float()
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x_global = self.norm(x)            # torch.Size([32, 65, 384])
        concat_f = torch.cat([x_global[:, 0], x_global[:, 1:].max(1)[0]], dim=-1)
        ret_global = self.cls_head_finetune(concat_f)


        # 模拟真实点云
        bool_masked_pos_all,_ = self._mask_center_all_views(center, view_points=view_points, k=10)      # torch.Size([32, 4, 64])只有true或者false


        neighborhood_all_masked = []
        center_all_masked = []
        x_global_gt = []
        for i in range(view_points.size(1)):
            bool_masked_pos = bool_masked_pos_all[:, i, :]
            neighborhood_masked_temp = neighborhood[bool_masked_pos].reshape(neighborhood.size(0), -1, neighborhood.size(2),neighborhood.size(3))
            center_masked_temp = center[bool_masked_pos].reshape(center.size(0), -1, center.size(2))

            x_global_temp = x_global[:,1:,:][bool_masked_pos].reshape(x_global.size(0), -1, x_global.size(2))

            neighborhood_all_masked.append(neighborhood_masked_temp.unsqueeze(1))
            center_all_masked.append(center_masked_temp.unsqueeze(1))

            x_global_gt.append(x_global_temp.unsqueeze(1))


        neighborhood_all_masked = torch.cat(neighborhood_all_masked, dim=1) # torch.Size([32, 4, 54, 32, 3])
        center_all_masked = torch.cat(center_all_masked, dim=1)             # torch.Size([32, 4, 54, 3])
        x_global_gt = torch.cat(x_global_gt, dim=1)

        loss1 = []
        loss2 = []
        x_partial_all = []

        ret_partial_all = []
        ret_partial_all2 = []
        for i in range(view_points.size(1)):
            points_temp = neighborhood_all_masked[:,i,:,:,:]
            center_temp = center_all_masked[:,i,:,:]
            group_input_tokens = self.encoder(points_temp)  # B G N

            cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
            cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

            pos = self.pos_embed(center_temp)

            x = torch.cat((cls_tokens, group_input_tokens), dim=1)
            pos = torch.cat((cls_pos, pos), dim=1)
            # transformer
            x = self.blocks(x, pos)
            x_partial = self.norm(x)  # torch.Size([32, 65, 384])
            concat_f = torch.cat([x_partial[:, 0], x_partial[:, 1:].max(1)[0]], dim=-1)
            ret_partial = self.cls_head_finetune(concat_f)

            x_partial_all.append(x_partial[:,1:,:].unsqueeze(1))
            ret_partial_all.append(ret_partial.unsqueeze(1))
            loss1.append(self.loss_func(x_partial[:,1:,:], x_global_gt[:,i,:,:]))

            # 计算每个缺失点云中每个patch距离它最近的k个patch
            bool_masked_pos_all, center_id = self._mask_center_all_views(center_temp, view_points=center_temp, k=10)       # torch.Size([32, 34, 34])

            x_local = []
            for j in range(center_temp.size(1)):
                bool_masked_pos = bool_masked_pos_all[:, j, :]
                neighborhood_masked_temp = points_temp[~bool_masked_pos].reshape(points_temp.size(0), -1,points_temp.size(2),points_temp.size(3))
                center_masked_temp = center_temp[~bool_masked_pos].reshape(center_temp.size(0), -1, center_temp.size(2))

                group_input_tokens_temp = self.encoder(neighborhood_masked_temp)  # B G N

                cls_tokens_temp = self.cls_token.expand(group_input_tokens_temp.size(0), -1, -1)
                cls_pos_temp = self.cls_pos.expand(group_input_tokens_temp.size(0), -1, -1)

                pos_temp = self.pos_embed(center_masked_temp)

                x_temp = torch.cat((cls_tokens_temp, group_input_tokens_temp), dim=1)
                pos_temp = torch.cat((cls_pos_temp, pos_temp), dim=1)
                # transformer
                x_temp = self.blocks(x_temp, pos_temp)
                x_partial_temp = self.norm(x_temp)
                x_local.append(x_partial_temp[:,1,:].unsqueeze(1))

            x_local = torch.cat(x_local, dim=1)

            loss2.append(self.loss_func(x_local, x_global_gt[:, i, :, :]))

            concat_f2 = torch.cat([x_partial[:, 0], x_local.max(1)[0]], dim=-1)
            ret_partial2 = self.cls_head_finetune(concat_f2)
            ret_partial_all2.append(ret_partial2.unsqueeze(1))


        ret_partial_all = torch.cat(ret_partial_all, dim=1)
        ret_partial_all = torch.cat([ret_global.unsqueeze(1),ret_partial_all],dim=1)

        ret_partial_all2 = torch.cat(ret_partial_all2, dim=1)
        ret_partial_all2 = torch.cat([ret_global.unsqueeze(1), ret_partial_all2], dim=1)





        return ret_partial_all, ret_partial_all2, loss1, loss2

