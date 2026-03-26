"""
planning heads 模块说明
======================

这部分改动的目标，是在 future diffusion 模型侧建立更强的 hist -> fut 显式连接，
让未来预测不只依赖 HistEncoder 输出的原始 context 序列，还额外依赖：

    [hist_context ; intent_token ; motion_token ; bridge_tokens]

最终这些 token 会一起送入 future DiT 的 cross-attention，作为 planning-aware 条件。

整体结构
--------
本文件实现了 4 个轻量 planning 子模块：

1. ContextPooler
   - 输入：HistEncoder 输出的 context，形状为 [B, T_ctx, H]
   - 作用：从同一份 history context 中抽取三类摘要
     - bridge_ctx：对全部 context 做全局平均池化，得到短时桥接分支的全局上下文
     - z_lat：用一个可学习查询向量做 attention pooling，提取横向意图相关信息
     - z_lon：用另一个可学习查询向量做 attention pooling，提取纵向意图相关信息
   - 原理：不改 HistEncoder 主体，只在输出序列上做轻量摘要，让不同 planning 分支共享同一份历史表征。

2. IntentHead
   - 输入：
     - context: [B, T_ctx, H]
     - hist_feat: [B, T_hist, 4]，当前 future 主路径固定使用 xyva 四维历史特征
   - 作用：
     - 预测横向意图 logits_lat
     - 预测纵向意图 logits_lon
     - 生成一个可送入 DiT cross-attn 的 intent_token
   - 意图获取方式：
     - 先用 ContextPooler 得到 z_lat、z_lon
     - 再从历史最后 K 帧中截取 tail state，并用 MLP 压缩成 hist_tail_embed
     - z_lat + hist_tail_embed -> 横向分类头
     - z_lon + hist_tail_embed -> 纵向分类头
   - token 构造方式：
     - 不是直接取 argmax，而是对 logits 做 softmax 得到概率 p_lat / p_lon
     - 再与可学习的意图 embedding 表相乘，得到软意图向量 e_lat / e_lon
     - 最后将 [z_lat, z_lon, e_lat, e_lon, hist_tail_embed] 拼接，经 MLP 生成 intent_token
   - 原理：
     - z_lat / z_lon 提供全局历史语义
     - hist_tail_embed 提供最近运动状态
     - 软意图 embedding 让 token 连续可导，便于与 diffusion 主分支联合训练

3. BridgeHead
   - 输入：
     - context: [B, T_ctx, H]
     - hist_feat: [B, T_hist, 4]
     - anchor_pos: [B, 1, 2]，历史最后一帧位置
   - 作用：
     - 预测未来前 tau 帧的短时 bridge rollout
     - 同时输出 bridge_tokens，作为 future DiT 的额外 cross 条件
   - 瞬时轨迹获取方式：
     - 用 ContextPooler 得到 bridge_ctx
     - 从历史最后 K 帧提取 hist tail，并编码为 hist_tail_embed
     - 将 bridge_ctx 与 hist_tail_embed 融合后，作为小型 GRU decoder 的初始状态
     - GRU 每一步输出一个 hidden state
     - hidden state -> bridge_vel
     - 通过 cumsum(bridge_vel) + anchor_pos 得到 bridge_pos
   - token 获取方式：
     - 将每个 decoder hidden state 线性投影到 hidden_dim，得到 bridge_tokens
   - 原理：
     - bridge 分支只负责短时连接，而不是完整 future 预测器
     - 它为 diffusion 主分支提供“从历史尾部如何平滑过渡到未来开头”的显式几何先验

4. MotionSummaryHead
   - 输入：
     - bridge_tokens: [B, tau, H]
     - bridge_aux：包含 bridge_pos、bridge_vel 等信息
   - 作用：将短时 bridge rollout 压缩成单个 motion_token
   - 聚合方式：
     - 取最后一个 bridge token
     - 取最后一帧 bridge_pos
     - 取 bridge_vel 的均值
     - 取 bridge_vel 的最后一帧
     - 拼接后经 MLP 映射成 motion_token
   - 原理：
     - bridge_tokens 保留逐步短时动态
     - motion_token 提供一个更紧凑的“短时运动摘要”，补充给 DiT cross-attn

最终 cross 条件
---------------
在 fut model 中，这些模块的输出会按如下顺序拼接：

    cross_tokens = [context ; intent_token ; motion_token ; bridge_tokens]

其中：
- context 提供完整历史上下文
- intent_token 提供长时行为/规划语义
- motion_token 提供短时运动摘要
- bridge_tokens 提供细粒度短时连接轨迹信息

这样做的核心思想是：
- 不改 DiT 主干和 self-attention
- 不改 HistEncoder 稀疏交互
- 只通过额外 planning tokens 增强 cross-attention 条件
- 用轻量模块把历史中的“意图”和“短时过渡结构”显式暴露给 future diffusion

---------------------------
- 当前默认维度下：
      - context: [B, 16, 128]
      - intent_token: [B, 1, 128]
      - motion_token: [B, 1, 128]
      - bridge_tokens: [B, 5, 128]
      - cross_tokens: [B, 23, 128]
  - 这套结构已经能向 future DiT 提供：
      - 完整历史记忆
      - 高层意图
      - 短时运动摘要
      - 短时桥接细节
  - 后续提高 cross-attn 的信息质量：
      - 更细粒度的 token 语义拆分
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def _build_tail_tensor(sequence, tail_k):
    # 统一抽取最后 K 帧；当历史长度不足时在前侧补零，保证输出形状稳定为 [B, K, D]。
    batch_size, time_steps, feat_dim = sequence.shape
    if time_steps >= tail_k:
        return sequence[:, -tail_k:, :]

    pad_len = tail_k - time_steps
    pad = sequence.new_zeros(batch_size, pad_len, feat_dim)
    return torch.cat([pad, sequence], dim=1)


class ContextPooler(nn.Module):
    # 从 HistEncoder 的序列上下文中提取 bridge / lateral intent / longitudinal intent 摘要。
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.q_lat = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)
        self.q_lon = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)

    def _attn_pool(self, context, query):
        # context: [B, T_ctx, H], query: [1, 1, H]
        batch_size = context.size(0)
        query_expanded = query.expand(batch_size, -1, -1)
        attn_logits = torch.matmul(query_expanded, context.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        attn_weights = F.softmax(attn_logits, dim=-1)
        pooled = torch.matmul(attn_weights, context).squeeze(1)  # [B, H]
        return pooled

    def forward(self, context):
        # bridge_ctx: [B, H], intent_ctx_lat/lon: [B, H]
        bridge_ctx = context.mean(dim=1)
        intent_ctx_lat = self._attn_pool(context, self.q_lat)
        intent_ctx_lon = self._attn_pool(context, self.q_lon)
        return intent_ctx_lat, intent_ctx_lon, bridge_ctx


class IntentHead(nn.Module):
    # 基于 context 摘要与历史尾部状态预测横向/纵向意图，并生成 planning token。
    def __init__(self, hidden_dim, tail_k, pooler):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.tail_k = int(tail_k)
        self.pooler = pooler

        tail_input_dim = self.tail_k * 4  # 轻量实现：默认使用 hist_xy + va 共 4 维。
        self.hist_tail_mlp = nn.Sequential(
            nn.Linear(tail_input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.lat_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 3),
        )
        self.lon_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 3),
        )
        self.intent_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 5, self.hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )

        self.lat_embedding = nn.Parameter(torch.randn(3, self.hidden_dim) * 0.02)
        self.lon_embedding = nn.Parameter(torch.randn(3, self.hidden_dim) * 0.02)

    def forward(self, context, hist_feat):
        # context: [B, T_ctx, H], hist_feat: [B, T_hist, 4]
        z_lat, z_lon, _ = self.pooler(context)

        hist_tail = _build_tail_tensor(hist_feat, self.tail_k)  # [B, K, 4]
        hist_tail_flat = hist_tail.reshape(hist_tail.size(0), -1)  # [B, K * 4]
        hist_tail_embed = self.hist_tail_mlp(hist_tail_flat)  # [B, H]

        logits_lat = self.lat_head(torch.cat([z_lat, hist_tail_embed], dim=-1))  # [B, 3]
        logits_lon = self.lon_head(torch.cat([z_lon, hist_tail_embed], dim=-1))  # [B, 3]

        p_lat = F.softmax(logits_lat, dim=-1)  # [B, 3]
        p_lon = F.softmax(logits_lon, dim=-1)  # [B, 3]
        e_lat = torch.matmul(p_lat, self.lat_embedding)  # [B, H]
        e_lon = torch.matmul(p_lon, self.lon_embedding)  # [B, H]

        intent_token = self.intent_mlp(
            torch.cat([z_lat, z_lon, e_lat, e_lon, hist_tail_embed], dim=-1)
        ).unsqueeze(1)  # [B, 1, H]

        aux = {
            "logits_lat": logits_lat,
            "logits_lon": logits_lon,
            "p_lat": p_lat,
            "p_lon": p_lon,
            "hist_tail_embed": hist_tail_embed,
            "z_lat": z_lat,
            "z_lon": z_lon,
        }
        return intent_token, aux


class BridgeHead(nn.Module):
    # 仅预测短时 bridge rollout，并同步输出可送入 DiT cross-attn 的 bridge tokens。
    def __init__(self, hidden_dim, tail_k, bridge_tau, pooler):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.tail_k = int(tail_k)
        self.bridge_tau = int(bridge_tau)
        self.pooler = pooler

        tail_input_dim = self.tail_k * 4  # 轻量实现：默认使用 hist_xy + va 共 4 维。
        self.hist_tail_mlp = nn.Sequential(
            nn.Linear(tail_input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.bridge_ctx_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.fuse_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.decoder = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.bridge_vel_proj = nn.Linear(self.hidden_dim, 2)
        self.bridge_token_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, context, hist_feat, anchor_pos):
        # context: [B, T_ctx, H], hist_feat: [B, T_hist, 4], anchor_pos: [B, 1, 2]
        _, _, bridge_ctx = self.pooler(context)
        hist_tail = _build_tail_tensor(hist_feat, self.tail_k)  # [B, K, 4]
        hist_tail_flat = hist_tail.reshape(hist_tail.size(0), -1)  # [B, K * 4]
        hist_tail_embed = self.hist_tail_mlp(hist_tail_flat)  # [B, H]
        bridge_ctx_embed = self.bridge_ctx_mlp(bridge_ctx)  # [B, H]
        decoder_init = self.fuse_mlp(torch.cat([bridge_ctx_embed, hist_tail_embed], dim=-1))  # [B, H]

        decoder_inputs = context.new_zeros(context.size(0), self.bridge_tau, self.hidden_dim)  # [B, tau, H]
        bridge_hidden, _ = self.decoder(decoder_inputs, decoder_init.unsqueeze(0))  # [B, tau, H]

        bridge_vel = self.bridge_vel_proj(bridge_hidden)  # [B, tau, 2]
        bridge_pos = torch.cumsum(bridge_vel, dim=1) + anchor_pos  # [B, tau, 2]
        bridge_tokens = self.bridge_token_proj(bridge_hidden)  # [B, tau, H]

        aux = {
            "bridge_vel": bridge_vel,
            "bridge_pos": bridge_pos,
            "bridge_hidden": bridge_hidden,
            "anchor_pos": anchor_pos,
            "bridge_ctx": bridge_ctx,
            "hist_tail_embed": hist_tail_embed,
        }
        return bridge_tokens, aux


class MotionSummaryHead(nn.Module):
    # 从短时 bridge rollout 中提炼一个运动摘要 token。
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.motion_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + 6, self.hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )

    def forward(self, bridge_tokens, bridge_aux):
        # bridge_tokens: [B, tau, H]
        bridge_pos = bridge_aux["bridge_pos"]  # [B, tau, 2]
        bridge_vel = bridge_aux["bridge_vel"]  # [B, tau, 2]

        last_bridge_token = bridge_tokens[:, -1, :]  # [B, H]
        last_bridge_pos = bridge_pos[:, -1, :]  # [B, 2]
        mean_bridge_vel = bridge_vel.mean(dim=1)  # [B, 2]
        last_bridge_vel = bridge_vel[:, -1, :]  # [B, 2]

        motion_token = self.motion_mlp(
            torch.cat([last_bridge_token, last_bridge_pos, mean_bridge_vel, last_bridge_vel], dim=-1)
        ).unsqueeze(1)  # [B, 1, H]
        return motion_token
