"""
planning heads 模块
===================

本文件实现轻量 planning 子模块：
1. ContextPooler：从 HistEncoder context 提取意图语义摘要。
2. IntentHead：基于 [x,y,dx,dy] 特征 + context 预测横/纵向意图，生成单一 intent_token。
3. BridgeHead：基于 hist + context 预测短时 bridge 轨迹并输出 bridge tokens（无需 intent 引导）。
4. FutureReconHead：Future 重建分支，仅训练时使用，为 BridgeHead 提供对齐目标。
5. AlignmentProjector：对齐投影层，将 BridgeHead 隐层投影到可比空间。

最终 cross 条件：
    cross_tokens = [context ; intent_token ; bridge_tokens]

TimeAlign 对齐机制（参考 ICLR 2026）：
- FutureReconHead 从 GT future 学习分布表示
- 通过 Local + Global alignment 将 BridgeHead 的表示拉向 future 分布
- 推理时只使用 BridgeHead，FutureReconHead 零开销
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def _build_tail_tensor(sequence, tail_k):
    """统一抽取最后 K 帧；当历史长度不足时在前侧补零，保证输出形状稳定为 [B, K, D]。"""
    batch_size, time_steps, feat_dim = sequence.shape
    if time_steps >= tail_k:
        return sequence[:, -tail_k:, :]

    pad_len = tail_k - time_steps
    pad = sequence.new_zeros(batch_size, pad_len, feat_dim)
    return torch.cat([pad, sequence], dim=1)


def _build_motion_features(hist_xy, tail_k):
    """构造运动特征 [x, y, dx, dy]。

    从历史轨迹中提取最后 K 帧的位置和一阶差分特征，用于 IntentHead 和 BridgeHead。

    Args:
        hist_xy: [B, T_hist, 2] - 历史轨迹 xy 坐标
        tail_k: int - 提取的尾部帧数

    Returns:
        motion_feat: [B, 4*K] - 展平的运动特征
    """
    # 取最后 K 帧
    hist_tail = _build_tail_tensor(hist_xy, tail_k)  # [B, K, 2]

    # 原始位置
    x = hist_tail[..., 0]  # [B, K]
    y = hist_tail[..., 1]  # [B, K]

    # 计算一阶差分
    dx = torch.zeros_like(x)
    dy = torch.zeros_like(y)
    dx[:, 1:] = x[:, 1:] - x[:, :-1]  # [B, K]，首帧差分为 0
    dy[:, 1:] = y[:, 1:] - y[:, :-1]  # [B, K]

    # 组合特征 [x, y, dx, dy] 并展平
    motion_feat = torch.stack([x, y, dx, dy], dim=-1)  # [B, K, 4]
    motion_feat = motion_feat.reshape(motion_feat.size(0), -1)  # [B, 4*K]

    return motion_feat


class ContextPooler(nn.Module):
    """从 HistEncoder 的序列上下文中提取意图相关的语义摘要。

    使用可学习的 query 向量做 attention pooling，提取与意图预测相关的上下文信息。
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        # 单一意图 query，用于提取融合的意图上下文
        self.q_intent = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)

    def _attn_pool(self, context, query):
        """Attention pooling: 使用 query 从 context 中提取相关信息。"""
        # context: [B, T_ctx, H], query: [1, 1, H]
        batch_size = context.size(0)
        query_expanded = query.expand(batch_size, -1, -1)
        attn_logits = torch.matmul(query_expanded, context.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        attn_weights = F.softmax(attn_logits, dim=-1)
        pooled = torch.matmul(attn_weights, context).squeeze(1)  # [B, H]
        return pooled

    def forward(self, context):
        """
        Args:
            context: [B, T_ctx, H] - HistEncoder 输出的上下文序列

        Returns:
            intent_ctx: [B, H] - 意图相关的上下文摘要
            bridge_ctx: [B, H] - bridge 分支使用的全局上下文（均值池化）
        """
        intent_ctx = self._attn_pool(context, self.q_intent)
        bridge_ctx = context.mean(dim=1)
        return intent_ctx, bridge_ctx


class IntentHead(nn.Module):
    """意图预测头：基于 [x,y,dx,dy] + context 预测横纵向意图，生成单一 intent_token。

    架构设计（方案 A）：
    1. 特征提取：从 hist_xy 构造 [x, y, dx, dy] 特征
    2. Context 融合：使用 attention pooling 从 context 提取意图相关信息
    3. 分别预测：独立的 lat_head 和 lon_head 进行 3 分类
    4. Token 生成：融合特征 + 软意图 embedding 生成单一 intent_token

    优势：
    - 训练稳定：3 类分类比 9 类更容易收敛
    - 数据高效：不受联合分布长尾问题影响
    - 鲁棒性强：单维度错误不会完全误导 DiT
    """
    def __init__(self, hidden_dim, tail_k, pooler):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.tail_k = int(tail_k)
        self.pooler = pooler

        # 运动特征编码器：输入 [x, y, dx, dy] * tail_k
        motion_feat_dim = 4 * self.tail_k
        self.motion_encoder = nn.Sequential(
            nn.Linear(motion_feat_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # 特征融合：motion + context
        self.fuse_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # 横向意图分类头
        self.lat_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 3),
        )

        # 纵向意图分类头
        self.lon_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 3),
        )

        # 意图 Embedding 表：用于生成软意图向量
        self.lat_embedding = nn.Parameter(torch.randn(3, self.hidden_dim) * 0.02)
        self.lon_embedding = nn.Parameter(torch.randn(3, self.hidden_dim) * 0.02)

        # Token 生成：融合 [h_fused, e_lat, e_lon] 生成单一 intent_token
        self.token_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )

    def forward(self, context, hist_xy):
        """
        Args:
            context: [B, T_ctx, H] - HistEncoder 输出的上下文序列
            hist_xy: [B, T_hist, 2] - 历史轨迹 xy 坐标

        Returns:
            intent_token: [B, 1, H] - 单一意图 token
            aux: dict - 包含 logits_lat, logits_lon, p_lat, p_lon 等辅助信息
        """
        # 1. 特征提取（使用模块级函数）
        motion_feat = _build_motion_features(hist_xy, self.tail_k)  # [B, 4*K]
        h_motion = self.motion_encoder(motion_feat)  # [B, H]

        # 2. Context 融合
        h_ctx, _ = self.pooler(context)  # [B, H]
        h_fused = self.fuse_mlp(torch.cat([h_motion, h_ctx], dim=-1))  # [B, H]

        # 3. 分别预测横纵向意图
        logits_lat = self.lat_head(h_fused)  # [B, 3]
        logits_lon = self.lon_head(h_fused)  # [B, 3]

        # 4. 计算软概率
        p_lat = F.softmax(logits_lat, dim=-1)  # [B, 3]
        p_lon = F.softmax(logits_lon, dim=-1)  # [B, 3]

        # 5. 生成软意图向量
        e_lat = torch.matmul(p_lat, self.lat_embedding)  # [B, H]
        e_lon = torch.matmul(p_lon, self.lon_embedding)  # [B, H]

        # 6. 融合生成单一 intent_token
        token_input = torch.cat([h_fused, e_lat, e_lon], dim=-1)  # [B, 3H]
        intent_token = self.token_mlp(token_input).unsqueeze(1)  # [B, 1, H]

        aux = {
            "logits_lat": logits_lat,
            "logits_lon": logits_lon,
            "p_lat": p_lat,
            "p_lon": p_lon,
            "h_motion": h_motion,
            "h_ctx": h_ctx,
            "h_fused": h_fused,
            "e_lat": e_lat,
            "e_lon": e_lon,
        }
        return intent_token, aux


class BridgeHead(nn.Module):
    """Bridge 预测头：基于 hist + context 预测短时轨迹并输出 bridge tokens。

    使用历史运动特征和上下文信息，通过 cross-attention 预测短时 bridge 轨迹。
    注意：不使用 intent 引导，仅依赖 hist 和 context。
    """
    def __init__(self, hidden_dim, tail_k, bridge_tau, pooler):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.tail_k = int(tail_k)
        self.bridge_tau = int(bridge_tau)
        self.pooler = pooler

        # 运动特征编码器
        motion_feat_dim = 4 * self.tail_k
        self.motion_encoder = nn.Sequential(
            nn.Linear(motion_feat_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Step query：可学习的位置 embedding
        self.step_embedding = nn.Parameter(torch.randn(1, self.bridge_tau, self.hidden_dim) * 0.02)

        # Query 生成：融合 step + motion（移除 intent）
        self.query_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Cross-attention：从 context 中提取信息
        num_heads = 4 if (self.hidden_dim % 4 == 0) else 1
        self.step_cross_attn = nn.MultiheadAttention(self.hidden_dim, num_heads, batch_first=True)

        # GRU 解码器
        self.tiny_gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )

        # 输出投影
        self.bridge_vel_proj = nn.Linear(self.hidden_dim, 2)
        self.bridge_token_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, context, hist_xy):
        """
        Args:
            context: [B, T_ctx, H] - HistEncoder 输出的上下文序列
            hist_xy: [B, T_hist, 2] - 历史轨迹 xy 坐标

        Returns:
            bridge_tokens: [B, tau, H] - Bridge tokens
            aux: dict - 包含 bridge_vel, bridge_pos 等辅助信息
        """
        batch_size = context.size(0)

        # 1. 运动特征编码（使用模块级函数）
        motion_feat = _build_motion_features(hist_xy, self.tail_k)  # [B, 4*K]
        h_motion = self.motion_encoder(motion_feat)  # [B, H]

        # 2. 构建 step query（仅使用 step + motion，不使用 intent）
        step_tokens = self.step_embedding.expand(batch_size, -1, -1)  # [B, tau, H]
        h_motion_step = h_motion.unsqueeze(1).expand(-1, self.bridge_tau, -1)  # [B, tau, H]
        step_query = self.query_mlp(torch.cat([step_tokens, h_motion_step], dim=-1))  # [B, tau, H]

        # 3. Cross-attention 从 context 提取信息
        bridge_cond, _ = self.step_cross_attn(step_query, context, context, need_weights=False)  # [B, tau, H]

        # 4. GRU 解码
        bridge_hidden, _ = self.tiny_gru(bridge_cond)  # [B, tau, H]

        # 5. 输出：速度、位置、tokens
        bridge_vel = self.bridge_vel_proj(bridge_hidden)  # [B, tau, 2]
        bridge_pos = torch.cumsum(bridge_vel, dim=1)  # [B, tau, 2]，相对锚点坐标系
        bridge_tokens = self.bridge_token_proj(bridge_hidden)  # [B, tau, H]

        aux = {
            "bridge_vel": bridge_vel,
            "bridge_pos": bridge_pos,
            "bridge_hidden": bridge_hidden,
            "h_motion": h_motion,
            "step_query": step_query,
        }
        return bridge_tokens, aux


class FutureReconHead(nn.Module):
    """Future 重建分支：从 GT future 学习分布对齐的目标表示。

    借鉴 TimeAlign (ICLR 2026) 的设计思想：
    - 仅在训练时使用，为 BridgeHead 提供对齐目标
    - 通过重建任务学习 future 的内在分布表示
    - 推理时不参与，零额外开销

    架构设计：
    1. 将 future_xy 转换为 [x, y, dx, dy] 特征
    2. 通过 Encoder + GRU 提取时序隐层表示
    3. 通过 Decoder 重建原始 future
    """

    def __init__(self, hidden_dim, bridge_tau):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.bridge_tau = int(bridge_tau)

        # Future 编码器：将 [x, y, dx, dy] 编码为隐层表示
        self.future_encoder = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # 时序建模：捕获 future 的时序结构
        self.temporal_encoder = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )

        # 重建解码器：从隐层重建位移
        self.recon_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 2),  # 输出 [dx, dy]
        )

    def _build_future_features(self, future_xy):
        """构造 future 的运动特征 [x, y, dx, dy]。
        Args:
            future_xy: [B, T, 2] - future 轨迹 xy 坐标
        Returns:
            future_feat: [B, T, 4] - 运动特征
        """
        x = future_xy[..., 0]  # [B, T]
        y = future_xy[..., 1]  # [B, T]

        # 计算一阶差分
        dx = torch.zeros_like(x)
        dy = torch.zeros_like(y)
        dx[:, 1:] = x[:, 1:] - x[:, :-1]
        dy[:, 1:] = y[:, 1:] - y[:, :-1]

        # 组合特征 [x, y, dx, dy]
        future_feat = torch.stack([x, y, dx, dy], dim=-1)  # [B, T, 4]
        return future_feat

    def forward(self, future_xy):
        """
        Args:
            future_xy: [B, T_f, 2] - GT future 轨迹
        Returns:
            future_hidden: [B, τ, H] - future 的隐层表示（对齐目标）
            future_recon: [B, τ, 2] - 重建的 future 位移
            aux: dict - 辅助信息
        """
        B, T, _ = future_xy.shape
        tau = min(T, self.bridge_tau)

        future_tau = future_xy[:, :tau, :]  # [B, τ, 2]
        future_feat = self._build_future_features(future_tau)  # [B, τ, 4]
        h_enc = self.future_encoder(future_feat)  # [B, τ, H]
        future_hidden, _ = self.temporal_encoder(h_enc)  # [B, τ, H]
        future_recon = self.recon_decoder(future_hidden)  # [B, τ, 2]

        aux = {
            "future_feat": future_feat,
            "h_enc": h_enc,
        }
        return future_hidden, future_recon, aux


class AlignmentProjector(nn.Module):
    """对齐投影层：将 BridgeHead 的隐层投影到与 FutureReconHead 可比的空间。
    设计参考 TimeAlign 的非对称梯度流：
    - 梯度只回传到 BridgeHead（predict branch）
    - FutureReconHead 的梯度在对齐损失计算时被 stop_grad 阻断
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, bridge_hidden):
        return self.proj(bridge_hidden)


def compute_local_alignment_loss(h_x, h_y, margin=0.1):
    """计算局部对齐损失：逐 token 余弦相似度。
    目标：确保 BridgeHead 的每个时间步表示与 FutureReconHead 对应时间步对齐。
    Args:
        h_x: [B, τ, H] - BridgeHead 的投影隐层
        h_y: [B, τ, H] - FutureReconHead 的隐层（会被 detach）
        margin: float - 松弛边界
    Returns:
        L_local: 局部对齐损失
    """
    # 归一化
    h_x_norm = F.normalize(h_x, dim=-1)
    h_y_norm = F.normalize(h_y.detach(), dim=-1)  # stop_grad

    # 逐 token 余弦相似度
    cos_sim = (h_x_norm * h_y_norm).sum(dim=-1)  # [B, τ]

    # 使用 ReLU-based margin loss 代替 GELU，确保非负
    # 当 cos_sim 接近 1 时损失接近 0，当 cos_sim 较小时损失增大
    loss = F.relu(1 - cos_sim - margin).mean()

    return loss


def compute_global_alignment_loss(h_x, h_y, margin=0.1):
    """计算全局对齐损失：对齐 batch 内的相对距离结构。
    目标：确保 BridgeHead 和 FutureReconHead 在 batch 内的样本关系结构一致。
    Args:
        h_x: [B, τ, H] - BridgeHead 的投影隐层
        h_y: [B, τ, H] - FutureReconHead 的隐层（会被 detach）
        margin: float - 松弛边界
    Returns:
        L_global: 全局对齐损失
    """
    B, T, H = h_x.shape

    # 展平为 [B, τ*H]
    h_x_flat = h_x.reshape(B, -1)
    h_y_flat = h_y.detach().reshape(B, -1)

    # 计算 pairwise 距离矩阵
    # 使用 L2 距离
    dist_x = torch.cdist(h_x_flat, h_x_flat, p=2)  # [B, B]
    dist_y = torch.cdist(h_y_flat, h_y_flat, p=2)  # [B, B]

    # 归一化距离矩阵（避免尺度差异）
    dist_x_norm = dist_x / (dist_x.max() + 1e-6)
    dist_y_norm = dist_y / (dist_y.max() + 1e-6)

    # 距离矩阵差异
    diff = torch.abs(dist_x_norm - dist_y_norm)

    # 使用 ReLU-based margin loss 代替 GELU，确保非负
    loss = F.relu(diff - margin).mean()

    return loss


def compute_alignment_loss(h_x, h_y, margin_local=0.1, margin_global=0.1):
    """计算完整的对齐损失：Local + Global with dynamic weighting。
    Args:
        h_x: [B, τ, H] - BridgeHead 的投影隐层
        h_y: [B, τ, H] - FutureReconHead 的隐层
        margin_local: float - 局部对齐的松弛边界
        margin_global: float - 全局对齐的松弛边界
    Returns:
        loss_align: 总对齐损失
        aux: dict - 包含 L_local, L_global, alpha, beta
    """
    L_local = compute_local_alignment_loss(h_x, h_y, margin_local)
    L_global = compute_global_alignment_loss(h_x, h_y, margin_global)

    # 动态权重平衡 (TimeAlign Eq.7)
    # 添加最小值保护，避免除零和异常大的权重
    eps = 1e-4
    L_local_safe = torch.clamp(L_local, min=eps)
    L_global_safe = torch.clamp(L_global, min=eps)
    total = L_local_safe + L_global_safe

    alpha = total / L_local_safe
    beta = total / L_global_safe

    # 限制权重范围，避免某一项过度主导
    alpha = torch.clamp(alpha, min=0.5, max=2.0)
    beta = torch.clamp(beta, min=0.5, max=2.0)

    # 加权求和
    loss_align = alpha * L_local + beta * L_global

    aux = {
        "L_local": L_local,
        "L_global": L_global,
        "alpha": alpha,
        "beta": beta,
    }
    return loss_align, aux
