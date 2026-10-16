import torch
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS, Hook

__all__ = ['GateMonitorHook']


@HOOKS.register_module()
class GateMonitorHook(Hook):
    """门控/融合诊断 hook：每隔 N iter 打印 occ 融合门控统计。

    读取模型在 forward_train 计算 occ 融合时暂存的 ``_occ_gate_stat``：

    - ``alpha_mean`` / ``alpha_std``：最终 `alpha = sigmoid(2D门控 logit + occ_fuse_logit)`
      在整个 (Dx, Dy, Dz, C) 占用体积上的均值/标准差。注意 **alpha 是“主OCC（图像+融合特征）
      权重”**，激光OCC 权重为 (1-alpha)：alpha 低 → 激光OCC 主导（被重用），alpha 高 → 主OCC
      主导（激光OCC 融合层参与少）。alpha 空间分布有 std 说明按位置自适应生效。
    - ``spatial_gate_logit_mean/std``：2D 门控网络预测值（sigmoid/偏置之前）的统计，
      std 显著 > 0 说明网络学到了位置差异（近场信激光、远场信图像）。
    - ``occ_fuse_logit``：全局先验偏置参数当前值（初值 = log(occ_fusion_weight/(1-·))，
      默认对数 0.7/0.3 ≈ 0.847 → 初始 alpha≈0.7）。它下降 → alpha 被推向 0（激光OCC 主导）。
    - ``pts_loss_occ``：激光 OCC 分支加权（0.5）aux CE 损失，>0 即激光分支在参与训练。

    用法（config custom_hooks 追加）:
        dict(type='GateMonitorHook', after_iter=50, priority='HIGH')
    """

    def __init__(self, after_iter=50):
        super().__init__()
        self.after_iter = int(after_iter)

    @master_only
    def after_train_iter(self, runner):
        if runner.iter == 0 or runner.iter % self.after_iter != 0:
            return
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        stat = getattr(model, '_occ_gate_stat', None)
        if stat is None:
            return
        alpha_mean = stat.get('alpha_mean', float('nan'))
        alpha_std = stat.get('alpha_std', float('nan'))
        gate_mean = stat.get('spatial_gate_logit_mean', float('nan'))
        gate_std = stat.get('spatial_gate_logit_std', float('nan'))
        fuse_logit = stat.get('occ_fuse_logit', float('nan'))
        pts_loss_occ = stat.get('pts_loss_occ', float('nan'))
        runner.logger.info(
            f'[GateMonitor] iter={runner.iter} '
            f'alpha_mean={alpha_mean:.4f} alpha_std={alpha_std:.4f} '
            f'spatial_gate_logit_mean={gate_mean:.4f} std={gate_std:.4f} '
            f'occ_fuse_logit={fuse_logit:.4f} pts_loss_occ={pts_loss_occ:.4f} | '
            f'(alpha=主OCC权重,低→激光OCC主导; pts_loss_occ>0→激光分支在训练; '
            f'gate_logit_std>0→空间自适应生效)')