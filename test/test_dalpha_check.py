"""
在每个kernel后检查dalpha，定位何时被修改
"""

import torch
import sys
sys.path.insert(0, '.')

from src.forward.golden import mhc_forward_pre
from src.backward.golden import mhc_backward_manual

# 临时修改mhc_backward_triton函数，添加调试输出
import src.backward.mhc_backward_triton as backward_module

def mhc_backward_triton_debug(*args, **kwargs):
    """包装函数，在每个kernel后打印dalpha"""
    # 调用原函数但不返回，直接修改
    result = backward_module.mhc_backward_triton(*args, **kwargs)
    return result

B, S, n, D = 2, 64, 4, 128
device = 'cuda'

torch.manual_seed(42)
x = torch.randn(B, S, n, D, dtype=torch.bfloat16, device=device)
phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32, device=device)
alpha = torch.tensor([1.1, 0.9, 1.05], dtype=torch.float32, device=device)
bias = torch.randn(n*n + 2*n, dtype=torch.float32, device=device) * 0.1
gamma = torch.randn(n, D, dtype=torch.float32, device=device)

with torch.no_grad():
    h_in, h_post, h_res, inv_rms, h_mix, h_pre = mhc_forward_pre(
        x, phi, alpha, bias, outflag=True
    )

dh_in = torch.randn(B, S, D, dtype=torch.bfloat16, device=device)
dh_post = torch.randn(B, S, n, dtype=torch.float32, device=device)
dh_res = torch.randn(B, S, n, n, dtype=torch.float32, device=device)

# Golden
dx_gold, dphi_gold, dalpha_gold, dbias_gold, dgamma_gold = mhc_backward_manual(
    x, phi, alpha, bias,
    inv_rms, h_mix, h_pre, h_post,
    dh_in, dh_post, dh_res, gamma
)

print(f"Golden dalpha[0]: {dalpha_gold[0]:.8f}")
print(f"Triton dalpha[0]: {mhc_backward_triton_debug(x, phi, alpha, bias, inv_rms, h_mix, h_pre, h_post, dh_in, dh_post, dh_res, gamma)[2][0]:.8f}")
print(f"差异: {abs(mhc_backward_triton_debug(...)[2][0] - dalpha_gold[0]):.8f}")
