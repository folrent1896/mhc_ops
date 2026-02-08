#!/bin/bash
# GPU 修复脚本 - 解决 NVIDIA RTX 3070 无法检测的问题

echo "========================================"
echo "修复 NVIDIA GPU 配置"
echo "========================================"
echo ""

# 1. 添加用户到必要的组
echo "[1/4] 将用户添加到 video 和 render 组..."
sudo usermod -aG video $USER
sudo usermod -aG render $USER
echo "✓ 用户组已更新"
echo ""

# 2. 加载 NVIDIA 内核模块
echo "[2/4] 加载 NVIDIA 内核模块..."
sudo modprobe nvidia
sudo modprobe nvidia-modeset
sudo modprobe nvidia-uvm
sudo modprobe nvidia-drm
echo "✓ 内核模块已加载"
echo ""

# 3. 创建设备节点（如果需要）
echo "[3/4] 创建设备节点..."
sudo nvidia-smi || echo "注意：nvidia-smi 可能需要重新登录"
echo ""

# 4. 验证配置
echo "[4/4] 验证配置..."
echo ""
echo "内核模块状态："
lsmod | grep nvidia
echo ""
echo "设备文件："
ls -l /dev/nvidia* 2>/dev/null || echo "设备节点尚未创建"
echo ""

echo "========================================"
echo "修复完成！"
echo "========================================"
echo ""
echo "重要提示："
echo "1. 注销并重新登录以使用户组生效"
echo "2. 或者重启系统"
echo "3. 然后运行 'nvidia-smi' 验证"
echo ""
