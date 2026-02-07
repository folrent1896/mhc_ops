#!/bin/bash
# 快速测试脚本 - 运行所有测试

set -e

echo "=================================="
echo "MHC Forward Pre - 运行测试套件"
echo "=================================="
echo ""

# 检查 Python 环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到 Python"
    exit 1
fi

echo "Python 版本: $(python --version)"
echo ""

# 检查 CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA 可用: 是"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1
else
    echo "CUDA 可用: 否 (将使用 CPU)"
fi
echo ""

# 运行快速测试
echo "=================================="
echo "运行快速测试..."
echo "=================================="
python test/quick_test.py

echo ""
echo "=================================="
echo "运行性能基准测试..."
echo "=================================="
python test/benchmark.py

echo ""
echo "=================================="
echo "所有测试完成！"
echo "=================================="
