#!/bin/bash

# ===== 分布式训练环境配置 =====
# 集群节点列表（按需修改）
HOSTS="te1 te2 te3 te4"

# 每台服务器的GPU数量
GPUS_PER_NODE=1

# 主节点配置
MASTER_ADDR="te1"  # 推荐使用IP而不是主机名
MASTER_PORT=6000   # 确保端口未被占用

# ===== NCCL 优化配置 =====
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0       # 使用 ifconfig 确认网卡名称
export NCCL_IB_DISABLE=1             # 禁用InfiniBand（如果没有IB设备）
export NCCL_P2P_LEVEL=NVL            # NVLink优化
export NCCL_ALGO=Ring                # 默认使用Ring算法

# ===== PyTorch 分布式配置 =====
export PYTHONUNBUFFERED=1            # 实时输出日志
export CUDA_LAUNCH_BLOCKING=1        # 更准确的CUDA错误定位
export TORCH_DISTRIBUTED_DEBUG=INFO  # PyTorch分布式调试信息

# ===== DeepSpeed 超时配置 =====
export PDSH_RCMD_TYPE=ssh            # 确保使用ssh连接
export GLOO_SOCKET_IFNAME=eth0       # Gloo后端网络接口
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1   # 异步错误处理
export TORCH_NCCL_BLOCKING_WAIT=1          # 超时设置（秒）
export NCCL_CHECK_DISABLE=0          # 启用NCCL检查

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IGNORE_CPU_AFFINITY=1

# ===== 生成 hostfile =====
echo "生成 hostfile..."
rm -f hostfile
for host in $HOSTS; do
    echo "$host slots=$GPUS_PER_NODE" >> hostfile
done

# ===== 启动命令 =====
echo "===== 分布式训练启动参数 ====="
echo "主机列表:    $HOSTS"
echo "每节点GPU:   $GPUS_PER_NODE"
echo "主节点:      $MASTER_ADDR:$MASTER_PORT"
echo "NCCL 配置:   NCCL_IB_DISABLE=$NCCL_IB_DISABLE, NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
echo "============================"


deepspeed --hostfile=hostfile \
    --master_addr="te1" \
    --master_port="8000" \
    --num_nodes=3 \
    --num_gpus=1 \
    scripts/fast_finetune.py

# ===== 错误处理 =====
if [ $? -ne 0 ]; then
    echo "[错误] 训练启动失败！可能原因："
    echo "1. 节点间SSH无密码登录未配置"
    echo "2. 端口 $MASTER_PORT 被占用"
    echo "3. GPU设备不可用（尝试运行 nvidia-smi 检查）"
    echo "4. NCCL通信失败（检查网卡配置）"
    echo "完整日志请查看 train.log"
    exit 1
fi