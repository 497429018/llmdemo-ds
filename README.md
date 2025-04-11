## 一、基础环境准备
## 基础软件
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget tmux htop ncdu unzip build-essential unzip rsync pdsh

sudo apt-get update && sudo apt-get install ninja-build

sudo hostnamectl set-hostname --static "te1" 
echo "172.18.81.132 te1" >> /etc/hosts
echo "172.18.81.133 te2" >> /etc/hosts
echo "172.18.81.134 te3" >> /etc/hosts
echo "172.18.81.135 te4" >> /etc/hosts

## 配置密码登录
vim /etc/ssh/sshd_config
PasswordAuthentication yes

systemctl restart sshd

passwd


## 免密
# 主节点执行
cd ~/.ssh
ssh-keygen -t rsa
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

ssh-copy-id -i ~/.ssh/id_rsa.pub root@te1
ssh-copy-id -i ~/.ssh/id_rsa.pub root@te2

## pdsh 远程访问
echo ssh | sudo tee /etc/pdsh/rcmd_default
pdsh -S -w te1 hostname

# 安装 conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /data/Miniconda3.sh
bash /data/Miniconda3.sh -b -p /data/miniconda3

echo 'export PATH="/data/miniconda3/bin:$PATH"' >> ~/.bashrc
source /data/miniconda3/bin/activate
source ~/.bashrc

# 安装 conda 训练环境
conda create -n llm python=3.10 -y
conda activate llm
echo 'conda activate llm' >> ~/.bashrc
source ~/.bashrc


##### 磁盘复现还原环境 begin
fdisk -l
mkdir /data
mount /dev/vdb1 /data

echo 'export PATH="/data/miniconda3/bin:$PATH"' >> ~/.bashrc
echo 'source /data/miniconda3/bin/activate' >> ~/.bashrc
echo 'conda activate llm' >> ~/.bashrc
source ~/.bashrc

# 保护模型加载多线程安全
echo 'export TOKENIZERS_PARALLELISM=false' >> ~/.bashrc
source ~/.bashrc

##### 磁盘复现还原环境 end

#  复制数据到其他节点
rsync -avh --progress [源路径] [目标路径]
rsync -avh --progress  /data/llmdemo-ds te2:/data 
rsync -avh --progress  /data/llmdemo-ds te3:/data
rsync -avh --progress  /data/llmdemo-ds te4:/data


## 构建自己的conda 环境
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets evaluate accelerate tensorboard sentencepiece peft  einops tiktoken bitsandbytes deepspeed
pip install huggingface_hub

## 基础模型存放
sudo mkdir -p /data/models
huggingface-cli download Qwen/Qwen2.5-7B --resume-download --local-dir /data/models/qwen2.5-7b
huggingface-cli download Qwen/Qwen2.5-14B --resume-download --local-dir /data/models/qwen2.5-14b


### 环境验证测试
rsync -avh --progress [源路径] [目标路径]
rsync -avz /root/.cache/huggingface/. te2:/root/.cache/huggingface/

rsync -avz --progress /data/huggingface/hub/. /root/.cache/huggingface/hub

# 检查硬件资源
nvidia-smi
python -c "import torch; print(torch.cuda.nccl.version())"
python -c "import torch; torch.distributed.init_process_group(backend='nccl'); print('NCCL test passed')"

# 检查端口占用
netstat -tulnp | grep :8000
kill -9 PID

### 显存监控
watch -n 1 "nvidia-smi --query-gpu=memory.used --format=csv"

## tensorboard 查看模型执行情况
tensorboard --logdir=logs --port 9090 \
--bind_all --load_fast=false

## 后台执行代码 避免终端退出 任务也终止了
chmod +x your_script.sh
nohup ./hf.sh > df.log 2>&1 & 

### 测试环境验证
deepspeed --hostfile=hostfile --master_addr=te1 test/test_multinode.py
deepspeed --hostfile=hostfile --master_addr=te1 test/test_multinode_enhanced.py

### 模型测试
bash run.sh