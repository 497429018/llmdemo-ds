# test_multinode_enhanced.py
# deepspeed --hostfile=hostfile --master_addr=te1 test_multinode_enhanced.py
import torch
import torch.distributed as dist
import os
import time

def main():
    # 初始化分布式环境
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f'cuda:{local_rank}')
    
    # 打印设备信息
    print(f"Rank {rank}/{world_size} (local_rank {local_rank}) using {torch.cuda.get_device_name(device)}")
    
    # 带宽测试
    if rank == 0:
        data = torch.randn(1000000, device=device)
        dist.send(data, dst=1)
        print(f"Rank 0 sent {data.numel()} elements to Rank 1")
    elif rank == 1:
        data = torch.empty(1000000, device=device)
        dist.recv(data, src=0)
        print(f"Rank 1 received {data.numel()} elements from Rank 0")
    
    # 同步所有节点
    dist.barrier()
    
    # 集体通信测试
    tensor = torch.tensor([local_rank], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank} all_reduce result: {tensor.item()}")

if __name__ == "__main__":
    main()
    dist.destroy_process_group()