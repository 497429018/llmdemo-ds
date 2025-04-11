import torch
import torch.distributed as dist
import os

def main():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # 简单的广播操作
    tensor = torch.tensor([rank]).cuda()
    dist.broadcast(tensor, src=0)
    
    print(f"Rank {rank}/{world_size} received tensor: {tensor.item()}")

if __name__ == "__main__":
    main()
    dist.destroy_process_group()