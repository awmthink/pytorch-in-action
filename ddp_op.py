import torch
import torch.distributed as dist

import torch.multiprocessing as mp
import os


def all_reduce(rank):
    t = torch.ones((5, 5), device=rank) * rank
    # t = p0.t + p1.t + p2.t + p3.t
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    assert t.mean().item() == 6  # [0, 1, 2, 3]


def reduce(rank):
    t = torch.ones((5, 5), device=rank) * rank
    dist.reduce(t, dst=0, op=dist.ReduceOp.SUM)
    # print(f"{os.getpid()}: {t.mean().item()}")
    if rank == 0:
        assert t.mean().item() == 6
    else:
        # 在gloo中结果不对
        assert t.mean().item() == rank


def boardcast(rank):
    t = torch.ones((5, 5), device=rank) * rank
    # 将rank 3的进程中的 t 广播到其他进程中
    dist.broadcast(t, src=3)
    assert t.mean().item() == 3


def all_gather(rank):
    t = torch.ones((5, 5), device=rank) * rank
    outputs = []
    for _ in range(dist.get_world_size()):
        outputs.append(torch.zeros((5, 5), device=rank))
    dist.all_gather(outputs, t)
    gather = torch.concat(outputs, dim=0)
    assert gather.shape == torch.Size([20, 5])
    assert gather.float().mean() == torch.tensor([0, 1.0, 2.0, 3.0]).mean()


def reduce_scatter(rank):
    world_size = dist.get_world_size()
    t = torch.ones((world_size * 5, 5), device=rank) * rank
    l = torch.split(t, 5, dim=0)
    reduce_rst = torch.zeros((5, 5), device=rank)
    dist.reduce_scatter(reduce_rst, list(l), dist.ReduceOp.SUM)
    assert reduce_rst.mean().item() == 6


def main_process(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "25321"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    dist.init_process_group(backend="nccl")
    all_reduce(rank)
    reduce(rank)
    boardcast(rank)
    all_gather(rank)
    reduce_scatter(rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    nprocs = 4
    mp.spawn(main_process, nprocs=nprocs, args=(nprocs,), join=True)
