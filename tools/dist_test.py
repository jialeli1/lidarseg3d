import argparse
import copy
import json
import os
import sys

from det3d import datasets

try:
    import apex
except:
    print("No APEX!")
import numpy as np
import torch
import yaml
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.trainer import get_dist_info, load_checkpoint
from det3d.torchie.trainer.utils import all_gather, synchronize
from torch.nn.parallel import DistributedDataParallel
import pickle 
import time 

import os.path as osp
import datetime
from det3d.core.utils.common_utils import create_logger

import torch.nn as nn



def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", required=True, help="the dir to save logs and models")
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
    parser.add_argument(
        "--txt_result",
        type=bool,
        default=False,
        help="whether to save results to standard KITTI format of txt type",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--speed_test", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--testset", action="store_true")
    parser.add_argument('--tcp_port', type=int, default=17888, help='tcp port for distrbuted training')

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():

    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)

    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        os.environ["MASTER_PORT"] = str(args.tcp_port)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        cfg.gpus = torch.distributed.get_world_size()
    else:
        cfg.gpus = args.gpus

    # init logger before other steps
    # logger = get_root_logger(cfg.log_level)

    # creat the detailed log file like OpenPCDet
    if torchie.is_str(cfg.work_dir):
        work_dir = osp.abspath(cfg.work_dir)
        torchie.mkdir_or_exist(work_dir)
    elif cfg.work_dir is None:
        work_dir = None
    else:
        raise TypeError("'work_dir' must be a str or None")
    log_file = osp.join(work_dir, 'log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = create_logger(log_file, rank=cfg.local_rank, log_level=cfg.log_level)


    logger.info("Distributed testing: {}".format(distributed))
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")



    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    
    if args.testset:
        print("Use Test Set")
        dataset = build_dataset(cfg.data.test)
    else:
        print("Use Val Set")
        dataset = build_dataset(cfg.data.val)

    data_loader = build_dataloader(
        dataset,
        batch_size=cfg.data.samples_per_gpu if not args.speed_test else 1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )


    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    logger.info(f"loading parameters from: {args.checkpoint}")


    # put model on gpus
    sync_bn_type = cfg.get("sync_bn_type", "apex")
    if distributed:
        # NOTE: 这里是否需要转换呢？
        if sync_bn_type == "apex":
            model = apex.parallel.convert_syncbn_model(model)
            logger.info("Do run apex.parallel.convert_syncbn_model(model)")
        elif sync_bn_type == "torch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            logger.info("Do run torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)")
        elif sync_bn_type == "none":
            logger.info("Do NOT convert syncbn")
        else:
            raise NotImplementedError
        model = DistributedDataParallel(
            model.cuda(cfg.local_rank),
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            # broadcast_buffers=False,
            find_unused_parameters=True,
        )
    else:
        # model = fuse_bn_recursively(model)
        logger.info("Do NOT convert syncbn")
        model = model.cuda()

    model.eval()
    mode = "val"

    logger.info(f"work dir: {args.work_dir}")
    if cfg.local_rank == 0:
        prog_bar = torchie.ProgressBar(len(data_loader.dataset) // cfg.gpus)

    detections = {}
    cpu_device = torch.device("cpu")

    start = time.time()

    start = int(len(dataset) / 3)
    end = int(len(dataset) * 2 /3)
    # start = 10
    # end = 20
    
    time_start = 0 
    time_end = 0 

    for i, data_batch in enumerate(data_loader):
        if i == start:
            torch.cuda.synchronize()
            time_start = time.time()

        if i == end:
            torch.cuda.synchronize()
            time_end = time.time()

        with torch.no_grad():
            outputs = batch_processor(
                model, data_batch, train_mode=False, local_rank=args.local_rank,
            )

        for output in outputs:
            token = output["metadata"]["token"]
            for k, v in output.items():
                if k not in ["metadata"]:
                    output[k] = v.to(cpu_device)
            detections.update(
                {token: output,}
            )
            if args.local_rank == 0:
                prog_bar.update()
        
    
    
    synchronize()

    all_predictions = all_gather(detections)


    if args.speed_test:
        logger.info("\n Total time per frame: {}".format((time_end -  time_start) / (end - start)))
    
    if args.local_rank != 0:
        return

    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)


    result_dict, _ = dataset.evaluation(
        copy.deepcopy(predictions), 
        output_dir=args.work_dir, 
        testset=args.testset, 
    )


    if result_dict is not None:
        logger.info(f"\nEvaluation results:")
        for k, v in result_dict["results"].items():
            # print(f"Evaluation {k}: {v}")
            logger.info(f"Evaluation {k}: {v}")


    if args.txt_result:
        assert False, "No longer support kitti"

if __name__ == "__main__":
    main()
