import argparse
import json
import os
import sys
import os.path as osp
import datetime
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

import numpy as np
import torch
import yaml
from det3d.datasets import build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
    set_random_seed_torch,
)
from det3d.core.utils.common_utils import create_logger
from det3d import torchie


import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", help="the dir to save logs and models")
    parser.add_argument("--resume_from", help="the checkpoint file to resume from")
    parser.add_argument("--load_from", help="the checkpoint file to load from")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="whether to evaluate the checkpoint during training",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--autoscale-lr",
        action="store_true",
        help="automatically scale lr with the number of gpus",
    )
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')

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
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.load_from is not None:
        cfg.load_from = args.load_from

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        os.environ["MASTER_PORT"] = str(args.tcp_port)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        cfg.gpus = torch.distributed.get_world_size()

    if args.autoscale_lr:
        cfg.lr_config.lr_max = cfg.lr_config.lr_max * cfg.gpus

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
    log_file = osp.join(work_dir, 'log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = create_logger(log_file, rank=cfg.local_rank, log_level=cfg.log_level)
    
    logger.info("Distributed training: {}".format(distributed))
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")


    if args.local_rank == 0:
        # copy important files to backup
        backup_dir = os.path.join(cfg.work_dir, "det3d")
        os.makedirs(backup_dir, exist_ok=True)
        # os.system("cp -r * %s/" % backup_dir)
        # logger.info(f"Backup source files to {cfg.work_dir}/det3d")


    # set random seeds
    if args.seed is not None:
        # logger.info("Set random seed to {}".format(args.seed))
        # set_random_seed(args.seed)
        set_random_seed_torch(args.seed, logger=logger)


    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)


    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))

    if cfg.checkpoint_config is not None:
        # save det3d version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            config=cfg.text, CLASSES=datasets[0].CLASSES
        )


    model.CLASSES = datasets[0].CLASSES
    # NOTE: We recommend using ""torch" instead of "apex" for sync_bn_type.
    sync_bn_type = cfg.get("sync_bn_type", "apex")

    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger,
        sync_bn_type=sync_bn_type,
    )


if __name__ == "__main__":
    main()
