import argparse
import sys
from utils import create_logger, seed_set
from utils.demo_visualize import demo_visualize
from utils.script import *
import numpy as np
sys.path.append(os.getcwd())
from config import Config, update_config
import torch
from tensorboardX import SummaryWriter
from utils.training import Trainer
from utils.evaluation import compute_stats

from data_loader.dataset_amass import DatasetAMASS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', default='h36m', help='h36m or humaneva or amass')
    parser.add_argument('--mode', default='train', help='train / eval / pred')
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    parser.add_argument('--milestone', type=list, default=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400])
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--save_model_interval', type=int, default=100)
    parser.add_argument('--ckpt', type=str, default='./checkpoints/h36m_ckpt.pt')
    parser.add_argument('--ema', type=bool, default=True)
    parser.add_argument('--vis_col', type=int, default=10)
    parser.add_argument('--vis_row', type=int, default=3)
    args = parser.parse_args()

    """setup"""
    seed_set(args.seed)
    # seed_set(6) 

    cfg = Config(f'{args.cfg}', test=(args.mode != 'train'))
    cfg = update_config(cfg, vars(args))

    if cfg.dataset == 'amass':
        dataset = {'train': DatasetAMASS('train'), 'test': DatasetAMASS('test')}
    else:
        dataset, dataset_multi_test = dataset_split(cfg)


    """logger"""
    tb_logger = SummaryWriter(cfg.tb_dir)
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
    display_exp_setting(logger, cfg)
    """model"""
    model, diffusion = create_model_and_diffusion(cfg)

    logger.info(">>> total params: {:.2f}M".format(
        sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    if args.mode == 'train':
        trainer = Trainer(
            model=model,
            diffusion=diffusion,
            dataset=dataset,
            cfg=cfg,
            logger=logger,
            tb_logger=tb_logger)
        trainer.loop()

    elif args.mode == 'eval':
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt)
        if cfg.dataset == 'amass':
            multimodal_dict = get_multimodal_gt_full_amass(logger, dataset['test'], args, cfg)
        else:
            multimodal_dict = get_multimodal_gt_full(logger, dataset_multi_test, args, cfg)
        compute_stats(diffusion, multimodal_dict, model, logger, cfg)

    else:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt)
        demo_visualize(args.mode, cfg, model, diffusion, dataset)
