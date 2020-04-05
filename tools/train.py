import argparse
import os
import sys
from os import mkdir
from apex import amp
import shutil
import torch.nn.functional as F

sys.path.append('..')
sys.path.append('.')

from config import cfg
from data import make_data_loader
from engine.trainer import do_train
from modeling import build_model
from solver import make_optimizer, WarmupMultiStepLR
from layers import make_loss
from utils.logger import setup_logger
from torch.utils.tensorboard import SummaryWriter
import torch
torch.cuda.set_device(0)

cfg.merge_from_file('configs/config.yml')
cfg.freeze()

output_dir = cfg.OUTPUT_DIR
writer = SummaryWriter(log_dir=os.path.join(output_dir,'tensorboard'))
logger = setup_logger("rendering_model", output_dir, 0)
logger.info("Running with config:\n{}".format(cfg))
shutil.copyfile('configs/config.yml', '%s/config.yml'%output_dir)

train_loader, vertex_list,dataset = make_data_loader(cfg, is_train=True)
model = build_model(cfg,vertex_list)
optimizer = make_optimizer(cfg, model)
scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
loss_fn = make_loss()
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

do_train(
        cfg,
        model,
        train_loader,
        None,
        optimizer,
        scheduler,
        loss_fn,
        writer
    )