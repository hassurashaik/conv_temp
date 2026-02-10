import argparse
import os
import sys
import torch

# allow importing from ./options
sys.path.append('./options')

from trainer import Trainer
from Conv_TasNet import ConvTasNet
from DataLoaders import make_dataloader
from option import parse
from utils import get_logger


def main():
    # -----------------------------
    # Read arguments
    # -----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt',
        type=str,
        required=True,
        help='Path to option YAML file'
    )
    args = parser.parse_args()

    # -----------------------------
    # Parse YAML options
    # -----------------------------
    opt = parse(args.opt, is_tain=True)
    logger = get_logger(__name__)

    # -----------------------------
    # Build model
    # -----------------------------
    logger.info('Building Conv-TasNet model')
    net = ConvTasNet(**opt['net_conf'])

    # -----------------------------
    # Build dataloaders
    # -----------------------------
    logger.info('Building train & validation dataloaders')

    train_loader = make_dataloader(
        is_train=True,
        data_kwargs=opt['datasets']['train'],
        num_workers=opt['datasets']['num_workers'],
        chunk_size=opt['datasets']['chunk_size'],
        batch_size=opt['datasets']['batch_size'],
    )

    val_loader = make_dataloader(
        is_train=False,
        data_kwargs=opt['datasets']['val'],
        num_workers=opt['datasets']['num_workers'],
        chunk_size=opt['datasets']['chunk_size'],
        batch_size=opt['datasets']['batch_size'],
    )

    # -----------------------------
    # Build trainer
    # -----------------------------
    logger.info('Building Trainer')

    trainer = Trainer(
        net=net,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        checkpoint=opt['train']['checkpoint'],
        lr=opt['train']['lr'],
        clip_norm=opt['train']['clip_norm'],
        patience=opt['train']['patience'],
        factor=opt['train']['factor'],
        min_lr=opt['train']['min_lr'],
        num_epochs=opt['train']['num_epochs'],
        log_interval=opt['train']['log_interval'],
    )

    # -----------------------------
    # AUTO-RESUME FROM LAST CHECKPOINT
    # -----------------------------
    ckpt_path = os.path.join(opt['train']['checkpoint'], "last.pt")

    if os.path.exists(ckpt_path):
        logger.info(f"Resuming training from checkpoint: {ckpt_path}")
        trainer.load(ckpt_path)
    else:
        logger.info("No checkpoint found, starting training from scratch")

    # -----------------------------
    # Start training
    # -----------------------------
    trainer.run()


if __name__ == "__main__":
    main()
