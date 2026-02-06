import torch
import time
import os
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

from SI_SNR import si_snr_loss
from Conv_TasNet import check_parameters
from utils import get_logger


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


class Trainer:
    def __init__(
        self,
        net,
        train_dataloader,
        val_dataloader,
        checkpoint="checkpoint",
        lr=1e-3,
        clip_norm=5.0,
        patience=3,
        factor=0.5,
        min_lr=1e-6,
        num_epochs=100,
        log_interval=100,
    ):
        assert torch.cuda.is_available(), "CUDA is required"

        self.device = torch.device("cuda")
        self.net = net.to(self.device)

        self.train_loader = train_dataloader
        self.val_loader = val_dataloader

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True,
        )

        self.scaler = GradScaler()
        self.clip_norm = clip_norm
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.cur_epoch = 0

        os.makedirs(checkpoint, exist_ok=True)
        self.checkpoint = checkpoint
        self.logger = get_logger(os.path.join(checkpoint, "trainer.log"))

        self.logger.info(
            f"Model params: {check_parameters(self.net):.2f} M"
        )

    # ---------------------------------------------------
    # Train one epoch
    # ---------------------------------------------------
    def train_epoch(self):
        self.net.train()
        losses = []
        start = time.time()

        for step, batch in enumerate(self.train_loader, 1):
            batch = to_device(batch, self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast():
                ests = self.net(batch["mix"])
                loss = si_snr_loss(ests, batch)

            self.scaler.scale(loss).backward()

            if self.clip_norm:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.net.parameters(), self.clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            losses.append(loss.item())

            if step % self.log_interval == 0:
                self.logger.info(
                    f"Epoch {self.cur_epoch:03d} | "
                    f"Iter {step:05d} | "
                    f"LR {self.optimizer.param_groups[0]['lr']:.3e} | "
                    f"Loss {sum(losses[-self.log_interval:]) / self.log_interval:.4f}"
                )

        avg_loss = sum(losses) / len(losses)
        self.logger.info(
            f"Epoch {self.cur_epoch:03d} TRAIN | "
            f"Loss {avg_loss:.4f} | "
            f"Time {(time.time() - start)/60:.2f} min"
        )
        return avg_loss

    # ---------------------------------------------------
    # Validation
    # ---------------------------------------------------
    def validate(self):
        self.net.eval()
        losses = []
        start = time.time()

        with torch.no_grad():
            for batch in self.val_loader:
                batch = to_device(batch, self.device)
                ests = self.net(batch["mix"])
                loss = si_snr_loss(ests, batch)
                losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        self.logger.info(
            f"Epoch {self.cur_epoch:03d} VAL   | "
            f"Loss {avg_loss:.4f} | "
            f"Time {(time.time() - start)/60:.2f} min"
        )
        return avg_loss

    # ---------------------------------------------------
    # Checkpoint
    # ---------------------------------------------------
    def save(self, name):
        torch.save(
            {
                "epoch": self.cur_epoch,
                "model": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            os.path.join(self.checkpoint, name),
        )

    # ---------------------------------------------------
    # Run training
    # ---------------------------------------------------
    def run(self):
        train_losses, val_losses = [], []
        best_loss = float("inf")
        no_improve = 0

        for epoch in range(1, self.num_epochs + 1):
            self.cur_epoch = epoch

            train_loss = self.train_epoch()
            val_loss = self.validate()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            self.scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                no_improve = 0
                self.save("best.pt")
                self.logger.info("✅ New best model saved")
            else:
                no_improve += 1

            self.save("last.pt")

            if no_improve >= 10:
                self.logger.info("⛔ Early stopping")
                break

        self.plot_losses(train_losses, val_losses)

    # ---------------------------------------------------
    # Plot loss
    # ---------------------------------------------------
    def plot_losses(self, train_losses, val_losses):
        plt.figure()
        plt.plot(train_losses, label="train")
        plt.plot(val_losses, label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.checkpoint, "loss_curve.png"))
