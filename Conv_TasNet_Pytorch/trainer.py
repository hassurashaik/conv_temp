import os
import time
import torch
import matplotlib.pyplot as plt

from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler

from SI_SNR import pit_si_snr_loss
from Conv_TasNet import check_parameters
from utils import get_logger


# -----------------------------
# Move batch to device
# -----------------------------
def to_device(batch, device):
    def move(x):
        if torch.is_tensor(x):
            return x.to(device)
        elif isinstance(x, list):
            return [move(i) for i in x]
        elif isinstance(x, tuple):
            return tuple(move(i) for i in x)
        else:
            return x

    return {k: move(v) for k, v in batch.items()}


# =============================
# Trainer
# =============================
class Trainer:
    def __init__(
        self,
        net,
        train_dataloader,
        val_dataloader,
        checkpoint,
        lr,
        clip_norm,
        patience,
        factor,
        min_lr,
        num_epochs,
        log_interval,
    ):
        # -----------------------------
        # Checkpoint dir
        # -----------------------------
        os.makedirs(checkpoint, exist_ok=True)
        self.checkpoint = checkpoint
        self.logger = get_logger(os.path.join(checkpoint, "trainer.log"))

        # -----------------------------
        # Device
        # -----------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net.to(self.device)

        self.use_amp = (self.device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)

        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model params: {check_parameters(self.net):.2f} M")

        # -----------------------------
        # Data
        # -----------------------------
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader

        # -----------------------------
        # Optimizer & Scheduler
        # -----------------------------
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

        # -----------------------------
        # Training config
        # -----------------------------
        self.clip_norm = clip_norm
        self.num_epochs = num_epochs
        self.log_interval = log_interval

        # Resume-critical state
        self.start_epoch = 1
        self.best_loss = float("inf")

        self._printed_shapes = False

    # ---------------------------------------------------
    # Load checkpoint (BACKWARD COMPATIBLE)
    # ---------------------------------------------------
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)

        # Required
        self.net.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

        # Optional (old checkpoints may not have these)
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        else:
            self.logger.warning("⚠️ No scheduler state in checkpoint. Using fresh scheduler.")

        if self.use_amp and "scaler" in ckpt and ckpt["scaler"] is not None:
            self.scaler.load_state_dict(ckpt["scaler"])
        else:
            self.logger.warning("⚠️ No AMP scaler state in checkpoint.")

        self.start_epoch = ckpt["epoch"] + 1
        self.best_loss = ckpt.get("best_loss", float("inf"))

        self.logger.info(
            f"✅ Resumed from {path} | "
            f"Start epoch: {self.start_epoch} | "
            f"Best loss: {self.best_loss:.4f}"
        )

    # ---------------------------------------------------
    # Save checkpoint (NEW FORMAT)
    # ---------------------------------------------------
    def save(self, name, epoch):
        torch.save(
            {
                "epoch": epoch,
                "model": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict() if self.use_amp else None,
                "best_loss": self.best_loss,
            },
            os.path.join(self.checkpoint, name),
        )

    # ---------------------------------------------------
    # Train one epoch
    # ---------------------------------------------------
    def train_epoch(self, epoch):
        self.net.train()
        losses = []
        start = time.time()

        for step, batch in enumerate(self.train_loader, 1):
            batch = to_device(batch, self.device)

            if not self._printed_shapes:
                self.logger.info(
                    f"[DEBUG] mix {batch['mix'].shape}, "
                    f"ref {batch['ref'].shape}"
                )

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=self.use_amp):
                ests = self.net(batch["mix"])
                loss = pit_si_snr_loss(ests, batch["ref"])

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(
                    (p for p in self.net.parameters() if p.grad is not None),
                    self.clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                clip_grad_norm_(self.net.parameters(), self.clip_norm)
                self.optimizer.step()

            losses.append(loss.item())

            if step % self.log_interval == 0:
                self.logger.info(
                    f"Epoch {epoch:03d} | "
                    f"Iter {step:05d} | "
                    f"LR {self.optimizer.param_groups[0]['lr']:.3e} | "
                    f"Loss {sum(losses[-self.log_interval:]) / self.log_interval:.4f}"
                )

            self._printed_shapes = True

        avg_loss = sum(losses) / len(losses)
        self.logger.info(
            f"Epoch {epoch:03d} TRAIN | "
            f"Loss {avg_loss:.4f} | "
            f"Time {(time.time() - start)/60:.2f} min"
        )

        return avg_loss

    # ---------------------------------------------------
    # Validation
    # ---------------------------------------------------
    def validate(self, epoch):
        self.net.eval()
        losses = []
        start = time.time()

        with torch.no_grad():
            for batch in self.val_loader:
                batch = to_device(batch, self.device)
                ests = self.net(batch["mix"])
                loss = pit_si_snr_loss(ests, batch["ref"])
                losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)

        self.logger.info(
            f"Epoch {epoch:03d} VAL   | "
            f"Loss {avg_loss:.4f} | "
            f"Time {(time.time() - start)/60:.2f} min"
        )

        self.logger.info(
            f"LR after scheduler: {self.optimizer.param_groups[0]['lr']:.3e}"
        )

        return avg_loss

    # ---------------------------------------------------
    # Training loop
    # ---------------------------------------------------
    def run(self):
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            self.scheduler.step(val_loss)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save("best.pt", epoch)
                self.logger.info("✅ New best model saved")

            self.save("last.pt", epoch)

        self.logger.info("🎉 Training completed")
