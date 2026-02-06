import torch
import time
import os
from contextlib import nullcontext
from utils import get_logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import data_parallel
from torch.nn.utils import clip_grad_norm_
from SI_SNR import si_snr_loss
import matplotlib.pyplot as plt
from Conv_TasNet import check_parameters


# =========================
# Utilities
# =========================
def to_device(batch, device):
    def move(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, list):
            return [i.to(device) for i in x]
        else:
            raise RuntimeError("Unsupported data type")
    return {k: move(v) for k, v in batch.items()}


# =========================
# Trainer
# =========================
class Trainer:
    """
    Trainer for Conv-TasNet (CPU + GPU safe)
    """

    def __init__(
        self,
        net,
        checkpoint="checkpoint",
        optimizer="adam",
        gpuid=(),
        optimizer_kwargs=None,
        clip_norm=5.0,
        min_lr=1e-6,
        patience=3,
        factor=0.5,
        logging_period=100,
        resume=None,
        num_epochs=100,
    ):

        # -------- Device --------
        if isinstance(gpuid, int):
            gpuid = (gpuid,)

        self.use_cuda = torch.cuda.is_available() and len(gpuid) > 0
        self.gpuid = gpuid
        self.device = torch.device(f"cuda:{gpuid[0]}") if self.use_cuda else torch.device("cpu")

        # -------- Local checkpoint --------
        self.checkpoint = checkpoint
        os.makedirs(self.checkpoint, exist_ok=True)

        # -------- Drive checkpoint (BEST ONLY) --------
        self.drive_dir = "/content/drive/MyDrive/ConvTasNet_Checkpoints"
        os.makedirs(self.drive_dir, exist_ok=True)
        self.drive_best = os.path.join(self.drive_dir, "best.pt")

        # -------- Logger --------
        self.logger = get_logger(os.path.join(self.checkpoint, "trainer.log"), file=False)
        self.logger.info(f"Using device: {self.device}")

        self.clip_norm = clip_norm
        self.logging_period = logging_period
        self.cur_epoch = 0
        self.num_epochs = num_epochs

        # -------- Resume / Fresh --------
        if resume and resume.get("resume_state", False) and os.path.exists(self.drive_best):
            ckpt = torch.load(self.drive_best, map_location="cpu")
            self.cur_epoch = ckpt["epoch"]
            net.load_state_dict(ckpt["model_state_dict"])
            self.net = net.to(self.device)
            self.optimizer = self.create_optimizer(
                optimizer, optimizer_kwargs, ckpt["optim_state_dict"]
            )
            self.logger.info(f"Resumed from epoch {self.cur_epoch}")
        else:
            self.net = net.to(self.device)
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)

        # -------- Scheduler --------
        self.param = check_parameters(self.net)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

        self.logger.info(
            f"Model ready | Params: {self.param:.2f} M | CUDA: {self.use_cuda}"
        )

    # =========================
    # Optimizer
    # =========================
    def create_optimizer(self, name, kwargs, state=None):
        opts = {
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
            "adam": torch.optim.Adam,
            "adadelta": torch.optim.Adadelta,
            "adagrad": torch.optim.Adagrad,
            "adamax": torch.optim.Adamax,
        }
        opt = opts[name](self.net.parameters(), **kwargs)
        if state:
            opt.load_state_dict(state)
        return opt

    # =========================
    # Save BEST only
    # =========================
    def save_best_checkpoint(self):
        state = {
            "epoch": self.cur_epoch,
            "model_state_dict": self.net.state_dict(),
            "optim_state_dict": self.optimizer.state_dict(),
        }

        torch.save(state, os.path.join(self.checkpoint, "best.pt"))
        torch.save(state, self.drive_best)

        self.logger.info(
            f"💾 Best model saved to Drive: {self.drive_best} (epoch {self.cur_epoch})"
        )

    # =========================
    # Forward
    # =========================
    def forward_net(self, mix):
        if self.use_cuda:
            return data_parallel(self.net, mix, device_ids=self.gpuid)
        return self.net(mix)

    # =========================
    # Train one epoch
    # =========================
    def train(self, dataloader):
        self.net.train()
        losses = []

        for step, batch in enumerate(dataloader, 1):
            batch = to_device(batch, self.device)

            if step == 1:
                print("🔥 Training is running...")

            self.optimizer.zero_grad()
            ests = self.forward_net(batch["mix"])
            loss = si_snr_loss(ests, batch)
            loss.backward()

            clip_grad_norm_(self.net.parameters(), self.clip_norm)
            self.optimizer.step()

            losses.append(loss.item())

            if step % self.logging_period == 0:
                avg = sum(losses[-self.logging_period:]) / self.logging_period
                self.logger.info(
                    f"<epoch:{self.cur_epoch}, iter:{step}, loss:{avg:.4f}>"
                )

        return sum(losses) / len(losses)

    # =========================
    # Validation
    # =========================
    def val(self, dataloader):
        self.net.eval()
        losses = []

        with torch.no_grad():
            for batch in dataloader:
                batch = to_device(batch, self.device)
                ests = self.forward_net(batch["mix"])
                loss = si_snr_loss(ests, batch)
                losses.append(loss.item())

        return sum(losses) / len(losses)

    # =========================
    # Run
    # =========================
    def run(self, train_loader, val_loader=None):
        best_loss = float("inf")
        train_losses, val_losses = [], []

        ctx = torch.cuda.device(self.gpuid[0]) if self.use_cuda else nullcontext()

        with ctx:
            self.logger.info("🚀 Starting training loop")
            print("🚀 Starting training loop")

            while self.cur_epoch < self.num_epochs:
                self.cur_epoch += 1

                train_loss = self.train(train_loader)
                val_loss = self.val(val_loader) if val_loader else train_loss

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_best_checkpoint()
                    self.logger.info(
                        f"Epoch {self.cur_epoch}: new best loss {best_loss:.4f}"
                    )
                else:
                    self.logger.info("No improvement")

                self.scheduler.step(val_loss)

        # -------- Plot --------
        plt.figure()
        plt.plot(train_losses, label="train")
        plt.plot(val_losses, label="val")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("SI-SNR Loss")
        plt.savefig("conv_tasnet_loss.png")
