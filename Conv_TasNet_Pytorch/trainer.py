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


def to_device(dicts, device):
    """
    Move batch data to device
    """
    def move(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, list):
            return [i.to(device) for i in x]
        else:
            raise RuntimeError("Unsupported data type")

    return {k: move(v) for k, v in dicts.items()}


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
        clip_norm=None,
        min_lr=0,
        patience=0,
        factor=0.5,
        logging_period=100,
        resume=None,
        stop=10,
        num_epochs=100,
    ):

        # -------- Device setup --------
        if isinstance(gpuid, int):
            gpuid = (gpuid,)

        self.use_cuda = torch.cuda.is_available() and len(gpuid) > 0
        self.gpuid = gpuid

        if self.use_cuda:
            self.device = torch.device(f"cuda:{gpuid[0]}")
        else:
            self.device = torch.device("cpu")

        # -------- Checkpoint --------
        if checkpoint and not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        self.checkpoint = checkpoint

        # -------- Logger --------
        self.logger = get_logger(os.path.join(checkpoint, "trainer.log"), file=False)
        self.logger.info(f"Using device: {self.device}")

        self.clip_norm = clip_norm
        self.logging_period = logging_period
        self.cur_epoch = 0
        self.stop = stop
        self.num_epochs = num_epochs

        # -------- Resume or fresh --------
        if resume and resume.get("resume_state", False):
            cpt_path = os.path.join(resume["path"], checkpoint, "best.pt")
            cpt = torch.load(cpt_path, map_location="cpu")
            self.cur_epoch = cpt["epoch"]
            net.load_state_dict(cpt["model_state_dict"])
            self.net = net.to(self.device)
            self.optimizer = self.create_optimizer(
                optimizer, optimizer_kwargs, cpt["optim_state_dict"]
            )
            self.logger.info(f"Resumed from epoch {self.cur_epoch}")
        else:
            self.net = net.to(self.device)
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)

        # -------- Params & scheduler --------
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

    def create_optimizer(self, optimizer, kwargs, state=None):
        opts = {
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
            "adam": torch.optim.Adam,
            "adadelta": torch.optim.Adadelta,
            "adagrad": torch.optim.Adagrad,
            "adamax": torch.optim.Adamax,
        }
        opt = opts[optimizer](self.net.parameters(), **kwargs)
        if state:
            opt.load_state_dict(state)
        return opt

    def save_checkpoint(self, best=True):
        torch.save(
            {
                "epoch": self.cur_epoch,
                "model_state_dict": self.net.state_dict(),
                "optim_state_dict": self.optimizer.state_dict(),
            },
            os.path.join(self.checkpoint, "best.pt" if best else "last.pt"),
        )

    def forward_net(self, mix):
        if self.use_cuda:
            return data_parallel(self.net, mix, device_ids=self.gpuid)
        else:
            return self.net(mix)

    def train(self, dataloader):
        self.net.train()
        losses = []
        start = time.time()

        for step, egs in enumerate(dataloader, 1):
            egs = to_device(egs, self.device)
            if step == 1:
                print("🔥 Training is running...")

            self.optimizer.zero_grad()
            ests = self.forward_net(egs["mix"])
            loss = si_snr_loss(ests, egs)
            loss.backward()

            if self.clip_norm:
                clip_grad_norm_(self.net.parameters(), self.clip_norm)

            self.optimizer.step()
            losses.append(loss.item())

            if step % self.logging_period == 0:
                avg = sum(losses[-self.logging_period:]) / self.logging_period
                self.logger.info(
                    f"<epoch:{self.cur_epoch}, iter:{step}, loss:{avg:.4f}>"
                )

        return sum(losses) / len(losses)

    def val(self, dataloader):
        self.net.eval()
        losses = []
        start = time.time()

        with torch.no_grad():
            for egs in dataloader:
                egs = to_device(egs, self.device)
                ests = self.forward_net(egs["mix"])
                loss = si_snr_loss(ests, egs)
                losses.append(loss.item())

        return sum(losses) / len(losses)

    def run(self, train_loader, val_loader):
        train_losses, val_losses = [], []

        ctx = torch.cuda.device(self.gpuid[0]) if self.use_cuda else nullcontext()

        with ctx:
            # val_loss = self.val(val_loader)
            # best_loss = val_loss
            best_loss = float("inf")

            self.save_checkpoint(best=True)
            self.logger.info("🚀 Starting training loop")
            print("🚀 Starting training loop")


            while self.cur_epoch < self.num_epochs:
                self.cur_epoch += 1

                train_loss = self.train(train_loader)
                
                # val_loss = self.val(val_loader)
                val_loss = train_loss  # dummy value


                train_losses.append(train_loss)
                val_losses.append(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_checkpoint(best=True)
                    self.logger.info(
                        f"Epoch {self.cur_epoch}: new best loss {best_loss:.4f}"
                    )
                else:
                    self.logger.info("No improvement")

                self.scheduler.step(val_loss)
                self.save_checkpoint(best=False)

        # -------- Plot loss --------
        plt.figure()
        plt.plot(train_losses, label="train")
        plt.plot(val_losses, label="val")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig("conv_tasnet_loss.png")
