import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    pipeline,
)
import datasets
import pytorch_lightning as pl
from argparse import ArgumentParser
from utils import TrainCollator, ValCollator
import multiprocessing
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import math
import wandb
import os

CPU_MODEL = None


class Model(pl.LightningModule):
    def __init__(self, tokenizer, train_dataset, val_dataset, hparams):
        super().__init__()

        self.hparams = hparams

        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        if self.hparams.use_english_weights:
            self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        else:
            self.gpt = GPT2LMHeadModel(GPT2Config.from_pretrained("gpt2"))

        wte = self.gpt.transformer.wte

        if hparams.wte_path is not None:
            wte.weight = nn.Parameter(torch.load(hparams.wte_path))
        else:
            mean, std = wte.weight.mean().item(), wte.weight.std().item()
            wte.weight = nn.Parameter(torch.normal(mean, std, size=wte.weight.size()))

        # tie input and output embeddings
        self.gpt.lm_head.weight = self.gpt.transformer.wte.weight

        # used to generate samples, must be on CPU
        global CPU_MODEL
        CPU_MODEL = GPT2LMHeadModel(self.gpt.config)

    def forward(self, *args, **kwargs):
        return self.gpt.forward(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        schedulers = []

        if self.hparams.use_onecycle:
            schedulers.append(
                torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.lr)
            )

        return [optimizer], schedulers

    def training_step(self, batch, batch_idx):
        input_ids = batch
        labels = input_ids.clone()

        loss = self.forward(input_ids=input_ids, labels=labels)[0]

        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        labels = input_ids.clone()
        labels[~attention_mask] = -100

        loss = (
            self.forward(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )[0]
            * attention_mask.sum()
        )
        return {"loss": loss.item(), "size": attention_mask.sum().item()}

    def validation_epoch_end(self, outputs):
        loss_sum = sum(x["loss"] for x in outputs)
        size = sum(x["size"] for x in outputs)

        loss = loss_sum / size
        perplexity = math.exp(loss)

        table = wandb.Table(columns=["Prompt", "Output"])

        CPU_MODEL.load_state_dict(self.gpt.state_dict())

        generate = pipeline(
            "text-generation",
            model=CPU_MODEL,
            tokenizer=self.tokenizer,
            config={"max_length": 6000},
        )

        for prompt in open("prompts.txt").read().split("\n"):
            table.add_data(
                prompt,
                generate(prompt, pad_token_id=self.tokenizer.eos_token_id)[0][
                    "generated_text"
                ],
            )

        if len(outputs) > self.hparams.num_sanity_val_steps:
            self.logger.experiment.log(
                {"val_loss": loss, "val_perplexity": perplexity, "examples": table}
            )

    @staticmethod
    def get_parser():
        parser = ArgumentParser()
        parser.add_argument(
            "--batch_size", type=int,
        )
        parser.add_argument("--max_length", type=int, help="Maximum context length.")
        parser.add_argument(
            "--wte_path", type=str, help="Path to .pth WTE embeddings to load."
        )
        parser.add_argument(
            "--use_english_weights",
            action="store_true",
            help="Whether to start from weights of english GPT2.",
        )
        parser.add_argument(
            "--use_onecycle",
            action="store_true",
            help="Whether to use 1cycle learning rate policy (if true, max_lr = lr).",
        )
        parser.add_argument("--lr", type=float, help="Learning rate.")
        parser = pl.Trainer.add_argparse_args(parser)

        parser.set_defaults(
            batch_size=2,
            max_length=1024,
            lr=1e-3,
            max_epochs=1,
            wte_path=None,
            use_english_weights=False,
        )
        return parser

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=TrainCollator(
                self.tokenizer, self.hparams.max_length, self.train_dataset
            ),
            num_workers=multiprocessing.cpu_count(),
            batch_size=self.hparams.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=ValCollator(self.tokenizer, self.hparams.max_length),
            num_workers=multiprocessing.cpu_count(),
            batch_size=self.hparams.batch_size,
            shuffle=False,
        )


if __name__ == "__main__":
    parser = Model.get_parser()

    parser.add_argument(
        "--train_slice", type=str, default=":", help="Slice of training data to use.",
    )
    parser.add_argument(
        "--val_slice", type=str, default=":", help="Slice of validation data to use.",
    )
    parser.add_argument(
        "--no_upload", action="store_true", help="Skip uploading model checkpoints.",
    )
    logger = WandbLogger(project="gerpt2")
    parser.set_defaults(logger=logger)

    args = parser.parse_args()

    tokenizer = GPT2Tokenizer(
        "data/used/german_tokenizer/vocab.json", "data/used/german_tokenizer/merges.txt"
    )
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = datasets.load_dataset(
        "json",
        data_files="data/used/train.tokens.json",
        split=f"train[{args.train_slice}]",
        cache_dir="data/used/train_tokens",
    )

    val_dataset = datasets.load_dataset(
        "json",
        data_files="data/used/val.tokens.json",
        split=f"train[{args.val_slice}]",
        cache_dir="data/used/val_tokens",
    )

    model = Model(tokenizer, train_dataset, val_dataset, args)

    checkpoint_dir = os.path.join(logger.experiment.dir, "checkpoints")
    wandb.save(os.path.join(checkpoint_dir, "*.cpkt"))

    callbacks = [LearningRateMonitor(logging_interval="step")]

    if not args.no_upload:
        callbacks.append(
            ModelCheckpoint(
                monitor="loss",
                dirpath=checkpoint_dir,
                filename="gpt2-{epoch:02d}-{loss:.2f}",
                save_top_k=-1,
                mode="min",
            )
        )

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer.tune(model)
    trainer.fit(model)
