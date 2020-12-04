from transformers import (
    HfArgumentParser,
    GPT2LMHeadModel,
    GPT2Config,
    GPT2Tokenizer,
    Trainer,
)
from dataclasses import dataclass
from transformers.integrations import WandbCallback
from transformers.training_args import TrainingArguments
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch import nn
import torch
from utils import Collator
import wandb
import os
import shutil
from coolname import generate_slug
import h5pickle


@dataclass
class ExtraArgs:
    max_length: int
    use_onecycle: bool
    max_n_train: int = None
    max_n_val: int = None
    wte_path: str = None
    use_english_weights: bool = False


class GPT2WandbCallback(WandbCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            ckpt_path = os.path.join(wandb.run.dir, "checkpoints")

            if os.path.exists(ckpt_path):
                shutil.rmtree(ckpt_path)
            shutil.copytree(args.output_dir, ckpt_path)

        super().on_save(args, state, control, **kwargs)


class GPT2Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.extra_args = kwargs.pop("extra_args")

        super().__init__(*args, **kwargs)
        print(f"Model is on device {self.model.device}.")
        self.model.tie_weights()

    # adapted to not use scheduler
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        if self.lr_scheduler is None:
            if self.extra_args.use_onecycle:
                self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    self.args.learning_rate,
                    epochs=self.args.num_train_epochs,
                    steps_per_epoch=self.args.steps_per_epoch,
                )
            else:
                self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lambda epoch: 1
                )

    def get_eval_dataloader(self, _):
        eval_sampler = self._get_eval_sampler(self.eval_dataset)

        return DataLoader(
            self.eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=Collator(self.extra_args.max_length),
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_train_dataloader(self):
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=Collator(self.extra_args.max_length),
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )


def get_model(extra_args):
    if extra_args.use_english_weights:
        model = GPT2LMHeadModel.from_pretrained("gpt2")
    else:
        model = GPT2LMHeadModel(GPT2Config.from_pretrained("gpt2"))

    wte = model.transformer.wte
    if extra_args.wte_path is not None:
        wte.weight = nn.Parameter(torch.load(extra_args.wte_path))
    else:
        mean, std = wte.weight.mean().item(), wte.weight.std().item()
        wte.weight = nn.Parameter(torch.normal(mean, std, size=wte.weight.size()))

    # tie input and output embeddings
    model.lm_head.weight = model.transformer.wte.weight
    model.tie_weights()

    return model


def main():
    json_file_parser = ArgumentParser()
    json_file_parser.add_argument("--config_file", type=str, default=None)
    json_file_parser.add_argument("--tpu_num_cores", type=int, default=None)
    json_parser_args = json_file_parser.parse_args()

    parser = HfArgumentParser([TrainingArguments, ExtraArgs])

    if json_parser_args.config_file is None:
        training_args, extra_args = parser.parse_args_into_dataclasses()
    else:
        training_args, extra_args = parser.parse_json_file(json_parser_args.config_file)

    with h5pickle.File(
        "data/train.hdf5", "r", libver="latest", swmr=True, skip_cache=False
    ) as f:
        train_dataset = f["train"]
        val_dataset = f["val"]

        if extra_args.max_n_train is not None:
            train_dataset = train_dataset[: extra_args.max_n_train]

        if extra_args.max_n_val is not None:
            val_dataset = val_dataset[: extra_args.max_n_val]

        model = get_model(extra_args)

        tokenizer = GPT2Tokenizer(
            "data/german_tokenizer_cc/vocab.json",
            "data/german_tokenizer_cc/merges.txt",
        )
        tokenizer.pad_token = tokenizer.eos_token

        name = generate_slug(2)

        if json_parser_args.tpu_num_cores is not None:
            training_args.tpu_num_cores = json_parser_args.tpu_num_cores

        training_args.remove_unused_columns = False
        steps_per_epoch = int(
            len(train_dataset)
            / training_args.per_device_train_batch_size
            / training_args.gradient_accumulation_steps
            / training_args.tpu_num_cores
        )
        training_args.steps_per_epoch = steps_per_epoch
        training_args.eval_steps = steps_per_epoch
        training_args.save_steps = steps_per_epoch
        training_args.run_name = name
        training_args.output_dir = os.path.join("checkpoints", name)

        trainer = GPT2Trainer(
            model,
            training_args,
            extra_args=extra_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            callbacks=[GPT2WandbCallback],
        )
        trainer.remove_callback(WandbCallback)

        trainer.train()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
