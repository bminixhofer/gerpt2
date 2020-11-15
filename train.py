from transformers import (
    HfArgumentParser,
    GPT2LMHeadModel,
    GPT2Config,
    GPT2Tokenizer,
    Trainer,
)
import datasets
from dataclasses import dataclass
from transformers.training_args import TrainingArguments
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch import nn
import torch
from utils import ValCollator, TrainCollator


@dataclass
class ExtraArgs:
    train_slice: str
    val_slice: str
    max_length: int
    wte_path: str = None
    use_english_weights: bool = False


class GPT2Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.extra_args = kwargs.pop("extra_args")

        super().__init__(*args, **kwargs)

    def get_eval_dataloader(self, _):
        eval_sampler = self._get_eval_sampler(self.eval_dataset)

        return DataLoader(
            self.eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=ValCollator(self.tokenizer, self.extra_args.max_length),
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_train_dataloader(self):
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=TrainCollator(
                self.tokenizer, self.extra_args.max_length, self.train_dataset
            ),
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


if __name__ == "__main__":
    json_file_parser = ArgumentParser()
    json_file_parser.add_argument("--config_file", type=str, default=None)
    json_file_path = json_file_parser.parse_args().config_file

    parser = HfArgumentParser([TrainingArguments, ExtraArgs])

    if json_file_path is None:
        training_args, extra_args = parser.parse_args_into_dataclasses()
    else:
        training_args, extra_args = parser.parse_json_file(json_file_path)

    train_dataset = datasets.load_dataset(
        "json",
        data_files="data/used/train.tokens.json",
        split=f"train[{extra_args.train_slice}]",
        cache_dir="data/used/train_tokens",
    )

    val_dataset = datasets.load_dataset(
        "json",
        data_files="data/used/val.tokens.json",
        split=f"train[{extra_args.val_slice}]",
        cache_dir="data/used/val_tokens",
    )

    model = get_model(extra_args)

    tokenizer = GPT2Tokenizer(
        "data/used/german_tokenizer/vocab.json", "data/used/german_tokenizer/merges.txt"
    )
    tokenizer.pad_token = tokenizer.eos_token

    training_args.remove_unused_columns = False
    steps_per_epoch = int(
        len(train_dataset)
        / training_args.per_device_train_batch_size
        / training_args.gradient_accumulation_steps
    )
    training_args.eval_steps = steps_per_epoch
    training_args.save_steps = steps_per_epoch
    training_args.logging_steps = 50

    trainer = GPT2Trainer(
        model,
        training_args,
        extra_args=extra_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
