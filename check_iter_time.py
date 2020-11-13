from torch.utils.data import DataLoader
import datasets
from utils import TrainCollator
from transformers import GPT2Tokenizer
import multiprocessing
import argparse
from tqdm.auto import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)

    args = parser.parse_args()

    tokenizer = GPT2Tokenizer(
        "data/used/german_tokenizer/vocab.json", "data/used/german_tokenizer/merges.txt"
    )
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = datasets.load_dataset(
        "json",
        data_files="data/used/train.tokens.json",
        split="train",
        cache_dir="data/used/train_tokens",
    )

    loader = DataLoader(
        train_dataset,
        collate_fn=TrainCollator(tokenizer, args.max_length, train_dataset),
        num_workers=multiprocessing.cpu_count(),
        batch_size=args.batch_size,
        shuffle=True,
    )

    for batch in tqdm(loader):
        pass
