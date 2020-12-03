from transformers import GPT2TokenizerFast
import click
from tqdm.auto import tqdm
import random
import h5py
import numpy as np

CHUNK_SIZE = 1_000


def get_samples(tokenizer, texts, val_chance, max_length):
    tokens = tokenizer(texts)["input_ids"]
    current_t = []

    samples = []
    val_samples = []

    while len(tokens) > 0:
        if len(current_t) >= max_length:
            while len(current_t) >= max_length:
                sample = [current_t.pop(0) for _ in range(max_length)]

                if random.random() < val_chance:
                    val_samples.append(sample)
                else:
                    samples.append(sample)

            current_t = []

        t = [tokenizer.eos_token_id] + tokens.pop()
        current_t += t

    return samples, val_samples


def write_samples(samples, dataset, max_length):
    while len(samples) >= CHUNK_SIZE:
        chunk = np.zeros((CHUNK_SIZE, max_length))

        for i in range(CHUNK_SIZE):
            chunk[i] = samples.pop()

        prev_length = dataset.shape[0]
        dataset.resize(prev_length + CHUNK_SIZE, axis=0)
        dataset[prev_length:, :] = chunk


@click.command()
@click.option("--tokenizer", help="Name or path of trained tokenizer.")
@click.option(
    "--text_path",
    help="Path to .txt file to tokenize. Texts are assuemd to be delimited by an empty line.",
)
@click.option(
    "--out_path",
    help="Path to output file. Will be written as HDF5 with datasets `train` and `val`.",
)
@click.option(
    "--max_length", type=int, default=1024, help="The maximum number of tokens.",
)
@click.option(
    "--batch_size",
    type=int,
    default=50_000,
    help="How many texts to tokenize in parallel.",
)
@click.option(
    "--val_chance",
    type=float,
    default=0.05,
    help="How many texts to tokenize in parallel.",
)
def main(
    tokenizer, text_path, out_path, val_chance, max_length, batch_size,
):
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer)
    with open(text_path) as f, h5py.File(out_path, "w", libver="latest") as hdf5_f:
        texts = []

        hdf5_f.swmr_mode = True  # need concurrent reads from pytorch
        train_dataset = hdf5_f.create_dataset(
            "train",
            (0, max_length),
            maxshape=(None, max_length),
            dtype=np.int32,
            chunks=(CHUNK_SIZE, max_length),
        )
        val_dataset = hdf5_f.create_dataset(
            "val",
            (0, max_length),
            maxshape=(None, max_length),
            dtype=np.int32,
            chunks=(CHUNK_SIZE, max_length),
        )

        current_text = ""

        train_samples = []
        val_samples = []

        for line in tqdm(f):
            if line.isspace():
                texts.append(current_text)
                current_text = ""

                if len(texts) == batch_size:
                    batch_train_samples, batch_val_samples = get_samples(
                        tokenizer, texts, val_chance, max_length,
                    )
                    train_samples += batch_train_samples
                    val_samples += batch_val_samples

                    write_samples(train_samples, train_dataset, max_length)
                    write_samples(val_samples, val_dataset, max_length)

                    texts = []
            else:
                current_text += line

        if len(current_text) > 0:
            texts.append(current_text)

        if len(texts) > 0:
            batch_train_samples, batch_val_samples = get_samples(
                tokenizer, texts, val_chance, max_length
            )

        write_samples(train_samples, train_dataset, max_length)
        write_samples(val_samples, val_dataset, max_length)


if __name__ == "__main__":
    main()
