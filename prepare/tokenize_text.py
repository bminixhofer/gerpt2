from transformers import GPT2TokenizerFast
import click
import json
from tqdm.auto import tqdm
import random


def write_tokens(
    tokenizer, texts, train_filehandle, val_filehandle, val_chance, max_length
):
    tokens = tokenizer(texts)["input_ids"]
    current_t = []

    while len(tokens) > 0:
        if len(current_t) >= max_length:
            line = json.dumps({"tokens": current_t[:max_length]}) + "\n"

            if random.random() < val_chance:
                val_filehandle.write(line)
            else:
                train_filehandle.write(line)

            current_t = []

        t = [tokenizer.eos_token_id] + tokens.pop()
        current_t += t


@click.command()
@click.option("--tokenizer", help="Name or path of trained tokenizer.")
@click.option(
    "--text_path",
    help="Path to .txt file to tokenize. Texts are assuemd to be delimited by an empty line.",
)
@click.option(
    "--train_out_path",
    help="Path to output file. Tokens will be written in JSON Lines format"
    " where each line has one key 'tokens'.",
)
@click.option(
    "--val_out_path",
    help="Path to output file. Tokens will be written in JSON Lines format"
    " where each line has one key 'tokens'.",
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
    tokenizer,
    text_path,
    train_out_path,
    val_out_path,
    val_chance,
    max_length,
    batch_size,
):
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer)
    with open(text_path) as f, open(train_out_path, "w") as train_token_f, open(
        val_out_path, "w"
    ) as val_token_f:
        texts = []

        current_text = ""

        for line in tqdm(f):
            if line.isspace():
                texts.append(current_text)
                current_text = ""

                if len(texts) == batch_size:
                    write_tokens(
                        tokenizer,
                        texts,
                        train_token_f,
                        val_token_f,
                        val_chance,
                        max_length,
                    )
                    texts = []
            else:
                current_text += line

        if len(current_text) > 0:
            texts.append(current_text)

        if len(texts) > 0:
            write_tokens(
                tokenizer, texts, train_token_f, val_token_f, val_chance, max_length
            )


if __name__ == "__main__":
    main()
