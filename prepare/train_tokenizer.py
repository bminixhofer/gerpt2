from transformers import GPT2TokenizerFast
import tokenizers
import click
from pathlib import Path


@click.command()
@click.option("--text_path", type=str, help="Path to input .txt file.")
@click.option("--out_directory", type=str, help="Path to tokenizer output directory.")
def main(text_path, out_directory):
    Path(out_directory).mkdir(exist_ok=True, parents=True)

    english_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    german_tokenizer = tokenizers.ByteLevelBPETokenizer()

    german_tokenizer.train(
        [text_path],
        vocab_size=english_tokenizer.vocab_size,
        special_tokens=["<|endoftext|>"],
        show_progress=True,
    )
    german_tokenizer.save_model(out_directory)


if __name__ == "__main__":
    main()
