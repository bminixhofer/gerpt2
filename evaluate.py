from transformers import GPT2Tokenizer, GPT2LMHeadModel
import click
from tqdm.auto import tqdm
import torch
from datasets import load_dataset


@click.command()
@click.option(
    "--model",
    "identifier",
    help="identifier of the pretrained model to use, passed to .from_pretrained",
)
@click.option(
    "--text_path", default=None, help="Path to text file to use for evaluation."
)
@click.option(
    "--use_german_wikipedia",
    default=False,
    is_flag=True,
    help="Whether to use the German Wikipedia for evaluation instead of a text file.",
)
@click.option(
    "--max_n_chars", default=1_000_000, help="Maximum number of chars to load."
)
def main(identifier, text_path, use_german_wikipedia, max_n_chars):
    # german wikipedia XOR a text path has to be set
    assert use_german_wikipedia == (text_path is None)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = GPT2Tokenizer.from_pretrained(identifier)
    model = GPT2LMHeadModel.from_pretrained(identifier).to(device)

    if use_german_wikipedia:
        text = ""
        dataset = load_dataset("wikipedia", "20200501.de", split="train")

        for article in dataset["text"]:
            text += article + "\n"
            if len(text) > max_n_chars:
                text = text[:max_n_chars]
                break
    else:
        with open(text_path) as f:
            text = f.read(max_n_chars)

    encodings = tokenizer(text, return_tensors="pt")

    # taken from https://huggingface.co/transformers/perplexity.html
    max_length = model.config.n_positions
    stride = 512
    end_loc = 0

    lls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc).item()
    print("Perplexity:", ppl)


if __name__ == "__main__":
    main()
