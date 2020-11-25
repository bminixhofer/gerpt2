from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import click


@click.command()
@click.option(
    "--model",
    "identifier",
    help="identifier of the pretrained model to use, passed to .from_pretrained",
)
@click.option("--prompt", help="Prompt passed to the model.")
@click.option("--n", default=1, help="Number of responses to generate")
@click.option("--max_length", default=500, help="Length in tokens of the response.")
def main(identifier, prompt, n, max_length):
    tokenizer = AutoTokenizer.from_pretrained(identifier)
    model = AutoModelForCausalLM.from_pretrained(identifier)

    print("PROMPT:", prompt)
    print()

    for _ in range(n):
        print(
            tokenizer.decode(
                model.generate(
                    torch.tensor(
                        [tokenizer.eos_token_id] + tokenizer.encode(prompt)
                    ).unsqueeze(0),
                    do_sample=True,
                    bad_words_ids=[[0]],
                    max_length=max_length,
                )[0]
            )
        )
        print("-" * 20)
        print()


if __name__ == "__main__":
    main()
