from tqdm.auto import tqdm
import click
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sklearn.metrics.pairwise import cosine_similarity
import torch
import json


def load_vectors(path, max_n=200_000):
    with open(path) as f:
        ids = {}
        dim = int(f.readline().strip().split()[1])
        vectors = np.zeros((max_n, dim))

        i = 0
        for line in tqdm(f, total=max_n):
            if i == max_n:
                break

            parts = line.split()
            name = parts[0]

            if name in ids:
                continue

            try:
                values = np.array([float(x) for x in parts[1:]])
                vectors[i] = values
                ids[name] = i
                i += 1
            except ValueError:
                pass

        return vectors, ids


def get_tokenizer_embeddings(tokenizer, vectors, ids, freqs):
    embs = {value: ([], [], set()) for value in tokenizer.get_vocab().values()}

    for lower_key, value in tqdm(ids.items()):
        for key in [lower_key, lower_key.title()]:
            tokenized = tokenizer.encode(
                key, add_special_tokens=False
            ) + tokenizer.encode(" " + key, add_special_tokens=False)

            for token_id in tokenized:
                if key not in embs[token_id][2] and key in freqs:
                    embs[token_id][0].append(vectors[value])
                    embs[token_id][1].append(freqs[key])
                    embs[token_id][2].add(key)

    embs_matrix = np.zeros((len(embs), vectors.shape[1]))

    for i in range(len(embs_matrix)):
        if len(embs[i][2]) == 0:
            continue

        freqs = np.array(embs[i][1], dtype=np.float32)
        freqs /= freqs.sum()

        embs_matrix[i] = (np.stack(embs[i][0]) * freqs[:, np.newaxis]).sum(axis=0)

    return embs_matrix


@click.command()
@click.option("--german_tokenizer", help="Name or path of trained German tokenizer.")
@click.option("--english_tokenizer", help="Name or path of trained English tokenizer.")
@click.option(
    "--gpt2_model", help="Name or path of trained English GPT2 model to use WTE from."
)
@click.option(
    "--german_freqs", help="Path to a .json file with frequencies of german words."
)
@click.option(
    "--english_freqs", help="Path to a .json file with frequencies of english words."
)
@click.option(
    "--german_vecs",
    help="German aligned word vectors from https://fasttext.cc/docs/en/aligned-vectors.html.",
)
@click.option(
    "--english_vecs",
    help="English aligned word vectors from https://fasttext.cc/docs/en/aligned-vectors.html.",
)
@click.option(
    "--out_path", help="Path to store the German WTE matrix at as .pt file.",
)
def main(
    german_tokenizer,
    english_tokenizer,
    gpt2_model,
    german_freqs,
    english_freqs,
    german_vecs,
    english_vecs,
    out_path,
):
    german_freqs = json.load(open(german_freqs))
    english_freqs = json.load(open(english_freqs))

    de_vectors, de_ids = load_vectors(german_vecs)
    en_vectors, en_ids = load_vectors(english_vecs)

    german_tokenizer = GPT2TokenizerFast.from_pretrained(german_tokenizer)
    english_tokenizer = GPT2TokenizerFast.from_pretrained(english_tokenizer)

    model = GPT2LMHeadModel.from_pretrained(gpt2_model)

    en_tok_embs = get_tokenizer_embeddings(
        english_tokenizer, en_vectors, en_ids, english_freqs
    )
    de_tok_embs = get_tokenizer_embeddings(
        german_tokenizer, de_vectors, de_ids, german_freqs
    )

    def get_closest(token_id, similarities=None):
        if (de_tok_embs[token_id] == 0).all():
            return None, None

        if similarities is None:
            similarities = cosine_similarity(
                de_tok_embs[token_id][np.newaxis, :], en_tok_embs
            )[0]

        best = np.argsort(similarities)[::-1]

        best = english_tokenizer.convert_ids_to_tokens(best)
        de_token = german_tokenizer.convert_ids_to_tokens([token_id])[0]
        space_before = de_token.startswith("Ġ")

        best = [x for x in best if x.startswith("Ġ") == space_before]
        en_token = best[0]

        return en_token, de_token

    print("Some sample mappings:")

    for token_id in np.random.choice(list(german_tokenizer.get_vocab().values()), 30):
        en_token, de_token = get_closest(token_id)

        print(f"{de_token} -> {en_token}")

    german_wte_weight = torch.zeros_like(model.transformer.wte.weight)
    mean, std = (
        model.transformer.wte.weight.mean().item(),
        model.transformer.wte.weight.std().item(),
    )
    similarities = cosine_similarity(de_tok_embs, en_tok_embs)

    en_vocab = english_tokenizer.get_vocab()
    n_matched = 0

    for token_id in tqdm(range(len(german_wte_weight))):
        en_token, _ = get_closest(token_id, similarities=similarities[token_id])

        if en_token is None:
            german_wte_weight[token_id] = torch.normal(
                mean, std, size=(german_wte_weight.shape[1],)
            )
        else:
            en_token_id = en_vocab[en_token]
            german_wte_weight[token_id] = model.transformer.wte.weight[en_token_id]

            n_matched += 1

    print(f"Matching token found for {n_matched} of {len(en_vocab)} tokens.")
    torch.save(german_wte_weight, out_path)


if __name__ == "__main__":
    main()
