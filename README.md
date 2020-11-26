# GerPT2

A small German GPT2.

See the [GPT2 model card](https://huggingface.co/gpt2) for considerations on limitations and bias. See the [GPT2 documentation](https://huggingface.co/transformers/model_doc/gpt2.html) for details on GPT2.

## Comparison to [dbmdz/german-gpt2](https://huggingface.co/dbmdz/german-gpt2)

I evaluated both GerPT2 and the other German GPT2, [dbmdz/german-gpt2](https://huggingface.co/dbmdz/german-gpt2) on the [CC-100](http://data.statmt.org/cc-100/) dataset and on the German Wikipedia:

|                   | CC-100 (PPL) | Wikipedia (PPL) |
|-------------------|--------------|-----------------|
| dbmdz/german-gpt2 | 49.47        | 62.92           |
| GerPT2            | __24.78__    | __35.33__       |
|                   |              |                 |

See the script `evaluate.py` in the [GerPT2 Github repository](https://github.com/bminixhofer/gerpt2) for the code.

## Usage

![GerPT2 usage](https://user-images.githubusercontent.com/13353204/100330362-288d0b80-2fcf-11eb-82e7-ed71c7140a88.png)

Also, two tricks might improve the generated text:

```python
output = model.generate(
    # during training an EOS token was used to mark the beginning of each text
    # so it can help to insert it at the start
    torch.tensor(
        [tokenizer.eos_token_id] + tokenizer.encode(prompt)
    ).unsqueeze(0),
    do_sample=True,
    # try setting bad_words_ids=[[0]] to disallow generating an EOS token, without this the model is
    # prone to ending generation early because a significant number of texts from the training corpus
    # is quite short
    bad_words_ids=[[0]],
    max_length=max_length,
)[0]
print(tokenizer.decode(output))
```

## Training details

GerPT2 is trained on the entire German data (67GB) from the [CC-100 Corpus](http://data.statmt.org/cc-100/) and weights were initialized from the [English GPT2 model](https://huggingface.co/gpt2). 
GerPT2 was trained with:

- a batch size of 256
- using OneCycle learning rate with a maximum of 5e-3
- with AdamW with a weight decay of 0.01
- for 7 epochs

Training took roughly 6 days on 8 TPUv3 cores.

To train GerPT2, follow these steps. Scripts are located in the [Github repository](https://github.com/bminixhofer/gerpt2):

0. Download and unzip training data from http://data.statmt.org/cc-100/.
1. Train a tokenizer using `prepare/train_tokenizer.py`. As training data for the tokenizer I used a random subset of 5% of the CC-100 data.
2. (optionally) generate a German input embedding matrix with `prepare/generate_aligned_wte.py`. This uses a neat trick to semantically map tokens from the English tokenizer to tokens from the German tokenizer using aligned word embeddings. E. g.:

```
ĠMinde -> Ġleast
Ġjed -> Ġwhatsoever
flughafen -> Air
vermittlung -> employment
teilung -> ignment
ĠInterpretation -> Ġinterpretation
Ġimport -> Ġimported
hansa -> irl
genehmigungen -> exempt
ĠAuflist -> Ġlists
Ġverschwunden -> Ġdisappeared
ĠFlyers -> ĠFlyers
Kanal -> Channel
Ġlehr -> Ġteachers
Ġnahelie -> Ġconvenient
gener -> Generally
mitarbeiter -> staff
```

This helps a lot on a trial run I did, although I wasn't able to do a full comparison due to budget and time constraints. To use this WTE matrix it can be passed via the `wte_path` to the training script. Credit to [this blogpost](https://medium.com/@pierre_guillou/faster-than-training-from-scratch-fine-tuning-the-english-gpt-2-in-any-language-with-hugging-f2ec05c98787) for the idea of initializing GPT2 from English weights. 

3. Tokenize the corpus using `prepare/tokenize_text.py`. This generates files for train and validation tokens in JSON Lines format.
4. Run the training script `train.py`! `run.sh` shows how this was executed for the full run with config `configs/tpu.json`.

## License

GerPT2 is licensed under the MIT License.

## Acknowledgements

Thanks to [Hugging Face](https://huggingface.co) for awesome tools and infrastructure.
Special thanks to [PetFinder.my](https://www.petfinder.my/) for generously sponsoring the resources used for training.
