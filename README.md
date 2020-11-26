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

Load the model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("benjamin/gerpt2")
model = AutoModelForCausalLM.from_pretrained("benjamin/gerpt2")
```

And use it with [Transformer's pipelines](https://huggingface.co/transformers/main_classes/pipelines.html):

```python
from transformers import pipeline

prompt = "In einer schockierenden Entdeckung fanden Wissenschaftler eine Herde Einhörner, die in einem abgelegenen, zuvor unerforschten Tal in den Anden lebten. Noch überraschender für die Forscher war die Tatsache, dass die Einhörner perfekt Deutsch sprachen."

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(pipe(prompt)[0]["generated_text"])
```

alternatively, two tricks can improve the generated text:

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

The prompt from above could generate something like this:

```
In einer schockierenden Entdeckung fanden Wissenschaftler eine Herde Einhörner, die in einem abgelegenen, zuvor unerforschten Tal in den Anden lebten. Noch überraschender für die Forscher war die Tatsache, dass die Einhörner perfekt Deutsch sprachen. Der Wissenschaftler Jean-Pierre Avon, ein Experte für den Umgang mit Einhörnern, war schockiert über den Fund. Er war nach Südamerika gereist und hat eine Entdeckung gemacht, die sich mehr als jemals zuvor für deutsche Forscher als nützlich erwiesen hat.
Die Wissenschaftler, die zum ersten Mal auf die Entdeckung der Einhörner im Jahr 2005 kamen, konnten eine Menge über die Existenz der Tiere erzählen und sagen, dass sie die menschliche Sprache erlernen und verstehen. Die Forscher analysierten auch die verschiedenen Arten von Einhörnern mit ähnlichen Eigenschaften und entdeckten so, dass es sich um eine besonders frühe Art von Einhörnern handelt.
Auch die verschiedenen Arten von Einhörnern, die die Forscher identifizierten, wurden in der folgenden Tabelle präsentiert: Von den fünf Arten der Diehörner sind zwei Arten aktiv und ihre Zahl ist größer als die Anzahl der aktiven Individuen, von denen jedes eine eine eigene Gruppe ist.
In der gleichen Ausgabe, die im Oktober 2009 veröffentlicht wurde, haben die Forscher eine erste Entdeckung gemacht, die den Unterschied zwischen den Arten einer anderen Art macht. Die Forscher stellten fest, dass die Individuen, die in Europa geboren wurden, viel älter waren als die Arten, die in den USA geboren wurden. Die Forscher konnten auch die Herkunft der Einhörner feststellen, da die verschiedenen Arten unterschiedliche Wurzeln hatten und sich in verschiedenen Regionen der Erde, wie Deutschland, befinden.
Wenn die Forscher von ihrem Forscherteam hören, dass das Verschwinden eines Einhörners und die Suche nach dem Entdecker eines Vogels für einige Zeit die natürliche Lebensweise beeinträchtigt, kann es durchaus sein, dass sie ein kleines Problem vor Augen haben. Das ist natürlich kein Grund zur Sorge.
Einhörner haben eine lange Geschichte. Sie haben Vorfahren aus der Zeit, als das Wort „Einhörner“ noch als "ein großes Wesen" oder „Pferde" (oder "Narren" oder "Maulwürmer") in der Familie des Menschen vorkam. Die Rasse hatte eine lange Geschichte in dem Land, als es noch ein Hirtendorf war. Später kamen die Menschen als "Pferde" in die Schweiz, um ihr Wissen über eine kleine Herde Einhörner weiterzugeben.
```

(picked from 10 samples)

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