from torch.nn.utils.rnn import pad_sequence
import torch
import random


class ValCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(x["tokens"][: self.max_length]) for x in batch],
            batch_first=True,
            padding_value=self.tokenizer.eos_token_id,
        )
        attention_mask = pad_sequence(
            [
                torch.ones(min(len(x["tokens"]), self.max_length), dtype=torch.bool)
                for x in batch
            ],
            batch_first=True,
            padding_value=self.tokenizer.eos_token_id,
        )

        return input_ids, attention_mask


class TrainCollator:
    def __init__(self, tokenizer, max_length, dataset):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = dataset

    def __call__(self, batch):
        output = torch.full((len(batch), self.max_length), -1)

        tokenized = [x["tokens"] for x in batch]
        for i, ids in enumerate(tokenized):
            if len(ids) > self.max_length:
                offset = random.randint(0, len(ids) - self.max_length)
                output[i] = torch.tensor(ids[offset : offset + self.max_length])

        pad = True

        while pad:
            lengths = (output != -1).sum(1)

            if (lengths == self.max_length).all():
                break

            for i, ids in enumerate(tokenized):
                if lengths[i] < self.max_length:
                    pad_tokens = self.dataset[random.randint(0, len(self.dataset) - 1)][
                        "tokens"
                    ]
                    pad_tokens = pad_tokens[: self.max_length - lengths[i]]

                    output[i, lengths[i] : lengths[i] + len(pad_tokens)] = torch.tensor(
                        pad_tokens
                    )

        return output
