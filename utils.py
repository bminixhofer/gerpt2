import torch
import random


class ValCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        input_ids = torch.zeros((len(batch), self.max_length), dtype=torch.long)
        labels = torch.full((len(batch), self.max_length), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), self.max_length), dtype=torch.bool)

        tokenized = [x["tokens"] for x in batch]
        for i, ids in enumerate(tokenized):
            length = min(len(ids), self.max_length)

            input_ids[i, :length] = torch.tensor(ids[:length], dtype=torch.long)
            labels[i, :length] = torch.tensor(ids[:length], dtype=torch.long)
            attention_mask[i, :length] = 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


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

        return {
            "input_ids": output,
            "labels": output.clone(),
        }
