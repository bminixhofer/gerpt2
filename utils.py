import torch


class Collator:
    def __init__(self, max_length):
        self.max_length = max_length

    def __call__(self, batch):
        tokenized = [torch.from_numpy(x) for x in batch]
        output = torch.stack(tokenized, 0)

        assert output.shape[1] == self.max_length

        return {
            "input_ids": output,
            "labels": output.clone(),
        }
