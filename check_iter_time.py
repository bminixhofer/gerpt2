from torch.utils.data import DataLoader
from utils import Collator
import multiprocessing
import argparse
import h5py
from tqdm.auto import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)

    args = parser.parse_args()

    with h5py.File("prepare/train.hdf5", "r", libver="latest", swmr=True) as f:
        train_dataset = f["train"]

        loader = DataLoader(
            train_dataset,
            collate_fn=Collator(args.max_length),
            num_workers=multiprocessing.cpu_count(),
            batch_size=args.batch_size,
            shuffle=True,
        )

        for batch in tqdm(loader):
            pass
