import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .corpus import read_parallel


class ParallelDataset(Dataset):
    def __init__(
        self,
        src_path,
        tgt_path,
        src_tokenizer,
        tgt_tokenizer,
        max_len,
        max_samples=None,
    ):
        pairs = read_parallel(src_path, tgt_path, max_samples=max_samples)
        self.samples = []
        self.skipped_too_long = 0

        for src_text, tgt_text in pairs:
            src_ids = src_tokenizer.encode(src_text)
            tgt_ids = tgt_tokenizer.encode(tgt_text)

            if not src_ids or not tgt_ids:
                continue
            if len(src_ids) > max_len or len(tgt_ids) + 1 > max_len:
                self.skipped_too_long += 1
                continue

            self.samples.append((
                torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(
                    [tgt_tokenizer.sos_id] + tgt_ids,
                    dtype=torch.long,
                ),
                torch.tensor(
                    tgt_ids + [tgt_tokenizer.eos_id],
                    dtype=torch.long,
                ),
                src_text,
                tgt_text,
            ))

        if not self.samples:
            raise ValueError(
                'no samples remain after tokenization and length filtering'
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def make_collate_fn(src_pad_id, tgt_pad_id, include_text=False):
    def collate_fn(batch):
        src = pad_sequence(
            [sample[0] for sample in batch],
            batch_first=True,
            padding_value=src_pad_id,
        )
        tgt_input = pad_sequence(
            [sample[1] for sample in batch],
            batch_first=True,
            padding_value=tgt_pad_id,
        )
        tgt_output = pad_sequence(
            [sample[2] for sample in batch],
            batch_first=True,
            padding_value=tgt_pad_id,
        )
        if include_text:
            return (
                src,
                tgt_input,
                tgt_output,
                [sample[3] for sample in batch],
                [sample[4] for sample in batch],
            )
        return src, tgt_input, tgt_output

    return collate_fn

