import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        if not pairs:
            raise ValueError('pairs must not be empty ')
        for index, (src_text, tgt_text) in enumerate(pairs):
            if not src_text.split():
                raise ValueError(f'sample {index} source must contain at least one token')
            if not tgt_text.split():
                raise ValueError(f'sample {index} target must contain at least one token')

        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        src_text, tgt_text = self.pairs[index]

        src_ids = self.src_vocab.encode(src_text)

        tgt_input_ids = self.tgt_vocab.encode(tgt_text, add_sos=True)

        tgt_output_ids = self.tgt_vocab.encode(tgt_text, add_eos=True)

        src = torch.tensor(src_ids,dtype=torch.long)
        tgt_input = torch.tensor(tgt_input_ids, dtype=torch.long)
        tgt_output = torch.tensor(tgt_output_ids, dtype=torch.long)

        return src, tgt_input, tgt_output


def make_collate_fn(src_pad_id, tgt_pad_id):
    def collate_fn(batch):
        src_sequences = []
        tgt_input_sequences = []
        tgt_output_sequences = []

        for src, tgt_input,  tgt_output in batch:
            src_sequences.append(src)
            tgt_input_sequences.append(tgt_input)
            tgt_output_sequences.append(tgt_output)

        src_batch = pad_sequence(
            src_sequences,
            batch_first=True,
            padding_value=src_pad_id,
        )

        tgt_input_batch = pad_sequence(
            tgt_input_sequences,
            batch_first=True,
            padding_value=tgt_pad_id,
        )

        tgt_output_batch = pad_sequence(
            tgt_output_sequences,
            batch_first=True,
            padding_value=tgt_pad_id,
        )

        return src_batch, tgt_input_batch, tgt_output_batch
    return collate_fn