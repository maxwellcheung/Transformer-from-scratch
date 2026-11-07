import  torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class LanguageModelDataset(Dataset):
    def __init__(self, texts, vocab):
        self.texts = list(texts)

        if not self.texts:
            raise ValueError('texts must not be empty')

        for index, text in  enumerate(self.texts):
            if not text.split():
                raise ValueError(f'sample {index} must contain at least one token')

        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]

        token_ids = self.vocab.encode(text, add_sos=True, add_eos=True,)

        tokens = torch.tensor(token_ids, dtype=torch.long,)
        
        input_ids = tokens[:-1]
        targets = tokens[1:]

        return input_ids, targets

def make_lm_collate_fn(pad_id):
    def collate_fn(batch):
        input_sequences = []
        target_sequences = []

        for input_ids, targets in batch:
            input_sequences.append(input_ids)
            target_sequences.append(targets)

        input_batch = pad_sequence(
            input_sequences,
            batch_first=True,
            padding_value=pad_id,
        )

        target_batch = pad_sequence(
            target_sequences,
            batch_first=True,
            padding_value=pad_id,
        )
            
        return input_batch, target_batch

    return collate_fn