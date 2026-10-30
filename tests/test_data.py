import torch
from seq2seq.vocab import Vocab
import pytest

def test_translation_dataset_returns_shifted_long_tensors():
    from seq2seq.data import TranslationDataset

    pairs = [
        ('i love you', '我 爱 你'),
        ('hello', '你 好'),
    ]

    src_vocab = Vocab(src for src, _ in pairs)
    tgt_vocab = Vocab(tgt for _, tgt in pairs)

    dataset = TranslationDataset(pairs, src_vocab, tgt_vocab)

    src, tgt_input, tgt_output = dataset[0]

    assert len(dataset) == 2

    assert src.dtype == torch.long
    assert tgt_input.dtype == torch.long
    assert tgt_output.dtype == torch.long

    assert src.tolist() == [4, 5, 6]
    assert tgt_input.tolist() == [1, 5, 6, 4]
    assert tgt_output.tolist() == [5, 6, 4, 2]


def test_collate_fn_pads_variable_length_sequences():
    from seq2seq.data import TranslationDataset, make_collate_fn

    pairs = [
        ('i love you', '我 爱 你'),
        ('hello', '你 好'),
    ]

    src_vocab = Vocab(src for src, _ in pairs)
    tgt_vocab = Vocab(tgt for _, tgt in pairs)

    dataset = TranslationDataset(pairs, src_vocab, tgt_vocab)
    batch = [dataset[0], dataset[1]]

    collate_fn = make_collate_fn(
        src_vocab.pad_id,
        tgt_vocab.pad_id,
    )

    src, tgt_input, tgt_output = collate_fn(batch)

    assert src.tolist() == [
        [4, 5, 6],
        [7, 0, 0],
    ]

    assert tgt_input.tolist() == [
        [1, 5, 6, 4],
        [1, 4, 7, 0],
    ]

    assert tgt_output.tolist() == [
        [5, 6, 4, 2],
        [4, 7, 2, 0]
    ]


def test_translation_dataset_rejects_empty_pairs():
    from seq2seq.data import TranslationDataset
    src_vocab = Vocab(['hello'])
    tgt_vocab = Vocab(['你'])

    with pytest.raises(ValueError, match='pairs'):
        TranslationDataset([], src_vocab, tgt_vocab)

def test_translation_dataset_rejects_empty_source_with_sample_index():
    from seq2seq.data import TranslationDataset

    src_vocab = Vocab(['hello world'])
    tgt_vocab = Vocab(['你 好'])

    pairs = [
        ('hello', '你'),
        ('   ', '好'),
    ]

    with pytest.raises(ValueError, match='sample 1'):
        TranslationDataset(pairs, src_vocab, tgt_vocab)


def test_translation_dataset_rejects_empty_target_with_sample_index():
    from seq2seq.data import TranslationDataset

    src_vocab = Vocab(['hello world'])
    tgt_vocab = Vocab(['你 好'])

    pairs = [
        ('hello', '你'),
        ('world', ' \t '),
    ]

    with pytest.raises(ValueError, match='sample 1'):
        TranslationDataset(pairs, src_vocab, tgt_vocab)