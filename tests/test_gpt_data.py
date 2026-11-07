import torch


def test_language_model_dataset_returns_shifted_tokens():
    from gpt.data import LanguageModelDataset
    from seq2seq.vocab import Vocab

    texts = [
        'i love you',
        'hello',
    ]

    vocab = Vocab(texts)

    dataset = LanguageModelDataset(
        texts,
        vocab,
    )

    input_ids, targets = dataset[0]

    assert input_ids.dtype == torch.long
    assert targets.dtype == torch.long

    assert input_ids.tolist() == [
        vocab.sos_id,
        vocab.word2idx['i'],
        vocab.word2idx['love'],
        vocab.word2idx['you'],
    ]

    assert targets.tolist() == [
        vocab.word2idx['i'],
        vocab.word2idx['love'],
        vocab.word2idx['you'],
        vocab.eos_id,
    ]


def test_language_model_collate_pads_sequences():
    from gpt.data import (
        LanguageModelDataset,
        make_lm_collate_fn,
    )
    from seq2seq.vocab import Vocab

    texts = [
        'i love you',
        'hello',
    ]

    vocab = Vocab(texts)

    dataset = LanguageModelDataset(
        texts,
        vocab,
    )

    collate_fn = make_lm_collate_fn(
        vocab.pad_id
    )

    input_ids, targets = collate_fn([
        dataset[0],
        dataset[1],
    ])

    assert input_ids.tolist() == [
        [
            vocab.sos_id,
            vocab.word2idx['i'],
            vocab.word2idx['love'],
            vocab.word2idx['you'],
        ],
        [
            vocab.sos_id,
            vocab.word2idx['hello'],
            vocab.pad_id,
            vocab.pad_id,
        ],
    ]

    assert targets.tolist() == [
        [
            vocab.word2idx['i'],
            vocab.word2idx['love'],
            vocab.word2idx['you'],
            vocab.eos_id,
        ],
        [
            vocab.word2idx['hello'],
            vocab.eos_id,
            vocab.pad_id,
            vocab.pad_id,
        ],
    ]

    assert input_ids.shape == targets.shape