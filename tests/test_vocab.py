import pytest
from seq2seq.vocab import Vocab

# 测试token与词表一致性
def test_vocab_builds_and_encodes_with_same_tokenization():
    vocab = Vocab(['i love you', 'i love cats'])

    ids = vocab.encode(
        'i love birds',
        add_sos=True,
        add_eos=True,
    )

    assert vocab.decode(ids) == 'i love <unk>'

# 测试仅四个特殊token的词表
def test_vocab_rejects_corpus_without_tokens():
    with pytest.raises(ValueError, match='token'):
        Vocab([" ", "\t"])

def test_decode_rejects_out_of_range_ids():
    vocab = Vocab(["i love you"])

    for invalid_id in (-1, len(vocab)):
        with pytest.raises(
            ValueError,
            match=f"token id: {invalid_id}",
        ):
            vocab.decode([invalid_id])

def test_encode_default_to_no_boundary_tokens():
    vocab = Vocab(['i love you'])
    ids = vocab.encode('i love')

    assert ids == [
        vocab.word2idx['i'],
        vocab.word2idx['love'],
    ]