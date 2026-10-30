import torch
import pytest

def test_greedy_decode_stops_at_eos_and_restores_mode():
    from seq2seq.inference import greedy_decode
    from seq2seq.model import (
        Seq2SeqTransformer,
        TransformerConfig,
    )

    class CountingTransformer(Seq2SeqTransformer):
        def __init__(self, config):
            super().__init__(config)
            self.encode_calls = 0

        def encode(self, src_tokens):
            self.encode_calls += 1
            return super().encode(src_tokens)

    config = TransformerConfig(
        src_vocab_size=10,
        tgt_vocab_size=10,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=1,
        dropout=0.0,
        src_pad_id=0,
        tgt_pad_id=0,
        max_len=16,
    )

    model = CountingTransformer(config)

    sos_id = 1
    eos_id = 2

    # 强制输出层始终选择 EOS。
    with torch.no_grad():
        model.output_projection.weight.zero_()
        model.output_projection.bias.zero_()
        model.output_projection.bias[eos_id] = 10.0

    src_tokens = torch.tensor([
        [4, 5, 0],
        [6, 7, 8],
    ])

    # 验证函数结束后能够恢复训练模式。
    model.train()

    generated = greedy_decode(
        model,
        src_tokens,
        sos_id=sos_id,
        eos_id=eos_id,
        max_new_tokens=5,
    )

    expected = torch.tensor([
        [sos_id, eos_id],
        [sos_id, eos_id],
    ])

    torch.testing.assert_close(
        generated.cpu(),
        expected,
    )

    assert model.encode_calls == 1
    assert model.training is True


def test_checkpoint_round_trip_preserves_logits_and_vocabs(
    tmp_path,
):
    from seq2seq.checkpoint import (
        load_checkpoint,
        save_checkpoint,
    )
    from seq2seq.model import (
        Seq2SeqTransformer,
        TransformerConfig,
    )
    from seq2seq.vocab import Vocab

    torch.manual_seed(0)

    src_vocab = Vocab([
        'i love you',
        'hello',
    ])

    tgt_vocab = Vocab([
        '我 爱 你',
        '你 好',
    ])

    config = TransformerConfig(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=1,
        dropout=0.0,
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
        max_len=16,
    )

    model = Seq2SeqTransformer(config)
    model.eval()

    src_tokens = torch.tensor([
        src_vocab.encode('i love you'),
    ])

    tgt_tokens = torch.tensor([
        tgt_vocab.encode(
            '我 爱',
            add_sos=True,
        ),
    ])

    with torch.no_grad():
        expected_logits = model(
            src_tokens,
            tgt_tokens,
        )

    checkpoint_path = (
        tmp_path
        / 'model.pt'
    )

    training_args = {
        'epoch': 3,
        'learning_rate': 0.001,
    }

    save_checkpoint(
        checkpoint_path,
        model,
        src_vocab,
        tgt_vocab,
        training_args,
    )

    (
        loaded_model,
        loaded_src_vocab,
        loaded_tgt_vocab,
        metadata,
    ) = load_checkpoint(
        checkpoint_path,
        device='cpu',
    )

    loaded_model.eval()

    with torch.no_grad():
        actual_logits = loaded_model(
            src_tokens,
            tgt_tokens,
        )

    torch.testing.assert_close(
        actual_logits,
        expected_logits,
    )

    assert (
        loaded_src_vocab.idx2word
        == src_vocab.idx2word
    )

    assert (
        loaded_tgt_vocab.idx2word
        == tgt_vocab.idx2word
    )

    assert metadata['training_args'] == training_args


def make_valid_checkpoint(path):
    from seq2seq.checkpoint import save_checkpoint
    from seq2seq.model import (
        Seq2SeqTransformer,
        TransformerConfig,
    )
    from seq2seq.vocab import Vocab

    src_vocab = Vocab([
        'i love you',
        'hello',
    ])

    tgt_vocab = Vocab([
        '我 爱 你',
        '你 好',
    ])

    config = TransformerConfig(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=1,
        dropout=0.0,
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
        max_len=16,
    )

    model = Seq2SeqTransformer(config)

    save_checkpoint(
        path,
        model,
        src_vocab,
        tgt_vocab,
        {'epoch': 1},
    )

@pytest.mark.parametrize(
    'missing_key',
    [
        'model_config',
        'src_vocab_tokens',
        'tgt_vocab_tokens',
    ],
)
def test_load_checkpoint_rejects_missing_keys(
    tmp_path,
    missing_key,
):
    from seq2seq.checkpoint import load_checkpoint

    checkpoint_path = tmp_path / 'model.pt'

    make_valid_checkpoint(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location='cpu',
        weights_only=True,
    )

    checkpoint.pop(missing_key)

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    with pytest.raises(
        ValueError,
        match=missing_key,
    ):
        load_checkpoint(
            checkpoint_path,
            device='cpu',
        )

def test_load_checkpoint_rejects_mismatched_weights(
    tmp_path,
):
    from seq2seq.checkpoint import load_checkpoint

    checkpoint_path = tmp_path / 'model.pt'

    make_valid_checkpoint(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location='cpu',
        weights_only=True,
    )

    # 原权重使用 d_model=8，
    # 但现在要求按照 d_model=12 重建模型。
    checkpoint['model_config']['d_model'] = 12

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    with pytest.raises(
        ValueError,
        match='weights.*do not match',
    ):
        load_checkpoint(
            checkpoint_path,
            device='cpu',
        )