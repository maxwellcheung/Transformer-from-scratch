from pathlib import Path

import pytest
import torch


def write_lines(path, lines):
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def test_parallel_reader_rejects_misalignment(tmp_path):
    from case.corpus import iter_parallel

    src = tmp_path / 'a.zh'
    tgt = tmp_path / 'a.en'
    write_lines(src, ['一', '二'])
    write_lines(tgt, ['one'])
    with pytest.raises(ValueError, match='different line counts'):
        list(iter_parallel(src, tgt))


def test_sentencepiece_dataset_and_checkpoint_round_trip(tmp_path):
    pytest.importorskip('sentencepiece')
    from case.checkpoint import load_checkpoint, save_checkpoint
    from case.data import ParallelDataset, make_collate_fn
    from case.tokenizer import SentencePieceTokenizer, train_tokenizer
    from seq2seq.model import Seq2SeqTransformer, TransformerConfig

    src = tmp_path / 'train.zh'
    tgt = tmp_path / 'train.en'
    write_lines(src, ['我爱你', '你好世界', '我喜欢猫'] * 10)
    write_lines(tgt, ['i love you', 'hello world', 'i like cats'] * 10)
    src_model = train_tokenizer(src, tmp_path / 'zh', vocab_size=32)
    tgt_model = train_tokenizer(tgt, tmp_path / 'en', vocab_size=32)
    src_tok = SentencePieceTokenizer(model_file=src_model)
    tgt_tok = SentencePieceTokenizer(model_file=tgt_model)

    dataset = ParallelDataset(src, tgt, src_tok, tgt_tok, max_len=32)
    batch = make_collate_fn(src_tok.pad_id, tgt_tok.pad_id)([
        dataset[0], dataset[1]
    ])
    assert batch[0].dtype == torch.long
    assert batch[1].shape == batch[2].shape

    config = TransformerConfig(
        len(src_tok), len(tgt_tok), 8, 2, 16, 1, 0.0,
        src_tok.pad_id, tgt_tok.pad_id, 32,
    )
    model = Seq2SeqTransformer(config)
    before = model(batch[0], batch[1])
    path = tmp_path / 'model.pt'
    save_checkpoint(path, model, src_tok, tgt_tok, {}, 1, 1.0)
    loaded, loaded_src, loaded_tgt, metadata = load_checkpoint(path)
    after = loaded(batch[0], batch[1])

    assert torch.equal(before, after)
    assert loaded_src.decode(loaded_src.encode('我爱你')) == '我爱你'
    assert loaded_tgt.decode(loaded_tgt.encode('i love you')) == 'i love you'
    assert metadata['epoch'] == 1


def test_train_evaluate_and_translate_end_to_end(tmp_path):
    pytest.importorskip('sentencepiece')
    pytest.importorskip('sacrebleu')
    pytest.importorskip('swanlab')
    from case.evaluate import main as evaluate_main
    from case.train import main as train_main
    from case.translate import main as translate_main

    data = tmp_path / 'data'
    data.mkdir()
    chinese = ['我爱你', '你好世界', '我喜欢猫', '谢谢你'] * 8
    english = ['i love you', 'hello world', 'i like cats', 'thank you'] * 8
    for split in ('train', 'dev', 'test'):
        write_lines(data / f'{split}.zh', chinese)
        write_lines(data / f'{split}.en', english)

    output = tmp_path / 'run'
    train_main([
        '--data-dir', str(data),
        '--output-dir', str(output),
        '--device', 'cpu',
        '--precision', 'fp32',
        '--epochs', '1',
        '--batch-size', '8',
        '--vocab-size', '32',
        '--d-model', '8',
        '--d-ff', '16',
        '--num-heads', '2',
        '--num-layers', '1',
        '--dropout', '0.0',
        '--warmup-steps', '1',
        '--num-workers', '0',
        '--swanlab',
        '--swanlab-mode', 'offline',
        '--swanlab-project', 'transformer-case-test',
        '--swanlab-experiment-name', 'offline-smoke-test',
    ])
    checkpoint = output / 'best.pt'
    assert checkpoint.is_file()

    assert (output / 'swanlog').is_dir()
    scores = evaluate_main([
        '--checkpoint', str(checkpoint),
        '--data-dir', str(data),
        '--device', 'cpu',
        '--batch-size', '8',
        '--max-new-tokens', '8',
        '--output', str(output / 'predictions.tsv'),
    ])
    assert all(torch.isfinite(torch.tensor(score)) for score in scores)
    assert (output / 'predictions.tsv').is_file()

    translate_main([
        '--checkpoint', str(checkpoint),
        '--device', 'cpu',
        '--max-new-tokens', '8',
        '--text', '我爱你',
    ])

