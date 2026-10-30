import csv
import math


def test_train_main_writes_one_epoch_metrics(
    tmp_path,
):
    from seq2seq.train import main

    output_dir = tmp_path / 'run'

    returned_dir = main([
        '--epochs', '1',
        '--batch-size', '2',
        '--learning-rate', '0.01',
        '--d-model', '8',
        '--d-ff', '16',
        '--num-heads', '2',
        '--num-layers', '1',
        '--dropout', '0.0',
        '--max-samples', '2',
        '--device', 'cpu',
        '--output-dir', str(output_dir),
    ])

    assert returned_dir == output_dir

    metrics_path = (
        output_dir
        / 'metrics.csv'
    )

    assert metrics_path.exists()

    with metrics_path.open(
        'r',
        encoding='utf-8',
        newline='',
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert reader.fieldnames == [
        'epoch',
        'train_loss',
        'token_accuracy',
        'learning_rate',
    ]

    assert len(rows) == 1

    row = rows[0]

    assert int(row['epoch']) == 1

    train_loss = float(
        row['train_loss']
    )

    token_accuracy = float(
        row['token_accuracy']
    )

    learning_rate = float(
        row['learning_rate']
    )

    assert math.isfinite(train_loss)
    assert train_loss > 0

    assert 0.0 <= token_accuracy <= 1.0

    assert learning_rate == 0.01



    checkpoint_path = (
        output_dir
        / 'checkpoint.pt'
    )

    assert checkpoint_path.exists()

    from seq2seq.checkpoint import load_checkpoint

    (
        loaded_model,
        loaded_src_vocab,
        loaded_tgt_vocab,
        metadata,
    ) = load_checkpoint(
        checkpoint_path,
        device='cpu',
    )

    assert loaded_model.config.d_model == 8
    assert loaded_model.config.num_layers == 1

    assert len(loaded_src_vocab) > 4
    assert len(loaded_tgt_vocab) > 4

    assert metadata['training_args']['epochs'] == 1

