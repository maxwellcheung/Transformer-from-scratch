import argparse
import csv
import random
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import (
    TranslationDataset,
    make_collate_fn,
)
from .model import (
    Seq2SeqTransformer,
    TransformerConfig,
)
from .training import train_step
from .vocab import Vocab

from .checkpoint import save_checkpoint

DEFAULT_PAIRS = [
    ('i love you', '我 爱 你'),
    ('hello', '你 好'),
    ('i love cats', '我 爱 猫'),
    ('you love dogs', '你 爱 狗'),
    ('good morning', '早上 好'),
    ('thank you', '谢谢 你'),
]

def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=0.001)

    parser.add_argument('--d-model', type=int, default=32)
    parser.add_argument('--d-ff', type=int, default=64)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max-len', type=int, default=64)

    parser.add_argument('--max-samples', type=int, default=6)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-dir', default=None)

    return parser

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    positive_values = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_samples': args.max_samples,
        'max_grad_norm': args.max_grad_norm,
    }

    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(
                f'{name} must be positive, got {value}'
            )

    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise ValueError(
                'CUDA device requested but CUDA is unavailable'
            )

    device = torch.device(args.device)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    pairs = DEFAULT_PAIRS[:args.max_samples]

    # 创建词表和Dataset
    src_texts = [
        src_text for src_text, _ in pairs
    ]

    tgt_texts = [
        tgt_text for _, tgt_text in pairs
    ]

    src_vocab = Vocab(src_texts)
    tgt_vocab = Vocab(tgt_texts)

    dataset = TranslationDataset(
        pairs,
        src_vocab,
        tgt_vocab,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(
            src_vocab.pad_id,
            tgt_vocab.pad_id,
        ),
        generator=generator,
    )

    # 创建模型和优化器
    config = TransformerConfig(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        dropout=args.dropout,
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
        max_len=args.max_len,
    )

    model = Seq2SeqTransformer(config)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    # 创建实验目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime(
            '%Y%m%d_%H%M%S_%f'
        )

        output_dir = (
            Path('outputs')
            / f'seq2seq_{timestamp}'
        )
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    metrics_path = output_dir / 'metrics.csv'

    # 训练与指标聚合
    with metrics_path.open(
        'w',
        encoding='utf-8',
        newline='',
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                'epoch',
                'train_loss',
                'token_accuracy',
                'learning_rate',
            ],
        )

        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            total_loss = 0.0
            total_correct_tokens = 0
            total_valid_tokens = 0

            for batch in data_loader:
                metrics = train_step(
                    model,
                    batch,
                    optimizer,
                    pad_id=tgt_vocab.pad_id,
                    max_grad_norm=args.max_grad_norm,
                )

                # 每个 batch 的 loss 是有效 token 的平均值。
                # 乘回 token 数，得到该 batch 的 loss 总和。
                total_loss += (
                    metrics['loss']
                    * metrics['valid_tokens']
                )

                total_correct_tokens += (
                    metrics['correct_tokens']
                )

                total_valid_tokens += (
                    metrics['valid_tokens']
                )

            train_loss = (
                total_loss
                / total_valid_tokens
            )

            token_accuracy = (
                total_correct_tokens
                / total_valid_tokens
            )

            learning_rate = (
                optimizer.param_groups[0]['lr']
            )

            writer.writerow({
                'epoch': epoch,
                'train_loss': train_loss,
                'token_accuracy': token_accuracy,
                'learning_rate': learning_rate,
            })

    # 保存checkpoint
    training_args = dict(
        vars(args)
    )

    save_checkpoint(
        output_dir / 'checkpoint.pt',
        model,
        src_vocab,
        tgt_vocab,
        training_args,
    )


    return output_dir

if __name__ == '__main__':
    main()