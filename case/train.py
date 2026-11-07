import argparse
import csv
import math
import random
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from seq2seq.model import Seq2SeqTransformer, TransformerConfig
from seq2seq.training import token_accuracy, token_cross_entropy

from .checkpoint import save_checkpoint
from .data import ParallelDataset, make_collate_fn
from .tokenizer import SentencePieceTokenizer, train_tokenizer
from .tracking import init_swanlab


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device, precision):
    if device.type != 'cuda' or precision == 'fp32':
        return nullcontext()
    dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    return torch.autocast(device_type='cuda', dtype=dtype)


def run_epoch(
    model,
    loader,
    pad_id,
    device,
    precision,
    optimizer=None,
    scheduler=None,
    scaler=None,
    max_grad_norm=1.0,
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    grad_context = nullcontext() if training else torch.no_grad()
    with grad_context:
        for src, tgt_input, tgt_output in loader:
            src = src.to(device, non_blocking=True)
            tgt_input = tgt_input.to(device, non_blocking=True)
            tgt_output = tgt_output.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with autocast_context(device, precision):
                logits = model(src, tgt_input)
                loss = token_cross_entropy(logits, tgt_output, pad_id)

            correct, valid = token_accuracy(logits, tgt_output, pad_id)

            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item() * valid
            total_correct += correct
            total_tokens += valid

    return {
        'loss': total_loss / total_tokens,
        'accuracy': total_correct / total_tokens,
    }


def build_scheduler(optimizer, warmup_steps, total_steps):
    def multiplier(step):
        current = step + 1
        if warmup_steps > 0 and current <= warmup_steps:
            return current / warmup_steps
        progress = (current - warmup_steps) / max(
            1, total_steps - warmup_steps
        )
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Train the hand-written Transformer on a real corpus.'
    )
    parser.add_argument('--data-dir', default='case/data/opus100_en_zh')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--src-lang', default='zh')
    parser.add_argument('--tgt-lang', default='en')
    parser.add_argument('--vocab-size', type=int, default=8000)
    parser.add_argument('--max-len', type=int, default=64)
    parser.add_argument('--max-train-samples', type=int)
    parser.add_argument('--max-valid-samples', type=int)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-steps', type=int, default=400)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--num-workers', type=int, default=4)

    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--d-ff', type=int, default=1024)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--precision', choices=('fp32', 'bf16', 'fp16'), default='bf16'
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--swanlab', action='store_true',
        help='record this run with SwanLab',
    )
    parser.add_argument(
        '--swanlab-project', default='transformer-from-scratch',
    )
    parser.add_argument('--swanlab-experiment-name')
    parser.add_argument('--swanlab-workspace')
    parser.add_argument(
        '--swanlab-mode',
        choices=('online', 'offline', 'local', 'disabled'),
        default='online',
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError('epochs and batch-size must be positive')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')

    set_seed(args.seed)
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer_dir = output_dir / 'tokenizers'

    src_model = train_tokenizer(
        data_dir / f'train.{args.src_lang}',
        tokenizer_dir / args.src_lang,
        vocab_size=args.vocab_size,
    )
    tgt_model = train_tokenizer(
        data_dir / f'train.{args.tgt_lang}',
        tokenizer_dir / args.tgt_lang,
        vocab_size=args.vocab_size,
    )
    src_tokenizer = SentencePieceTokenizer(model_file=src_model)
    tgt_tokenizer = SentencePieceTokenizer(model_file=tgt_model)

    train_dataset = ParallelDataset(
        data_dir / f'train.{args.src_lang}',
        data_dir / f'train.{args.tgt_lang}',
        src_tokenizer,
        tgt_tokenizer,
        args.max_len,
        args.max_train_samples,
    )
    valid_dataset = ParallelDataset(
        data_dir / f'dev.{args.src_lang}',
        data_dir / f'dev.{args.tgt_lang}',
        src_tokenizer,
        tgt_tokenizer,
        args.max_len,
        args.max_valid_samples,
    )
    print(
        f'train={len(train_dataset)} (skipped={train_dataset.skipped_too_long}), '
        f'valid={len(valid_dataset)} (skipped={valid_dataset.skipped_too_long})'
    )

    collate = make_collate_fn(src_tokenizer.pad_id, tgt_tokenizer.pad_id)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=args.num_workers > 0,
    )

    config = TransformerConfig(
        src_vocab_size=len(src_tokenizer),
        tgt_vocab_size=len(tgt_tokenizer),
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        dropout=args.dropout,
        src_pad_id=src_tokenizer.pad_id,
        tgt_pad_id=tgt_tokenizer.pad_id,
        max_len=args.max_len,
    )
    model = Seq2SeqTransformer(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = build_scheduler(optimizer, args.warmup_steps, total_steps)
    scaler = None
    if device.type == 'cuda' and args.precision == 'fp16':
        scaler = torch.amp.GradScaler('cuda')


    swanlab_run = init_swanlab(
        args,
        output_dir,
        train_dataset,
        valid_dataset,
        model,
    )
    metrics_path = output_dir / 'metrics.csv'
    best_loss = float('inf')
    with metrics_path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=[
            'epoch', 'train_loss', 'train_accuracy',
            'valid_loss', 'valid_accuracy', 'learning_rate',
        ])
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model, train_loader, tgt_tokenizer.pad_id, device,
                args.precision, optimizer, scheduler, scaler,
                args.max_grad_norm,
            )
            valid_metrics = run_epoch(
                model, valid_loader, tgt_tokenizer.pad_id, device,
                args.precision,
            )
            learning_rate = optimizer.param_groups[0]['lr']
            row = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'valid_loss': valid_metrics['loss'],
                'valid_accuracy': valid_metrics['accuracy'],
                'learning_rate': learning_rate,
            }
            writer.writerow(row)
            file.flush()
            print(
                f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} "
                f"valid_loss={row['valid_loss']:.4f} "
                f"valid_acc={row['valid_accuracy']:.4f} lr={learning_rate:.2e}"
            )

            if valid_metrics['loss'] < best_loss:
                best_loss = valid_metrics['loss']
                save_checkpoint(
                    output_dir / 'best.pt',
                    model,
                    src_tokenizer,
                    tgt_tokenizer,
                    vars(args),
                    epoch,
                    best_loss,
                )

            if swanlab_run is not None:
                swanlab_run.log({
                    'train/loss': row['train_loss'],
                    'train/token_accuracy': row['train_accuracy'],
                    'validation/loss': row['valid_loss'],
                    'validation/token_accuracy': row['valid_accuracy'],
                    'optimization/learning_rate': learning_rate,
                    'best/validation_loss': best_loss,
                }, step=epoch)

    save_checkpoint(

        output_dir / 'last.pt',
        model,
        src_tokenizer,
        tgt_tokenizer,
        vars(args),
        args.epochs,
        valid_metrics['loss'],
    )
    print(f'Best validation loss: {best_loss:.4f}')
    print(f'Checkpoint: {output_dir / "best.pt"}')
    if swanlab_run is not None:
        swanlab_run.finish()
    return output_dir


if __name__ == '__main__':
    main()

