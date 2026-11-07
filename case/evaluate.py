import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from seq2seq.inference import greedy_decode

from .checkpoint import load_checkpoint
from .data import ParallelDataset, make_collate_fn


def load_sacrebleu():
    try:
        from sacrebleu.metrics import BLEU, CHRF
    except ImportError as error:
        raise RuntimeError(
            'sacrebleu is required; run '
            '`python -m pip install -r case/requirements.txt`'
        ) from error
    return BLEU, CHRF


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', default='case/data/opus100_en_zh')
    parser.add_argument('--src-lang', default='zh')
    parser.add_argument('--tgt-lang', default='en')
    parser.add_argument('--split', default='test')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='predictions.tsv')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    model, src_tokenizer, tgt_tokenizer, metadata = load_checkpoint(
        args.checkpoint, args.device
    )
    data_dir = Path(args.data_dir)
    dataset = ParallelDataset(
        data_dir / f'{args.split}.{args.src_lang}',
        data_dir / f'{args.split}.{args.tgt_lang}',
        src_tokenizer,
        tgt_tokenizer,
        model.config.max_len,
        args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(
            src_tokenizer.pad_id,
            tgt_tokenizer.pad_id,
            include_text=True,
        ),
    )

    sources = []
    references = []
    hypotheses = []
    for src, _, _, src_texts, tgt_texts in loader:
        generated = greedy_decode(
            model,
            src,
            tgt_tokenizer.sos_id,
            tgt_tokenizer.eos_id,
            min(args.max_new_tokens, model.config.max_len - 1),
        )
        hypotheses.extend(
            tgt_tokenizer.decode(row.tolist()) for row in generated.cpu()
        )
        sources.extend(src_texts)
        references.extend(tgt_texts)

    BLEU, CHRF = load_sacrebleu()
    bleu = BLEU(tokenize='13a').corpus_score(hypotheses, [references])
    chrf = CHRF().corpus_score(hypotheses, [references])
    print(f'checkpoint_epoch={metadata["epoch"]}')
    print(f'BLEU={bleu.score:.2f}')
    print(f'chrF={chrf.score:.2f}')

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['source', 'reference', 'prediction'])
        writer.writerows(zip(sources, references, hypotheses))
    print(f'Predictions: {output}')
    return bleu.score, chrf.score


if __name__ == '__main__':
    main()

