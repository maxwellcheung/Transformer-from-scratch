import argparse

import torch

from seq2seq.inference import greedy_decode

from .checkpoint import load_checkpoint


def translate(model, src_tokenizer, tgt_tokenizer, text, max_new_tokens):
    src_ids = src_tokenizer.encode(text)
    if not src_ids:
        raise ValueError('input text contains no tokens')
    if len(src_ids) > model.config.max_len:
        raise ValueError(
            f'input has {len(src_ids)} tokens, model limit is '
            f'{model.config.max_len}'
        )
    src = torch.tensor([src_ids], dtype=torch.long)
    generated = greedy_decode(
        model,
        src,
        tgt_tokenizer.sos_id,
        tgt_tokenizer.eos_id,
        min(max_new_tokens, model.config.max_len - 1),
    )
    return tgt_tokenizer.decode(generated[0].cpu().tolist())


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--text')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--device', default='cuda:0')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    model, src_tokenizer, tgt_tokenizer, _ = load_checkpoint(
        args.checkpoint, args.device
    )

    if args.text is not None:
        print(translate(
            model, src_tokenizer, tgt_tokenizer,
            args.text, args.max_new_tokens,
        ))
        return

    print('Interactive Chinese-to-English translation. Empty input exits.')
    while True:
        try:
            text = input('中文> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            break
        print('English>', translate(
            model, src_tokenizer, tgt_tokenizer,
            text, args.max_new_tokens,
        ))


if __name__ == '__main__':
    main()

