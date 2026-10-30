from pathlib import Path

import torch

from .model import (
    Seq2SeqTransformer,
    TransformerConfig,
)
from .vocab import Vocab

def save_checkpoint(
    path,
    model,
    src_vocab,
    tgt_vocab,
    training_args,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        'model_config': dict(
            vars(model.config)
        ),
        'model_state_dict': model.state_dict(),
        'src_vocab_tokens': list(
            src_vocab.idx2word
        ),
        'tgt_vocab_tokens': list(
            tgt_vocab.idx2word
        ),
        'training_args': dict(
            training_args
        ),
    }

    torch.save(
        checkpoint,
        path,
    )

def load_checkpoint(path, device):
    path = Path(path)

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    required_keys = {
        'model_config',
        'model_state_dict',
        'src_vocab_tokens',
        'tgt_vocab_tokens',
        'training_args',
    }

    missing_keys = (
        required_keys
        - checkpoint.keys()
    )

    if missing_keys:
        raise ValueError(
            'checkpoint is missing keys: '
            f'{sorted(missing_keys)}'
        )

    config = TransformerConfig(
        **checkpoint['model_config']
    )

    model = Seq2SeqTransformer(config)
    model = model.to(device)

    try:
        model.load_state_dict(
            checkpoint['model_state_dict']
        )
    except RuntimeError as error:
        raise ValueError(
            'checkpoint model weights '
            'do not match model_config'
        ) from error

    src_vocab = Vocab.from_idx2word(
        checkpoint['src_vocab_tokens']
    )

    tgt_vocab = Vocab.from_idx2word(
        checkpoint['tgt_vocab_tokens']
    )

    metadata = {
        'training_args': checkpoint[
            'training_args'
        ],
    }

    return (
        model,
        src_vocab,
        tgt_vocab,
        metadata,
    )