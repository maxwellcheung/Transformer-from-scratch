from pathlib import Path

import torch

from seq2seq.model import Seq2SeqTransformer, TransformerConfig

from .tokenizer import SentencePieceTokenizer


def save_checkpoint(
    path,
    model,
    src_tokenizer,
    tgt_tokenizer,
    training_args,
    epoch,
    validation_loss,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'format_version': 1,
        'model_config': dict(vars(model.config)),
        'model_state_dict': model.state_dict(),
        'src_tokenizer_model': src_tokenizer.serialized_model(),
        'tgt_tokenizer_model': tgt_tokenizer.serialized_model(),
        'training_args': dict(training_args),
        'epoch': epoch,
        'validation_loss': validation_loss,
    }, path)


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(
        Path(path),
        map_location=device,
        weights_only=True,
    )
    if checkpoint.get('format_version') != 1:
        raise ValueError('unsupported case checkpoint format')

    config = TransformerConfig(**checkpoint['model_config'])
    model = Seq2SeqTransformer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    src_tokenizer = SentencePieceTokenizer(
        serialized_model=checkpoint['src_tokenizer_model']
    )
    tgt_tokenizer = SentencePieceTokenizer(
        serialized_model=checkpoint['tgt_tokenizer_model']
    )
    metadata = {
        'training_args': checkpoint['training_args'],
        'epoch': checkpoint['epoch'],
        'validation_loss': checkpoint['validation_loss'],
    }
    return model, src_tokenizer, tgt_tokenizer, metadata

