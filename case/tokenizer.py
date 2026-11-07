from pathlib import Path


def _sentencepiece():
    try:
        import sentencepiece as spm
    except ImportError as error:
        raise RuntimeError(
            'sentencepiece is required; run '
            '`python -m pip install -r case/requirements.txt`'
        ) from error
    return spm


class SentencePieceTokenizer:
    """Small adapter exposing the interface used by this project."""

    def __init__(self, model_file=None, serialized_model=None):
        spm = _sentencepiece()
        self.processor = spm.SentencePieceProcessor()

        if (model_file is None) == (serialized_model is None):
            raise ValueError(
                'provide exactly one of model_file or serialized_model'
            )

        if model_file is not None:
            loaded = self.processor.load(str(model_file))
        else:
            loaded = self.processor.load_from_serialized_proto(
                serialized_model
            )

        if not loaded:
            raise ValueError('failed to load SentencePiece model')

        self.pad_id = self.processor.pad_id()
        self.sos_id = self.processor.bos_id()
        self.eos_id = self.processor.eos_id()
        self.unk_id = self.processor.unk_id()

        expected = (0, 1, 2, 3)
        actual = (
            self.pad_id,
            self.sos_id,
            self.eos_id,
            self.unk_id,
        )
        if actual != expected:
            raise ValueError(
                'tokenizer special IDs must be '
                '(pad=0, bos=1, eos=2, unk=3), '
                f'got {actual}'
            )

    def __len__(self):
        return self.processor.get_piece_size()

    def encode(self, text, add_sos=False, add_eos=False):
        ids = self.processor.encode(text, out_type=int)
        if add_sos:
            ids.insert(0, self.sos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        clean_ids = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id == self.eos_id:
                break
            if token_id not in (self.pad_id, self.sos_id):
                clean_ids.append(token_id)
        return self.processor.decode(clean_ids)

    def serialized_model(self):
        return self.processor.serialized_model_proto()


def train_tokenizer(
    input_path,
    model_prefix,
    vocab_size=8000,
    model_type='unigram',
    character_coverage=0.9995,
):
    if vocab_size <= 4:
        raise ValueError('vocab_size must be greater than 4')

    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    model_prefix = Path(model_prefix)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    spm = _sentencepiece()
    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece='<pad>',
        bos_piece='<sos>',
        eos_piece='<eos>',
        unk_piece='<unk>',
        hard_vocab_limit=False,
        shuffle_input_sentence=True,
        input_sentence_size=1_000_000,
    )
    return model_prefix.with_suffix('.model')

