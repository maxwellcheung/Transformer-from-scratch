from collections import Counter


class Vocab:

    SPECIAL_TOKENS = ('<pad>', '<sos>', '<eos>', '<unk>')

    def __init__(self, texts):
        counts = Counter()
        first_seen = {}
        for text in texts:
            for token in text.split():
                counts[token] += 1
                first_seen.setdefault(token, len(first_seen))

        if not counts:
            raise ValueError('texts must contain at least one token')

        tokens = sorted(
            counts,
            key=lambda token: (-counts[token], first_seen[token])
        )

        self.idx2word = list(self.SPECIAL_TOKENS) + tokens
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}

        self.pad_id = self.word2idx['<pad>']
        self.sos_id = self.word2idx['<sos>']
        self.eos_id = self.word2idx['<eos>']
        self.unk_id = self.word2idx['<unk>']

    def __len__(self):
        return len(self.idx2word)

    def encode(self, text, add_sos=False, add_eos=False):
        # 把空格分词的文本转换成id列表
        ids = [self.word2idx.get(token, self.unk_id) for token in text.split()]
        if add_sos:
            ids.insert(0, self.sos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        # 把id转换成文本，遇到 EOS 立即停止；不显示特殊 token
        words = []
        for token_id in ids:
            if not 0 <= token_id < len(self.idx2word):
                raise ValueError(f'out of range token id: {token_id}')
            word = self.idx2word[token_id]
            if word == '<eos>':
                break
            if word not in {'<sos>', '<pad>'}:
                words.append(word)
        return " ".join(words)

    @classmethod
    def from_idx2word(cls, idx2word):
        tokens = list(idx2word)

        if (
            len(tokens) < len(cls.SPECIAL_TOKENS)
            or tuple(tokens[:4]) != cls.SPECIAL_TOKENS
        ):
            raise ValueError(
                'vocab state must start with '
                f'{cls.SPECIAL_TOKENS}'
            )

        if len(set(tokens)) != len(tokens):
            raise ValueError(
                'vocab state contains duplicate tokens'
            )

        vocab = cls.__new__(cls)

        vocab.idx2word = tokens

        vocab.word2idx = {
            word: index
            for index, word in enumerate(tokens)
        }

        vocab.pad_id = vocab.word2idx['<pad>']
        vocab.sos_id = vocab.word2idx['<sos>']
        vocab.eos_id = vocab.word2idx['<eos>']
        vocab.unk_id = vocab.word2idx['<unk>']

        return vocab