import random
import re
import unicodedata
from pathlib import Path


WHITESPACE = re.compile(r'\s+')


def normalize_text(text):
    text = unicodedata.normalize('NFKC', text)
    return WHITESPACE.sub(' ', text).strip()


def iter_parallel(src_path, tgt_path):
    src_path = Path(src_path)
    tgt_path = Path(tgt_path)

    with src_path.open('r', encoding='utf-8') as src_file, \
            tgt_path.open('r', encoding='utf-8') as tgt_file:
        line_number = 0
        while True:
            src_line = src_file.readline()
            tgt_line = tgt_file.readline()

            if not src_line and not tgt_line:
                break

            line_number += 1
            if not src_line or not tgt_line:
                raise ValueError(
                    'parallel files have different line counts; '
                    f'first mismatch at line {line_number}'
                )

            yield normalize_text(src_line), normalize_text(tgt_line)


def clean_pairs(
    pairs,
    max_chars=500,
    max_length_ratio=8.0,
    deduplicate=True,
):
    seen = set()
    stats = {
        'read': 0,
        'kept': 0,
        'empty': 0,
        'too_long': 0,
        'bad_ratio': 0,
        'duplicate': 0,
    }
    cleaned = []

    for src_text, tgt_text in pairs:
        stats['read'] += 1

        if not src_text or not tgt_text:
            stats['empty'] += 1
            continue

        if len(src_text) > max_chars or len(tgt_text) > max_chars:
            stats['too_long'] += 1
            continue

        ratio = max(len(src_text), len(tgt_text)) / max(
            1, min(len(src_text), len(tgt_text))
        )
        if ratio > max_length_ratio:
            stats['bad_ratio'] += 1
            continue

        pair = (src_text, tgt_text)
        if deduplicate and pair in seen:
            stats['duplicate'] += 1
            continue

        seen.add(pair)
        cleaned.append(pair)
        stats['kept'] += 1

    return cleaned, stats


def deterministic_sample(pairs, max_samples, seed):
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    if max_samples is not None:
        pairs = pairs[:max_samples]
    return pairs


def write_parallel(pairs, src_path, tgt_path):
    src_path = Path(src_path)
    tgt_path = Path(tgt_path)
    src_path.parent.mkdir(parents=True, exist_ok=True)
    tgt_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with src_path.open('w', encoding='utf-8') as src_file, \
            tgt_path.open('w', encoding='utf-8') as tgt_file:
        for src_text, tgt_text in pairs:
            src_file.write(src_text + '\n')
            tgt_file.write(tgt_text + '\n')
            count += 1
    return count


def read_parallel(src_path, tgt_path, max_samples=None):
    pairs = list(iter_parallel(src_path, tgt_path))
    if max_samples is not None:
        pairs = pairs[:max_samples]
    if not pairs:
        raise ValueError('parallel corpus contains no sentence pairs')
    return pairs

