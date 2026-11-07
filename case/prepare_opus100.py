import argparse
import tarfile
import urllib.request
from pathlib import Path

from .corpus import (
    clean_pairs,
    deterministic_sample,
    iter_parallel,
    write_parallel,
)


OPUS_URL = (
    'https://object.pouta.csc.fi/OPUS-100/v1.0/'
    'opus-100-corpus-en-zh-v1.0.tar.gz'
)


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        print(f'Using cached archive: {destination}')
        return destination

    temporary = destination.with_suffix(destination.suffix + '.part')
    print(f'Downloading {url}')
    try:
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    marker = destination / '.complete'
    if marker.is_file():
        return destination

    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f'unsafe archive member: {member.name}')
        tar.extractall(destination, filter='data')
    marker.touch()
    return destination


def find_split_file(root, split, language):
    candidates = list(Path(root).rglob(f'*{split}.{language}'))
    candidates += list(Path(root).rglob(f'*.{split}.{language}'))
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f'expected one {split}.{language} file, found {candidates}'
        )
    return candidates[0]


def prepare_split(
    extracted_dir,
    output_dir,
    split,
    max_samples,
    seed,
    max_chars,
    max_length_ratio,
):
    src_input = find_split_file(extracted_dir, split, 'zh')
    tgt_input = find_split_file(extracted_dir, split, 'en')
    pairs, stats = clean_pairs(
        iter_parallel(src_input, tgt_input),
        max_chars=max_chars,
        max_length_ratio=max_length_ratio,
    )
    pairs = deterministic_sample(pairs, max_samples, seed)
    count = write_parallel(
        pairs,
        Path(output_dir) / f'{split}.zh',
        Path(output_dir) / f'{split}.en',
    )
    print(f'{split}: wrote {count} pairs; cleaning stats={stats}')


def build_parser():
    parser = argparse.ArgumentParser(
        description='Download and prepare OPUS-100 Chinese-English data.'
    )
    parser.add_argument('--output-dir', default='case/data/opus100_en_zh')
    parser.add_argument('--cache-dir', default='case/data/downloads')
    parser.add_argument('--max-train-samples', type=int, default=10_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-chars', type=int, default=500)
    parser.add_argument('--max-length-ratio', type=float, default=8.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.max_train_samples < 0:
        raise ValueError('max-train-samples must be non-negative')
    max_train_samples = args.max_train_samples or None
    archive = download(
        OPUS_URL,
        Path(args.cache_dir) / 'opus-100-corpus-en-zh-v1.0.tar.gz',
    )
    extracted = safe_extract(
        archive,
        Path(args.cache_dir) / 'opus-100-en-zh',
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, max_samples in (
        ('train', max_train_samples),
        ('dev', None),
        ('test', None),
    ):
        prepare_split(
            extracted,
            output_dir,
            split,
            max_samples,
            args.seed,
            args.max_chars,
            args.max_length_ratio,
        )
    return output_dir


if __name__ == '__main__':
    main()

