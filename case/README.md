# Real Chinese-to-English training case

Run every command from `from_scratch/`.

## 1. Install the additional dependencies

```bash
/root/miniconda3/envs/torch310/bin/python -m pip install -r case/requirements.txt
```

## 2. Download, clean and sample OPUS-100

The first run uses 10,000 training pairs. Development and test sets remain intact.

```bash
/root/miniconda3/envs/torch310/bin/python -m case.prepare_opus100 \
  --output-dir case/data/opus100_en_zh \
  --max-train-samples 10000
```

## 3. Train

```bash
/root/miniconda3/envs/torch310/bin/python -m case.train \
  --data-dir case/data/opus100_en_zh \
  --output-dir case/outputs/opus10k_run01 \
  --device cuda:0 \
  --precision bf16 \
  --epochs 20 \
  --batch-size 64
```

Training creates `metrics.csv`, `best.pt`, `last.pt`, and the two tokenizer files.
The checkpoint embeds both tokenizer models and can therefore be moved alone.

## 4. Evaluate on the held-out test set

```bash
/root/miniconda3/envs/torch310/bin/python -m case.evaluate \
  --checkpoint case/outputs/opus10k_run01/best.pt \
  --data-dir case/data/opus100_en_zh \
  --device cuda:0 \
  --output case/outputs/opus10k_run01/test_predictions.tsv
```

## 5. Translate

One sentence:

```bash
/root/miniconda3/envs/torch310/bin/python -m case.translate \
  --checkpoint case/outputs/opus10k_run01/best.pt \
  --device cuda:0 \
  --text '我爱你。'
```

Interactive mode (omit `--text`):

```bash
/root/miniconda3/envs/torch310/bin/python -m case.translate \
  --checkpoint case/outputs/opus10k_run01/best.pt \
  --device cuda:0
```

## Scale to more data

Prepare a fresh directory with `--max-train-samples 100000`, then use a fresh
training output directory. To retain every available training pair, pass
`--max-train-samples 0`.


## SwanLab experiment tracking

Log in once for cloud tracking:

```bash
/root/miniconda3/envs/torch310/bin/swanlab login
```

Enable tracking by adding these options to `case.train`:

```text
--swanlab
--swanlab-project transformer-from-scratch
--swanlab-experiment-name opus100k-run01
--swanlab-mode online
```

`online` synchronizes to swanlab.cn. Use `--swanlab-mode offline` when the
machine cannot access the cloud; logs are written under the run's `swanlog/`
directory. Training still writes `metrics.csv` independently.
