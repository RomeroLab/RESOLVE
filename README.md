# RESOLVE

One CSV and one schedule produce a 64-label protein campaign. Our method is ESM-2 keep/tune. The same contract also runs the ten external baselines.

A campaign always labels 64 sequences. The first 16 are a shared initial set, drawn from the CSV path and the seed. The other 48 are chosen by the schedule.

| Schedule | Rounds | Per round |
|---|---:|---:|
| 2x24 | 2 | 24 |
| 3x16 | 3 | 16 |
| 4x12 | 4 | 12 |
| 6x8 | 6 | 8 |
| 8x6 | 8 | 6 |
| 12x4 | 12 | 4 |

The endpoint is `hits_at_64`: how many of the registered elites are among those 64 labels. A tie with the best baseline is not a win for our method. Recall is that count divided by the elite count.

## Layout

```
RESOLVE/
  README.md                 this file
  pyproject.toml
  scripts/setup_conda.sh    create a fresh conda env named resolve
  examples/test_input.csv   a small input file in the required shape
  src/resolve_method/
    cli.py                  command line
    data.py                 CSV loading
    scores.py               rank targets, the shared initial 16, elite count
    kernel.py               one Gaussian world on ESM-2
    pool.py                 the ten worlds and how their probabilities mix
    policy.py               leave the frozen top only when two futures agree
    lora.py                 ESM-2 650M and the round LoRA
    campaign.py             the keep/tune loop
    linear_head.py          the Bayesian linear head shared by several baselines
    baselines.py            the ten external methods
```

## Environment

The setup script installs Python 3.13, torch 2.11.0 (CUDA 12.8), transformers 4.57.6, peft 0.20.0, scikit-learn 1.9.0, numpy, and scipy. It creates a new env and does not change any other env.

```bash
bash scripts/setup_conda.sh
conda activate resolve
```

If those packages are already installed, run from this directory with `PYTHONPATH=src` and skip the install.

## CSV input

The only dataset argument is the path to one CSV. There is no separate name. The path you pass, together with the seed, fixes the initial 16. The same path and the same seed reproduce the same 16. Moving or renaming the file changes the draw. Row order is the candidate index, so reordering the rows also changes which sequences those indices point to.

The file needs a header and at least 64 data rows. A ready-made example is `examples/test_input.csv` (80 sequences, 10 marked elites). It begins:

```csv
sequence,label,elite
AAAAACDEFGHIKLMNPQRS,50.0000,1
CAAACDEFGHIKLMNPQRST,49.0000,1
DAAADEFGHIKLMNPQRSTV,48.0000,1
```

Required columns, matched without regard to case:

| Role | Accepted header |
|---|---|
| sequence | `sequence`, `seq`, `mutated_sequence`, or `aa_sequence` |
| label | `label`, `score`, `fitness`, `target`, or `dms_score` |

Each sequence is a string of the 20 amino acids `ACDEFGHIKLMNPQRSTVWY`. Lower case is accepted and upper-cased. A gap, a stop, or any other character is an error. The label is a finite number. Higher means better.

`elite` is optional. Use `1`, `true`, or `yes` for an elite and `0` otherwise. The marked set must be a non-empty proper subset of the rows. If the column is absent, the elites are the `max(10, ceil(0.01 n))` largest labels, and a tie goes to the smaller index. The example file includes the column, so those 10 marked rows are the elites. `examples/test_input.csv` has 80 rows, so the default rule would also have chosen 10.

For `frozen_esm2_masked_marginal_greedy` the sequences must all have the same length. Pass `--reference` as the wild-type sequence, or omit it and the reference is the per-column majority residue.

## Run our method

```bash
PYTHONPATH=src python -m resolve_method \
  --csv examples/test_input.csv \
  --schedule 3x16 \
  --seeds 0 \
  --gpu 1 \
  --out results/test_input__3x16
```

`--method` defaults to `ours`. Several schedules can be comma-separated.

Our method uses ESM-2 650M only. Keep is the frozen mean-pooled embedding, L2-normalised per row. Tune is the same model with a LoRA trained each round on the labels revealed so far. The batch stays on the frozen-kernel top unless two virtual-label futures both show a gain and agree. Both batches are valued with the frozen probability.

## Run the baselines

```bash
PYTHONPATH=src python -m resolve_method \
  --csv examples/test_input.csv \
  --schedule 3x16,4x12 \
  --method baselines \
  --gpu 1 \
  --out results/test_input__baselines
```

`--method all` runs ours and the ten baselines. `--method linucb,uniform_random` runs a subset. `uniform_random` is the only method that does not load ESM-2, so it is also the only one that can omit `--gpu`.

`summary.csv` has one row per method, schedule, and seed, with `hits_at_64` and `recall`. `campaign.json` stores the chosen indices. For our method the JSON also stores whether each round stayed frozen or left for the LoRA top.

## What was changed when the baselines were rerun

The ten methods keep their own surrogates and their own acquisition rules. Around those rules, every method uses the same contract:

1. They no longer use the driver and the initial set they were first published with. Every method, including ours, starts from the same 16 indices for a given CSV path and seed, then spends the same 48 queries.
2. The schedule is one of the six above. It is not fixed at four rounds of 16.
3. A round commits the whole batch before any label inside that batch is revealed. Returned indices are de-duplicated and then sorted.
4. Baselines read the raw ESM-2 mean pool. Our method L2-normalises those same vectors before the kernel sees them. Masked marginal does not use the pool at all; it scores sequences with the language model.
5. The elite set and `hits_at_64` are the endpoint. Intermediate counts at 16, 32, and 48 labels are stored on each baseline campaign and are not the comparison.
6. Methods that would otherwise draw a random first batch fit on the shared 16 labels from round one.
7. The EvolvePro forest uses one worker (`n_jobs=1`).
8. Each ALDE ensemble member is seeded before its weights are drawn.

Nothing in a baseline reads another method's trajectory, and nothing is fit to campaign return.

## The ten baselines

`uniform_random`. Each round draws its batch uniformly from the sequences not yet labelled. The draw depends on the CSV path, the seed, and the round. It does not use ESM-2.

`frozen_esm2_masked_marginal_greedy`. Zero-shot score from Meier et al. (NeurIPS 2021): mask every substituted site at once and sum the log odds of the mutant residue against the wild-type residue. The ranking never reads a label, so for one seed the set of 64 labels is the same on every schedule. The order inside the 48 queries can still change, because each batch is sorted by index. Sequences must be aligned. Pass `--reference` for the wild type; if you omit it, the reference is the per-column majority residue, with an amino-acid tie broken by `ACDEFGHIKLMNPQRSTVWY`.

`frozen_esm2_bayes_ucb_beta_2.0`. Bayesian linear regression on the raw mean pool, then `mean + 2 * predictive scale`. The head uses prior precision 1, observation noise 1, and jitter `1e-8`, standardised on the revealed rows only.

`diversity_kcenter`. Greedy k-center in that same raw pool. Already labelled sequences seed the covered set. A distance tie goes to the smaller index.

`linucb`. Not a linear bandit on the embedding. LinUCB (Li et al., WWW 2010) chooses among five experts: posterior mean, UCB with coefficient 0.5, expected improvement, predictive scale, and k-center. The context is ten public numbers (round, budget remaining, label spread, how much the experts overlap). The reward, seen only after the batch is labelled, is the fraction of the batch that beats the best label already in hand. Ridge 1, exploration 1.

`batchbald_regression`. The regression form of BatchBALD. On the linear head the mutual information of a batch is `0.5 log det(I + K/sigma^2)`, so the greedy step adds the point with the largest variance given the points already in the batch. Search is limited to the 512 pool rows with the largest marginal scale, or the whole pool if it is smaller.

`one_batch_voi_topk`. One-batch value of information. The action is to report `take` sequences with the highest posterior mean, and the utility is the sum of those means. A candidate is scored by how much a fantasy label at that candidate raises the utility. Search is limited to 256 rows, or the whole pool if it is smaller. Inside the batch, a chosen point is conditioned at its current mean.

`gp_rollout_nonmyopic`. Non-myopic rollout on the same linear head (Lam et al., NeurIPS 2016; Lee et al., UAI 2020). The value of a point is its expected improvement plus the expected improvement collected by a myopic continuation over the queries still left, including later rounds. Fantasies are three Gauss-Hermite nodes. The continuation is a sampled greedy-EI policy, and a batch is filled without revealing inside the batch. Search is limited to 96 rows, widened if the remaining budget needs more. The utility is the best observed value, not the hidden elite set. On the last round the future term is zero.

`evolvepro_style_rf_sqrt`. A random forest of 100 trees, `max_features="sqrt"`, fit to the revealed raw embeddings and labels. The batch is the top of that forest's predictions.

`alde_style_dnn_ts`. An ensemble of five networks, each with two hidden layers of width 256. Each member is trained on a 90% bootstrap of the revealed rows for up to 100 epochs, with learning rate `1e-3` and early stopping after 10 non-improving epochs. Features and labels are standardised on the training rows. Each pick in the batch draws one member and takes that member's argmax. The draw is which network is consulted, not a random tie break.
