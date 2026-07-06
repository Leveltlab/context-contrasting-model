# context-contrasting-model

Standalone repository for the contextual-contrasting model-scatter simulation used
for the final paper figures.

The repository includes the reference output in `done-final-fix/`. A default run
is configured to reproduce those results while writing new artifacts to
`outputs/`, so the reference folder is not overwritten.

## Setup

```bash
conda env create -f environment.yml
conda activate context-contrasting-model
python -m pip install -e .
```

## Reproduce `done-final-fix`

From the repository root, run:

```bash
python -m context_contrasting_model.run_model_scatter
```

The command above is equivalent to the final run settings:

```bash
python -m context_contrasting_model.run_model_scatter \
  --n-jobs 10 \
  --plot-center-panels \
  --output-dir outputs \
  --export-panels \
  --image-format svg \
  --fam-examples 2 3 -NO/+O +NO 13 6 \
  --nov-examples +NO/+Om 11 7 2 6 \
  --n-samples 300 \
  --n-steps-per-phase 400
```

The default run also uses `--test-trials 5`, `--training-trials 7`,
`--training-stimulus-order randomized`, `--seed 7151`,
`--transition-sampling data-like`, `--zscore-std-floor 0.04`,
`--response-tail-fraction 1.0`, `--threshold 0.3`, and
`--axis-clip-percentile 99.0`.

## Outputs

The default run writes:

- `outputs/sample_responses.csv`
- `outputs/transition_table.csv`
- `outputs/sampled_config_parameters.csv`
- `outputs/sampled_configs.json`
- `outputs/metadata.json`
- `outputs/figures/*.svg`
- `outputs/center_panels/*.svg`
- `outputs/summaries/*.csv`
- `outputs/highlight_examples/`

To compare a fresh run against the embedded reference:

```bash
python scripts/verify_reproduction.py
```

The verifier compares CSV/JSON outputs byte-for-byte. For SVG/EPS files it
normalizes Matplotlib-generated timestamps and internal object IDs before
comparison; the rendered plot content must still match.
