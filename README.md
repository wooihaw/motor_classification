# Motor Load Classifier

Classify motor operating mode — **Low**, **Moderate**, or **High** load — from
`.wav` audio recordings. The pipeline splits recordings into Train/Test sets,
segments each file into overlapping 30-second chunks, extracts FFT-based
spectral features, and trains a RandomForest classifier.

## Repository contents

```
motor-load-classifier/
├── README.md
├── pyproject.toml
└── motor_load_classifier.py
```

## Requirements

- Python 3.9+
- [git](https://git-scm.com/)
- [uv](https://docs.astral.sh/uv/) (installed below)

No manual virtual environment setup is needed — `uv` handles that for you.

## 1. Install uv

**macOS / Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative (any OS, if you already have Python/pip):**

```bash
pip install uv
```

Verify the install:

```bash
uv --version
```

## 2. Clone the repository

```bash
git clone https://github.com/<your-username>/motor-load-classifier.git
cd motor-load-classifier
```

## 3. Restore dependencies with uv

The dependencies (`numpy`, `scipy`, `scikit-learn`, `matplotlib`, `joblib`) are
declared in `pyproject.toml`. Restore them with:

```bash
uv sync
```

This creates a local `.venv/` and installs everything needed — no separate
`pip install` step required. A `uv.lock` file will also be generated the first
time you run this; commit it to the repo so everyone gets identical package
versions.

## 4. Prepare your dataset

Arrange your raw recordings into three folders, one per class, before running
the script:

```
raw_data/
├── Low/       *.wav
├── Moderate/  *.wav
└── High/      *.wav
```

## 5. Run the script

Use `uv run` so the script executes inside the managed environment (no need
to manually activate `.venv`):

```bash
uv run python motor_load_classifier.py \
  --low_dir raw_data/Low \
  --moderate_dir raw_data/Moderate \
  --high_dir raw_data/High \
  --output_root ./dataset_split \
  --chunk_duration 30 \
  --overlap 0.5
```

See all available options:

```bash
uv run python motor_load_classifier.py -h
```

| Argument            | Default                     | Description                                   |
|---------------------|------------------------------|------------------------------------------------|
| `--low_dir`          | *(required)*                 | Folder of raw "Low" load `.wav` files          |
| `--moderate_dir`     | *(required)*                 | Folder of raw "Moderate" load `.wav` files     |
| `--high_dir`         | *(required)*                 | Folder of raw "High" load `.wav` files         |
| `--output_root`      | `./dataset_split`            | Where `Train/` and `Test/` folders are created |
| `--test_ratio`       | `0.2`                         | Fraction of files per class reserved for test  |
| `--chunk_duration`   | `30.0`                        | Chunk length in seconds                        |
| `--overlap`          | `0.5`                         | Overlap fraction between consecutive chunks    |
| `--target_sr`        | `22050`                       | Sample rate all audio is resampled to          |
| `--n_bins`           | `200`                         | Number of log-spaced FFT frequency bins        |
| `--move_files`       | off                           | Move instead of copy files into Train/Test     |
| `--model_out`        | `motor_load_model.joblib`     | Path to save the trained model                 |
| `--seed`             | `42`                          | Random seed for reproducibility                |

## Output

After running, you'll get:

- `dataset_split/Train/{Low,Moderate,High}/` and `dataset_split/Test/{Low,Moderate,High}/` — the organized recordings
- `dataset_split/confusion_matrix_chunks.png` — chunk-level confusion matrix
- `dataset_split/confusion_matrix_files.png` — file-level confusion matrix (majority vote per recording)
- `motor_load_model.joblib` — the trained model, reloadable with `joblib.load(...)`
- Console output with chunk-level and file-level accuracy and classification reports

## License

Add a license of your choice (e.g. MIT) in a `LICENSE` file at the repo root.
