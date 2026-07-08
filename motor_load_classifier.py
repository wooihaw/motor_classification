"""
motor_load_classifier.py
=========================
Train a machine learning classifier to determine motor operating mode
(Low / Moderate / High load) from .wav audio recordings using FFT-based
spectral features.

PIPELINE
--------
1. Organize raw recordings (one folder per class) into a Train/Test split:
       <output_root>/Train/{Low,Moderate,High}
       <output_root>/Test/{Low,Moderate,High}
   The split happens at the FILE level (not the chunk level) so that
   segments from the same recording never appear in both Train and Test.
   This avoids data leakage and gives an honest estimate of performance.

2. Segment every audio file into fixed-length overlapping chunks
   (default: 30 s chunks, 50% overlap).

3. Extract FFT-based spectral features from every chunk (log-spaced
   frequency-bin energies + a few summary descriptors).

4. Train a RandomForest classifier (StandardScaler + RandomForest pipeline)
   on the Train chunks.

5. Evaluate on the Test chunks, reporting both:
       - chunk-level accuracy / confusion matrix / classification report
       - file-level accuracy (majority vote of all chunks belonging to the
         same original recording), which is usually the number you care
         about in practice.

DEPENDENCIES
------------
Only packages that ship with a standard scientific-Python install are used:
    numpy, scipy, scikit-learn, matplotlib, joblib
Install with:
    pip install numpy scipy scikit-learn matplotlib joblib

USAGE
-----
    python motor_load_classifier.py \
        --low_dir /path/to/Low \
        --moderate_dir /path/to/Moderate \
        --high_dir /path/to/High \
        --output_root ./dataset_split \
        --chunk_duration 30 \
        --overlap 0.5

Run `python motor_load_classifier.py -h` for all options.
"""

import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from scipy.io import wavfile
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")  # safe for headless / script execution
import matplotlib.pyplot as plt

CLASSES = ["Low", "Moderate", "High"]


# ---------------------------------------------------------------------------
# STEP 1 : Organize raw recordings into Train/Test folders
# ---------------------------------------------------------------------------
def organize_train_test(source_dirs, output_root, test_ratio=0.2, seed=42,
                         move_files=False):
    """
    Split each class's .wav files into Train/Test folders at the FILE level.

    source_dirs : dict, e.g. {"Low": "...", "Moderate": "...", "High": "..."}
    output_root : where <output_root>/Train/<class> and Test/<class> go
    test_ratio  : fraction of files (per class) reserved for testing
    move_files  : if False (default) files are copied, originals kept intact
    """
    rng = random.Random(seed)
    train_root = Path(output_root) / "Train"
    test_root = Path(output_root) / "Test"

    for cls in CLASSES:
        (train_root / cls).mkdir(parents=True, exist_ok=True)
        (test_root / cls).mkdir(parents=True, exist_ok=True)

        src_dir = Path(source_dirs[cls])
        wav_files = sorted(src_dir.glob("*.wav"))
        if not wav_files:
            print(f"[WARN] No .wav files found in {src_dir}")
            continue

        files = wav_files[:]
        rng.shuffle(files)

        n_test = int(round(len(files) * test_ratio))
        n_test = max(1, n_test) if len(files) > 1 else 0
        test_files = files[:n_test]
        train_files = files[n_test:]

        transfer = shutil.move if move_files else shutil.copy2
        for f in train_files:
            transfer(str(f), str(train_root / cls / f.name))
        for f in test_files:
            transfer(str(f), str(test_root / cls / f.name))

        print(f"[{cls:9s}] total={len(files):3d}  "
              f"train={len(train_files):3d}  test={len(test_files):3d}")

    return str(train_root), str(test_root)


# ---------------------------------------------------------------------------
# Audio loading (WAV only, no external audio libraries required)
# ---------------------------------------------------------------------------
def load_wav_mono(file_path, target_sr=22050):
    """Load a .wav file, convert to mono float32 in [-1, 1], resample to target_sr."""
    sr, data = wavfile.read(str(file_path))
    data = np.asarray(data)

    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)

    if data.ndim > 1:          # stereo / multi-channel -> mono
        data = data.mean(axis=1)

    if sr != target_sr:
        n_samples = int(round(len(data) * target_sr / sr))
        data = signal.resample(data, n_samples)

    return data.astype(np.float32), target_sr


# ---------------------------------------------------------------------------
# STEP 2 : Segment audio into overlapping chunks
# ---------------------------------------------------------------------------
def segment_audio(file_path, chunk_duration=30.0, overlap=0.5,
                   target_sr=22050, pad_short=True):
    """
    Load a wav file and slice it into overlapping fixed-length chunks.

    overlap : fraction of overlap between consecutive chunks (0.5 = 50%)
    pad_short : if a file is shorter than chunk_duration, zero-pad it into
                a single chunk instead of discarding it
    Returns a list of 1-D numpy arrays, each of length chunk_duration*target_sr.
    """
    y, sr = load_wav_mono(file_path, target_sr)
    chunk_len = int(round(chunk_duration * sr))
    hop_len = max(1, int(round(chunk_len * (1 - overlap))))

    if len(y) < chunk_len:
        if pad_short:
            y_padded = np.pad(y, (0, chunk_len - len(y)))
            return [y_padded]
        return []

    chunks = []
    start = 0
    last_start = -1
    while start + chunk_len <= len(y):
        chunks.append(y[start:start + chunk_len])
        last_start = start
        start += hop_len

    # Make sure the tail of the recording is captured even if the hop
    # stride doesn't land exactly on the end of the file.
    tail_start = len(y) - chunk_len
    if tail_start > last_start:
        chunks.append(y[tail_start:])

    return chunks


# ---------------------------------------------------------------------------
# STEP 3 : FFT feature extraction
# ---------------------------------------------------------------------------
def extract_fft_features(chunk, sr, n_bins=200, fmax=None):
    """
    Compute the FFT magnitude spectrum of a chunk and reduce it to a fixed
    length feature vector: mean energy in `n_bins` log-spaced frequency
    bins, plus a handful of summary descriptors (total energy, spectral
    centroid, dominant frequency, RMS).
    """
    n = len(chunk)
    windowed = chunk * np.hanning(n)          # reduce spectral leakage
    fft_vals = np.fft.rfft(windowed)
    magnitude = np.abs(fft_vals) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    if fmax is None:
        fmax = sr / 2.0

    low_edge = max(freqs[1], 1.0)
    bin_edges = np.logspace(np.log10(low_edge), np.log10(fmax), n_bins + 1)

    features = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        mask = (freqs >= bin_edges[i]) & (freqs < bin_edges[i + 1])
        if np.any(mask):
            features[i] = magnitude[mask].mean()

    total_energy = float(np.sum(magnitude ** 2))
    spectral_centroid = float(np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-12))
    dominant_freq = float(freqs[np.argmax(magnitude)])
    rms = float(np.sqrt(np.mean(chunk ** 2)))

    extra = np.array([total_energy, spectral_centroid, dominant_freq, rms],
                      dtype=np.float32)
    return np.concatenate([features, extra])


# ---------------------------------------------------------------------------
# STEP 2+3 combined : build a full feature dataset from a Train/ or Test/ folder
# ---------------------------------------------------------------------------
def build_feature_dataset(root_dir, chunk_duration=30.0, overlap=0.5,
                           target_sr=22050, n_bins=200):
    """
    Walk <root_dir>/<class>/*.wav, segment + FFT-featurize every chunk.
    Returns X (features), y (labels), file_ids (source filename per chunk,
    used later for file-level majority-vote evaluation).
    """
    X, y, file_ids = [], [], []
    root = Path(root_dir)

    for cls in CLASSES:
        cls_dir = root / cls
        wav_files = sorted(cls_dir.glob("*.wav"))
        print(f"  {cls_dir}: {len(wav_files)} file(s)")

        for f in wav_files:
            chunks = segment_audio(f, chunk_duration, overlap, target_sr)
            for chunk in chunks:
                feats = extract_fft_features(chunk, target_sr, n_bins)
                X.append(feats)
                y.append(cls)
                file_ids.append(f"{cls}/{f.name}")

    return np.array(X, dtype=np.float32), np.array(y), file_ids


# ---------------------------------------------------------------------------
# STEP 4 : Train the model
# ---------------------------------------------------------------------------
def train_model(X_train, y_train, random_state=42):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


# ---------------------------------------------------------------------------
# STEP 5 : Evaluate on Test set (chunk-level + file-level)
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, file_ids_test, output_dir="."):
    y_pred = model.predict(X_test)

    # ---- chunk-level metrics ----
    chunk_acc = accuracy_score(y_test, y_pred)
    print(f"\nChunk-level test accuracy: {chunk_acc:.4f}")
    print("\nChunk-level classification report:")
    print(classification_report(y_test, y_pred, labels=CLASSES, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
    _plot_confusion_matrix(cm, Path(output_dir) / "confusion_matrix_chunks.png",
                            title="Confusion Matrix (chunk-level)")

    # ---- file-level metrics (majority vote across a file's chunks) ----
    votes = defaultdict(list)
    truth = {}
    for fid, true_lbl, pred_lbl in zip(file_ids_test, y_test, y_pred):
        votes[fid].append(pred_lbl)
        truth[fid] = true_lbl

    file_true, file_pred = [], []
    for fid, preds in votes.items():
        majority = Counter(preds).most_common(1)[0][0]
        file_true.append(truth[fid])
        file_pred.append(majority)

    file_acc = accuracy_score(file_true, file_pred)
    print(f"\nFile-level test accuracy (majority vote): {file_acc:.4f}")
    print(f"Number of test files: {len(file_true)}")
    print("\nFile-level classification report:")
    print(classification_report(file_true, file_pred, labels=CLASSES, zero_division=0))

    cm_file = confusion_matrix(file_true, file_pred, labels=CLASSES)
    _plot_confusion_matrix(cm_file, Path(output_dir) / "confusion_matrix_files.png",
                            title="Confusion Matrix (file-level, majority vote)")

    return {
        "chunk_accuracy": chunk_acc,
        "file_accuracy": file_acc,
    }


def _plot_confusion_matrix(cm, out_path, title):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES)
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train a motor load (Low/Moderate/High) classifier from WAV recordings using FFT features."
    )
    parser.add_argument("--low_dir", required=True, help="Folder with 'Low' load .wav files")
    parser.add_argument("--moderate_dir", required=True, help="Folder with 'Moderate' load .wav files")
    parser.add_argument("--high_dir", required=True, help="Folder with 'High' load .wav files")
    parser.add_argument("--output_root", default="./dataset_split",
                         help="Where Train/ and Test/ folders will be created")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                         help="Fraction of files per class reserved for testing")
    parser.add_argument("--chunk_duration", type=float, default=30.0,
                         help="Chunk length in seconds")
    parser.add_argument("--overlap", type=float, default=0.5,
                         help="Fractional overlap between consecutive chunks (0-1)")
    parser.add_argument("--target_sr", type=int, default=22050,
                         help="Sample rate all audio is resampled to before FFT")
    parser.add_argument("--n_bins", type=int, default=200,
                         help="Number of log-spaced FFT frequency bins used as features")
    parser.add_argument("--move_files", action="store_true",
                         help="Move files into Train/Test instead of copying them")
    parser.add_argument("--model_out", default="motor_load_model.joblib",
                         help="Path to save the trained model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_dirs = {"Low": args.low_dir, "Moderate": args.moderate_dir, "High": args.high_dir}

    print("=== Step 1: Organizing Train/Test folders ===")
    train_root, test_root = organize_train_test(
        source_dirs, args.output_root, args.test_ratio, args.seed, args.move_files
    )

    print("\n=== Step 2+3: Segmenting & extracting FFT features (Train) ===")
    X_train, y_train, _ = build_feature_dataset(
        train_root, args.chunk_duration, args.overlap, args.target_sr, args.n_bins
    )
    print(f"Train feature matrix: {X_train.shape}")

    print("\n=== Step 2+3: Segmenting & extracting FFT features (Test) ===")
    X_test, y_test, file_ids_test = build_feature_dataset(
        test_root, args.chunk_duration, args.overlap, args.target_sr, args.n_bins
    )
    print(f"Test feature matrix: {X_test.shape}")

    print("\n=== Step 4: Training RandomForest model ===")
    model = train_model(X_train, y_train, args.seed)

    print("\n=== Step 5: Evaluating on Test set ===")
    evaluate_model(model, X_test, y_test, file_ids_test, output_dir=args.output_root)

    joblib.dump(model, args.model_out)
    print(f"\nModel saved to: {args.model_out}")


if __name__ == "__main__":
    main()
