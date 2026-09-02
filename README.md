# Performing Facial Recognition with Deep Learning

Course-end certification project: build a baseline CNN facial recognition model for **Face2Gene**, using the ORL (AT&T) Database of Faces (400 images, 40 subjects).

This project targets **Python 3.14** and uses **PyTorch** rather than TensorFlow, because TensorFlow does not currently publish wheels for Python 3.14. Everything else in the brief (CNN with conv/pool/FC layers, train/val/test split, iterate to >90% accuracy) is implemented the same way, just in PyTorch.

## Files

- `facial_recognition.ipynb` — the main notebook, with markdown explanations at every step
- `requirements.txt` — pinned dependencies
- `README.md` — this file

## 1. Set up your environment

Open a terminal in VS Code (`` Ctrl+` ``) in this project folder, then:

```powershell
# Create a virtual environment (use `python`, not `python3`, on Windows)
python -m venv .venv

# Activate it (PowerShell)
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation with an execution-policy error, run this once and then retry activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

You should see `(.venv)` at the start of your terminal prompt once it's active.

## 2. Install dependencies

```powershell
pip install -r requirements.txt
```

This installs PyTorch (CPU build), torchvision, scikit-learn, numpy, matplotlib, scipy, and Jupyter/ipykernel.

> **No GPU required.** The model is small (a few hundred thousand parameters) and trains in a couple of minutes on CPU. If you do have an NVIDIA GPU and want CUDA acceleration, install a CUDA-enabled `torch` build instead by following the selector at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) — check what's currently available for Python 3.14 there, since GPU wheel support tends to lag CPU wheel support for very new Python versions.

## 3. Open and run the notebook in VS Code

1. Install the **Python** and **Jupyter** extensions in VS Code if you don't already have them (Extensions panel, search "Python" and "Jupyter", both by Microsoft).
2. Open `facial_recognition.ipynb` in VS Code.
3. In the top-right of the notebook, click the kernel picker and select the Python interpreter from `.venv` (it will usually show as `.venv (Python 3.14.x)`). If it's not listed, use **Select Kernel > Python Environments...** and browse to `.venv\Scripts\python.exe`.
4. Click **Run All** at the top of the notebook (or run cells one at a time with `Shift+Enter`).

## 4. What happens when you run it

- The dataset (~4 MB) downloads automatically on first run via `sklearn.datasets.fetch_olivetti_faces` and is cached locally afterward — no manual Kaggle download needed.
- You'll see sample face images, training/validation loss and accuracy plots, a confusion matrix, and a handful of example predictions.
- The trained model weights are saved to `models/face_recognition_cnn.pt`.
- The notebook prints the final test accuracy and whether it cleared the 90% target from the brief.

## Troubleshooting

**`python3` not found / Microsoft Store popup**
On Windows the command is `python`, not `python3`. If you've already hit this, see Windows Settings → Apps → Advanced app settings → App execution aliases, and turn off the `python.exe`/`python3.exe` Store shortcuts.

**`pip install -r requirements.txt` fails on `torch`**
Make sure you're using Python 3.10–3.14 (check with `python --version`). If you're on an even newer Python release than 3.14, PyTorch wheels may not exist yet either — check [pytorch.org](https://pytorch.org/get-started/locally/) for current support.

**Dataset download fails / times out**
`fetch_olivetti_faces` needs an internet connection on first run only. If your network blocks the download host, try again on a different network, or download the dataset manually from the [Kaggle link in the project brief](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces) and adapt the loading cell as described in the notebook's data-loading section (Section 2).

**Kernel doesn't show `.venv` in VS Code**
Restart VS Code after creating the venv, then use **Python: Select Interpreter** from the Command Palette (`Ctrl+Shift+P`) to pick `.venv` explicitly, and re-pick it as the notebook kernel.

## Notes on the approach

- **Dataset:** ORL/AT&T Database of Faces, loaded via scikit-learn (64x64 grayscale, pixel values in [0, 1]).
- **Split:** stratified 70/15/15 train/validation/test, so each of the 40 subjects is represented in every split despite only having 10 images each.
- **Model:** a compact CNN (3 conv+pool blocks, then fully-connected classifier), with batch norm and dropout for regularization on this small dataset.
- **Training:** Adam optimizer, cross-entropy loss, learning-rate scheduling on validation loss plateau, best-checkpoint selection by validation accuracy.
- **Evaluation:** test accuracy, per-class precision/recall/F1, confusion matrix, and visual sample predictions.
- **Optional transfer-learning variant** included (a pretrained `mobilenet_v3_small` backbone) for comparison against the baseline CNN.
