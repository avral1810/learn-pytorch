# learn-pytorch

PyTorch learning workspace focused on:

1. CPU-first fundamentals
2. Basic neural networks
3. CNNs and vision concepts
4. Transformer fundamentals
5. Transition to Apple Metal (`mps`) later

Start here: [tutorials/README.md](./tutorials/README.md)

## Local Learning App

The repo also includes a local chapter-based PyTorch learning app under `quiz/`.

## Setup

```bash
bash setup.sh
```

This creates a dedicated virtual environment at `quiz/.venv` for the local app and installs `quiz/requirements.txt`.

The setup pins `numpy<2` because the current PyTorch build used by the app may fail with NumPy 2.x. If you already created `quiz/.venv`, rerun `bash setup.sh` to reconcile the environment.

## Run

```bash
bash run.sh
```

This starts the local app with `python -m quiz.app`, which keeps package imports working correctly.

Open:

```text
http://127.0.0.1:8000
```

## App Notes

- The browser shows chapter pages with lesson content, diagrams, and embedded coding quizzes.
- `Run` executes code in a fresh local Python subprocess using `quiz/.venv`.
- `Submit` executes hidden tests in a fresh local Python subprocess.
- The app is intended for trusted local use only.
- Every run starts fresh, so variables do not persist like a notebook.
