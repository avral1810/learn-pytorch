# Quiz App

Local HTML chapter website for learning beginner PyTorch with embedded coding quizzes.

## Environment

The quiz setup is self-enclosed.

- It creates and uses its own virtual environment at `quiz/.venv`
- It does not depend on the tutorial PDF environment at `.pdf-venv`
- It does not require a separate repo-level `.venv`
- The quiz server and code execution both run through `quiz/.venv`
- Chapter images are read from files in `quiz/content/assets/chapters/`
- If you replace an image file with a new one using the same filename, the webpage will show the new image

## Setup

```bash
bash quiz/setup.sh
```

## Run

```bash
bash quiz/run.sh
```

Then open:

```text
http://127.0.0.1:8000
```

## How It Works

- The browser shows chapter webpages with lesson content, diagrams, and a chapter-local quiz.
- `Run` executes your code in a fresh local Python subprocess using `quiz/.venv`.
- `Submit` executes the hidden tests for that question in a fresh local Python subprocess.
- The code runs only on your machine. Nothing is sent to a remote service.

## Notes

- This app is intended for trusted local use only.
- Every run starts fresh, so variables do not persist like a notebook.
- The first version focuses on chapters `00` through `05`.
