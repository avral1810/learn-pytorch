from __future__ import annotations

from copy import deepcopy


CHAPTERS = [
    {
        "id": "00",
        "number": "00",
        "title": "Tensors and Autograd",
        "lesson": "00_tensors_and_autograd",
        "summary": "Learn tensors, shapes, and the backward/update loop that powers training.",
        "image": "00_gradient_descent.png",
        "script_path": "tutorials/00_tensors_and_autograd.py",
        "pdf_path": "tutorials/pdfs/00_tensors_and_autograd.pdf",
        "question_ids": ["q00_add_tensors", "q01_trainable_scalar"],
        "previous_chapter": None,
        "next_chapter": "01",
        "sections": [
            {"type": "paragraph", "content": "PyTorch revolves around tensors and autograd. Tensors hold the numbers; autograd tracks how a final loss depends on trainable values."},
            {"type": "list", "title": "What to understand first", "items": ["A scalar has shape (). A matrix may have shape (2, 2).", "Elementwise operations keep the same shape when the inputs match.", "Calling backward() computes gradients for tensors with requires_grad=True."]},
            {"type": "code", "title": "Core update loop", "content": "loss.backward()\nwith torch.no_grad():\n    parameter -= learning_rate * parameter.grad\nparameter.grad.zero_()"},
            {"type": "callout", "title": "Beginner focus", "content": "Understand why only trainable parameters need requires_grad=True and why gradients must be cleared each step."},
        ],
    },
    {
        "id": "01",
        "number": "01",
        "title": "Shapes, Reshape, and Gradients",
        "lesson": "01_shapes_gradients_and_reshape",
        "summary": "Practice shape thinking so matrix multiplication, reshape, and broadcasting stop feeling mysterious.",
        "image": "01_matrix_multiplication.png",
        "script_path": "tutorials/01_shapes_gradients_and_reshape.py",
        "pdf_path": "tutorials/pdfs/01_shapes_gradients_and_reshape.pdf",
        "question_ids": ["q02_reshape_to_column"],
        "previous_chapter": "00",
        "next_chapter": "02",
        "sections": [
            {"type": "paragraph", "content": "Most beginner PyTorch bugs are shape bugs. If you can read a shape and predict how it changes after reshape, unsqueeze, or matrix multiplication, later chapters become much easier."},
            {"type": "list", "title": "Shape rules to memorize", "items": ["reshape changes arrangement, not values.", "unsqueeze adds a dimension of size 1.", "broadcasting lets smaller tensors expand over compatible dimensions."]},
            {"type": "code", "title": "Common reshape pattern", "content": "x = torch.tensor([1, 2, 3])\ncolumn = x.unsqueeze(1)  # shape (3, 1)"},
        ],
    },
    {
        "id": "02",
        "number": "02",
        "title": "Linear Regression From Scratch",
        "lesson": "02_linear_regression_from_scratch",
        "summary": "Build a regression model with raw tensors so the forward pass and loss are fully visible.",
        "image": "02_linear_regression_plot.png",
        "script_path": "tutorials/02_linear_regression_from_scratch.py",
        "pdf_path": "tutorials/pdfs/02_linear_regression_from_scratch.pdf",
        "question_ids": ["q03_linear_model", "q04_mse_loss"],
        "previous_chapter": "01",
        "next_chapter": "03",
        "sections": [
            {"type": "paragraph", "content": "Linear regression is the cleanest place to learn supervised learning. You pick a formula, compare prediction to target, compute a loss, and update the parameters."},
            {"type": "code", "title": "Model formula", "content": "preds = x @ weight + bias\nloss = ((preds - targets) ** 2).mean()"},
            {"type": "list", "title": "Shape story", "items": ["x shape (batch, 1)", "weight shape (1, 1)", "bias shape (1,)", "preds shape (batch, 1)"]},
        ],
    },
    {
        "id": "03",
        "number": "03",
        "title": "Linear Regression With nn.Module",
        "lesson": "03_linear_regression_with_nn_module",
        "summary": "See the same regression problem expressed with nn.Module, DataLoader, and an optimizer.",
        "image": "05_perceptron.png",
        "script_path": "tutorials/03_linear_regression_with_nn_module.py",
        "pdf_path": "tutorials/pdfs/03_linear_regression_with_nn_module.pdf",
        "question_ids": ["q05_tiny_linear_forward"],
        "previous_chapter": "02",
        "next_chapter": "04",
        "sections": [
            {"type": "paragraph", "content": "Nothing magical happens inside nn.Module. PyTorch just packages trainable tensors, the forward pass, and optimizer-friendly parameter access in a cleaner abstraction."},
            {"type": "code", "title": "Standard training pattern", "content": "preds = model(batch_x)\nloss = loss_fn(preds, batch_y)\noptimizer.zero_grad()\nloss.backward()\noptimizer.step()"},
        ],
    },
    {
        "id": "04",
        "number": "04",
        "title": "Logistic Regression From Scratch",
        "lesson": "04_logistic_regression_from_scratch",
        "summary": "Move from regression to binary classification with logits, sigmoid, and binary cross-entropy.",
        "image": "04_logistic_curve.png",
        "script_path": "tutorials/04_logistic_regression_from_scratch.py",
        "pdf_path": "tutorials/pdfs/04_logistic_regression_from_scratch.pdf",
        "question_ids": ["q06_logistic_probability"],
        "previous_chapter": "03",
        "next_chapter": "05",
        "sections": [
            {"type": "paragraph", "content": "Logistic regression still starts with a linear formula, but now you interpret the output as a probability by applying sigmoid to the raw logit."},
            {"type": "code", "title": "Probability path", "content": "logits = x @ weight + bias\nprobs = torch.sigmoid(logits)"},
            {"type": "callout", "title": "Important distinction", "content": "A logit is any real number. A probability must lie between 0 and 1."},
        ],
    },
    {
        "id": "05",
        "number": "05",
        "title": "Logistic Regression With nn.Module",
        "lesson": "05_logistic_regression_with_nn_module",
        "summary": "Use nn.Linear and BCEWithLogitsLoss so PyTorch handles the standard binary-classification stack.",
        "image": "05_perceptron.png",
        "script_path": "tutorials/05_logistic_regression_with_nn_module.py",
        "pdf_path": "tutorials/pdfs/05_logistic_regression_with_nn_module.pdf",
        "question_ids": ["q07_logistic_module_forward"],
        "previous_chapter": "04",
        "next_chapter": None,
        "sections": [
            {"type": "paragraph", "content": "In PyTorch you usually return raw logits from forward() and let BCEWithLogitsLoss handle the stable sigmoid-plus-loss computation during training."},
            {"type": "list", "title": "What to remember", "items": ["forward() returns logits, not probabilities", "torch.sigmoid is for interpretation during evaluation", "labels should match the output shape"]},
        ],
    },
]


CHAPTERS_BY_ID = {chapter["id"]: chapter for chapter in CHAPTERS}


def public_chapter_summaries() -> list[dict[str, object]]:
    return [
        {
            "id": chapter["id"],
            "number": chapter["number"],
            "title": chapter["title"],
            "lesson": chapter["lesson"],
            "summary": chapter["summary"],
            "image": chapter["image"],
            "question_count": len(chapter["question_ids"]),
        }
        for chapter in CHAPTERS
    ]


def get_chapter(chapter_id: str) -> dict[str, object] | None:
    chapter = CHAPTERS_BY_ID.get(chapter_id)
    if chapter is None:
        return None
    return deepcopy(chapter)
