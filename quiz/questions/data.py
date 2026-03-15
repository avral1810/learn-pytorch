from __future__ import annotations

from copy import deepcopy

from quiz.content.data import CHAPTERS


QUESTIONS = [
    {"id": "q00_add_tensors", "chapter_id": "00", "title": "Add Two Tensors", "lesson": "00_tensors_and_autograd", "prompt": "Write a function `add_tensors(a, b)` that returns the elementwise tensor sum.", "starter_code": """import torch


def add_tensors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Return the elementwise sum of a and b.
    raise NotImplementedError("Replace this line with your code.")
""", "visible_examples": ["Matching positions should be added together.", "A (2, 2) input pair should return shape (2, 2)."], "test_module": "quiz.questions.tests.q00_add_tensors"},
    {"id": "q01_trainable_scalar", "chapter_id": "00", "title": "Create A Trainable Scalar", "lesson": "00_tensors_and_autograd", "prompt": "Write `build_trainable_scalar(value)` so the returned scalar tensor has `requires_grad=True`.", "starter_code": """import torch


def build_trainable_scalar(value: float) -> torch.Tensor:
    # Return a scalar tensor that PyTorch can differentiate with respect to.
    raise NotImplementedError("Replace this line with your code.")
""", "visible_examples": ["The result should be a scalar tensor.", "The tensor must allow gradient tracking."], "test_module": "quiz.questions.tests.q01_trainable_scalar"},
    {"id": "q02_reshape_to_column", "chapter_id": "01", "title": "Reshape Into A Column", "lesson": "01_shapes_gradients_and_reshape", "prompt": "Write `reshape_to_column(values)` so a flat input becomes a tensor with shape `(n, 1)`.", "starter_code": """import torch


def reshape_to_column(values) -> torch.Tensor:
    # Convert values into a torch tensor and return shape (n, 1).
    raise NotImplementedError("Replace this line with your code.")
""", "visible_examples": ["Input [1, 2, 3] should become shape (3, 1).", "Value order should stay top-to-bottom."], "test_module": "quiz.questions.tests.q02_reshape_to_column"},
    {"id": "q03_linear_model", "chapter_id": "02", "title": "Manual Linear Model", "lesson": "02_linear_regression_from_scratch", "prompt": "Write `linear_model(x, weight, bias)` using the formula `x @ weight + bias`.", "starter_code": """import torch


def linear_model(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # Compute the model output using matrix multiplication and bias broadcasting.
    raise NotImplementedError("Replace this line with your code.")
""", "visible_examples": ["If x is (batch, 1) and weight is (1, 1), the result should be (batch, 1).", "The bias should broadcast across the batch."], "test_module": "quiz.questions.tests.q03_linear_model"},
    {"id": "q04_mse_loss", "chapter_id": "02", "title": "Mean Squared Error", "lesson": "02_linear_regression_from_scratch", "prompt": "Write `mean_squared_error(preds, targets)` so it returns `((preds - targets) ** 2).mean()`.", "starter_code": """import torch


def mean_squared_error(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Return the scalar mean squared error.
    raise NotImplementedError("Replace this line with your code.")
""", "visible_examples": ["The output should be a scalar tensor.", "Larger errors should contribute more to the loss."], "test_module": "quiz.questions.tests.q04_mse_loss"},
    {"id": "q05_tiny_linear_forward", "chapter_id": "03", "title": "Tiny nn.Module Forward", "lesson": "03_linear_regression_with_nn_module", "prompt": "Complete the `forward` method so it returns `self.linear(x)`.", "starter_code": """import torch
from torch import nn


class TinyLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return the model prediction using the linear layer above.
        raise NotImplementedError("Replace this line with your code.")
""", "visible_examples": ["The output shape should match `(batch, 1)`.", "Use the layer defined in `__init__`, not a new one."], "test_module": "quiz.questions.tests.q05_tiny_linear_forward"},
    {"id": "q06_logistic_probability", "chapter_id": "04", "title": "Logistic Probability", "lesson": "04_logistic_regression_from_scratch", "prompt": "Write `logistic_probability(x, weight, bias)` so it returns `sigmoid(x @ weight + bias)`.", "starter_code": """import torch


def logistic_probability(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    # Return probabilities between 0 and 1.
    raise NotImplementedError("Replace this line with your code.")
""", "visible_examples": ["Output values must stay between 0 and 1.", "For x shape (batch, 2) and weight shape (2, 1), output shape should be (batch, 1)."], "test_module": "quiz.questions.tests.q06_logistic_probability"},
    {"id": "q07_logistic_module_forward", "chapter_id": "05", "title": "Logistic Module Forward", "lesson": "05_logistic_regression_with_nn_module", "prompt": "Finish the `forward` method. Return raw logits from `self.linear(x)` and do not apply sigmoid.", "starter_code": """import torch
from torch import nn


class LogisticRegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return raw logits, not probabilities.
        raise NotImplementedError("Replace this line with your code.")
""", "visible_examples": ["The output shape should be (batch, 1).", "The returned values are logits, so they are not forced into 0..1."], "test_module": "quiz.questions.tests.q07_logistic_module_forward"},
]


QUESTIONS_BY_ID = {question["id"]: question for question in QUESTIONS}


def public_question_payload(question_ids: list[str]) -> list[dict[str, object]]:
    payload = []
    for question_id in question_ids:
        item = deepcopy(QUESTIONS_BY_ID[question_id])
        payload.append(
            {
                "id": item["id"],
                "chapter_id": item["chapter_id"],
                "title": item["title"],
                "lesson": item["lesson"],
                "prompt": item["prompt"],
                "starter_code": item["starter_code"],
                "visible_examples": item["visible_examples"],
            }
        )
    return payload


def get_public_chapter_detail(chapter_id: str) -> dict[str, object] | None:
    chapter = next((chapter for chapter in CHAPTERS if chapter["id"] == chapter_id), None)
    if chapter is None:
        return None
    detail = deepcopy(chapter)
    detail["questions"] = public_question_payload(detail["question_ids"])
    return detail
