from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class QuestionText:
    title: str
    prompt: str
    visible_examples_title: str = "Visible Checks"


@dataclass(frozen=True)
class QuestionTemplate:
    starter_code: str
    editor_title: str = "Your Answer"
    editor_note: str = "Write the code for this question here"


@dataclass(frozen=True)
class QuestionAnswer:
    lesson: str
    symbol_name: str | None = None


@dataclass(frozen=True)
class QuestionTests:
    module_name: str
    hidden_binary_key: str | None = None


@dataclass(frozen=True)
class QuizQuestion:
    id: str
    chapter_id: str
    text: QuestionText
    template: QuestionTemplate
    answer: QuestionAnswer
    tests: QuestionTests
    visible_examples: tuple[str, ...] = field(default_factory=tuple)

    def to_public_payload(self) -> dict[str, object]:
        return {
            "id": self.id,
            "chapter_id": self.chapter_id,
            "title": self.text.title,
            "lesson": self.answer.lesson,
            "prompt": self.text.prompt,
            "starter_code": self.template.starter_code,
            "visible_examples": list(self.visible_examples),
            "visible_examples_title": self.text.visible_examples_title,
            "answer_editor_title": self.template.editor_title,
            "answer_editor_note": self.template.editor_note,
        }

    def to_serializable(self) -> dict[str, object]:
        return {
            "id": self.id,
            "chapter_id": self.chapter_id,
            "text": {
                "title": self.text.title,
                "prompt": self.text.prompt,
                "visible_examples_title": self.text.visible_examples_title,
            },
            "template": {
                "starter_code": self.template.starter_code,
                "editor_title": self.template.editor_title,
                "editor_note": self.template.editor_note,
            },
            "answer": {
                "lesson": self.answer.lesson,
                "symbol_name": self.answer.symbol_name,
            },
            "tests": {
                "module_name": self.tests.module_name,
                "hidden_binary_key": self.tests.hidden_binary_key,
            },
            "visible_examples": list(self.visible_examples),
        }


@dataclass(frozen=True)
class Quiz:
    chapter_id: str
    questions: tuple[QuizQuestion, ...] = field(default_factory=tuple)

    @property
    def question_ids(self) -> list[str]:
        return [question.id for question in self.questions]

    def to_serializable(self) -> dict[str, object]:
        return {
            "chapter_id": self.chapter_id,
            "question_ids": self.question_ids,
            "questions": [question.to_serializable() for question in self.questions],
        }
