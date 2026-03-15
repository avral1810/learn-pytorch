from __future__ import annotations

import html
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)


ROOT = Path(__file__).resolve().parent
NOTES_DIR = ROOT / "notes"
PDF_DIR = ROOT / "pdfs"
ASSETS_DIR = ROOT / "assets"


LESSON_IMAGES = {
    "00_tensors_and_autograd": {
        "path": ASSETS_DIR / "00_gradient_descent.png",
        "caption": "Gradient descent visualization",
        "source": "https://commons.wikimedia.org/wiki/File:Gradient_descent.png",
    },
    "01_shapes_gradients_and_reshape": {
        "path": ASSETS_DIR / "00b_matrix_multiplication.png",
        "caption": "Matrix multiplication illustration",
        "source": "https://commons.wikimedia.org/wiki/File:Matrix_Multiplication.png",
    },
    "02_linear_regression_from_scratch": {
        "path": ASSETS_DIR / "01_linear_regression_plot.png",
        "caption": "Linear regression plot",
        "source": "https://commons.wikimedia.org/wiki/File:Linear_Regression_Plot.png",
    },
    "03_linear_regression_with_nn_module": {
        "path": ASSETS_DIR / "03_linear_regression.svg",
        "caption": "Linear regression diagram",
        "source": "https://commons.wikimedia.org/wiki/File:Linear_regression.svg",
    },
    "04_logistic_regression_from_scratch": {
        "path": ASSETS_DIR / "04_logistic_curve.png",
        "caption": "Logistic curve",
        "source": "https://commons.wikimedia.org/wiki/File:Logistic-curve.png",
    },
    "05_logistic_regression_with_nn_module": {
        "path": ASSETS_DIR / "03_perceptron.png",
        "caption": "Perceptron diagram",
        "source": "https://commons.wikimedia.org/wiki/File:Perceptron.png",
    },
    "06_manual_mlp_from_scratch": {
        "path": ASSETS_DIR / "02_xor_neural_network.jpg",
        "caption": "XOR neural network diagram",
        "source": "https://commons.wikimedia.org/wiki/File:XOR_neural_network_2-2-1.jpg",
    },
    "07_basic_nn_mlp_with_nn_module": {
        "path": ASSETS_DIR / "05_mlp_training_loop.svg",
        "caption": "MLP training loop",
        "source": "https://commons.wikimedia.org/wiki/File:MLP-training-loop.svg",
    },
    "08_cnn_basics": {
        "path": ASSETS_DIR / "03_convolutional_network.png",
        "caption": "Convolutional network illustration",
        "source": "https://commons.wikimedia.org/wiki/File:Convolutional_Network.png",
    },
    "09_vision_classifier": {
        "path": ASSETS_DIR / "04_convolutional_neural_network.png",
        "caption": "Convolutional neural network architecture",
        "source": "https://commons.wikimedia.org/wiki/File:Convolutional_Neural_Network.png",
    },
    "10_lstm_sequence_classifier": {
        "path": ASSETS_DIR / "09_lstm_cell.png",
        "caption": "LSTM cell diagram",
        "source": "https://commons.wikimedia.org/wiki/File:The_LSTM_cell.png",
    },
    "11_transformer_basics": {
        "path": ASSETS_DIR / "05_encoder_self_attention.png",
        "caption": "Encoder self-attention diagram",
        "source": "https://commons.wikimedia.org/wiki/File:Encoder_self-attention,_detailed_diagram.png",
    },
    "12_toy_gan": {
        "path": ASSETS_DIR / "11_gan_illustration.png",
        "caption": "Generative adversarial network illustration",
        "source": "https://commons.wikimedia.org/wiki/File:Rete_generativa_avversaria.png",
    },
    "13_device_cpu_to_mps": {
        "path": ASSETS_DIR / "06_microchip.jpg",
        "caption": "Chip photo for device-compute context",
        "source": "https://commons.wikimedia.org/wiki/File:Mac_Mini_M1_chip.jpg",
    },
    "14_basic_rnn_sequence_classifier": {
        "path": ASSETS_DIR / "14_rnn_unfold.svg",
        "caption": "Recurrent neural network unfold diagram",
        "source": "https://commons.wikimedia.org/wiki/File:Recurrent_neural_network_unfold.svg",
    },
}


def build_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "LessonTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#102a43"),
            fontSize=20,
            leading=24,
            spaceAfter=14,
        ),
        "meta": ParagraphStyle(
            "Meta",
            parent=styles["BodyText"],
            textColor=colors.HexColor("#486581"),
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=styles["Italic"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#334e68"),
            fontSize=9,
            leading=11,
            spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=9,
            leading=12,
            spaceAfter=0,
        ),
        "footer": ParagraphStyle(
            "Footer",
            parent=styles["BodyText"],
            textColor=colors.HexColor("#627d98"),
            fontSize=8,
            leading=10,
            spaceBefore=8,
        ),
    }


def fit_image(image_path: Path, max_width: float, max_height: float) -> Image:
    img = Image(str(image_path))
    width, height = img.imageWidth, img.imageHeight
    scale = min(max_width / width, max_height / height, 1.0)
    img.drawWidth = width * scale
    img.drawHeight = height * scale
    return img


def build_pdf(note_path: Path, styles: dict[str, ParagraphStyle]) -> None:
    slug = note_path.stem
    output_path = PDF_DIR / f"{slug}.pdf"
    lines = note_path.read_text(encoding="utf-8").splitlines()
    title = lines[0].strip() if lines else slug
    body = "\n".join(lines[2:] if len(lines) > 2 else lines)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title=title,
        author="Codex",
    )

    story = [Paragraph(html.escape(title), styles["title"])]

    image_info = LESSON_IMAGES.get(slug)
    if image_info:
        story.append(Paragraph("Companion diagram", styles["meta"]))
        story.append(fit_image(image_info["path"], max_width=6.5 * inch, max_height=2.6 * inch))
        story.append(Spacer(1, 0.08 * inch))
        story.append(Paragraph(html.escape(image_info["caption"]), styles["caption"]))
        story.append(
            Paragraph(
                html.escape(f"Image source: {image_info['source']}"),
                styles["meta"],
            )
        )

    story.append(Spacer(1, 0.12 * inch))
    story.append(Preformatted(body, styles["body"], dedent=0))
    story.append(Spacer(1, 0.1 * inch))
    story.append(
        Paragraph(
            "Read the matching Python file and run it while comparing the printed shapes, losses, and outputs against this handout.",
            styles["footer"],
        )
    )

    doc.build(story)


def main() -> None:
    PDF_DIR.mkdir(exist_ok=True)
    styles = build_styles()

    for note_path in sorted(NOTES_DIR.glob("*.txt")):
        build_pdf(note_path, styles)
        print(f"Built {PDF_DIR / (note_path.stem + '.pdf')}")


if __name__ == "__main__":
    main()
