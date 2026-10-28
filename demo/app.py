"""Gradio demo: classify an aerial / satellite image into one of 45 scene categories.

Usage:
    RESISC45_MODEL_PATH=runs/<run>/model.keras python demo/app.py
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from resisc45.inference import Classifier

HERE = Path(__file__).parent
GITHUB_URL = "https://github.com/esammostafa9-cloud/resisc45-scene-classification"
UNSURE_THRESHOLD = 0.5

METRICS = {
    "Test accuracy": "94.4%",
    "Macro F1": "0.944",
    "Classes": "45",
    "Test images": "4,725",
}

CSS = """
.gradio-container { max-width: 1100px !important; margin: auto; }
#header { text-align: center; }
#header h1 { margin-bottom: 0.2rem; }
#header p { color: var(--body-text-color-subdued); margin-top: 0; }
.result-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 16px;
    padding: 20px 24px;
    background: var(--block-background-fill);
}
.result-card .label { font-size: 1.9rem; font-weight: 700; line-height: 1.2; }
.result-card .caption { color: var(--body-text-color-subdued); font-size: 0.9rem; }
.badge {
    display: inline-block; margin-top: 10px; padding: 4px 12px; border-radius: 999px;
    font-weight: 600; font-size: 0.9rem;
}
.badge.sure { background: rgba(34, 197, 94, 0.15); color: rgb(22, 163, 74); }
.badge.unsure { background: rgba(245, 158, 11, 0.18); color: rgb(217, 119, 6); }
.metrics { display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; margin: 8px 0 4px; }
.metric {
    border: 1px solid var(--border-color-primary); border-radius: 12px;
    padding: 8px 16px; text-align: center; min-width: 110px;
}
.metric b { display: block; font-size: 1.25rem; }
.metric span { color: var(--body-text-color-subdued); font-size: 0.8rem; }
"""


def pretty(label: str) -> str:
    return label.replace("_", " ").capitalize()


def result_card(label: str, confidence: float, runner_up: str | None) -> str:
    sure = confidence >= UNSURE_THRESHOLD
    badge = (
        f'<span class="badge sure">{confidence:.1%} confident</span>'
        if sure
        else f'<span class="badge unsure">{confidence:.1%} — the model is unsure</span>'
    )
    hint = (
        f'<div class="caption" style="margin-top:8px">Could also be: {pretty(runner_up)}</div>'
        if runner_up and not sure
        else ""
    )
    return (
        '<div class="result-card">'
        '<div class="caption">Predicted scene</div>'
        f'<div class="label">{pretty(label)}</div>{badge}{hint}</div>'
    )


EMPTY_CARD = (
    '<div class="result-card"><div class="caption">Predicted scene</div>'
    '<div class="label">—</div>'
    '<div class="caption" style="margin-top:8px">Upload an image or pick an example.</div></div>'
)


def build_demo(classifier: Classifier) -> gr.Blocks:
    def classify(image_path: str | None):
        if not image_path:
            return EMPTY_CARD, None
        pred = classifier.predict_path(image_path, top_k=5)
        runner_up = pred.top_k[1][0] if len(pred.top_k) > 1 else None
        scores = {pretty(c): p for c, p in pred.top_k}
        return result_card(pred.label, pred.confidence, runner_up), scores

    examples = sorted(str(p) for p in (HERE / "examples").glob("*.jpg"))
    metrics_html = (
        '<div class="metrics">'
        + "".join(
            f'<div class="metric"><b>{v}</b><span>{k}</span></div>' for k, v in METRICS.items()
        )
        + "</div>"
    )
    classes = ", ".join(pretty(c) for c in classifier.class_names)

    with gr.Blocks(title="Aerial Scene Classifier") as demo:
        gr.HTML(
            '<div id="header"><h1>🛰️ Aerial Scene Classifier</h1>'
            "<p>Identify the land-use scene in a satellite or aerial image — "
            "ResNet-50 fine-tuned on NWPU-RESISC45</p></div>",
            padding=False,
        )
        gr.HTML(metrics_html, padding=False)

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                image = gr.Image(type="filepath", label="Aerial image", height=360)
                with gr.Row():
                    clear = gr.Button("Clear", variant="secondary")
                    submit = gr.Button("Classify", variant="primary")
                if examples:
                    gr.Examples(examples=examples, inputs=image, label="Try an example")
            with gr.Column(scale=4):
                card = gr.HTML(EMPTY_CARD, padding=False)
                scores = gr.Label(num_top_classes=5, label="Top-5 predictions", show_heading=False)

        with gr.Accordion("About this model", open=False):
            gr.Markdown(
                f"""
**Model.** ResNet-50 pretrained on ImageNet, trained in two phases: first the
classification head with a frozen backbone, then fine-tuning of the `conv4` and `conv5`
stages at a low learning rate.

**Data.** [NWPU-RESISC45](https://doi.org/10.1109/JPROC.2017.2675998) — 31,500 aerial
images, 700 per class, split 70 / 15 / 15. Metrics above are on the held-out test set.

**Known weak spots.** Look-alike scenes: *palace* vs *church*, *dense* vs *medium
residential*, *railway* vs *railway station*.

**Tip.** The model expects top-down imagery similar to Google Earth tiles. Ground-level
photos are outside what it was trained on, and predictions for them are not meaningful.

**All {len(classifier.class_names)} classes:** {classes}

Code: [GitHub]({GITHUB_URL})
"""
            )

        submit.click(classify, inputs=image, outputs=[card, scores])
        image.change(classify, inputs=image, outputs=[card, scores])
        clear.click(lambda: (None, EMPTY_CARD, None), outputs=[image, card, scores])

    return demo


if __name__ == "__main__":
    demo = build_demo(Classifier.from_env())
    demo.queue(default_concurrency_limit=2, max_size=20)
    demo.launch(
        css=CSS,
        theme=gr.themes.Soft(),
        max_file_size=20 * 1024 * 1024,
        show_error=False,
    )
