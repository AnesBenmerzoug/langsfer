import torch
from IPython.display import HTML
import matplotlib.pyplot as plt

__all__ = [
    "render_collapsible_text_blocks",
    "generate_text",
    "render_generated_text",
    "plot_losses",
]


def safe_html(text: str) -> str:
    """Replaces newlines with <br> but preserves other text exactly"""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


def render_collapsible_text_blocks(
    texts: list[str], height: str = "200px", width: str = "100%"
) -> HTML:
    """Renders a list of texts as collapsible HTML details blocks."""

    titles = [f"Sample {i + 1}" for i in range(len(texts))]

    # Construct the HTML string
    html_parts = []
    for i, text in enumerate(texts):
        safe_text = safe_html("\n" + text)
        block = f"""
        <details>
            <summary>{titles[i]}</summary>
            <pre style="padding: 10px; 
                        border-radius: 6px; 
                        border: 1px solid #e1e4e8; 
                        overflow-x: auto; 
                        height: {height}; 
                        width: {width};">
                {safe_text}
            </pre>
        </details>
        <br>
        """
        html_parts.append(block)

    full_html = "".join(html_parts)
    return HTML(full_html)


def generate_text(
    model,
    tokenizer,
    sample_texts: list[str],
    *,
    device: str = "cpu",
    max_length: int = 512,
) -> list[dict[str, str]]:
    """Generates texts using model from part of the sample texts."""
    sample_input_ids = tokenizer(sample_texts)["input_ids"]
    max_length = min([max_length] + list(map(len, sample_input_ids)))
    shortened_input_ids = [
        input_ids[: max_length // 2] for input_ids in sample_input_ids
    ]
    shortened_texts = [
        tokenizer.decode(input_ids, add_special_tokens=False)
        for input_ids in shortened_input_ids
    ]
    # Convert to tensor and add to device
    input_tensor = (
        torch.as_tensor(shortened_input_ids).reshape(len(sample_texts), -1).to(device)
    )
    # Create attention mask (1s for all tokens)
    attention_mask = torch.ones_like(input_tensor).to(device)
    # Move model to device to accelerate generation
    model.to(device)

    generated_token_ids = (
        model.generate(
            input_tensor,
            attention_mask=attention_mask,
            max_length=max_length,
            min_length=int(0.1 * max_length),
            top_p=0.9,
            temperature=0.3,
            repetition_penalty=2.0,
        )
        .detach()
        .cpu()
        .numpy()
    )
    generated_texts = [
        tokenizer.decode(token_ids, add_special_tokens=False)
        for token_ids in generated_token_ids
    ]

    results = []
    for shortened_text, sample_text, generated_text in zip(
        shortened_texts, sample_texts, generated_texts
    ):
        results.append(
            {
                "input": shortened_text,
                "original": sample_text[
                    len(shortened_text) : max(
                        len(generated_text), 2 * len(shortened_text)
                    )
                ],
                "generated": generated_text[len(shortened_text) :],
            }
        )

    return results


def render_generated_text(
    outputs: list[dict], height: str = "100px", width: str = "90%"
) -> HTML:
    """Renders structured data (Input, Original, Generated) as collapsible HTML blocks."""
    titles = [f"Sample {i + 1}" for i in range(len(outputs))]

    html_parts = []
    for i, output in enumerate(outputs):
        input_text = safe_html("\n" + output["input"])
        original_text = safe_html(output["original"])
        generated_text = safe_html(output["generated"])
        block = f"""
        <details>
            <summary>{titles[i]}</summary>
            <div style="display: flex; flex-direction: column; gap: 15px; padding: 10px;">
                <div style="padding: 10px; border-radius: 6px; border: 1px solid #e1e4e8; overflow-x: auto;">
                    <strong style="color: #d73a49; display: block; margin-bottom: 5px;">Input Text</strong>
                    <pre style="white-space: pre-wrap; height: {height}; overflow: auto; margin: 0; padding: 0;">{input_text}</pre>
                </div>
                <div style="padding: 10px; border-radius: 6px; border: 1px solid #e1e4e8; overflow-x: auto;">
                    <strong style="color: #6f42c1; display: block; margin-bottom: 5px;">Original Text</strong>
                    <pre style="white-space: pre-wrap; height: {height}; overflow: auto; margin: 0; padding: 0;">{original_text}</pre>
                </div>
                <div style="padding: 10px; border-radius: 6px; border: 1px solid #e1e4e8; overflow-x: auto;">
                    <strong style="color: #22863a; display: block; margin-bottom: 5px;">Generated Text</strong>
                    <pre style="white-space: pre-wrap; height: {height}; overflow: auto; margin: 0; padding: 0;">{generated_text}</pre>
                </div>
            </div>
        </details>
        <br>
        """
        html_parts.append(block)

    return HTML("".join(html_parts))


def plot_losses(log_history: list[dict]) -> None:
    """Plots the training and validation losses"""
    train_metrics = [item for item in log_history if "loss" in item]
    train_steps = [x["step"] for x in train_metrics]
    train_losses = [x["loss"] for x in train_metrics]

    val_metrics = [item for item in log_history if "eval_loss" in item]
    val_steps = [x["step"] for x in val_metrics]
    val_losses = [x["eval_loss"] for x in val_metrics]

    fig, ax = plt.subplots()
    ax.plot(train_steps, train_losses, label="Training Loss", linewidth=2)
    ax.plot(val_steps, val_losses, label="Validation Loss", linestyle="--")
    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.legend()
    return ax
