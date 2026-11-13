"""Optional Gradio demo. Easiest way to show ScholarLens from Google Colab,
because `share=True` gives a public link without any port forwarding.

    python -m app.gradio_demo
"""

import gradio as gr

from src.pipeline import ScholarLens

lens = ScholarLens()


def do_summary(text):
    return lens.summarize(text)["summary"] if text.strip() else "Paste a paper first."


def do_entities(text):
    if not text.strip():
        return []
    return [[e["text"], e["label"], e["count"]] for e in lens.entities(text)[:60]]


def do_qa(text, question):
    if not text.strip() or not question.strip():
        return "Paste a paper and type a question.", ""
    out = lens.ask(question, text)
    return out.get("answer") or out.get("note", "No answer."), out.get("evidence", "")


with gr.Blocks(title="ScholarLens") as demo:
    gr.Markdown("# ScholarLens\nSummarize a paper, extract its entities and ask it questions.")
    paper = gr.Textbox(lines=14, label="Paper text")
    with gr.Tab("Summary"):
        btn = gr.Button("Summarize")
        summary = gr.Textbox(label="Summary", lines=6)
        btn.click(do_summary, paper, summary)
    with gr.Tab("Entities"):
        btn = gr.Button("Extract entities")
        table = gr.Dataframe(headers=["Entity", "Type", "Mentions"], interactive=False)
        btn.click(do_entities, paper, table)
    with gr.Tab("Ask"):
        question = gr.Textbox(label="Question")
        btn = gr.Button("Ask")
        answer = gr.Textbox(label="Answer")
        evidence = gr.Textbox(label="Evidence passage", lines=4)
        btn.click(do_qa, [paper, question], [answer, evidence])

if __name__ == "__main__":
    demo.launch(share=True)
