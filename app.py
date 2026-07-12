import gradio as gr
from src.rag_pipeline import answer_query

DESCRIPTION = """
# Efficient LLM Inference — Research Assistant (RAG)

Ask questions about quantization, KV-cache optimization, and speculative
decoding, grounded in a curated corpus of ~30 arXiv papers.

Answers are generated only from retrieved paper excerpts — sources are shown
below each answer. This is a portfolio/demo project; always verify against
the original papers for anything you rely on.
"""

EXAMPLE_QUESTIONS = [
    "What is the difference between GPTQ and AWQ?",
    "How does speculative decoding speed up inference?",
    "What techniques exist for compressing the KV cache?",
    "What is PagedAttention and why does it help serving throughput?",
]


def respond(query, k):
    if not query.strip():
        return "Please enter a question.", ""
    try:
        answer, sources = answer_query(query, k=int(k))
    except Exception as e:
        return f"Error: {e}", ""
    sources_md = "\n".join(f"- {title} ({arxiv_id})" for title, arxiv_id in sources)
    return answer, sources_md


with gr.Blocks(title="Efficient LLM Inference RAG") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        query_box = gr.Textbox(label="Your question", placeholder="Ask about quantization, KV-cache, speculative decoding...", lines=2)

    k_slider = gr.Slider(minimum=2, maximum=8, value=5, step=1, label="Number of chunks to retrieve")

    submit_btn = gr.Button("Ask", variant="primary")

    answer_box = gr.Textbox(label="Answer", lines=10)
    sources_box = gr.Markdown(label="Sources")

    gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=query_box)

    submit_btn.click(fn=respond, inputs=[query_box, k_slider], outputs=[answer_box, sources_box])
    query_box.submit(fn=respond, inputs=[query_box, k_slider], outputs=[answer_box, sources_box])

if __name__ == "__main__":
    demo.launch()
