import gradio as gr

from app.rag_simple.generation.answer import answer_question


def format_context(chunks):
    result = "<h2 style='color: #ff7800;'>Context relevant</h2>\n\n"
    for chunk in chunks:
        result += f"<span style='color: #ff7800;'>Sursă: {chunk.metadata.get('source', '?')}</span>\n\n"
        result += chunk.page_content + "\n\n"
    return result


def chat(history):
    last_message = history[-1]["content"]
    prior = history[:-1]
    answer, chunks = answer_question(last_message, prior)
    history.append({"role": "assistant", "content": answer})
    return history, format_context(chunks)


def main():
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="Calea de Autovindecare — Asistent", theme=theme) as ui:
        gr.Markdown("# 🌱 Calea de Autovindecare\nÎntreabă orice despre macrobiotică și sănătate naturistă.")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="💬 Conversație", height=600, type="messages", show_copy_button=True
                )
                message = gr.Textbox(
                    label="Întrebarea ta",
                    placeholder="Întreabă orice despre macrobiotică...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="📚 Context regăsit",
                    value="*Contextul regăsit va apărea aici*",
                    container=True,
                    height=600,
                )

        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot, context_markdown])

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()
