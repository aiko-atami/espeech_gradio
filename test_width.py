import gradio as gr
with gr.Blocks(css=".gradio-container { max-width: 800px !important; }") as demo:
    gr.Markdown("Test")
demo.launch(prevent_thread_lock=True)
