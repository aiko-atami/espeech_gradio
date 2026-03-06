import gradio as gr
with gr.Blocks(css=".no-border { --block-border-width: 0px; border: none; box-shadow: none; }") as demo:
    with gr.Accordion("Test accordion", elem_classes="no-border"):
        gr.Markdown("Inside accordion")
demo.launch(prevent_thread_lock=True)
