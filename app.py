import gradio as gr
from score_chat import score_and_alert

def predict(chat, th):
    return score_and_alert(chat, th)

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Textbox(lines=8, label="Paste chat"),
            gr.Slider(0, 100, value=20, step=1, label="Threshold %")],
    outputs="json",
    title="Hash-Matching Scam Alert"
)
demo.launch()
