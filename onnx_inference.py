import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
import gradio as gr

# CONFIG 
MODEL_PATH = "model.onnx"
IMG_SIZE = 160

classes = [
    "Bridge", "Clean", "Crack", "LER",
    "Open", "Other", "Scratch", "Vias"
]

#  LOAD MODEL 
ort_session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# PREPROCESS 
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

#  PREDICTION 
def predict(img):
    img = img.convert("RGB")
    x = preprocess(img).unsqueeze(0).numpy().astype(np.float32)

    logits = ort_session.run(None, {"input": x})[0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / np.sum(exp, axis=1, keepdims=True)

    idx = np.argmax(probs)
    label = classes[idx]
    conf = probs[0][idx] * 100

    colors = {
        "Open": "#ff0033",
        "Bridge": "#ff9900",
        "Crack": "#ffee00",
        "LER": "#ff00ff",
        "Scratch": "#00ff66",
        "Vias": "#3399ff",
        "Clean": "#00ffcc",
        "Other": "#aaaaaa"
    }
    c = colors[label]

    html = f"""
<div style="
    width:100%;
    min-height:260px;
    text-align:center;
    padding:40px;
    border-radius:25px;
    background:radial-gradient(circle,#111,#000);
    border:3px solid {c};
    box-shadow:0 0 40px {c};
    animation:pulse 1.5s infinite;
">

<h1 style="font-size:42px;color:#ff4444;">
🚨 DEFECT ALERT 🚨
</h1>

<h2 style="
    font-size:40px;
    color:{c};
    text-shadow:0 0 25px {c};
">
{label.upper()}
</h2>

<p style="font-size:26px;color:white;">
Confidence: <b>{conf:.2f}%</b>
</p>
</div>

<style>
@keyframes pulse {{
  0% {{ box-shadow:0 0 20px {c}; }}
  50% {{ box-shadow:0 0 55px {c}; }}
  100% {{ box-shadow:0 0 20px {c}; }}
}}
</style>
"""
    return html

#  UI 
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    <h1 style="text-align:center;">Wafer Defect Classifier</h1>
    <p style="text-align:center;">Upload sample images & get <b>defect alert</b> if any</p>
    """)
    with gr.Row(equal_height=True):

        with gr.Column(scale=1, min_width=420):
            img = gr.Image(
                type="pil",
                label="Upload Wafer Image",
                height=320
            )

        with gr.Column(scale=1, min_width=420):
            output = gr.HTML()

    analyze_btn = gr.Button("🚀 ANALYZE", variant="primary")
    analyze_btn.click(predict, inputs=img, outputs=output)

demo.launch()
