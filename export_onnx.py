import os
import torch
import numpy as np
import onnxruntime as ort
from onnxruntime.transformers import optimizer

from model import CaptchaTransformer
from dataset import NUM_CLASSES

WEIGHT_PATH = "model_v2.pth"
TEMP_PATH = "_model_v2.onnx"
OUTPUT_PATH = "model_v2.onnx"
OPSET = 18


def export_raw():
    device = torch.device("cpu")

    model = CaptchaTransformer(
        num_classes=NUM_CLASSES,
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    model.eval()

    dummy = torch.randn(1, 1, 32, 128, device=device)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            TEMP_PATH,
            input_names=["input"],
            output_names=["logits"],
            opset_version=OPSET,
            dynamic_axes={
                "input": {0: "batch"},
                "logits": {0: "batch"},
            },
            do_constant_folding=True,
        )


def optimize():
    opt_model = optimizer.optimize_model(
        TEMP_PATH,
        model_type="bert",
        num_heads=4,
        hidden_size=256,
        opt_level=99,
    )
    opt_model.save_model_to_file(OUTPUT_PATH)


def verify():
    dummy = np.random.randn(1, 1, 32, 128).astype(np.float32)

    sess_raw = ort.InferenceSession(TEMP_PATH, providers=["CPUExecutionProvider"])
    sess_opt = ort.InferenceSession(OUTPUT_PATH, providers=["CPUExecutionProvider"])

    out_raw = sess_raw.run(["logits"], {"input": dummy})[0]
    out_opt = sess_opt.run(["logits"], {"input": dummy})[0]

    max_diff = np.max(np.abs(out_raw - out_opt))
    print(f"最大誤差: {max_diff:.6e}")
    if max_diff >= 1e-4:
        print("警告 誤差偏大 請檢查")


def cleanup():
    for path in [TEMP_PATH, TEMP_PATH + ".data"]:
        if os.path.exists(path):
            os.remove(path)

def main():
    export_raw()
    optimize()
    verify()
    cleanup()
    size = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"已輸出: {OUTPUT_PATH} ({size:.2f} MB)")


if __name__ == "__main__":
    main()