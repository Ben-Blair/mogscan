"""Export the trained MogNet to ONNX for onnxruntime-web and verify it matches PyTorch."""
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from train_cnn import MogNet

ROOT = Path(__file__).parent
DST = ROOT.parent / "app" / "public" / "models" / "mognet.onnx"


class Wrapped(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, image):
        return self.m(image)


def main():
    model = MogNet()
    model.load_state_dict(torch.load(ROOT / "out" / "cnn.pt", map_location="cpu"))
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    wrapped = Wrapped(model).eval()
    torch.onnx.export(wrapped, (x,), str(DST), input_names=["image"],
                      output_names=["score", "sex_logit"], opset_version=17, dynamo=False)
    model.eval()  # export restores the wrapper's previous mode; make sure we compare in eval mode
    sess = ort.InferenceSession(str(DST))
    o_score, o_sex = sess.run(None, {"image": x.numpy()})
    with torch.no_grad():
        t_score, t_sex = model(x)
    print("max abs diff", float(np.abs(o_score - t_score.numpy()).max()), float(np.abs(o_sex - t_sex.numpy()).max()))
    assert np.allclose(o_score, t_score.numpy(), atol=1e-4) and np.allclose(o_sex, t_sex.numpy(), atol=1e-4)
    print(f"exported {DST} ({DST.stat().st_size / 1e6:.1f} MB), outputs match PyTorch")


if __name__ == "__main__":
    main()
