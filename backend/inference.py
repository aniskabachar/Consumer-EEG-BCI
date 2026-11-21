import torch, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class BCIPredictor:
    CLASS_NAMES = {0: "FOCUSED", 1: "DROWSY"}

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cache = {}

    def _get_model(self, path):
        if path not in self._cache:
            from evaluation.models.bci_model import build_model
            m = build_model(n_channels=5)
            m.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True)
            )
            m.to(self.device).eval()
            self._cache[path] = m
        return self._cache[path]

    def predict(self, epoch: np.ndarray, model_path: str, noise: float = 0.0):
        x = torch.tensor(epoch, dtype=torch.float32).T.unsqueeze(0)
        if noise > 0:
            x = x + noise * (x.std() + 1e-8) * torch.randn_like(x)
        with torch.no_grad():
            probs = (
                torch.softmax(
                    self._get_model(model_path)(x.to(self.device)), 1
                ).cpu().numpy()[0]
            )
        pred = int(probs.argmax())
        return pred, float(probs[pred]), probs

    @property
    def device_label(self):
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.get_device_name(0).replace("NVIDIA ", "")[:18]
        return "CPU"
