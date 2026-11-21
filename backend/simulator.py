import numpy as np


class EEGSimulator:
    CHANNEL_NAMES = ["FT7", "FT8", "T7", "PZ", "O2"]
    FS = 200

    def __init__(self, data_path: str):
        d = np.load(data_path)
        self.epochs      = d["epochs"]
        self.labels      = d["labels"]
        self.subject_ids = d["subject_ids"]
        self.subject_map = {
            int(s): np.where(self.subject_ids == s)[0]
            for s in np.unique(self.subject_ids)
        }
        self.subjects = sorted(self.subject_map)

    def epoch(self, subject: int, pos: int):
        idx = self.subject_map[subject][pos % len(self.subject_map[subject])]
        return self.epochs[idx].copy(), int(self.labels[idx])

    def n(self, subject: int) -> int:
        return len(self.subject_map[subject])

    def balance(self, subject: int) -> dict:
        lbl = self.labels[self.subject_map[subject]]
        return {
            "alert":  float((lbl == 0).mean() * 100),
            "drowsy": float((lbl == 1).mean() * 100),
        }
