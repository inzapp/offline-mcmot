from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    exp = np.exp(values - values.max())
    return (exp / exp.sum()).astype(np.float32)


class AttributeModels:
    def __init__(self, cfg: dict):
        import onnxruntime as ort
        self.cfg = cfg
        self.gender = ort.InferenceSession(cfg["gender_model"], providers=["CPUExecutionProvider"])
        self.age = ort.InferenceSession(cfg["age_model"], providers=["CPUExecutionProvider"])

    @staticmethod
    def _input(session, image: np.ndarray, height: int, width: int) -> np.ndarray:
        info = session.get_inputs()[0]
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        if "float" in info.type:
            array = rgb.astype(np.float32) / 255.0
        else:
            array = rgb.astype(np.uint8)
        shape = info.shape
        if len(shape) == 4 and shape[1] == 3:
            array = np.transpose(array, (2, 0, 1))
        return array[None]

    def infer(self, image: np.ndarray) -> dict:
        g_input = self._input(self.gender, image, 64, 64)
        g_raw = np.asarray(self.gender.run(None, {self.gender.get_inputs()[0].name: g_input})[0]).reshape(-1)
        gender_score = float(g_raw[0])
        gender = "female" if gender_score < float(self.cfg["gender_threshold"]) else "male"
        a_input = self._input(self.age, image, 128, 64)
        a_raw = np.asarray(self.age.run(None, {self.age.get_inputs()[0].name: a_input})[0]).reshape(-1)
        age_index = int(np.argmax(a_raw))
        age_score = float(a_raw[age_index])
        labels = self.cfg["age_labels"]
        ranges = [(0, 14), (14, 20), (20, 50), (50, 70)]
        age_estimate = (ranges[age_index][0] + (ranges[age_index][1] - ranges[age_index][0]) * age_score
                        if age_index < len(ranges) else "")
        return {"gender": gender, "gender_score": gender_score,
                "age": labels[age_index] if age_index < len(labels) else str(age_index),
                "age_index": age_index, "age_score": age_score, "age_estimate": age_estimate,
                "age_vector": ";".join(f"{x:.8f}" for x in a_raw)}


class ReIDModel:
    def __init__(self, cfg: dict):
        import tensorflow as tf
        self.tf = tf
        self.model = tf.keras.models.load_model(cfg["model"], compile=False)
        self.rows = 128
        self.cols = 64
        config_path = Path(cfg["config"])
        if config_path.exists():
            import yaml
            values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.rows, self.cols = int(values["input_rows"]), int(values["input_cols"])

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        batch = []
        for image in images:
            resized = cv2.resize(image, (self.cols, self.rows), interpolation=cv2.INTER_LINEAR)
            batch.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        return self.model(np.stack(batch), training=False).numpy().astype(np.float32)


def infer_all(persons_path: Path, output_path: Path, cfg: dict) -> dict[str, np.ndarray]:
    with persons_path.open(encoding="utf-8", newline="") as stream:
        persons = list(csv.DictReader(stream))
    fields = ["local_uid", "crop_path", "gender", "gender_score", "age", "age_index", "age_estimate",
              "age_score", "age_vector", "reid_embedding_path", "status"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings: dict[str, np.ndarray] = {}
    valid = []
    for person in persons:
        image = cv2.imread(person["crop_path"]) if person.get("crop_path") else None
        if image is not None:
            valid.append((person, image))
    attr = AttributeModels(cfg["attributes"]) if valid else None
    reid = ReIDModel(cfg["reid"]) if valid else None
    batch_size = int(cfg["reid"].get("batch_size", 32))
    reid_dir = output_path.parent / "embeddings"
    reid_dir.mkdir(exist_ok=True)
    embedded: dict[str, np.ndarray] = {}
    for start in range(0, len(valid), batch_size):
        batch = valid[start:start + batch_size]
        vectors = reid.embed([image for _, image in batch])
        for (person, _), vector in zip(batch, vectors):
            embedded[person["local_uid"]] = vector
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for person in persons:
            uid, crop_path = person["local_uid"], person.get("crop_path", "")
            image = cv2.imread(crop_path) if crop_path else None
            if image is None:
                writer.writerow({"local_uid": uid, "crop_path": crop_path, "status": "unavailable"})
                continue
            try:
                result = attr.infer(image)
                vector = embedded[uid]
                safe = uid.replace(":", "_") + ".npy"
                embedding_path = reid_dir / safe
                np.save(embedding_path, vector)
                embeddings[uid] = vector
                writer.writerow({"local_uid": uid, "crop_path": crop_path, **result,
                                 "reid_embedding_path": str(embedding_path.resolve()), "status": "ok"})
            except Exception as exc:
                writer.writerow({"local_uid": uid, "crop_path": crop_path,
                                 "status": f"error:{type(exc).__name__}:{exc}"})
    return embeddings
