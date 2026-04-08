"""
Scenario6 : 380×380 + retinal preprocessing + lighter augmentation
"""

import os
import sys
import cv2
import argparse
import warnings
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="./data/DR_Image")
    p.add_argument("--model", type=str, default="dr_run9_model.keras")
    p.add_argument("--run-tag", type=str, default="dr_run9")
    p.add_argument("--eval-after", action="store_true")

    p.add_argument("--img", type=int, default=380)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--head-epochs", type=int, default=25)
    p.add_argument("--ft-epochs", type=int, default=35)
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--ft-lr", type=float, default=2e-6)
    p.add_argument("--unfreeze", type=int, default=220)

    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--focal-alpha", type=float, default=0.25)

    p.add_argument("--use-class-weights", action="store_true")
    p.add_argument("--cw-min", type=float, default=0.7)
    p.add_argument("--cw-max", type=float, default=1.8)

    p.add_argument("--clahe", action="store_true", help="Enable CLAHE in retinal preprocessing")
    p.add_argument("--mixed-precision", action="store_true")
    return p.parse_args()


def set_gpu_config(mixed_precision: bool):
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print("Could not set memory growth:", e)

    if mixed_precision:
        try:
            from tensorflow.keras import mixed_precision as mp
            mp.set_global_policy("mixed_float16")
            print("Mixed precision enabled: mixed_float16")
        except Exception as e:
            print("Could not enable mixed precision:", e)


@tf.keras.utils.register_keras_serializable(package="DR")
class CategoricalFocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, from_logits=False, name="categorical_focal_loss"):
        super().__init__(name=name)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.from_logits = bool(from_logits)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        if self.from_logits:
            y_pred = tf.nn.softmax(tf.cast(y_pred, tf.float32), axis=-1)
        else:
            y_pred = tf.cast(y_pred, tf.float32)

        eps = tf.constant(1e-7, dtype=tf.float32)
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        loss = -self.alpha * tf.pow(1.0 - p_t, self.gamma) * tf.math.log(p_t)
        return loss

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "from_logits": self.from_logits,
        })
        return cfg


class RetinalSequence(keras.utils.Sequence):
    """
    Custom sequence with:
    - dataframe input
    - optional light augmentation
    - retinal preprocessing
    - one-hot labels

    This avoids limitations of ImageDataGenerator for image-level preprocessing.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        class_indices: dict,
        img_size=(380, 380),
        batch_size=8,
        shuffle=True,
        augment=False,
        clahe=False,
        seed=123,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.class_indices = class_indices
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.clahe = clahe
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(self.df))
        self.num_classes = len(self.class_indices)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_idx]

        x = np.zeros((len(batch_df), self.img_size[0], self.img_size[1], 3), dtype=np.float32)
        y = np.zeros((len(batch_df), self.num_classes), dtype=np.float32)

        for i, (_, row) in enumerate(batch_df.iterrows()):
            img = self.load_and_preprocess_image(row["Paths"])
            if self.augment:
                img = self.light_augment(img)

            img = preprocess_input(img.astype(np.float32))
            x[i] = img

            class_id = self.class_indices[row["Labels"]]
            y[i, class_id] = 1.0

        return x, y

    def load_and_preprocess_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.remove_black_border(img)
        img = self.circular_crop(img)

        if self.clahe:
            img = self.apply_clahe(img)

        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        return img

    def remove_black_border(self, img, threshold=10):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray > threshold
        if not np.any(mask):
            return img

        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        cropped = img[y0:y1, x0:x1]
        return cropped if cropped.size > 0 else img

    def circular_crop(self, img):
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        radius = int(min(center[0], center[1]) * 0.95)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)

        result = cv2.bitwise_and(img, img, mask=mask)

        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        coords = cv2.findNonZero((gray > 0).astype(np.uint8))
        if coords is not None:
            x, y, ww, hh = cv2.boundingRect(coords)
            result = result[y:y + hh, x:x + ww]

        return result

    def apply_clahe(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def light_augment(self, img):
        # rotation: +/-10 degrees
        if self.rng.random() < 0.8:
            angle = self.rng.uniform(-10, 10)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        # small zoom: 0.92 - 1.08
        if self.rng.random() < 0.7:
            scale = self.rng.uniform(0.92, 1.08)
            h, w = img.shape[:2]
            nh, nw = int(h * scale), int(w * scale)
            resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

            if scale >= 1.0:
                y0 = (nh - h) // 2
                x0 = (nw - w) // 2
                img = resized[y0:y0 + h, x0:x0 + w]
            else:
                canvas = np.zeros_like(img)
                y0 = (h - nh) // 2
                x0 = (w - nw) // 2
                canvas[y0:y0 + nh, x0:x0 + nw] = resized
                img = canvas

        # slight shifts
        if self.rng.random() < 0.7:
            h, w = img.shape[:2]
            tx = int(self.rng.uniform(-0.03, 0.03) * w)
            ty = int(self.rng.uniform(-0.03, 0.03) * h)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        # brightness: 0.9 - 1.1
        if self.rng.random() < 0.7:
            factor = self.rng.uniform(0.9, 1.1)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # horizontal flip
        if self.rng.random() < 0.5:
            img = cv2.flip(img, 1)

        return img


class DRRun9:
    def __init__(self, data_path: str, img: int, batch: int, seed: int, clahe: bool):
        self.data_path = Path(data_path)
        self.img_size = (img, img)
        self.batch_size = batch
        self.seed = seed
        self.clahe = clahe

        self.model = None
        self.train_seq = None
        self.valid_seq = None
        self.test_seq = None
        self.class_indices = None

    def load_dataset(self):
        print("Loading dataset...")
        if not self.data_path.exists():
            print(f"Error: Data path {self.data_path} does not exist!")
            return None

        imgpaths, labels = [], []
        class_dirs = [d for d in self.data_path.iterdir() if d.is_dir()]
        print(f"Found classes: {[d.name for d in class_dirs]}")

        for class_dir in class_dirs:
            for f in class_dir.iterdir():
                if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    imgpaths.append(str(f))
                    labels.append(class_dir.name)

        df = pd.DataFrame({"Paths": imgpaths, "Labels": labels})
        print(f"Total images loaded: {len(df)}")
        print("\nClass Distribution:")
        print(df["Labels"].value_counts())
        print()
        return df

    def create_generators(self, df):
        print("Creating data generators...")
        train, testval = train_test_split(df, test_size=0.2, shuffle=True, random_state=self.seed)
        valid, test = train_test_split(testval, test_size=0.5, shuffle=True, random_state=self.seed)

        print(f"Train set: {train.shape[0]} images")
        print(f"Validation set: {valid.shape[0]} images")
        print(f"Test set: {test.shape[0]} images\n")

        class_names = sorted(df["Labels"].unique().tolist())
        self.class_indices = {name: i for i, name in enumerate(class_names)}
        print("Class indices:", self.class_indices)

        self.train_seq = RetinalSequence(
            train, self.class_indices, img_size=self.img_size, batch_size=self.batch_size,
            shuffle=True, augment=True, clahe=self.clahe, seed=self.seed
        )
        self.valid_seq = RetinalSequence(
            valid, self.class_indices, img_size=self.img_size, batch_size=self.batch_size,
            shuffle=False, augment=False, clahe=self.clahe, seed=self.seed
        )
        self.test_seq = RetinalSequence(
            test, self.class_indices, img_size=self.img_size, batch_size=self.batch_size,
            shuffle=False, augment=False, clahe=self.clahe, seed=self.seed
        )

        return self.train_seq, self.valid_seq, self.test_seq

    def compute_capped_class_weights(self, cw_min: float, cw_max: float):
        y = np.array([self.class_indices[label] for label in self.train_seq.df["Labels"].values])
        classes = np.unique(y)
        w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        cw = {int(c): float(wi) for c, wi in zip(classes, w)}
        for k in cw:
            cw[k] = max(cw_min, min(cw_max, cw[k]))

        inv = {v: k for k, v in self.class_indices.items()}
        print("Class weights (balanced then capped):")
        for cid in sorted(cw.keys()):
            print(f"  {cid} ({inv[cid]}): {cw[cid]:.4f}")
        print()
        return cw

    def build_model(self, lr: float, gamma: float, alpha: float):
        print("Building model (EfficientNetB4 + smaller head + focal loss)...")

        base = EfficientNetB4(
            input_shape=(self.img_size[0], self.img_size[1], 3),
            include_top=False,
            weights="imagenet",
        )
        base.trainable = False

        num_classes = len(self.class_indices)

        inputs = keras.Input(shape=(self.img_size[0], self.img_size[1], 3))
        x = base(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation="elu")(x)  # smaller head
        outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

        self.model = keras.Model(inputs, outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=CategoricalFocalLoss(gamma=gamma, alpha=alpha),
            metrics=["accuracy"],
        )
        self.model.summary()
        return self.model

    def train_head(self, epochs, lr, class_weight, model_path, gamma, alpha):
        print(f"\nStage A: Train head (base frozen) for {epochs} epochs, lr={lr} ...")
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=CategoricalFocalLoss(gamma=gamma, alpha=alpha),
            metrics=["accuracy"],
        )
        callbacks = [
            ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=1),
            EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ]
        return self.model.fit(
            self.train_seq,
            validation_data=self.valid_seq,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )

    def fine_tune(self, epochs, ft_lr, unfreeze_last_n, class_weight, model_path, gamma, alpha):
        print(f"\nStage B: Fine-tune last {unfreeze_last_n} base layers for {epochs} epochs, lr={ft_lr} ...")

        base = None
        for layer in self.model.layers:
            if isinstance(layer, keras.Model) and layer.name.startswith("efficientnetb4"):
                base = layer
                break
        if base is None:
            base = self.model.layers[1]

        base.trainable = True
        if unfreeze_last_n > 0 and unfreeze_last_n < len(base.layers):
            for l in base.layers[:-unfreeze_last_n]:
                l.trainable = False

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=ft_lr),
            loss=CategoricalFocalLoss(gamma=gamma, alpha=alpha),
            metrics=["accuracy"],
        )
        callbacks = [
            ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=1),
            EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
        ]
        return self.model.fit(
            self.train_seq,
            validation_data=self.valid_seq,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )

    def evaluate(self):
        print("\nEvaluating model...")
        train_score = self.model.evaluate(self.train_seq, verbose=0)
        valid_score = self.model.evaluate(self.valid_seq, verbose=0)
        test_score = self.model.evaluate(self.test_seq, verbose=0)

        print(f"Train - Loss: {train_score[0]:.4f}, Accuracy: {train_score[1]:.4f}")
        print(f"Valid - Loss: {valid_score[0]:.4f}, Accuracy: {valid_score[1]:.4f}")
        print(f"Test  - Loss: {test_score[0]:.4f}, Accuracy: {test_score[1]:.4f}\n")
        return {"train": train_score, "valid": valid_score, "test": test_score}


def main():
    args = parse_args()
    set_gpu_config(args.mixed_precision)

    if not os.path.exists(args.data):
        print(f"Error: Data path {args.data} does not exist!")
        sys.exit(1)

    runner = DRRun9(args.data, args.img, args.batch, args.seed, args.clahe)
    df = runner.load_dataset()
    if df is None:
        sys.exit(1)

    runner.create_generators(df)

    if args.use_class_weights:
        class_weight = runner.compute_capped_class_weights(args.cw_min, args.cw_max)
    else:
        class_weight = None
        print("Class weights: DISABLED\n")

    runner.build_model(args.head_lr, args.focal_gamma, args.focal_alpha)

    runner.train_head(
        args.head_epochs, args.head_lr, class_weight, args.model,
        args.focal_gamma, args.focal_alpha
    )

    if os.path.exists(args.model):
        runner.model = keras.models.load_model(args.model, compile=False)
        runner.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.ft_lr),
            loss=CategoricalFocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha),
            metrics=["accuracy"],
        )

    if args.ft_epochs > 0:
        runner.fine_tune(
            args.ft_epochs, args.ft_lr, args.unfreeze, class_weight, args.model,
            args.focal_gamma, args.focal_alpha
        )

    if os.path.exists(args.model):
        runner.model = keras.models.load_model(args.model, compile=False)
        runner.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.ft_lr),
            loss=CategoricalFocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha),
            metrics=["accuracy"],
        )

    runner.evaluate()

    print(f"Saved best model to: {args.model}")
    print("Run9 training complete.")

    if args.eval_after:
        cmd = [
            sys.executable, "evaluate.py",
            "--data", args.data,
            "--model", args.model,
            "--run-tag", args.run_tag,
            "--img", str(args.img),
            "--batch", str(args.batch),
            "--seed", str(args.seed),
        ]
        print("\nRunning evaluate.py (CPU-only) to save confusion matrix + reports + plots:")
        print(" ".join(cmd))
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()