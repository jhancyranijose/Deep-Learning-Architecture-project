"""
Scenario5 : 380×380 + severity-aware cost-sensitive loss
"""

import os
import sys
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="./data/DR_Image")
    p.add_argument("--model", type=str, default="dr_run8_model.keras")
    p.add_argument("--run-tag", type=str, default="dr_run8")
    p.add_argument("--eval-after", action="store_true")

    p.add_argument("--img", type=int, default=380)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--head-epochs", type=int, default=25)
    p.add_argument("--ft-epochs", type=int, default=40)
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--ft-lr", type=float, default=1e-6)
    p.add_argument("--unfreeze", type=int, default=260)

    # cost-sensitive settings
    p.add_argument("--cost-power", type=float, default=2.0,
                   help="Cost grows as |i-j|**cost_power (2.0 is strong, 1.0 is mild)")
    p.add_argument("--cost-scale", type=float, default=1.0,
                   help="Scales expected cost multiplier strength (1.0 default)")

    # class weights optional
    p.add_argument("--use-class-weights", action="store_true")
    p.add_argument("--cw-min", type=float, default=0.7)
    p.add_argument("--cw-max", type=float, default=1.8)

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


def build_cost_matrix(num_classes: int, power: float) -> np.ndarray:
    """
    Cost(i, j) = |i - j| ** power
    i=true class index, j=predicted class index
    """
    C = np.zeros((num_classes, num_classes), dtype=np.float32)
    for i in range(num_classes):
        for j in range(num_classes):
            C[i, j] = float(abs(i - j) ** power)
    return C


@tf.keras.utils.register_keras_serializable(package="DR")
class CostSensitiveCELoss(keras.losses.Loss):
    """
    Cross-entropy multiplied by expected ordinal cost under predicted distribution.

    expected_cost = sum_j p_j * cost(true, j)
    loss = CE(y_true, y_pred) * (1 + cost_scale * expected_cost)

    - If model predicts far from true class -> higher penalty.
    - If near -> smaller penalty.
    """
    def __init__(self, cost_matrix, cost_scale=1.0, name="cost_sensitive_ce"):
        super().__init__(name=name)
        self.cost_matrix = tf.constant(cost_matrix, dtype=tf.float32)
        self.cost_scale = float(cost_scale)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        eps = tf.constant(1e-7, dtype=tf.float32)
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

        # Standard categorical cross entropy (per-sample)
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)  # shape (B,)

        # true class index
        true_idx = tf.argmax(y_true, axis=-1, output_type=tf.int32)  # (B,)

        # cost row for each sample: cost(true, all_classes)
        costs = tf.gather(self.cost_matrix, true_idx)  # (B, K)

        # expected cost under predicted probs
        exp_cost = tf.reduce_sum(costs * y_pred, axis=-1)  # (B,)

        return ce * (1.0 + self.cost_scale * exp_cost)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "cost_scale": self.cost_scale,
            "cost_matrix": self.cost_matrix.numpy().tolist(),
        })
        return cfg


class DRRun8:
    def __init__(self, data_path: str, img: int, batch: int, seed: int):
        self.data_path = Path(data_path)
        self.img_size = (img, img)
        self.batch_size = batch
        self.seed = seed

        self.model = None
        self.train_gen = None
        self.valid_gen = None
        self.test_gen = None

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

        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=18,
            zoom_range=0.18,
            width_shift_range=0.08,
            height_shift_range=0.08,
            brightness_range=(0.80, 1.20),
            horizontal_flip=True,
            fill_mode="nearest",
        )
        valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        self.train_gen = train_datagen.flow_from_dataframe(
            train, x_col="Paths", y_col="Labels",
            target_size=self.img_size, class_mode="categorical",
            color_mode="rgb", shuffle=True, batch_size=self.batch_size
        )
        self.valid_gen = valid_datagen.flow_from_dataframe(
            valid, x_col="Paths", y_col="Labels",
            target_size=self.img_size, class_mode="categorical",
            color_mode="rgb", shuffle=True, batch_size=self.batch_size
        )
        self.test_gen = test_datagen.flow_from_dataframe(
            test, x_col="Paths", y_col="Labels",
            target_size=self.img_size, class_mode="categorical",
            color_mode="rgb", shuffle=False, batch_size=self.batch_size
        )

        print("Class indices:", self.train_gen.class_indices)
        return self.train_gen, self.valid_gen, self.test_gen

    def compute_capped_class_weights(self, cw_min: float, cw_max: float):
        y = self.train_gen.classes
        classes = np.unique(y)
        w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        cw = {int(c): float(wi) for c, wi in zip(classes, w)}
        for k in cw:
            cw[k] = max(cw_min, min(cw_max, cw[k]))

        inv = {v: k for k, v in self.train_gen.class_indices.items()}
        print("Class weights (balanced then capped):")
        for cid in sorted(cw.keys()):
            print(f"  {cid} ({inv[cid]}): {cw[cid]:.4f}")
        print()
        return cw

    def build_model(self, lr: float, loss_obj: keras.losses.Loss):
        print("Building model (EfficientNetB4 + GAP head + cost-sensitive CE)...")

        base = EfficientNetB4(
            input_shape=(self.img_size[0], self.img_size[1], 3),
            include_top=False,
            weights="imagenet",
        )
        base.trainable = False

        num_classes = len(self.train_gen.class_indices)

        inputs = keras.Input(shape=(self.img_size[0], self.img_size[1], 3))
        x = base(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.35)(x)
        x = layers.Dense(256, activation="elu")(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Dense(128, activation="elu")(x)
        outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

        self.model = keras.Model(inputs, outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=loss_obj,
            metrics=["accuracy"],
        )
        self.model.summary()
        return self.model

    def train_head(self, epochs, lr, class_weight, model_path, loss_obj):
        print(f"\nStage A: Train head (base frozen) for {epochs} epochs, lr={lr} ...")
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=loss_obj,
            metrics=["accuracy"],
        )
        callbacks = [
            ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=1),
            EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ]
        return self.model.fit(
            self.train_gen,
            validation_data=self.valid_gen,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )

    def fine_tune(self, epochs, ft_lr, unfreeze_last_n, class_weight, model_path, loss_obj):
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
            loss=loss_obj,
            metrics=["accuracy"],
        )
        callbacks = [
            ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=1),
            EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
        ]
        return self.model.fit(
            self.train_gen,
            validation_data=self.valid_gen,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )

    def evaluate(self):
        print("\nEvaluating model...")
        train_score = self.model.evaluate(self.train_gen, verbose=0)
        valid_score = self.model.evaluate(self.valid_gen, verbose=0)
        test_score = self.model.evaluate(self.test_gen, verbose=0)
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

    runner = DRRun8(args.data, args.img, args.batch, args.seed)
    df = runner.load_dataset()
    if df is None:
        sys.exit(1)

    runner.create_generators(df)

    num_classes = len(runner.train_gen.class_indices)

    # IMPORTANT: Ensure correct severity order (matches class_indices printed)
    # Your class_indices currently:
    #   Healthy:0, Mild:1, Moderate:2, Proliferate:3, Severe:4
    # But semantic severity order is:
    #   Healthy(0), Mild(1), Moderate(2), Severe(3), Proliferate(4)
    #
    # If your mapping is NOT severity order, the cost matrix will be wrong.
    #
    # So we build cost matrix in "current class index order" to avoid mismatch.
    # If you later enforce severity mapping, then this becomes true ordinal.
    cost_matrix = build_cost_matrix(num_classes, power=args.cost_power)

    loss_obj = CostSensitiveCELoss(cost_matrix=cost_matrix, cost_scale=args.cost_scale)

    if args.use_class_weights:
        class_weight = runner.compute_capped_class_weights(args.cw_min, args.cw_max)
    else:
        class_weight = None
        print("Class weights: DISABLED\n")

    runner.build_model(args.head_lr, loss_obj)

    # Stage A
    runner.train_head(args.head_epochs, args.head_lr, class_weight, args.model, loss_obj)

    # Reload best
    if os.path.exists(args.model):
        runner.model = keras.models.load_model(args.model, compile=False)
        runner.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.ft_lr),
            loss=loss_obj,
            metrics=["accuracy"],
        )

    # Stage B
    if args.ft_epochs > 0:
        runner.fine_tune(args.ft_epochs, args.ft_lr, args.unfreeze, class_weight, args.model, loss_obj)

    # Reload best
    if os.path.exists(args.model):
        runner.model = keras.models.load_model(args.model, compile=False)
        runner.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.ft_lr),
            loss=loss_obj,
            metrics=["accuracy"],
        )

    runner.evaluate()
    print(f"Saved best model to: {args.model}")
    print("Run8 training complete.")

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