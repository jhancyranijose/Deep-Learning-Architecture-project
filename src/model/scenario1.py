"""
Scenario1 : Improved head + class weighting + two-stage fine-tuning
"""

import os
import sys
import argparse
import warnings
import logging
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("Could not set memory growth:", e)

from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="./data/DR_Image", help="Dataset root folder")
    p.add_argument("--model", type=str, default="dr_run2_model.keras", help="Where to save model")
    p.add_argument("--img", type=int, default=224, help="Image size (img x img)")
    p.add_argument("--batch", type=int, default=20, help="Batch size")
    p.add_argument("--seed", type=int, default=123, help="Random seed for split")
    p.add_argument("--head-epochs", type=int, default=25, help="Epochs for frozen-base head training")
    p.add_argument("--ft-epochs", type=int, default=10, help="Epochs for fine-tuning")
    p.add_argument("--head-lr", type=float, default=3e-4, help="LR for head training")
    p.add_argument("--ft-lr", type=float, default=1e-5, help="LR for fine-tuning")
    p.add_argument("--unfreeze", type=int, default=40, help="Unfreeze last N layers of EfficientNet during fine-tune")
    p.add_argument("--eval-after", action="store_true", help="Run evaluate.py after training (CPU-only, saves outputs)")
    p.add_argument("--run-tag", type=str, default="dr_run2", help="Run tag used by evaluate.py (folders)")
    return p.parse_args()


class DRRun2:
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
        """Same directory loading style as dr5.py"""
        print("Loading dataset...")
        if not self.data_path.exists():
            print(f"Error: Data path {self.data_path} does not exist!")
            return None

        imgpaths, labels = [], []
        class_dirs = [d for d in self.data_path.iterdir() if d.is_dir()]
        if not class_dirs:
            print(f"Error: No subdirectories found in {self.data_path}")
            return None

        print(f"Found classes: {[d.name for d in class_dirs]}")
        for class_dir in class_dirs:
            for f in class_dir.iterdir():
                if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    imgpaths.append(str(f))
                    labels.append(class_dir.name)

        if not imgpaths:
            print("Error: No images found!")
            return None

        df = pd.DataFrame({"Paths": imgpaths, "Labels": labels})
        print(f"Total images loaded: {len(df)}")
        print("\nClass Distribution:")
        print(df["Labels"].value_counts())
        print()
        return df

    def create_generators(self, df):
        """Same split logic as dr5.py, but better preprocessing + augmentation."""
        print("Creating data generators...")

        train, testval = train_test_split(df, test_size=0.2, shuffle=True, random_state=self.seed)
        valid, test = train_test_split(testval, test_size=0.5, shuffle=True, random_state=self.seed)

        print(f"Train set: {train.shape[0]} images")
        print(f"Validation set: {valid.shape[0]} images")
        print(f"Test set: {test.shape[0]} images\n")

        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=15,
            zoom_range=0.10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            brightness_range=(0.85, 1.15),
            horizontal_flip=True,
            fill_mode="nearest",
        )
        valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        self.train_gen = train_datagen.flow_from_dataframe(
            train,
            x_col="Paths",
            y_col="Labels",
            target_size=self.img_size,
            class_mode="categorical",
            color_mode="rgb",
            shuffle=True,
            batch_size=self.batch_size,
        )
        self.valid_gen = valid_datagen.flow_from_dataframe(
            valid,
            x_col="Paths",
            y_col="Labels",
            target_size=self.img_size,
            class_mode="categorical",
            color_mode="rgb",
            shuffle=True,
            batch_size=self.batch_size,
        )
        self.test_gen = test_datagen.flow_from_dataframe(
            test,
            x_col="Paths",
            y_col="Labels",
            target_size=self.img_size,
            class_mode="categorical",
            color_mode="rgb",
            shuffle=False,
            batch_size=self.batch_size,
        )

        return self.train_gen, self.valid_gen, self.test_gen

    def get_class_weights(self):
        """Key fix: makes Severe/Proliferate matter in training."""
        y = self.train_gen.classes
        classes = np.unique(y)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

        inv = {v: k for k, v in self.train_gen.class_indices.items()}
        print("Class weights:")
        for cid, w in class_weight.items():
            print(f"  {cid} ({inv[cid]}): {w:.4f}")
        print()
        return class_weight

    def build_model(self, lr: float):
        print("Building model (EfficientNetB3 + GAP head)...")

        base = EfficientNetB3(
            input_shape=(self.img_size[0], self.img_size[1], 3),
            include_top=False,
            weights="imagenet",
        )
        base.trainable = False

        inputs = keras.Input(shape=(self.img_size[0], self.img_size[1], 3))
        x = base(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)  # ✅ better than Flatten for generalization
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation="elu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation="elu")(x)
        num_classes = len(self.train_gen.class_indices)
        outputs = layers.Dense(num_classes, activation="softmax")(x)

        self.model = keras.Model(inputs, outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model.summary()
        return self.model

    def train_head(self, epochs: int, lr: float, class_weight: dict, model_path: str):
        print(f"\nStage A: Train head (base frozen) for {epochs} epochs, lr={lr} ...")

        callbacks = [
            ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=1),
            EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ]

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return self.model.fit(
            self.train_gen,
            validation_data=self.valid_gen,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )

    def fine_tune(self, epochs: int, ft_lr: float, unfreeze_last_n: int, class_weight: dict, model_path: str):
        print(f"\nStage B: Fine-tune last {unfreeze_last_n} base layers for {epochs} epochs, lr={ft_lr} ...")

        base = None
        for layer in self.model.layers:
            if isinstance(layer, keras.Model) and layer.name.startswith("efficientnetb3"):
                base = layer
                break
        if base is None:
            base = self.model.layers[1]

        base.trainable = True

        if unfreeze_last_n > 0:
            for l in base.layers[:-unfreeze_last_n]:
                l.trainable = False

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=ft_lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = [
            ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=1),
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
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

    if not os.path.exists(args.data):
        print(f"Error: Data path {args.data} does not exist!")
        sys.exit(1)

    runner = DRRun2(args.data, args.img, args.batch, args.seed)

    df = runner.load_dataset()
    if df is None:
        sys.exit(1)

    runner.create_generators(df)
    class_weight = runner.get_class_weights()

    runner.build_model(lr=args.head_lr)

    runner.train_head(
        epochs=args.head_epochs,
        lr=args.head_lr,
        class_weight=class_weight,
        model_path=args.model,
    )

    if os.path.exists(args.model):
        runner.model = keras.models.load_model(args.model)

    if args.ft_epochs > 0:
        runner.fine_tune(
            epochs=args.ft_epochs,
            ft_lr=args.ft_lr,
            unfreeze_last_n=args.unfreeze,
            class_weight=class_weight,
            model_path=args.model,
        )

    if os.path.exists(args.model):
        runner.model = keras.models.load_model(args.model)

    runner.evaluate()

    print(f"Saved best model to: {args.model}")
    print("Run2 training complete.")

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
        print("\nRunning evaluation.py (CPU-only) to save confusion matrix + reports + plots:")
        print(" ".join(cmd))
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()