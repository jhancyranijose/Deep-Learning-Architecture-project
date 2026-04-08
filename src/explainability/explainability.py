#!/usr/bin/env python3
"""
explainability.py

Use a saved Run7 model (dr_run7_model.keras) to generate Grad-CAM overlays,
raw heatmaps, and optional LIME explanations.

Designed for models trained by dr_run7.py:
- EfficientNetB4
- input size 380x380 by default
- preprocess_input from keras EfficientNet
- optional focal loss used during training

Examples

1) Explain 32 samples from the Run7 test split
python explainability.py \
  --data ./data/DR_Image \
  --model dr_run7_model.keras \
  --xai-dir xai_outputs \
  --xai-samples 32

2) Explain one specific image
python explainability.py \
  --data ./data/DR_Image \
  --model dr_run7_model.keras \
  --image ./data/DR_Image/Moderate\ DR/example.jpg \
  --xai-dir xai_outputs_single

3) Add LIME for a single image
python explainability.py \
  --data ./data/DR_Image \
  --model dr_run7_model.keras \
  --image ./data/DR_Image/Moderate\ DR/example.jpg \
  --xai-dir xai_outputs_single \
  --lime
"""

import os
import sys
import argparse
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

# Optional LIME
try:
    import lime
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    LIME_AVAILABLE = True
except Exception:
    LIME_AVAILABLE = False

warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Dataset root with class subfolders")
    p.add_argument("--model", type=str, required=True, help="Saved Run7 model .keras file")
    p.add_argument("--img", type=int, default=380)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--xai-dir", type=str, default="xai_outputs")
    p.add_argument("--xai-samples", type=int, default=32, help="Number of test samples to explain")
    p.add_argument("--image", type=str, default=None, help="Explain a single image instead of test split")
    p.add_argument("--class-index", type=int, default=None, help="Optional fixed class index to explain; default=predicted class")
    p.add_argument("--lime", action="store_true", help="Also generate LIME for single image or test samples")
    p.add_argument("--lime-samples", type=int, default=500, help="LIME num_samples")
    return p.parse_args()


@tf.keras.utils.register_keras_serializable(package="DR")
class CategoricalFocalLoss(keras.losses.Loss):
    """
    Serializable multiclass focal loss for one-hot labels.
    Compatible with dr_run7_model.keras loading.
    """
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


def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def set_gpu_config():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print("Could not set memory growth:", e)


class Run7Explainability:
    def __init__(self, data_path: str, model_path: str, img: int, batch: int, seed: int):
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.img_size = (img, img)
        self.batch_size = batch
        self.seed = seed

        self.model = None
        self.test_gen = None
        self.df_test = None

    # -------------------------------
    # Data
    # -------------------------------
    def load_dataset(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

        imgpaths, labels = [], []
        class_dirs = sorted([d for d in self.data_path.iterdir() if d.is_dir()], key=lambda x: x.name)
        if not class_dirs:
            raise ValueError(f"No class subdirectories found in {self.data_path}")

        print(f"Found classes: {[d.name for d in class_dirs]}")

        for class_dir in class_dirs:
            for f in class_dir.iterdir():
                if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    imgpaths.append(str(f))
                    labels.append(class_dir.name)

        df = pd.DataFrame({"Paths": imgpaths, "Labels": labels})
        print(f"Total images loaded: {len(df)}")
        print("\nClass distribution:")
        print(df["Labels"].value_counts())
        print()
        return df

    def create_test_generator(self, df):
        # Match dr_run7.py split logic: 80/10/10 with stratify
        train_df, testval_df = train_test_split(
            df,
            test_size=0.2,
            shuffle=True,
            random_state=self.seed,
            stratify=df["Labels"],
        )
        valid_df, test_df = train_test_split(
            testval_df,
            test_size=0.5,
            shuffle=True,
            random_state=self.seed,
            stratify=testval_df["Labels"],
        )

        self.df_test = test_df.reset_index(drop=True)

        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        self.test_gen = test_datagen.flow_from_dataframe(
            self.df_test,
            x_col="Paths",
            y_col="Labels",
            target_size=self.img_size,
            class_mode="categorical",
            color_mode="rgb",
            shuffle=False,
            batch_size=self.batch_size,
        )
        print("Class indices:", self.test_gen.class_indices)
        return self.test_gen

    # -------------------------------
    # Model
    # -------------------------------
    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        print(f"Loading model: {self.model_path}")
        self.model = keras.models.load_model(self.model_path, compile=False)

        # compile is not required for inference, but helpful for evaluate/predict consistency
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=2e-6),
            loss=CategoricalFocalLoss(),
            metrics=["accuracy"],
        )
        print("Loaded model successfully.")
        self.model.summary()
        return self.model

    # -------------------------------
    # Image utils
    # -------------------------------
    def load_raw_image(self, image_path: str):
        img = keras.utils.load_img(image_path, target_size=self.img_size)
        return keras.utils.img_to_array(img)

    def preprocess_image(self, raw_img: np.ndarray):
        x = np.expand_dims(raw_img.copy(), axis=0).astype(np.float32)
        x = preprocess_input(x)
        return x

    # -------------------------------
    # Robust Grad-CAM for nested Run7 model
    # -------------------------------
    def get_backbone(self):
        for layer in self.model.layers:
            if isinstance(layer, keras.Model) and "efficientnet" in layer.name.lower():
                return layer
        raise ValueError("Could not find EfficientNet backbone in loaded model.")

    def get_last_conv_layer_name(self, backbone):
        for layer in reversed(backbone.layers):
            if isinstance(layer, layers.Conv2D):
                return layer.name
        raise ValueError("Could not find a Conv2D layer in the backbone.")

    def _apply_layer_in_inference(self, layer, x):
        try:
            return layer(x, training=False)
        except TypeError:
            return layer(x)

    def build_classifier_from_last_conv(self, last_conv_layer_name: str):
        """
        Build a classifier model that starts from the chosen last conv activation
        and replays:
          1) remaining layers inside the EfficientNet backbone
          2) outer classifier head layers from the saved Run7 model

        This avoids the graph-mixing KeyError you hit before.
        """
        backbone = self.get_backbone()
        last_conv_layer = backbone.get_layer(last_conv_layer_name)

        classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:], name="gradcam_input")
        x = classifier_input

        # Replay remaining backbone layers after the chosen conv layer
        passed = False
        for layer in backbone.layers:
            if layer.name == last_conv_layer_name:
                passed = True
                continue
            if passed:
                x = self._apply_layer_in_inference(layer, x)

        # Replay outer model head after backbone
        backbone_idx = None
        for i, layer in enumerate(self.model.layers):
            if layer is backbone:
                backbone_idx = i
                break

        if backbone_idx is None:
            raise ValueError("Backbone index not found in outer model.")

        for layer in self.model.layers[backbone_idx + 1:]:
            x = self._apply_layer_in_inference(layer, x)

        classifier_model = keras.Model(classifier_input, x, name="gradcam_classifier")
        return classifier_model

    def make_gradcam_heatmap(self, preprocessed_batch: np.ndarray, pred_index: int = None, last_conv_layer_name: str = None):
        backbone = self.get_backbone()
        if last_conv_layer_name is None:
            last_conv_layer_name = self.get_last_conv_layer_name(backbone)

        last_conv_model = keras.Model(
            backbone.input,
            backbone.get_layer(last_conv_layer_name).output,
            name="last_conv_model",
        )
        classifier_model = self.build_classifier_from_last_conv(last_conv_layer_name)

        x = tf.convert_to_tensor(preprocessed_batch, dtype=tf.float32)

        with tf.GradientTape() as tape:
            last_conv_output = last_conv_model(x, training=False)
            tape.watch(last_conv_output)
            preds = classifier_model(last_conv_output, training=False)

            if pred_index is None:
                pred_index = int(tf.argmax(preds[0]))
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_output)
        if grads is None:
            raise RuntimeError("Gradients are None. Grad-CAM could not be computed.")

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = last_conv_output[0]

        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.nn.relu(heatmap)

        max_val = tf.reduce_max(heatmap)
        heatmap = tf.where(max_val > 0, heatmap / max_val, heatmap)

        return heatmap.numpy(), pred_index, last_conv_layer_name

    # -------------------------------
    # Save visuals
    # -------------------------------
    def save_gradcam_visualization(
        self,
        raw_img: np.ndarray,
        heatmap: np.ndarray,
        out_overlay_path: str,
        out_heatmap_path: str,
        title: str = "",
        alpha: float = 0.35,
    ):
        heatmap_resized = tf.image.resize(
            heatmap[..., np.newaxis],
            self.img_size,
            method="bilinear"
        ).numpy().squeeze()

        heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

        # raw heatmap
        plt.figure(figsize=(5, 5))
        plt.imshow(heatmap_resized, cmap="jet")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_heatmap_path, dpi=180, bbox_inches="tight", pad_inches=0)
        plt.close()

        # overlay
        plt.figure(figsize=(6, 6))
        plt.imshow(raw_img.astype("uint8"))
        plt.imshow(heatmap_resized, cmap="jet", alpha=alpha)
        if title:
            plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_overlay_path, dpi=180, bbox_inches="tight", pad_inches=0.1)
        plt.close()

    # -------------------------------
    # LIME
    # -------------------------------
    def predict_for_lime(self, images: np.ndarray):
        """
        LIME passes images in RGB, usually float64, unnormalized.
        We convert to float32 and EfficientNet preprocess_input.
        """
        x = images.astype(np.float32)
        x = preprocess_input(x)
        preds = self.model.predict(x, verbose=0)
        return preds

    def save_lime_visualization(self, raw_img: np.ndarray, out_path: str, num_samples: int = 500):
        if not LIME_AVAILABLE:
            print("LIME not installed; skipping LIME output.")
            return

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            raw_img.astype("double"),
            self.predict_for_lime,
            top_labels=5,
            hide_color=0,
            num_samples=num_samples,
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False,
        )

        plt.figure(figsize=(6, 6))
        plt.imshow(mark_boundaries(temp / 255.0, mask))
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.1)
        plt.close()

    # -------------------------------
    # Explain one image
    # -------------------------------
    def explain_one_image(self, image_path: str, out_dir: str, class_index: int = None, make_lime: bool = False, lime_samples: int = 500):
        out_dir = Path(out_dir)
        overlay_dir = out_dir / "overlays"
        heatmap_dir = out_dir / "heatmaps"
        lime_dir = out_dir / "lime"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        if make_lime:
            lime_dir.mkdir(parents=True, exist_ok=True)

        raw_img = self.load_raw_image(image_path)
        x = self.preprocess_image(raw_img)

        preds = self.model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        pred_conf = float(preds[pred_idx])

        heatmap, used_idx, last_conv_name = self.make_gradcam_heatmap(
            x,
            pred_index=class_index if class_index is not None else pred_idx,
        )

        stem = Path(image_path).stem
        overlay_path = overlay_dir / f"{stem}_overlay.png"
        heatmap_path = heatmap_dir / f"{stem}_heatmap.png"

        title = f"pred={pred_idx} conf={pred_conf:.3f} explained={used_idx} conv={last_conv_name}"
        self.save_gradcam_visualization(
            raw_img=raw_img,
            heatmap=heatmap,
            out_overlay_path=str(overlay_path),
            out_heatmap_path=str(heatmap_path),
            title=title,
        )

        print(f"Saved overlay: {overlay_path}")
        print(f"Saved heatmap: {heatmap_path}")

        if make_lime:
            lime_path = lime_dir / f"{stem}_lime.png"
            self.save_lime_visualization(raw_img, str(lime_path), num_samples=lime_samples)
            print(f"Saved LIME: {lime_path}")

    # -------------------------------
    # Explain test split
    # -------------------------------
    def explain_test_predictions(self, out_dir: str, max_samples: int = 32, class_index: int = None, make_lime: bool = False, lime_samples: int = 500):
        if self.test_gen is None:
            raise ValueError("Test generator not initialized. Call create_test_generator() first.")

        out_dir = Path(out_dir)
        overlay_dir = out_dir / "overlays"
        heatmap_dir = out_dir / "heatmaps"
        lime_dir = out_dir / "lime"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        if make_lime:
            lime_dir.mkdir(parents=True, exist_ok=True)

        idx_to_class = {v: k for k, v in self.test_gen.class_indices.items()}
        filepaths = list(self.test_gen.filepaths)
        true_classes = list(self.test_gen.classes)

        n = min(max_samples, len(filepaths))
        records = []

        print(f"Generating explanations for {n} test samples...")

        for i in range(n):
            image_path = filepaths[i]
            true_idx = int(true_classes[i])
            true_label = idx_to_class[true_idx]

            raw_img = self.load_raw_image(image_path)
            x = self.preprocess_image(raw_img)

            preds = self.model.predict(x, verbose=0)[0]
            pred_idx = int(np.argmax(preds))
            pred_label = idx_to_class[pred_idx]
            confidence = float(preds[pred_idx])

            heatmap, used_idx, last_conv_name = self.make_gradcam_heatmap(
                x,
                pred_index=class_index if class_index is not None else pred_idx,
            )

            stem = Path(image_path).stem
            overlay_path = overlay_dir / f"{i:04d}_{stem}_overlay.png"
            heatmap_path = heatmap_dir / f"{i:04d}_{stem}_heatmap.png"

            title = (
                f"true={true_label} | pred={pred_label} ({confidence:.3f}) | "
                f"explained={idx_to_class[used_idx]} | conv={last_conv_name}"
            )

            self.save_gradcam_visualization(
                raw_img=raw_img,
                heatmap=heatmap,
                out_overlay_path=str(overlay_path),
                out_heatmap_path=str(heatmap_path),
                title=title,
            )

            lime_path = None
            if make_lime:
                lime_path = lime_dir / f"{i:04d}_{stem}_lime.png"
                self.save_lime_visualization(raw_img, str(lime_path), num_samples=lime_samples)

            records.append({
                "index": i,
                "image_path": image_path,
                "true_class_idx": true_idx,
                "true_label": true_label,
                "pred_class_idx": pred_idx,
                "pred_label": pred_label,
                "confidence": confidence,
                "explained_class_idx": used_idx,
                "explained_label": idx_to_class[used_idx],
                "last_conv_layer": last_conv_name,
                "overlay_path": str(overlay_path),
                "heatmap_path": str(heatmap_path),
                "lime_path": str(lime_path) if lime_path is not None else "",
            })

            print(f"[{i+1}/{n}] {Path(image_path).name} -> pred={pred_label} ({confidence:.4f})")

        summary_path = out_dir / "gradcam_summary.csv"
        pd.DataFrame(records).to_csv(summary_path, index=False)

        print(f"\nSaved summary: {summary_path}")
        print(f"Overlay dir: {overlay_dir}")
        print(f"Heatmap dir: {heatmap_dir}")
        if make_lime:
            print(f"LIME dir: {lime_dir}")


def main():
    args = parse_args()
    set_seed(args.seed)
    set_gpu_config()

    runner = Run7Explainability(
        data_path=args.data,
        model_path=args.model,
        img=args.img,
        batch=args.batch,
        seed=args.seed,
    )

    runner.load_model()

    if args.image is not None:
        runner.explain_one_image(
            image_path=args.image,
            out_dir=args.xai_dir,
            class_index=args.class_index,
            make_lime=args.lime,
            lime_samples=args.lime_samples,
        )
        return

    df = runner.load_dataset()
    runner.create_test_generator(df)

    runner.explain_test_predictions(
        out_dir=args.xai_dir,
        max_samples=args.xai_samples,
        class_index=args.class_index,
        make_lime=args.lime,
        lime_samples=args.lime_samples,
    )


if __name__ == "__main__":
    main()