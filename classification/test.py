import torch
from pathlib import Path
from ultralytics import YOLO
import cv2
import csv
import re

class Test:
    def __init__(self, weights, imgsz=384, batch=64, device=None):
        self.weights = Path(weights)

        if device is None:
            device = "0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.imgsz = imgsz
        self.batch = batch

        print(f"Loading model from: {self.weights}")
        self.model = YOLO(str(self.weights))

    def predict_single(self, image_path, save_dir=None, prefix=""):
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        results = self.model.predict(
            source=str(img_path),
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            save=False
        )
        r = results[0]
        if r.probs is None:
            raise RuntimeError("Not a classification model or no probabilities returned.")

        # we only consider the Top1 result
        top1_idx = int(r.probs.top1)
        label = r.names[top1_idx]
        score = float(getattr(r.probs, "top1conf", r.probs.data[top1_idx]))
        print(f">>> {img_path.name} -> {label} ({score:.4f})")

        # visulization on image and save
        out_path = None
        if save_dir is not None:
            # generate output path
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"{prefix}{img_path.stem}_{label}_{score}.jpg"
            out_path = save_dir / out_name

            img = cv2.imread(str(img_path))
            # show label and score on image
            text = f"{label}: {score:.4f}"
            # adjust font scale and thickness for better visibility
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            x1, y1 = 10, 10
            x2, y2 = x1 + tw + 20, y1 + th + 20
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
            alpha = 0.5
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
            cv2.putText(
                img,
                text,
                (x1 + 10, y1 + th + 5),
                font,
                font_scale,
                (0, 255, 0),
                thickness,
                cv2.LINE_AA
            )
            cv2.imwrite(str(out_path), img)
            print(f"Saved to: {out_path}")

        return label, score, out_path
    
    def predict_folder_save_images(self, folder, save_dir, prefix=""):
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # find all image files in the folder
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        img_paths = [
            p for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix.lower() in exts
        ]
        if not img_paths:
            print(f"No images found in: {folder}")
            return
        print(f"Found {len(img_paths)} images in {folder}")
        print(f"Saving annotated results to: {save_dir}")
        for i, img_path in enumerate(img_paths, start=1):
            print(f"[{i}/{len(img_paths)}] Processing {img_path.name} ...")
            self.predict_single(
                image_path=str(img_path),
                save_dir=save_dir,
                prefix=prefix
            )
        print("All images processed.")

    def extract_number(self, s):
        nums = re.findall(r"\d+", s)
        if not nums:
            return float('inf')
        return int(nums[-1])
        
    def predict_folder_to_csv(self, folder, csv_path):
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        print(f"Predicting on folder: {folder}")
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        img_paths = [p for p in folder.iterdir() if p.suffix.lower() in exts]
        if len(img_paths) == 0:
            print("No images found in folder.")
            return

        # use sorted number order
        img_paths = sorted(img_paths, key=lambda p: self.extract_number(p.stem))
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "filename", "label", "name"])
            for img_path in img_paths:
                results = self.model.predict(
                    source=str(img_path),
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False
                )
                r = results[0]
                top1_idx = int(r.probs.top1)
                name = r.names[top1_idx]  # cat / dog
                label = 1 if name == "dog" else 0
                id_no_ext = img_path.stem
                writer.writerow([id_no_ext, img_path.name, label, name])
        print(f"CSV saved to: {csv_path}")

if __name__ == "__main__":
    weights = "/workspace/classification_projects/classification_task11/weights/best.pt"
    tester = Test(
        weights=weights,
        imgsz=384,
        batch=64,
        device=None
    )

    tester.predict_folder_save_images(folder="/workspace/code/test", save_dir="/workspace/code/image_results", prefix="test_predict_")

    test_folder = "/workspace/code/test"
    output_csv = "/workspace/code/test_results.csv"
    tester.predict_folder_to_csv(test_folder, output_csv)
