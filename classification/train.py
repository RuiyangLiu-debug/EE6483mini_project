import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

import torch
import torch.multiprocessing as mp
mp.set_start_method("fork", force=True)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

try:
    import cv2
    cv2.setNumThreads(0)
except Exception:
    pass

torch.backends.cudnn.benchmark = True

from ultralytics import YOLO


class Train:
    def __init__(self, path, model, scale, epochs, imgsz, batch, device,
                 workers, cache, compile_model, deterministic):
        self.path = Path(path)
        self.model = YOLO(model)

        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.device = device
        self.workers = workers
        self.cache = cache  # 'ram' / 'disk' / False

        self.compile_model = compile_model
        self.deterministic = deterministic

    def run(self, name, project):
        results = self.model.train(
            data=str(self.path),
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            device=self.device,
            name=name,
            project=project,
            verbose=True,
            workers=self.workers,
            cache=self.cache,
            amp=True,
            compile=self.compile_model,
            deterministic=self.deterministic,
            val=True,
            plots=True
        )

        print(f"Training complete. Results saved to: {results.save_dir}")
        best = Path(results.save_dir) / "weights" / "best.pt"
        if best.exists():
            print(f"Best model weights: {best}")
        else:
            print("Warning: best.pt not found!")
        return best

    def validate(self, weights_path=None):
        model_path = weights_path or "runs/classify/train/weights/best.pt"
        print(f"Validating model: {model_path}")
        model = YOLO(model_path)
        metrics = model.val(data=str(self.path), imgsz=self.imgsz, batch=self.batch)
        print("Validation metrics:", metrics)
        return metrics


if __name__ == "__main__":
    train_data_path = "/workspace/code/datasets"
    model = "/workspace/code/classification/yolo11s-cls.yaml"
    scale = "s"
    epoch = 150
    imgsz = 384
    batch_size = 64
    workers = 8
    cache = "disk"
    compile_model = False
    deterministic = False

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device = "0"
        print(f"Detected GPU: {gpu_name}")
    else:
        device = "cpu"
        print("No GPU found, using CPU")

    trainer = Train(
        train_data_path, model, scale, epoch, imgsz, batch_size,
        device, workers, cache, compile_model, deterministic
    )
    task_name = "classification_task"
    project_name = "classification_projects"
    best_weights = trainer.run(task_name, project_name)
