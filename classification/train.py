import os
import sys
from pathlib import Path

# 0) 先把各类数值库的内部线程数锁到 1，避免与 DataLoader 多进程抢核
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

# 1) 选择合适的多进程启动方式（Linux 下 fork 最省开销）
import torch
import torch.multiprocessing as mp
mp.set_start_method("fork", force=True)
torch.set_num_threads(1)          # 计算核内线程
torch.set_num_interop_threads(1)  # 内核间并行

# 2) 关掉 OpenCV 的内部线程，把并行交给 DataLoader 进程
try:
    import cv2
    cv2.setNumThreads(0)
except Exception:
    pass

# 3) 选用更快的卷积算法：非确定性时开启 cudnn.benchmark
torch.backends.cudnn.benchmark = True

from ultralytics import YOLO


class Train:
    def __init__(self, path, model, scale, epochs, imgsz, batch, device,
                 workers, cache, compile_model, deterministic):
        self.path = Path(path)

        # ★ 如用 .yaml 会自动构建模型；也可直接传权重 .pt 进行 finetune
        self.model = YOLO(model)

        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.device = device

        # ★ workers 就是 PyTorch DataLoader 的 num_workers（多进程数）
        self.workers = workers

        # ★ 强烈建议使用 'ram'（前提是内存够），减少 I/O + 解码开销
        #   'disk' 也行，比 False 好很多
        self.cache = cache  # 'ram' / 'disk' / False

        self.compile_model = compile_model
        self.deterministic = deterministic

    def run(self, name, project):
        # Ultralytics 会把这些 kwargs 透传给内部的 DataLoader / 训练循环
        results = self.model.train(
            data=str(self.path),
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            device=self.device,
            name=name,
            project=project,
            verbose=True,

            # ★ 多进程数据加载（关键）
            workers=self.workers,

            # ★ 缓存数据（强烈推荐 'ram'，内存不够再用 'disk'）
            cache=self.cache,

            # 训练侧优化
            amp=True,                 # 混合精度
            compile=self.compile_model,
            deterministic=self.deterministic,

            # 其余保持默认
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
    # === 你的原始超参数 ===
    train_data_path = "/workspace/code/classification/datasets"
    model = "yolo11s-cls.yaml"
    scale = "s"
    epoch = 150
    # ★ 小模型易“喂不饱”，建议适当加大 batch/imgsz 提高 GPU 占用
    imgsz = 384          # 320->384/448 让 GPU 更忙
    batch_size = 64     # 显存允许可继续加到 192/256
    workers = 8          # 8~12 之间找稳态；若抖动大先用 8
    cache = "disk"        # ← 关键改动：'ram' 或 'disk'，不要 False
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
