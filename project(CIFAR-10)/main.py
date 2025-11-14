# E:\Work\NTU\courses\6483\yolov11\main.py
from pathlib import Path
from ultralytics import YOLO
import torch

# —— 可选：全局设置 runs_dir / datasets_dir（部分版本支持）
try:
    from ultralytics.utils import SETTINGS
except Exception:
    SETTINGS = None

def main():
    BASE = Path(r"E:\Work\NTU\courses\6483\yolov11")
    RUNS = BASE / "runs"       # 训练/验证输出（best.pt、results、混淆矩阵等）
    DATA = BASE / "data"       # 数据集根目录（cifar10 会下载到这里）
    RUNS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    # 更新 Ultralytics 全局保存位置（若你的版本支持 SETTINGS）
    if SETTINGS:
        SETTINGS.update({
            "runs_dir": str(RUNS),
            "datasets_dir": str(DATA),
        })

    print("CUDA available:", torch.cuda.is_available())

    project = str(RUNS)       # 显式指定输出到 E:\...\yolov11\runs
    name    = "y11s_safe_win"

    model = YOLO("yolo11s-cls.pt")
    model.train(
        data="cifar10",       # 将自动下载到 DATA/cifar10（若 SETTINGS 生效）
        imgsz=32,
        epochs=100,
        batch=32,
        workers=4,
        amp=True,
        device=0,
        project=project,      # ← 产物根目录
        name=name,            # ← 子目录名：runs\y11s_safe_win\...

        auto_augment="randaugment",
        mixup=0.2, cutmix=0.2, erasing=0.3,
        cos_lr=True,
        warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
    )

    best = Path(project) / name / "weights" / "best.pt"
    print("[INFO] best.pt:", best.resolve())

    # 测试集评估也指定同一个 project，输出会在 runs 下新建 val 目录
    metrics = YOLO(str(best)).val(
        data="cifar10", imgsz=32, split="test", device=0,
        project=project, name=f"{name}_test"
    )
    print(f"Top-1={metrics.top1:.3f}  Top-5={metrics.top5:.3f}")
    print("[INFO] test outputs:", metrics.save_dir)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
