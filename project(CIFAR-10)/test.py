from ultralytics import YOLO
import multiprocessing


def main():
    best = r"E:\Work\NTU\courses\6483\yolov11\runs\y11s_safe_win\weights\best.pt"
    model = YOLO(best)
    metrics = model.val(
        data=r"E:\Work\NTU\courses\6483\yolov11\test\cifar-10.yaml",
        split="test",
        imgsz=32,
        device=0,
        project=r"E:\Work\NTU\courses\6483\yolov11\runs_cifar10_local",
        name="y11s_safe_win_test",
        exist_ok=True,
        workers=0,         # 建议先设为 0，避免多进程再出幺蛾子
    )
    print("Top-1:", metrics.top1, "Top-5:", metrics.top5)
    print("saved to:", metrics.save_dir)


if __name__ == "__main__":
    multiprocessing.freeze_support()  # 按报错提示加上这一句
    main()
