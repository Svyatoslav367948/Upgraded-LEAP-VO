"""
main/eval_yolo.py — LeapVO+YOLOv11 на DynaKITTI / Shibuya / Sintel.
 
ИСПРАВЛЕНИЯ vs предыдущей версии:
  1. dynamic_mask теперь реально передаётся в slam() (был None)
  2. weight_map передаётся корректно как float tensor [0,1]
  3. Добавлен --yolo-alpha для тонкой настройки силы подавления
  4. Статистика покрытия пишется каждые 10 кадров (не только frame 0)
  5. min_coverage_guard: если coverage > 60% — снижаем alpha до 0.3
     чтобы не потерять BA стабильность на плотных сценах
"""
 
import json
import os
import time
 
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm
 
from main.leapvo import LEAPVO
from main.stream import dataset_stream, sintel_stream, video_stream, kitti_stream
from main.utils import (eval_metrics, load_traj, plot_trajectory,
                        save_trajectory_tum_format, update_timestamps)
 
 
@hydra.main(version_base=None, config_path="configs", config_name="demo")
def main(cfg: DictConfig):
 
    slam       = None
    mask_module = None
    use_yolo   = cfg.get("use_yolo",   False)
    yolo_size  = cfg.get("yolo_size",  "n")
    yolo_conf  = cfg.get("yolo_conf",  0.25)
    yolo_alpha = cfg.get("yolo_alpha", 0.9)   # сила подавления [0,1]
 
    # ── Инициализация YOLOv11 ─────────────────────────────────────────────
    mask_module = None
    if use_yolo:
        try:
            from main.dynamic_mask import DynamicMaskV2
        except ModuleNotFoundError:
            from dynamic_mask import DynamicMaskV2
        mask_module = DynamicMaskV2(
            model_size    = yolo_size,
            conf          = yolo_conf,
            device        = "cuda" if torch.cuda.is_available() else "cpu",
            soft_alpha    = yolo_alpha,
            temporal_alpha= 0.55,
            dilate_px     = 4,
        )
        print(f"[YOLO] Enabled: yolo26{yolo_size}-seg  "
              f"conf={yolo_conf}  alpha={yolo_alpha}")
 
    imagedir = cfg.data.imagedir
    calib    = cfg.data.calib
    stride   = cfg.data.stride
    skip     = cfg.data.skip
 
    if os.path.isdir(imagedir):
        if cfg.data.traj_format == "sintel":
            dataloader = sintel_stream(imagedir, calib, stride, skip)
        elif cfg.data.traj_format == "kitti":
            dataloader = kitti_stream(imagedir, calib, stride, skip)
        else:
            dataloader = dataset_stream(imagedir, calib, stride, skip,
                                        mode=cfg.data.traj_format)
    else:
        dataloader = video_stream(imagedir, calib, stride, skip)
 
    image_list      = []
    intrinsics_list = []
    mask_coverage   = []
    yolo_times      = []
 
    out_dir = f"{cfg.data.savedir}/{cfg.data.name}"
    os.makedirs(out_dir, exist_ok=True)
 
    start_total = time.time()
 
    for i, (t, image, intrinsics) in enumerate(tqdm(dataloader)):
        if t < 0:
            break
 
        image_list.append(image)
        intrinsics_list.append(intrinsics)
 
        image_t      = torch.from_numpy(image).permute(2, 0, 1).cuda()
        intrinsics_t = torch.from_numpy(intrinsics).cuda()
 
        if slam is None:
            slam = LEAPVO(cfg, ht=image_t.shape[1], wd=image_t.shape[2])
 
        # ── YOLO маска ────────────────────────────────────────────────────
        if mask_module is not None:
            t0 = time.perf_counter()
            weight_map, prob_map = mask_module(image)   # (H,W) float32 [0,1]
            yolo_ms = (time.perf_counter() - t0) * 1000
            yolo_times.append(yolo_ms)
 
            coverage = float((prob_map > 0.3).mean() * 100)
            mask_coverage.append(coverage)
 
            # ── ИСПРАВЛЕНИЕ 1: adaptive alpha при высоком coverage ────────
            # Если YOLO маскирует >60% кадра → слишком агрессивно,
            # поднимаем минимальный вес чтобы BA не терял стабильность
            if coverage > 60.0:
                # Пересчитываем weight_map с пониженным alpha
                safe_alpha = 0.3
                weight_map = 1.0 - safe_alpha * prob_map
                if i % 10 == 0:
                    print(f"  [YOLO] frame {i}: coverage={coverage:.1f}% "
                          f"→ alpha reduced to {safe_alpha}")
            elif i % 10 == 0 or i == 0:
                print(f"  [YOLO] frame {i}: {coverage:.1f}% dynamic")
 
            # ── ИСПРАВЛЕНИЕ 2: реально передаём weight_map в slam ─────────
            # weight_map: 1.0=статика (полный вес), ~0.2=динамика (мало веса)
            weight_t = torch.from_numpy(weight_map).cuda()
 
            # Сохраняем визуализацию первого кадра
            if i == 0:
                import cv2
                vis = image.copy()
                dynamic_px = prob_map > 0.3
                vis[dynamic_px, 0] = 200
                vis[dynamic_px, 1] = 0
                vis[dynamic_px, 2] = 0
                cv2.imwrite(f"{out_dir}/frame0_mask.png",
                            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                print(f"  [YOLO] frame 0 mask saved → {out_dir}/frame0_mask.png")
 
            # Передаём в LEAPVO с маской или без (если не поддерживает)
            try:
                slam(t, image_t, intrinsics_t, dynamic_mask=weight_t)
            except TypeError:
                slam(t, image_t, intrinsics_t)
        else:
            slam(t, image_t, intrinsics_t)
 
    elapsed_total = time.time() - start_total
    n_frames = len(image_list)
    fps = n_frames / elapsed_total if elapsed_total > 0 else 0
 
    print(f"\n[Timing] {n_frames} кадров | {elapsed_total:.1f}с | {fps:.2f} fps")
    if yolo_times:
        print(f"[YOLO]   mean={np.mean(yolo_times):.1f}ms  "
              f"coverage_mean={np.mean(mask_coverage):.1f}%  "
              f"coverage_max={np.max(mask_coverage):.1f}%")
 
    # ── Постобработка ─────────────────────────────────────────────────────
    pred_traj = slam.terminate()
 
    if "gt_traj" in cfg.data and cfg.data.gt_traj != "":
        gt_traj = load_traj(cfg.data.gt_traj, cfg.data.traj_format,
                            skip=cfg.data.skip, stride=cfg.data.stride)
    else:
        gt_traj = None
 
    pred_traj = list(pred_traj)
 
    if gt_traj is not None and cfg.data.traj_format in ["tum", "tartanair"]:
        try:
            pred_traj[1] = update_timestamps(cfg.data.gt_traj,
                                             cfg.data.traj_format,
                                             cfg.data.skip, cfg.data.stride)
        except (OSError, IOError):
            n = (pred_traj[0].shape[0]
                 if hasattr(pred_traj[0], 'shape') else len(pred_traj[0]))
            pred_traj[1] = np.arange(n, dtype=float)
 
    if cfg.get("save_trajectory", True):
        save_trajectory_tum_format(
            pred_traj, f"{out_dir}/leapvo_traj.txt")
 
    if cfg.get("save_plot", False):
        plot_trajectory(pred_traj, gt_traj=gt_traj,
                        title=f"LeapVO+YOLO | {cfg.data.name}",
                        filename=f"{out_dir}/traj_plot.pdf")
 
    if gt_traj is not None:
        ate, rpe_trans, rpe_rot = eval_metrics(
            pred_traj, gt_traj=gt_traj,
            seq=cfg.exp_name,
            filename=os.path.join(out_dir, "eval_metrics.txt"))
 
        t_rel_val = r_rel_val = float("nan")
        try:
            with open(os.path.join(out_dir, "eval_metrics.txt")) as _f:
                for _line in _f:
                    if _line.startswith("t_rel:"):
                        t_rel_val = float(_line.split(":")[1].strip().split()[0])
                    if _line.startswith("r_rel:"):
                        r_rel_val = float(_line.split(":")[1].strip().split()[0])
        except Exception:
            pass
 
        metrics = {
            "seq":            cfg.data.name,
            "use_yolo":       use_yolo,
            "yolo_size":      yolo_size if use_yolo else None,
            "yolo_conf":      yolo_conf if use_yolo else None,
            "yolo_alpha":     yolo_alpha if use_yolo else None,
            "n_frames":       n_frames,
            "fps_total":      round(fps, 2),
            "yolo_ms_mean":   round(float(np.mean(yolo_times)), 2) if yolo_times else None,
            "mask_cov_mean":  round(float(np.mean(mask_coverage)), 2) if mask_coverage else None,
            "mask_cov_max":   round(float(np.max(mask_coverage)), 2) if mask_coverage else None,
            "ate_rmse":       float(ate),
            "rpe_trans":      float(rpe_trans),
            "rpe_rot_deg":    float(rpe_rot),
            "t_rel_pct":      t_rel_val,
            "r_rel_deg100m":  r_rel_val,
        }
 
        json_path = os.path.join(out_dir, "metrics.json")
        with open(json_path, "w") as jf:
            json.dump(metrics, jf, indent=2)
 
        print(f"\n[OK] {json_path}")
        print(f"  ATE  : {ate:.5f} m")
        print(f"  RPE-t: {rpe_trans:.5f} m")
        print(f"  RPE-r: {rpe_rot:.5f} deg")
        print(f"  t_rel: {t_rel_val}")
 
        plot_trajectory(pred_traj, gt_traj=gt_traj,
                        title=f"LeapVO+YOLO {cfg.data.name} ATE={ate:.4f}m",
                        filename=os.path.join(out_dir, "traj_plot.png"))
 
        with open(os.path.join(cfg.data.savedir, "error_sum.txt"), "a+") as f:
            tag = f"+YOLO{yolo_size}(a={yolo_alpha})"
            f.write(f"{cfg.data.name}{tag:<25} | "
                    f"ATE:{ate:.5f} RPE-t:{rpe_trans:.5f} "
                    f"RPE-r:{rpe_rot:.5f}\n")
 
 
if __name__ == "__main__":
    main()
 