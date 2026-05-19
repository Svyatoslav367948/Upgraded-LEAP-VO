"""
main/dynamic_mask.py v2 — улучшенный модуль динамической маскировки.
 
Улучшения по сравнению с v1:
  1. Soft weighting вместо hard masking (вариант 2)
     weight = 1 - p_dynamic * alpha
     Патчи в динамических зонах не удаляются, а получают меньший вес.
 
  2. Temporal mask smoothing (вариант 6)
     mask_t = alpha * mask_t + (1-alpha) * mask_{t-1}
     Убирает покадровый шум детектора.
 
  3. Motion-aware intersection (вариант 4)
     dynamic_final = yolo_mask AND (motion_score > threshold)
     Припаркованные машины и стоящие люди НЕ маскируются.
     Только реально движущиеся объекты подавляются.
 
  4. Адаптивный порог покрытия
     Если YOLO маскирует >MAX_COVERAGE кадра — масштабируем
     веса чтобы не терять BA стабильность.
"""
 
import numpy as np
import torch
from typing import Optional, Tuple
 
# Классы COCO потенциально динамических объектов
DYNAMIC_CLASSES = {
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    6,   # train
    7,   # truck
    14,  # bird
    15,  # cat
    16,  # dog
    17,  # horse
    18,  # sheep
    19,  # cow
}
 
# Классы с высокой вероятностью движения (более агрессивное подавление)
HIGH_DYNAMIC_CLASSES = {0, 1, 2, 3, 5, 7}  # person, car, motorcycle, bus, truck
 
# Максимальная доля замаскированных пикселей до масштабирования весов
MAX_COVERAGE = 0.45
 
 
class DynamicMaskV2:
    """
    Улучшенный модуль динамической маскировки для LeapVO.
 
    Параметры:
      model_size      : "n", "s", "m" — размер YOLOv11 модели
      conf            : порог детекции
      soft_alpha      : сила подавления (0=нет эффекта, 1=полное удаление)
                        Рекомендуется 0.7–0.85 для баланса точность/стабильность
      temporal_alpha  : сглаживание по времени (0=нет памяти, 1=полная память)
                        0.3 означает 30% предыдущего кадра
      motion_thresh   : порог motion score для intersection (0=только YOLO)
      dilate_px       : расширение маски в пикселях
    """
 
    def __init__(
        self,
        model_size: str = "m",
        conf: float = 0.25,
        device: str = "cuda",
        soft_alpha: float = 0.55,
        temporal_alpha: float = 0.55,
        motion_thresh: float = 0.0,
        dilate_px: int = 4,
    ):
        from ultralytics import YOLO
 
        model_name = f"yolo26{model_size}-seg.pt"
        print(f"[DynamicMaskV2] Loading {model_name} ...")
        self.model  = YOLO(model_name)
        self.model.to(device)
 
        self.conf           = conf
        self.device         = device
        self.soft_alpha     = soft_alpha
        self.temporal_alpha = temporal_alpha
        self.motion_thresh  = motion_thresh
        self.dilate_px      = dilate_px
 
        # Состояние для temporal smoothing
        self._prev_prob_map: Optional[np.ndarray] = None
 
        print(
            f"[DynamicMaskV2] Ready | soft_alpha={soft_alpha} "
            f"temporal={temporal_alpha} motion_thresh={motion_thresh}"
        )
 
    def reset(self):
        """Сброс состояния между последовательностями."""
        self._prev_prob_map = None
 
    def _dilate(self, mask: np.ndarray) -> np.ndarray:
        if self.dilate_px <= 0:
            return mask
        import cv2
        k = self.dilate_px * 2 + 1
        kernel = np.ones((k, k), np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
 
    def _yolo_prob_map(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Возвращает карту вероятности динамики (H, W) float32 [0,1].
        Объекты HIGH_DYNAMIC_CLASSES получают вес 1.0,
        остальные DYNAMIC_CLASSES — вес 0.6.
        """
        H, W = frame_rgb.shape[:2]
        prob = np.zeros((H, W), dtype=np.float32)
 
        results = self.model(
            frame_rgb,
            conf=self.conf,
            verbose=False,
            classes=list(DYNAMIC_CLASSES),
        )
 
        if results and results[0].masks is not None:
            masks_data = results[0].masks.data      # (N, H', W')
            classes    = results[0].boxes.cls.cpu().numpy().astype(int)
            confs      = results[0].boxes.conf.cpu().numpy()
 
            for i, (cls_id, confidence) in enumerate(zip(classes, confs)):
                if cls_id not in DYNAMIC_CLASSES:
                    continue
 
                # Базовый вес: высокодинамичные классы сильнее
                base_weight = 1.0 if cls_id in HIGH_DYNAMIC_CLASSES else 0.6
 
                m = masks_data[i].cpu().numpy()
                if m.shape != (H, W):
                    import cv2
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
 
                # Вес = base_weight * уверенность детектора
                prob = np.maximum(prob, m * base_weight * float(confidence))
 
        return prob
 
    @torch.no_grad()
    def __call__(
        self,
        frame_rgb: np.ndarray,
        motion_map: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вычисляет soft weight map для одного кадра.
 
        Вход:
          frame_rgb  : np.ndarray (H, W, 3) uint8 RGB
          motion_map : np.ndarray (H, W) float32 — motion score из LeapVO
                       (опционально, для intersection)
 
        Выход:
          weight_map : np.ndarray (H, W) float32 [0, 1]
                       1.0 = статика (полный вес)
                       0.0 = высокая динамика (нет веса)
          prob_map   : np.ndarray (H, W) float32 [0, 1] — вероятность динамики
        """
        H, W = frame_rgb.shape[:2]
 
        # 1. YOLO вероятностная карта
        prob_map = self._yolo_prob_map(frame_rgb)
 
        # 2. Temporal smoothing (вариант 6)
        if self._prev_prob_map is not None and self._prev_prob_map.shape == (H, W):
            prob_map = (
                self.temporal_alpha * self._prev_prob_map
                + (1 - self.temporal_alpha) * prob_map
            )
        self._prev_prob_map = prob_map.copy()
 
        # 3. Motion-aware intersection (вариант 4)
        # Если передан motion_map — подавляем только реально движущиеся объекты
        if motion_map is not None and self.motion_thresh > 0:
            # Нормируем motion_map в [0,1]
            motion_norm = motion_map.astype(np.float32)
            if motion_norm.max() > 1e-6:
                motion_norm = motion_norm / motion_norm.max()
            # Intersection: динамический только если YOLO И motion > threshold
            motion_mask = motion_norm > self.motion_thresh
            # Мягкое пересечение: prob * motion_weight
            prob_map = prob_map * (
                0.3 + 0.7 * motion_norm  # статичные объекты сохраняют 30% веса
            ) * motion_mask.astype(np.float32) + prob_map * 0.3 * (~motion_mask)
        
        # 4. Дилатация бинарной маски для краёв объектов
        binary_mask = prob_map > 0.3
        if self.dilate_px > 0:
            binary_mask = self._dilate(binary_mask)
            # Применяем дилатацию только к prob_map через маску
            prob_map = np.where(binary_mask, np.maximum(prob_map, 0.3), prob_map)
 
        # 5. Soft weight map (вариант 2)
        # weight = 1 - soft_alpha * prob
        # При prob=1.0: weight = 1 - soft_alpha (например 0.2 при alpha=0.8)
        # При prob=0.0: weight = 1.0 (полный вес)
        weight_map = np.clip(1.0 - self.soft_alpha * prob_map, 0.25, 1.0)
 
        # 6. Адаптивное масштабирование при высоком покрытии
        coverage = (prob_map > 0.3).mean()
        if coverage > MAX_COVERAGE:
            # Слишком много замаскировано — поднимаем минимальный вес
            # чтобы BA не терял стабильность
            scale = MAX_COVERAGE / coverage
            weight_map = weight_map * scale + (1 - scale) * np.ones_like(weight_map)
 
        return weight_map.astype(np.float32), prob_map
 
    def filter_patches_soft(
        self,
        patches_xy: np.ndarray,
        weight_map: np.ndarray,
        min_keep_ratio: float = 0.6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Возвращает веса для каждого патча (soft, не удаляет).
 
        patches_xy     : (N, 2) координаты центров патчей (x, y)
        weight_map     : (H, W) float32 карта весов
        min_keep_ratio : минимальная доля патчей с weight > 0.5
 
        Возвращает:
          patches_xy   : те же патчи (не фильтруем жёстко)
          patch_weights: (N,) float32 веса для каждого патча
        """
        H, W = weight_map.shape
        N = len(patches_xy)
 
        if N == 0:
            return patches_xy, np.ones(0, dtype=np.float32)
 
        xs = patches_xy[:, 0].astype(int).clip(0, W - 1)
        ys = patches_xy[:, 1].astype(int).clip(0, H - 1)
 
        # Усредняем вес в окрестности 4×4 пикселя вокруг центра патча
        patch_weights = np.zeros(N, dtype=np.float32)
        for i, (x, y) in enumerate(zip(xs, ys)):
            x0, x1 = max(0, x - 2), min(W, x + 3)
            y0, y1 = max(0, y - 2), min(H, y + 3)
            patch_weights[i] = weight_map[y0:y1, x0:x1].mean()
 
        # Проверка стабильности: если слишком мало хороших патчей — поднимаем
        good_ratio = (patch_weights > 0.5).mean()
        if good_ratio < min_keep_ratio:
            # Нормируем чтобы лучшие патчи получили вес 1.0
            min_w = patch_weights.min()
            max_w = patch_weights.max()
            if max_w > min_w + 1e-6:
                patch_weights = (patch_weights - min_w) / (max_w - min_w)
                patch_weights = 0.3 + 0.7 * patch_weights  # диапазон [0.3, 1.0]
 
        return patches_xy, patch_weights
 
    def get_coverage_stats(self, prob_map: np.ndarray) -> dict:
        """Статистика покрытия для логирования."""
        binary = prob_map > 0.3
        return {
            "coverage_pct": float(binary.mean() * 100),
            "high_dynamic_pct": float((prob_map > 0.7).mean() * 100),
            "mean_prob": float(prob_map.mean()),
        }
DynamicMask = DynamicMaskV2
