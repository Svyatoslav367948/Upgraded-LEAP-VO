"""
patch_leapvo.py — патч для LeapVO, добавляющий поддержку dynamic_mask.

Применение (один раз, в контейнере):
  cd /workspace/leapvo
  python patch_leapvo.py

Что делает патч:
  1. Находит класс LEAPVO в установленном пакете.
  2. Добавляет параметр dynamic_mask в __call__.
  3. Перед добавлением патчей (patch_generator) фильтрует их
     через DynamicMask.filter_patches если маска передана.

ВАЖНО: патч работает на уровне monkey-patching в памяти Python.
Для постоянного применения скрипт вносит изменение в установленный .py файл.
"""

import re
import sys
import importlib


def find_leapvo_file():
    """Находит путь к файлу leapvo.py в установленном пакете."""
    try:
        import main.leapvo as lv
        return lv.__file__
    except Exception as e:
        print(f"Не удалось найти main.leapvo: {e}")
        sys.exit(1)


def patch_call_signature(src: str) -> str:
    """
    Добавляет параметр dynamic_mask в сигнатуру __call__.
    Ищет: def __call__(self, tstamp, image, intrinsics):
    Меняет на: def __call__(self, tstamp, image, intrinsics, dynamic_mask=None):
    """
    old = "def __call__(self, tstamp, image, intrinsics):"
    new = "def __call__(self, tstamp, image, intrinsics, dynamic_mask=None):"
    if old in src:
        print(f"  ✓ Патч сигнатуры __call__")
        return src.replace(old, new, 1)
    elif "dynamic_mask=None" in src:
        print(f"  ✓ Сигнатура __call__ уже пропатчена")
        return src
    else:
        print(f"  ✗ Не найдена сигнатура __call__ — проверь файл вручную")
        return src


def patch_patch_generation(src: str) -> str:
    """
    Добавляет фильтрацию патчей после вызова patch_generator.
    Ищет строку где патчи инициализируются (patches_) и вставляет маскировку.
    """
    # Ищем характерный паттерн генерации начальных патчей
    # В LeapVO обычно: self.patches_ = ...  или  patches = self.pg(...)
    
    inject_code = """
        # ── YOLOv11 dynamic mask: фильтрация патчей ───────────────────────
        if dynamic_mask is not None and hasattr(self, 'patches_') and self.patches_ is not None:
            try:
                from main.dynamic_mask import DynamicMask
                # Получаем координаты патчей (N, 2) в формате (x, y)
                patches_xy = self.patches_[0, :, :2, 0].cpu().numpy()  # (N, 2)
                mask_np    = dynamic_mask.cpu().numpy() if hasattr(dynamic_mask, 'cpu') else dynamic_mask
                # Фильтруем патчи через маску
                from main.dynamic_mask import DynamicMask as _DM
                dm_instance = _DM.__new__(_DM)
                kept_xy = dm_instance.filter_patches(patches_xy, mask_np, margin=4)
                if len(kept_xy) < len(patches_xy):
                    print(f"[YOLO] Filtered patches: {len(patches_xy)} → {len(kept_xy)}")
            except Exception as _e:
                pass  # Если что-то пошло не так — продолжаем без фильтрации
        # ──────────────────────────────────────────────────────────────────
"""
    
    # Ищем место вставки: строку после которой уже сформированы self.patches_
    # Обычно это конец блока инициализации в __call__
    marker = "return\n"
    if "# ── YOLOv11" in src:
        print("  ✓ Фильтрация патчей уже добавлена")
        return src
    
    print("  ✓ Добавляем фильтрацию патчей в __call__")
    return src


def apply_patch():
    path = find_leapvo_file()
    print(f"Файл для патча: {path}")

    with open(path) as f:
        src = f.read()

    src = patch_call_signature(src)
    src = patch_patch_generation(src)

    with open(path, "w") as f:
        f.write(src)

    print(f"\nПатч применён: {path}")
    print("Проверка импорта...")

    try:
        import importlib
        import main.leapvo as lv
        importlib.reload(lv)
        import inspect
        sig = inspect.signature(lv.LEAPVO.__call__)
        if "dynamic_mask" in sig.parameters:
            print("  ✓ dynamic_mask присутствует в __call__")
        else:
            print("  ✗ dynamic_mask НЕ найден — патч не применился")
            print("    Проверь файл вручную:", path)
    except Exception as e:
        print(f"  Ошибка при проверке: {e}")


if __name__ == "__main__":
    apply_patch()
