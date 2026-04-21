import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import threading
import cv2
from PIL import Image, ImageTk
import numpy as np
import torch
from datetime import datetime
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import get_device, load_config
from src.model import DETROreDetector
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DETRDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ForGerion - DETR Detection")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 700)

        self.setup_style()

        self.model = None
        self.device = None
        self.config = None
        self.current_image = None
        self.current_image_path = None
        self.results = None
        self.is_processing = False

        self.setup_ui()
        self.load_model()

    def setup_style(self):
        """Настройка стиля и цветовой схемы"""
        self.bg_primary = "#0a0e27"
        self.bg_secondary = "#1a1f3a"
        self.accent = "#00d9ff"
        self.accent_dark = "#00b8d4"
        self.text_primary = "#ffffff"
        self.text_secondary = "#b0b8d4"
        self.success = "#00ff88"
        self.warning = "#ff6b6b"

        self.root.configure(bg=self.bg_primary)

        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TFrame", background=self.bg_primary)
        style.configure(
            "TLabel", background=self.bg_primary, foreground=self.text_primary
        )
        style.configure(
            "TButton", background=self.bg_secondary, foreground=self.text_primary
        )
        style.map("TButton", background=[("active", self.accent_dark)])

    def setup_ui(self):
        """Создаёт интерфейс"""
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Левая панель
        left_panel = tk.Frame(main, bg=self.bg_secondary, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=0, pady=0)
        left_panel.pack_propagate(False)

        # Заголовок
        title_frame = tk.Frame(left_panel, bg=self.bg_secondary, height=100)
        title_frame.pack(fill=tk.X, padx=0, pady=0)
        title_frame.pack_propagate(False)

        title_label = tk.Label(
            title_frame,
            text="DETR Detection",
            font=("Courier New", 24, "bold"),
            fg=self.accent,
            bg=self.bg_secondary,
        )
        title_label.pack(pady=30)

        # Режим 1 - Одно изображение
        mode1_frame = tk.Frame(left_panel, bg=self.bg_secondary)
        mode1_frame.pack(fill=tk.X, padx=20, pady=15)

        mode1_title = tk.Label(
            mode1_frame,
            text="Single Image",
            font=("Courier New", 14, "bold"),
            fg=self.accent,
            bg=self.bg_secondary,
        )
        mode1_title.pack(pady=(0, 10))

        mode1_desc = tk.Label(
            mode1_frame,
            text="Обработай одно изображение и посмотри результат",
            font=("Courier New", 9),
            fg=self.text_secondary,
            bg=self.bg_secondary,
            wraplength=250,
            justify=tk.LEFT,
        )
        mode1_desc.pack(pady=(0, 15))

        btn1 = tk.Button(
            mode1_frame,
            text="▶ Выбрать изображение",
            command=lambda: self.show_mode(1),
            bg=self.accent,
            fg=self.bg_primary,
            font=("Courier New", 11, "bold"),
            relief=tk.FLAT,
            padx=15,
            pady=12,
            cursor="hand2",
        )
        btn1.pack(fill=tk.X)

        # Разделитель
        sep1 = tk.Frame(left_panel, bg=self.accent, height=1)
        sep1.pack(fill=tk.X, padx=20, pady=25)

        # Режим 2 - Batch обработка
        mode2_frame = tk.Frame(left_panel, bg=self.bg_secondary)
        mode2_frame.pack(fill=tk.X, padx=20, pady=15)

        mode2_title = tk.Label(
            mode2_frame,
            text="Batch Processing",
            font=("Courier New", 14, "bold"),
            fg=self.accent,
            bg=self.bg_secondary,
        )
        mode2_title.pack(pady=(0, 10))

        mode2_desc = tk.Label(
            mode2_frame,
            text="Обработай множество изображений одновременно",
            font=("Courier New", 9),
            fg=self.text_secondary,
            bg=self.bg_secondary,
            wraplength=250,
            justify=tk.LEFT,
        )
        mode2_desc.pack(pady=(0, 15))

        btn2 = tk.Button(
            mode2_frame,
            text="▶ Выбрать папку",
            command=lambda: self.show_mode(2),
            bg=self.accent,
            fg=self.bg_primary,
            font=("Courier New", 11, "bold"),
            relief=tk.FLAT,
            padx=15,
            pady=12,
            cursor="hand2",
        )
        btn2.pack(fill=tk.X)

        # Confidence threshold
        threshold_frame = tk.Frame(left_panel, bg=self.bg_secondary)
        threshold_frame.pack(fill=tk.X, padx=20, pady=15)

        tk.Label(
            threshold_frame,
            text="Confidence",
            font=("Courier New", 10, "bold"),
            fg=self.accent,
            bg=self.bg_secondary,
        ).pack()

        self.threshold_var = tk.DoubleVar(value=0.5)
        threshold_scale = tk.Scale(
            threshold_frame,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.threshold_var,
            bg=self.bg_secondary,
            fg=self.text_primary,
            troughcolor=self.bg_primary,
            highlightthickness=0,
        )
        threshold_scale.pack(fill=tk.X)

        # Статус модели внизу
        status_frame = tk.Frame(left_panel, bg=self.bg_secondary)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=20)

        self.model_status = tk.Label(
            status_frame,
            text="⏳ Загрузка модели...",
            font=("Courier New", 9),
            fg=self.text_secondary,
            bg=self.bg_secondary,
        )
        self.model_status.pack()

        # Правая панель
        self.right_panel = tk.Frame(main, bg=self.bg_primary)
        self.right_panel.pack(
            side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=30, pady=30
        )

        self.content_frame = tk.Frame(self.right_panel, bg=self.bg_primary)
        self.content_frame.pack(fill=tk.BOTH, expand=True)

        self.show_welcome()

    def show_welcome(self):
        """Показывает приветственный экран"""
        self.clear_content()

        welcome_frame = tk.Frame(self.content_frame, bg=self.bg_primary)
        welcome_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=40)

        welcome_title = tk.Label(
            welcome_frame,
            text="Добро пожаловать",
            font=("Courier New", 32, "bold"),
            fg=self.accent,
            bg=self.bg_primary,
        )
        welcome_title.pack(pady=(60, 20))

        welcome_text = tk.Label(
            welcome_frame,
            text="Выбери режим работы слева",
            font=("Courier New", 14),
            fg=self.text_secondary,
            bg=self.bg_primary,
        )
        welcome_text.pack(pady=20)

    def show_mode(self, mode):
        """Показывает экран выбранного режима"""
        if mode == 1:
            self.show_single_image_mode()
        else:
            self.show_batch_mode()

    def show_single_image_mode(self):
        """Режим одного изображения"""
        self.clear_content()

        container = tk.Frame(self.content_frame, bg=self.bg_primary)
        container.pack(fill=tk.BOTH, expand=True)

        # Левая половина - оригинальное изображение
        left = tk.Frame(container, bg=self.bg_secondary, relief=tk.RAISED, bd=1)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))

        tk.Label(
            left,
            text="ORIGINAL",
            font=("Courier New", 10, "bold"),
            fg=self.accent,
            bg=self.bg_secondary,
        ).pack(pady=10)

        self.image_label = tk.Label(left, bg=self.bg_secondary)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        btn_frame_left = tk.Frame(left, bg=self.bg_secondary)
        btn_frame_left.pack(fill=tk.X, padx=10, pady=10)

        tk.Button(
            btn_frame_left,
            text="📂 Выбрать изображение",
            command=self.select_single_image,
            bg=self.accent,
            fg=self.bg_primary,
            font=("Courier New", 10, "bold"),
            relief=tk.FLAT,
            pady=8,
            cursor="hand2",
        ).pack(fill=tk.X)

        # Правая половина - результат
        right = tk.Frame(container, bg=self.bg_secondary, relief=tk.RAISED, bd=1)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(15, 0))

        tk.Label(
            right,
            text="DETECTION",
            font=("Courier New", 10, "bold"),
            fg=self.accent,
            bg=self.bg_secondary,
        ).pack(pady=10)

        self.result_label = tk.Label(right, bg=self.bg_secondary)
        self.result_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        btn_frame_right = tk.Frame(right, bg=self.bg_secondary)
        btn_frame_right.pack(fill=tk.X, padx=10, pady=10)

        btn_detect = tk.Button(
            btn_frame_right,
            text="🔍 Обнаружить",
            command=self.detect_single,
            bg=self.success,
            fg=self.bg_primary,
            font=("Courier New", 10, "bold"),
            relief=tk.FLAT,
            pady=8,
            cursor="hand2",
        )
        btn_detect.pack(fill=tk.X)

        btn_save = tk.Button(
            btn_frame_right,
            text="💾 Сохранить",
            command=self.save_single_result,
            bg=self.accent,
            fg=self.bg_primary,
            font=("Courier New", 10, "bold"),
            relief=tk.FLAT,
            pady=8,
            cursor="hand2",
        )
        btn_save.pack(fill=tk.X, pady=(5, 0))

    def show_batch_mode(self):
        """Режим batch обработки"""
        self.clear_content()

        main_frame = tk.Frame(self.content_frame, bg=self.bg_primary)
        main_frame.pack(fill=tk.BOTH, expand=True)

        title = tk.Label(
            main_frame,
            text="Batch Processing",
            font=("Courier New", 20, "bold"),
            fg=self.accent,
            bg=self.bg_primary,
        )
        title.pack(pady=(0, 30))

        info_frame = tk.Frame(main_frame, bg=self.bg_secondary, relief=tk.RAISED, bd=1)
        info_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            info_frame,
            text="Выбери папку с изображениями для обработки",
            font=("Courier New", 11),
            fg=self.text_primary,
            bg=self.bg_secondary,
        ).pack(pady=15, padx=15)

        path_frame = tk.Frame(main_frame, bg=self.bg_secondary, relief=tk.RAISED, bd=1)
        path_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            path_frame,
            text="📁 Папка:",
            font=("Courier New", 10, "bold"),
            fg=self.accent,
            bg=self.bg_secondary,
        ).pack(anchor=tk.W, padx=15, pady=(10, 0))

        self.batch_path_label = tk.Label(
            path_frame,
            text="Не выбрана",
            font=("Courier New", 9),
            fg=self.text_secondary,
            bg=self.bg_secondary,
        )
        self.batch_path_label.pack(anchor=tk.W, padx=15, pady=(0, 10))

        button_frame = tk.Frame(main_frame, bg=self.bg_primary)
        button_frame.pack(fill=tk.X, padx=10, pady=20)

        tk.Button(
            button_frame,
            text="Выбрать папку",
            command=self.select_batch_folder,
            bg=self.accent,
            fg=self.bg_primary,
            font=("Courier New", 11, "bold"),
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor="hand2",
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            button_frame,
            text=" Обработать",
            command=self.process_batch,
            bg=self.success,
            fg=self.bg_primary,
            font=("Courier New", 11, "bold"),
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor="hand2",
        ).pack(side=tk.LEFT, padx=5)

        self.batch_progress = ttk.Progressbar(main_frame, mode="indeterminate")
        self.batch_progress.pack(fill=tk.X, padx=10, pady=10)

        self.batch_status = tk.Label(
            main_frame,
            text="Готов к обработке",
            font=("Courier New", 10),
            fg=self.text_secondary,
            bg=self.bg_primary,
        )
        self.batch_status.pack(pady=10)

        results_frame = tk.Frame(
            main_frame, bg=self.bg_secondary, relief=tk.RAISED, bd=1
        )
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(
            results_frame,
            text="📊 Результаты",
            font=("Courier New", 11, "bold"),
            fg=self.accent,
            bg=self.bg_secondary,
        ).pack(anchor=tk.W, padx=15, pady=(10, 0))

        self.batch_results_text = tk.Text(
            results_frame,
            bg=self.bg_primary,
            fg=self.text_primary,
            font=("Courier New", 9),
            height=8,
            relief=tk.FLAT,
            padx=10,
            pady=10,
        )
        self.batch_results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.batch_results_text.config(state=tk.DISABLED)

    def clear_content(self):
        """Очищает контент панель"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()

    def load_model(self):
        """Загружает модель в отдельном потоке"""

        def load():
            try:
                self.config = load_config("config.yaml")
                self.device = get_device()
                self.model = DETROreDetector(
                    num_classes=self.config["model"]["num_classes"],
                    model_name=self.config["model"]["name"],
                ).to(self.device)

                checkpoint = torch.load(
                    self.config["paths"]["model_path"], map_location=self.device
                )

                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint

                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("model."):
                        new_key = k[6:]
                        new_state_dict[new_key] = v
                    else:
                        new_state_dict[k] = v

                self.model.model.load_state_dict(new_state_dict, strict=False)
                self.root.after(0, self.on_model_loaded)
            except Exception as e:
                self.root.after(0, lambda: self.on_model_error(str(e)))

        thread = threading.Thread(target=load, daemon=True)
        thread.start()

    def on_model_loaded(self):
        """Вызывается когда модель загружена"""
        self.model_status.config(text="✓ Модель готова", fg=self.success)

    def on_model_error(self, error):
        """Вызывается при ошибке загрузки"""
        self.model_status.config(text=f"✗ Ошибка: {error[:30]}", fg=self.warning)
        messagebox.showerror("Ошибка", f"Не удалось загрузить модель:\n{error}")

    def select_single_image(self):
        """Выбирает одно изображение"""
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png"), ("All", "*.*")]
        )

        if path:
            self.current_image_path = Path(path)
            self.current_image = cv2.imread(path)
            self.display_image(self.current_image, self.image_label)
            self.result_label.config(image="")
            self.results = None

    def detect_single(self):
        """Запускает детекцию для одного изображения"""
        if self.current_image is None:
            messagebox.showwarning("Предупреждение", "Выбери изображение!")
            return

        if self.model is None:
            messagebox.showerror("Ошибка", "Модель не загружена")
            return

        thread = threading.Thread(target=self._detect_thread, daemon=True)
        thread.start()

    def _detect_thread(self):
        """Поток для детекции"""
        try:
            image = cv2.imread(str(self.current_image_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            transform = A.Compose(
                [
                    A.Resize(height=800, width=800),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )

            transformed = transform(image=image_rgb)
            image_tensor = transformed["image"].unsqueeze(0).to(self.device)

            predictions = self.model.predict(
                image_tensor, threshold=self.threshold_var.get()
            )

            self.results = predictions[0]
            annotated = self._annotate_image(image_rgb, self.results)

            self.root.after(0, lambda: self.display_image(annotated, self.result_label))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Ошибка", str(e)))

    def _annotate_image(self, image, predictions):
        """Рисует боксы на изображении"""
        class_names = {0: "Арматура", 1: "Большие камни"}
        colors = {0: (255, 0, 0), 1: (0, 165, 255)}

        h, w = image.shape[:2]
        annotated = image.copy()

        for box, label, score in zip(
            predictions["boxes"], predictions["labels"], predictions["scores"]
        ):
            if isinstance(box, torch.Tensor):
                box = box.cpu().numpy()
            if isinstance(label, torch.Tensor):
                label = label.cpu().item()
            if isinstance(score, torch.Tensor):
                score = score.cpu().item()

            cx, cy, bw, bh = box
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            color = colors.get(label, (255, 255, 255))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            class_name = class_names.get(label, f"Class {label}")
            text = f"{class_name}: {score:.2f}"
            cv2.putText(
                annotated,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        return annotated

    def save_single_result(self):
        """Сохраняет результат"""
        if self.results is None:
            messagebox.showwarning("Предупреждение", "Запусти детекцию!")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
        )

        if path:
            image_rgb = cv2.imread(str(self.current_image_path))
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
            annotated = self._annotate_image(image_rgb, self.results)
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, annotated_bgr)
            messagebox.showinfo("Успех", f"Сохранено: {path}")

    def select_batch_folder(self):
        """Выбирает папку для batch обработки"""
        path = filedialog.askdirectory(title="Выбери папку с изображениями")

        if path:
            self.batch_folder = Path(path)
            self.batch_path_label.config(text=str(self.batch_folder))

    def process_batch(self):
        """Обрабатывает папку с изображениями"""
        if not hasattr(self, "batch_folder"):
            messagebox.showwarning("Предупреждение", "Выбери папку!")
            return

        if self.model is None:
            messagebox.showerror("Ошибка", "Модель не загружена")
            return

        thread = threading.Thread(target=self._batch_thread, daemon=True)
        thread.start()

    def _batch_thread(self):
        """Поток для batch обработки"""
        try:
            image_files = list(self.batch_folder.rglob("*"))
            image_files = [
                f for f in image_files if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]

            if not image_files:
                self.root.after(
                    0,
                    lambda: messagebox.showwarning("Ошибка", "Изображения не найдены!"),
                )
                return

            total_images = len(image_files)

            self.root.after(0, lambda: self.batch_progress.start())
            self.root.after(
                0,
                lambda: self.batch_status.config(
                    text=f"⏳ Обработка: 0/{total_images}", fg=self.text_secondary
                ),
            )

            output_dir = self.batch_folder / "Detection_Results"
            output_dir.mkdir(exist_ok=True)

            stats = {
                "detected": 0,
                "no_detection": 0,
                "total": total_images,
                "processed": 0,
            }

            transform = A.Compose(
                [
                    A.Resize(height=800, width=800),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )

            for idx, img_file in enumerate(image_files, 1):
                image = cv2.imread(str(img_file))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                transformed = transform(image=image_rgb)
                image_tensor = transformed["image"].unsqueeze(0).to(self.device)

                predictions = self.model.predict(
                    image_tensor, threshold=self.threshold_var.get()
                )
                result = predictions[0]

                annotated = self._annotate_image(image_rgb, result)
                annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

                output_path = output_dir / img_file.name
                cv2.imwrite(str(output_path), annotated_bgr)

                if len(result["scores"]) > 0:
                    stats["detected"] += 1
                else:
                    stats["no_detection"] += 1

                stats["processed"] = idx

                progress_text = f"⏳ Обработка: {idx}/{total_images} | Найдено: {stats['detected']} | Без объектов: {stats['no_detection']}"
                self.root.after(
                    0, lambda text=progress_text: self.batch_status.config(text=text)
                )

            result_text = f"""
Обработка завершена

Статистика:
  • Всего изображений: {stats["total"]}
  • Обнаружены объекты: {stats["detected"]}
  • Без объектов: {stats["no_detection"]}
  • Процент с объектами: {(stats["detected"] / stats["total"] * 100):.1f}%

Результаты сохранены в:
{output_dir}
            """.strip()

            self.root.after(0, lambda: self.update_batch_results(result_text))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
        finally:
            self.root.after(0, lambda: self.batch_progress.stop())

    def update_batch_results(self, text):
        """Обновляет результаты batch"""
        self.batch_results_text.config(state=tk.NORMAL)
        self.batch_results_text.delete(1.0, tk.END)
        self.batch_results_text.insert(1.0, text)
        self.batch_results_text.config(state=tk.DISABLED)
        self.batch_status.config(text="✅ Готово!", fg=self.success)

    def display_image(self, cv_image, label):
        """Показывает изображение на лейбле"""
        if cv_image.shape[2] == 3:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv_image

        h, w = rgb_image.shape[:2]
        max_w, max_h = 600, 700

        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(rgb_image, (new_w, new_h))

        pil_image = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(pil_image)

        label.config(image=photo)
        label.image = photo


def main():
    root = tk.Tk()
    app = DETRDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
