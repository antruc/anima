"""Camera trap filter.

Scans a folder of camera trap images and videos with MegaDetector v6
(ONNX format, run with ONNX Runtime) and copies every file in which a person
or an animal is detected to an output folder. Original files are never moved
or modified.
"""

import ctypes
import locale
import logging
import os
import pathlib
import shutil
import sys
import threading

import cv2
import numpy as np
import onnxruntime as ort
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

from anima import dnd

log = logging.getLogger(__name__)

# MegaDetector v6 compact (MDV6-yolov10-c, AGPL-3.0), trained on camera trap
# images. Exported with NMS embedded: its output is already filtered.
# Export:
# YOLO("MDV6-yolov10-c.pt").export(format="onnx", imgsz=640, nms=True, conf=0.05)
MODEL_PATH = pathlib.Path(__file__).parent / "MDV6-yolov10-c.onnx"

# MegaDetector class ids that count as a detection: 0 animal, 1 person.
# 2 vehicle is ignored.
CLASSES = [0, 1]

# Minimum detection score (0 to 1) for a file to be copied.
# Calibrate with a folder of real field photos: raise it if empty frames are
# still copied (false positives), lower it if animals are missed.
# Values below 0.05 have no effect: the model was exported with conf=0.05, so
# weaker detections never reach this filter.
MIN_CONFIDENCE = 0.1

# Square input size the model was exported with, in pixels.
IMG_SIZE = 640

# Seconds between video frames sent to the model. Frames in between are
# skipped without decoding. Lower it if fast animals are missed in videos.
VIDEO_STEP_S = 1.0

# File extensions processed. Anything else in the folder is ignored.
IMG_EXT = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VID_EXT = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".wmv"}


def system_lang():
    """Return the OS UI language as a two-letter lowercase code, or ''."""
    if sys.platform == "win32":
        # UI language, not regional format: Spanish Windows with US dates is still "es"
        langid = ctypes.windll.kernel32.GetUserDefaultUILanguage()
        return locale.windows_locale.get(langid, "")[:2].lower()
    for var in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):  # gettext priority
        if os.environ.get(var):
            return os.environ[var][:2].lower()
    return ""


# UI strings per language. Any language other than these falls back to English.
TEXT = {
    "es": {
        "input_hint": "Selecciona o arrastra carpeta de origen",
        "output_hint": "Selecciona carpeta de destino",
        "input": "Entrada",
        "output": "Salida",
        "run": "Procesar",
        "cancel": "Cancelar",
        "input_dialog": "Carpeta de origen",
        "output_dialog": "Carpeta de destino",
        "drop_one": "Error: Arrastra una sola carpeta",
        "need_both": "Error: Debes seleccionar ambas carpetas",
        "same_folder": "Error: Origen y destino son la misma carpeta",
        "running": "Procesando...",
        "no_files": "Error: No se encontraron archivos soportados",
        "copy_failed": "Error: No se pudo copiar {}",
        "cancelled": "Cancelado",
        "done": "Listo",
        "failed": "Error: Hay un problema con tus archivos",
        "no_model": "Error: No se pudo cargar el modelo",
    },
    "en": {
        "input_hint": "Select or drag a source folder",
        "output_hint": "Select a destination folder",
        "input": "Input",
        "output": "Output",
        "run": "Process",
        "cancel": "Cancel",
        "input_dialog": "Source folder",
        "output_dialog": "Destination folder",
        "drop_one": "Error: Drag a single folder",
        "need_both": "Error: You must select both folders",
        "same_folder": "Error: Source and destination are the same folder",
        "running": "Processing...",
        "no_files": "Error: No supported files found",
        "copy_failed": "Error: Could not copy {}",
        "cancelled": "Cancelled",
        "done": "Done",
        "failed": "Error: There is a problem with your files",
        "no_model": "Error: Could not load the model",
    },
}
T = TEXT.get(system_lang(), TEXT["en"])


def preprocess(img):
    """Convert an OpenCV BGR image into the model input tensor.

    The image is scaled to fit IMG_SIZE keeping its aspect ratio and centered
    on a gray square canvas (letterbox), as the model was trained that way.

    Args:
        img: HxWx3 uint8 array in BGR order, as returned by OpenCV.

    Returns:
        1x3xIMG_SIZExIMG_SIZE float32 array, RGB, values in [0, 1].
    """
    h, w = img.shape[:2]
    r = IMG_SIZE / max(h, w)
    nw, nh = round(w * r), round(h * r)
    canvas = np.full((IMG_SIZE, IMG_SIZE, 3), 114, np.uint8)
    top, left = (IMG_SIZE - nh) // 2, (IMG_SIZE - nw) // 2
    canvas[top : top + nh, left : left + nw] = cv2.resize(img, (nw, nh))
    return canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255


def has_target(det, min_confidence=MIN_CONFIDENCE):
    """Tell whether the model output contains a target class.

    Args:
        det: Nx6 array, one row per detection: x1, y1, x2, y2, score, class.
        min_confidence: Minimum score for a detection to count.

    Returns:
        True if any detection of a class in CLASSES scores at least
        min_confidence.
    """
    confident = det[:, 4] >= min_confidence
    return bool(np.any(confident & np.isin(det[:, 5].astype(int), CLASSES)))


def pick_folder(paths):
    """Return the dropped folder as a pathlib.Path, or None.

    Only a single directory is accepted: a file, several items or nothing
    give None.
    """
    if len(paths) != 1:
        return None
    path = pathlib.Path(paths[0])
    return path if path.is_dir() else None


def same_folder(a, b):
    """Tell whether two paths point to the same folder.

    Resolves symlinks and ignores case on Windows. Never raises, so a folder
    deleted after being selected is reported later by the worker.
    """
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(
        os.path.realpath(b)
    )


def dest_path(src, folder):
    """Return where to copy src inside folder without overwriting anything.

    Cards from different cameras reuse names such as IMG_0001.JPG, so a taken
    name gets a numbered suffix. Returns None if the same file (equal size
    and modification time) is already there, so a rerun does not duplicate.
    """
    size, mtime = src.stat().st_size, src.stat().st_mtime
    dest, n = folder / src.name, 1
    while dest.exists():
        st = dest.stat()
        # 2 s tolerance: FAT cards store modification times in 2 s steps
        if st.st_size == size and abs(st.st_mtime - mtime) <= 2:
            return None
        dest = folder / f"{src.stem} ({n}){src.suffix}"
        n += 1
    return dest


def frames(path):
    """Yield the frames of an image or video file as BGR arrays.

    An image yields a single frame, a video yields one frame every
    VIDEO_STEP_S seconds, starting with the first one. Unreadable or corrupt
    files yield nothing, so they are skipped.

    Args:
        path: pathlib.Path of the file.
    """
    if path.suffix.lower() in IMG_EXT:
        # imdecode instead of imread: imread fails on non-ASCII paths on Windows
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            yield img
        return
    cap = cv2.VideoCapture(str(path))
    # Some containers report 0 fps: then every frame is used
    step = max(1, round(cap.get(cv2.CAP_PROP_FPS) * VIDEO_STEP_S))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
            for _ in range(step - 1):
                if not cap.grab():  # grab advances without decoding
                    return
    finally:
        cap.release()


class Anima(toga.App):
    """Main window: pick input and output folders, run, cancel, see progress.

    Processing runs in a background thread so the window stays responsive.
    The thread never touches widgets directly: every UI update is sent to the
    main event loop through ui().
    """

    def startup(self):
        """Build the window and initialize the app state."""
        self.input_path = None
        self.output_path = None
        self.session = None  # ONNX Runtime session, loaded on first run
        self.running = False  # set to False by Cancel to stop the worker

        # Layout: one row per control group, stacked vertically
        main_box = toga.Box(style=Pack(direction=COLUMN))
        input_box = toga.Box(style=Pack(direction=ROW, margin=10))
        output_box = toga.Box(
            style=Pack(direction=ROW, margin_right=10, margin_left=10)
        )
        progress_box = toga.Box(
            style=Pack(direction=ROW, margin_right=10, margin_left=10, margin_top=15)
        )
        status_box = toga.Box(
            style=Pack(direction=ROW, margin_right=10, margin_left=10)
        )
        run_box = toga.Box(style=Pack(direction=ROW, margin=10))

        self.input_label = toga.Label(
            T["input_hint"], style=Pack(margin_top=3, margin_left=10)
        )
        self.output_label = toga.Label(
            T["output_hint"], style=Pack(margin_top=3, margin_left=10)
        )
        self.status_label = toga.Label("", style=Pack(margin_top=3))

        self.input_button = toga.Button(T["input"], on_press=self.select_input_folder)
        self.output_button = toga.Button(
            T["output"], on_press=self.select_output_folder
        )
        self.run_button = toga.Button(
            T["run"], on_press=self.run_app, style=Pack(flex=2)
        )
        self.cancel_button = toga.Button(
            T["cancel"], on_press=self.cancel_run, style=Pack(flex=2), enabled=False
        )

        # Buttons disabled while a run is in progress (Cancel is the inverse)
        self.buttons = [self.input_button, self.output_button, self.run_button]

        input_box.add(self.input_button)
        input_box.add(self.input_label)

        output_box.add(self.output_button)
        output_box.add(self.output_label)

        self.progress = toga.ProgressBar(max=100, value=0, style=Pack(flex=1))
        progress_box.add(self.progress)

        status_box.add(self.status_label)

        run_box.add(self.run_button)
        run_box.add(self.cancel_button)

        main_box.add(input_box)
        main_box.add(output_box)
        main_box.add(progress_box)
        main_box.add(status_box)
        main_box.add(run_box)

        self.main_window = toga.Window(size=toga.Size(345, 155), resizable=True)
        self.main_window.content = main_box
        self.main_window.show()

        # Optional: the buttons still work if the native hook fails
        try:
            dnd.enable_drop(self.main_window, self.on_drop)
        except Exception:
            log.exception("Drag and drop unavailable")

    def on_drop(self, paths):
        """Use a folder dropped on the window as the input folder."""
        if self.running:
            return
        folder = pick_folder(paths)
        if folder is None:
            self.status_label.text = T["drop_one"]
            return
        self.set_input_folder(folder)

    def set_input_folder(self, path):
        """Set the folder to scan and show it in the input label."""
        self.input_path = path
        self.input_label.text = str(path)

    def run_app(self, widget):
        """Handle the Process button: validate, lock the UI, start the worker.

        Runs on the main thread.
        """
        if self.running:
            return
        if self.input_path is None or self.output_path is None:
            self.status_label.text = T["need_both"]
            return
        if same_folder(self.input_path, self.output_path):
            self.status_label.text = T["same_folder"]
            return

        self.running = True
        self.set_buttons(False)
        self.status_label.text = T["running"]
        self.progress.value = 0
        # Daemon: the thread dies with the app if the window is closed mid run
        threading.Thread(target=self.run_model, daemon=True).start()

    def run_model(self):
        """Process every supported file in the input folder.

        Runs on the worker thread. Subfolders are not scanned. The progress bar
        advances once per file, and every exit path calls finish() to unlock
        the UI.
        """
        try:
            files = sorted(
                p
                for p in pathlib.Path(self.input_path).iterdir()
                if p.is_file() and p.suffix.lower() in IMG_EXT | VID_EXT
            )
            if not files:
                self.ui(self.finish, T["no_files"], 0)
                return

            # Load the model once and reuse it on later runs
            if self.session is None:
                try:
                    self.session = ort.InferenceSession(
                        str(MODEL_PATH), providers=["CPUExecutionProvider"]
                    )
                except Exception:
                    log.exception("Model load failed: %s", MODEL_PATH)
                    self.ui(self.finish, T["no_model"], 0)
                    return
            self.ui(self.set_progress, 0, len(files))

            for i, path in enumerate(files):
                if self.detect_file(path):
                    try:
                        # Full destination path: a missing output folder
                        # raises instead of becoming a file with its name
                        dest = dest_path(path, pathlib.Path(self.output_path))
                        if dest is not None:
                            # copy2 keeps the original timestamps (capture date)
                            shutil.copy2(path, dest)
                    except OSError:
                        log.exception("Copy failed: %s", path)
                        self.ui(self.finish, T["copy_failed"].format(path.name), 0)
                        return
                if not self.running:
                    self.ui(self.finish, T["cancelled"], 0)
                    return
                self.ui(self.set_progress, i + 1, None)

            self.ui(self.finish, T["done"], len(files))
        except Exception:
            log.exception("Prediction failed")
            self.ui(self.finish, T["failed"], 0)

    def detect_file(self, path):
        """Return True if any frame of the file contains a target.

        Videos stop at the first matching frame, so the rest is not decoded.
        Returns False early if the run was cancelled. A file that cannot be
        read or decoded (zero bytes, locked, degenerate size) is logged and
        skipped, so one bad file does not abort the whole run.
        """
        try:
            for frame in frames(path):
                if not self.running:
                    return False
                det = self.session.run(None, {"images": preprocess(frame)})[0][0]
                if has_target(det):
                    return True
        except (cv2.error, OSError, ValueError):
            log.exception("Skipped unreadable file: %s", path)
        return False

    def cancel_run(self, widget):
        """Handle the Cancel button: ask the worker to stop.

        The worker checks the flag between frames, so it stops within one
        frame.
        """
        self.running = False

    def ui(self, fn, *args):
        """Schedule fn(*args) on the main event loop. Safe from any thread."""
        self.loop.call_soon_threadsafe(fn, *args)

    def set_progress(self, value, maximum):
        """Update the progress bar. A maximum of None keeps the current one."""
        if maximum is not None:
            self.progress.max = maximum
        self.progress.value = value

    def finish(self, message, progress_value):
        """End a run: show the result message and unlock the UI."""
        self.running = False
        self.status_label.text = message
        self.progress.value = progress_value
        self.set_buttons(True)

    def set_buttons(self, enabled):
        """Enable or disable the folder and Process buttons, Cancel the opposite."""
        for button in self.buttons:
            button.enabled = enabled
        self.cancel_button.enabled = not enabled

    async def select_input_folder(self, widget):
        """Handle the Input button: pick the folder to scan."""
        path_name = await self.dialog(toga.SelectFolderDialog(T["input_dialog"]))
        if path_name is not None:
            self.set_input_folder(path_name)

    async def select_output_folder(self, widget):
        """Handle the Output button: pick the folder where matches are copied."""
        path_name = await self.dialog(toga.SelectFolderDialog(T["output_dialog"]))
        if path_name is not None:
            self.output_path = path_name
            self.output_label.text = str(path_name)


def main():
    """Entry point used by Briefcase and __main__.py."""
    return Anima()
