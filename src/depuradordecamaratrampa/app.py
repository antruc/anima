import logging
import pathlib
import shutil
import threading

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

from ultralytics import YOLO

log = logging.getLogger(__name__)

# 0: person, 14: bird, 15: cat, 16: dog, 17: horse, 18: sheep,
# 19: cow, 20: elephant, 21: bear, 22: zebra, 23: giraffe
CLASSES = [0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


def should_copy(path, last_file, num_boxes):
    # Videos yield one result per frame, copy each file only once
    return num_boxes > 0 and path != last_file


class DepuradordeCamaraTrampa(toga.App):
    def startup(self):
        self.input_path = None
        self.output_path = None
        self.model = None
        self.running = False

        # Create Toga components
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
            "Seleccione carpeta de origen", style=Pack(margin_top=3, margin_left=10)
        )
        self.output_label = toga.Label(
            "Seleccione carpeta de destino", style=Pack(margin_top=3, margin_left=10)
        )
        self.status_label = toga.Label("", style=Pack(margin_top=3))

        self.input_button = toga.Button("Entrada", on_press=self.select_input_folder)
        self.output_button = toga.Button("Salida", on_press=self.select_output_folder)
        self.run_button = toga.Button(
            "Procesar", on_press=self.run_app, style=Pack(flex=2)
        )
        cancel_button = toga.Button(
            "Cancelar", on_press=self.cancel_run, style=Pack(flex=2)
        )

        self.buttons = [self.input_button, self.output_button, self.run_button]

        # Add components into boxes
        input_box.add(self.input_button)
        input_box.add(self.input_label)

        output_box.add(self.output_button)
        output_box.add(self.output_label)

        self.progress = toga.ProgressBar(max=100, value=0, style=Pack(flex=1))
        progress_box.add(self.progress)

        status_box.add(self.status_label)

        run_box.add(self.run_button)
        run_box.add(cancel_button)

        main_box.add(input_box)
        main_box.add(output_box)
        main_box.add(progress_box)
        main_box.add(status_box)
        main_box.add(run_box)

        # Create and show main window
        self.main_window = toga.Window(size=toga.Size(345, 155), resizable=True)
        self.main_window.content = main_box
        self.main_window.show()

    def run_app(self, widget):
        # Runs on the main thread: validate and lock the UI before starting
        if self.running:
            return
        if self.input_path is None or self.output_path is None:
            self.status_label.text = "Error: Debe seleccionar ambas carpetas"
            return
        if pathlib.Path(self.input_path) == pathlib.Path(self.output_path):
            self.status_label.text = "Error: Origen y destino son la misma carpeta"
            return

        self.running = True
        self.set_buttons(False)
        self.status_label.text = "Procesando..."
        self.progress.value = 0
        # Daemon: the thread dies with the app if the window is closed mid run
        threading.Thread(target=self.run_model, daemon=True).start()

    def run_model(self):
        try:
            # Load model once, reuse on later runs
            if self.model is None:
                model_path = pathlib.Path(__file__).parent / "yolo26n-dynamic.onnx"
                self.model = YOLO(model_path, task="detect")

            total = sum(
                1 for p in pathlib.Path(self.input_path).iterdir() if p.is_file()
            )
            self.ui(self.set_progress, 0, max(total, 1))

            results = self.model.predict(
                self.input_path,
                classes=CLASSES,
                stream=True,
                verbose=False,
            )
            last_file = ""
            for i, result in enumerate(results):
                if not self.running:
                    self.ui(self.finish, "Cancelado", 0)
                    return
                if should_copy(result.path, last_file, len(result.boxes)):
                    shutil.copy2(result.path, self.output_path)
                    last_file = result.path
                self.ui(self.set_progress, i + 1, None)

            self.ui(self.finish, "Listo", total)
        except FileNotFoundError:
            self.ui(self.finish, "Error: No se encontraron archivos soportados", 0)
        except Exception:
            log.exception("Prediction failed")
            self.ui(self.finish, "Error: Hay un problema con sus archivos", 0)

    def cancel_run(self, widget):
        self.running = False

    def ui(self, fn, *args):
        # All UI updates from the worker thread go through the event loop
        self.loop.call_soon_threadsafe(fn, *args)

    def set_progress(self, value, maximum):
        if maximum is not None:
            self.progress.max = maximum
        self.progress.value = value

    def finish(self, message, progress_value):
        self.running = False
        self.status_label.text = message
        self.progress.value = progress_value
        self.set_buttons(True)

    def set_buttons(self, enabled):
        for button in self.buttons:
            button.enabled = enabled

    async def select_input_folder(self, widget):
        path_name = await self.dialog(toga.SelectFolderDialog("Carpeta de origen"))
        if path_name is not None:
            self.input_path = path_name
            self.input_label.text = str(path_name)

    async def select_output_folder(self, widget):
        path_name = await self.dialog(toga.SelectFolderDialog("Carpeta de destino"))
        if path_name is not None:
            self.output_path = path_name
            self.output_label.text = str(path_name)


def main():
    return DepuradordeCamaraTrampa()
