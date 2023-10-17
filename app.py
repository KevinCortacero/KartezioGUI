import os
import time
import traceback

from kartezio.apps.segmentation import create_segmentation_model
from kartezio.callback import CallbackSave, Event
from kartezio.dataset import read_dataset

import sys
import functools
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QLabel, QHBoxLayout,
                               QFormLayout, QLineEdit, QComboBox, QFrame, QScrollArea, QListWidget, QTextEdit,
                               QStackedWidget, QSpacerItem, QSizePolicy, QProgressBar, QGroupBox, QStyle)
from PySide6.QtGui import QColor, QPalette, QPixmap, QFont, QIntValidator, QRegularExpressionValidator
from PySide6.QtCore import Qt, QObject, Signal, QRunnable, Slot, QThreadPool, QRegularExpression

import skimage.filters.edges
from kartezio.endpoint import EndpointWatershed, EndpointHoughCircle, EndpointThreshold
from kartezio.enums import JSON_ELITE
from kartezio.fitness import FitnessAP, FitnessIOU
from kartezio.inference import KartezioModel
from kartezio.model.components import KartezioCallback
from kartezio.training import train_model
from kartezio.utils.io import pack_one_directory, JsonSaver
from kartezio.preprocessing import TransformToHSV, TransformToHED, SelectChannels
import kartezio.utils.json_utils as json
from numena.io.drive import Directory
from numena.io.json import json_read, json_write


fitness_list = [FitnessIOU(), FitnessAP()]
preprocessing_list = [TransformToHSV(), TransformToHED()]
endpoint_list = [EndpointWatershed(), EndpointHoughCircle(), EndpointThreshold(threshold=128)]

fitness_names = [f.name for f in fitness_list]
preprocessing_names = [p.name for p in preprocessing_list]
preprocessing_names.insert(0, "None")
preprocessing_list.insert(0, None)
endpoint_names = [e.name for e in endpoint_list]

print(fitness_names)
print(preprocessing_names)
print(endpoint_names)
class KartezioSignals(QObject):
    finished = Signal(int)
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(tuple)


class KartezioTraining(QRunnable):
    def __init__(self, fn, index, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.index = index
        self.args = args
        self.kwargs = kwargs
        self.signals = KartezioSignals()

        # Add the callback to our kwargs
        self.kwargs['index_callback'] = index
        self.kwargs['progress_callback'] = self.signals.progress

    @Slot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit(self.index)  # Done


class KartezioDesktop(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kartezio Desktop App")
        self.setGeometry(100, 100, 720, 480)

        self.app_workdir = Directory("./")
        self.app_datasets = self.app_workdir.next(".local").next("datasets")
        self.dataset_map = {}
        for item in self.app_datasets.ls("*/META.json"):
            data = json_read(item)
            self.dataset_map[data["name"]] = item.parent
        print(self.dataset_map)

        self.app_models = self.app_workdir.next(".local").next("models")
        self.model_map = {}
        for item in self.app_models.ls("*/*/model.json"):
            self.model_map[item.parent.name] = KartezioModel(item, fitness=FitnessAP())
        print(self.model_map)

        # Apply the Dracula theme
        self.setup_dracula_theme()

        # Main Layout
        main_layout = QVBoxLayout()

        self.create_horizontal_menu(main_layout)

        # Content Area - Using a Stacked Widget
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        self.create_pages()
        self.stacked_widget.setCurrentIndex(0)

        # Set the layout
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.threadpool = QThreadPool()
        self.pbars = []
        self.sbars = []
        self.fps = []
        self.index = 0

    def open_dataset_directory(self):
        os.startfile(self.app_datasets._path)

    def setup_dracula_theme(self):
        self.setStyleSheet("""
            #GreenProgressBar {
                text-align: center;
                min-height: 16px;
                max-height: 16px;
                border-radius: 3px;
            }
            #GreenProgressBar::chunk {
                border-radius: 3px;
                background-color: #63c74d;
            }
           QPushButton {
    background-color: #34a8eb;
    color: #f8f8f2;
    border: none;
    border-radius: 5px;
    padding: 10px 10px;
    font-size: 16px;
    outline: none;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

QPushButton:hover {
    background-color: #4db1ff;
    box-shadow: 0 6px 8px rgba(0, 0, 0, 0.12);
}

QPushButton:pressed {
    background-color: #2b86d4;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
}


        """)

    def create_horizontal_menu(self, layout):
        menu_layout = QHBoxLayout()
        menu_items = ["Home", "Getting Started", "Create a Model", "View my Models", "Run Experiments", "FAQ"]
        self.pages = {menu_items[i]: i for i in range(len(menu_items))}
        for item in menu_items:
            btn = QPushButton(item)
            btn.setFlat(True)
            btn.clicked.connect(functools.partial(self.menu_clicked, item))
            menu_layout.addWidget(btn)

        layout.addLayout(menu_layout)

    def menu_clicked(self, item):
        index = self.pages[item]
        self.stacked_widget.setCurrentIndex(index)

    def create_pages(self):
        self.stacked_widget.addWidget(self.display_home())
        self.stacked_widget.addWidget(self.display_model_cards())
        self.stacked_widget.addWidget(self.display_training())
        self.stacked_widget.addWidget(self.display_inference())
        self.stacked_widget.addWidget(self.display_datasets())

    def display_home(self):
        widget = QWidget()
        layout = QVBoxLayout()

        pixmap = QPixmap(200, 200)
        pixmap.load("simple-no-bg.png")
        pixmap = pixmap.scaledToWidth(256)
        logo_label = QLabel()
        logo_label.setPixmap(pixmap)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        description = QLabel(
            "Kartezio GUI Application.\n\nManage models, undergo training sessions, perform inferences.")
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = description.font()
        font.setPointSize(12)
        description.setFont(font)

        layout.addStretch(1)
        layout.addWidget(logo_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(description, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)

        widget.setLayout(layout)
        return widget

    def display_model_cards(self):
        widget = QWidget()
        layout = QVBoxLayout()

        for i in range(3):
            card = self.create_model_card(f"Model {i + 1}", "Placeholder image here.")
            layout.addWidget(card)

        widget.setLayout(layout)
        return widget

    def create_model_card(self, title, description):
        card = QFrame()
        layout = QVBoxLayout()

        label_title = QLabel(title)
        font = label_title.font()
        font.setPointSize(14)
        label_title.setFont(font)

        label_description = QLabel(description)

        layout.addWidget(label_title)
        layout.addWidget(label_description)
        card.setLayout(layout)
        return card

    def display_training(self):
        widget = QWidget()
        layout = QVBoxLayout()
        self.form = QFormLayout()

        parameters = QHBoxLayout()
        self.model_name = QLineEdit(self)
        regex = QRegularExpression(r"^^[a-zA-Z0-9_]{1,32}$")
        validator = QRegularExpressionValidator(regex, self.model_name)
        self.model_name.setValidator(validator)
        parameters.addWidget(self.model_name)

        self.iterations = QLineEdit(self)
        self.iterations.setValidator(QIntValidator())
        self.iterations.setText(str(200))
        parameters.addWidget(QLabel("Iterations"))
        parameters.addWidget(self.iterations)

        self.max_functions = QLineEdit(self)
        self.max_functions.setValidator(QIntValidator())
        self.max_functions.setText(str(20))
        parameters.addWidget(QLabel("Max Functions"))
        parameters.addWidget(self.max_functions)

        datasets_infos = QHBoxLayout()
        self.dataset = QComboBox()
        self.dataset.addItems(self.dataset_map)
        datasets_infos.addWidget(self.dataset)
        dataset_link_btn = QPushButton("Open Location")
        dataset_link_btn.setFlat(True)
        dataset_link_btn.clicked.connect(self.open_dataset_directory)
        datasets_infos.addWidget(dataset_link_btn)

        components = QHBoxLayout()
        self.preprocessing = QComboBox()
        self.preprocessing.addItems(preprocessing_names)
        components.addWidget(self.preprocessing)
        self.endpoint = QComboBox()
        self.endpoint.addItems(endpoint_names)
        components.addWidget(QLabel("Endpoint"))
        components.addWidget(self.endpoint)
        self.fitness = QComboBox()
        self.fitness.addItems(fitness_names)
        components.addWidget(QLabel("Fitness"))
        components.addWidget(self.fitness)

        self.form.addRow("Model Name", parameters)
        self.form.addRow("Dataset", datasets_infos)
        self.form.addRow("Preprocessing", components)

        train_btn = QPushButton("Generate Algorithm")
        train_btn.clicked.connect(self.train_model)
        layout.addLayout(self.form)
        layout.addWidget(train_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        group = QGroupBox()
        self.models_group = QFormLayout()
        group.setLayout(self.models_group)
        models = QScrollArea()
        models.setWidget(group)
        models.setWidgetResizable(True)
        # models.setFixedHeight(200)
        layout.addWidget(models)
        widget.setLayout(layout)
        return widget

    def _train_model(self, model, output_path, preprocessing, dataset, fitness, model_name, index_callback, progress_callback):
        callbacks = [
            CallbackSaveModel(output_path, dataset, preprocessing, fitness, model_name),
            CallbackProgressBar(index_callback, progress_callback, model.generations)
        ]
        return train_model(model, dataset, output_path, preprocessing, callbacks=callbacks, pack=False)

    def print_output(self, s):
        print(s)

    def thread_complete(self, index):
        self.pbars[index].setValue(100)
        print("THREAD COMPLETE!")

    def progress_fn(self, state):
        index, progression, score, fps = state
        self.pbars[index].setValue(progression)
        self.sbars[index].setValue((1. - score) * 100)
        self.fps[index].setText(fps)

    def create_model_training_card(self):
        card = QFrame()
        layout = QVBoxLayout()

        # line two
        pbar = QProgressBar(objectName="GreenProgressBar")
        self.pbars.append(pbar)
        sbar = QProgressBar(objectName="GreenProgressBar")
        self.sbars.append(sbar)
        lfps = QLabel()
        self.fps.append(lfps)
        pbar.setValue(0)
        sbar.setValue(0)
        model_state_layout = QHBoxLayout()
        model_state_layout.addWidget(QLabel("Progression"))
        model_state_layout.addWidget(pbar)
        model_state_layout.addWidget(QLabel("Score"))
        model_state_layout.addWidget(sbar)
        model_state_layout.addWidget(QLabel("FPS"))
        model_state_layout.addWidget(lfps)
        layout.addLayout(model_state_layout)

        # line three
        model_param_layout = QHBoxLayout()
        model_param_layout.addWidget(QLabel("Iterations"))
        model_param_layout.addWidget(QLabel("TODO"))
        model_param_layout.addWidget(QLabel("Functions"))
        model_param_layout.addWidget(QLabel("TODO"))
        model_param_layout.addWidget(QLabel("Fitness"))
        model_param_layout.addWidget(QLabel("TODO"))
        model_param_layout.addWidget(QLabel("Preprocessing"))
        model_param_layout.addWidget(QLabel("TODO"))
        model_param_layout.addWidget(QLabel("Endpoint"))
        model_param_layout.addWidget(QLabel("TODO"))
        layout.addLayout(model_param_layout)
        card.setLayout(layout)
        return card
    def train_model(self):
        iterations = int(self.iterations.text())
        nodes = int(self.max_functions.text())
        model_name = f"{self.model_name.text()}_{int(round(time.time()))}"
        fitness = fitness_list[self.fitness.currentIndex()]
        preprocessing = preprocessing_list[self.preprocessing.currentIndex()]
        endpoint = endpoint_list[self.endpoint.currentIndex()]

        dataset_path = self.dataset_map[self.dataset.currentText()]
        dataset = read_dataset(dataset_path)

        model = create_segmentation_model(
            iterations, _lambda=5, inputs=dataset.inputs,
            nodes=nodes, outputs=endpoint.arity,
            fitness=fitness, endpoint=endpoint
        )

        out_dir = self.app_models.next(dataset_path.name)
        worker = KartezioTraining(self._train_model, self.index, model, out_dir._path, preprocessing, dataset, fitness, model_name)
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)
        self.models_group.addRow(model_name, self.create_model_training_card())
        self.index += 1
        self.threadpool.start(worker)

    def display_inference(self):
        widget = QWidget()
        layout = QVBoxLayout()
        model_selector = QComboBox()
        model_selector.addItems(["Model 1", "Model 2", "Model 3"])
        input_data = QTextEdit()
        infer_btn = QPushButton("Run Inference")
        layout.addWidget(QLabel("Select Model:"))
        layout.addWidget(model_selector)
        layout.addWidget(QLabel("Input Data:"))
        layout.addWidget(input_data)
        layout.addWidget(infer_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        widget.setLayout(layout)
        return widget

    def display_datasets(self):
        widget = QWidget()
        layout = QVBoxLayout()

        dataset_list = QListWidget()
        for i in range(10):
            dataset_list.addItem(f"Dataset {i + 1}")

        layout.addWidget(QLabel("Available Datasets:"))
        layout.addWidget(dataset_list)

        widget.setLayout(layout)
        return widget


class CallbackSaveModel(KartezioCallback):
    def __init__(self, workdir, dataset, preprocessing, fitness, model_name, frequency=1):
        super().__init__(frequency)
        self.workdir = Directory(workdir).next(model_name)
        self.dataset = dataset
        self.fitness = fitness.dumps()
        self.preprocessing = preprocessing
        if self.preprocessing is not None:
            self.preprocessing = self.preprocessing.dumps()
        self.json_saver = None

    def set_parser(self, parser):
        super().set_parser(parser)
        self.json_saver = JsonSaver(self.dataset, self.parser)

    def save_model(self, filepath, individual):
        json_data = {
            "dataset": json.from_dataset(self.dataset),
            "fitness": self.fitness,
            "preprocessing": self.preprocessing,
            "individual": json.from_individual(individual),
            "decoding": self.parser.dumps(),
        }
        json_write(filepath, json_data)

    def _callback(self, n, e_name, e_content):
        if e_name == Event.END_STEP or e_name == Event.END_LOOP:
            self.save_model(self.workdir / "model.json", e_content.individuals[0])

class CallbackProgressBar(KartezioCallback):
    def __init__(self, index, signal, max_iter):
        super().__init__()
        self.index = index
        self.__signal = signal
        self.max_iter = max_iter

    def _callback(self, n, e_name, e_content):
        fitness, time = e_content.get_best_fitness()
        if time == 0:
            fps = "'inf' "
        else:
            fps = str(int(round(1.0 / time)))
        data = (self.index, (n / self.max_iter) * 100, fitness, fps)
        self.__signal.emit(data)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KartezioDesktop()
    window.show()
    sys.exit(app.exec())
