import os
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QThread, QObject, pyqtSignal, QRunnable, QThreadPool
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QAction, QSpinBox, QLabel, QDoubleSpinBox, QMenu, QTableWidget, QPushButton, QWidget, \
    QVBoxLayout, QHBoxLayout
from mne.viz import plot_topomap

from scipy.signal import savgol_filter
from scipy.fft import fft, fftfreq, rfft
from PyQt5 import QtCore, QtGui, QtWidgets
from datetime import datetime
import mne
from scipy.fftpack import rfftfreq

from pynfb.generators import ch_names32
from pynfb.generators import ch_names27
from pynfb.inlets.montage import Montage

from pynfb.postprocessing.utils import runica
from pynfb.widgets import edf_saver
from vispy import scene

class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(int, object, object)
    progress = pyqtSignal(int)

class DPHandler(QRunnable):
    def __init__(self, y, c, scaler, shift):
        super().__init__()
        self.y = y
        self.c = c
        self.scaler = scaler
        self.shift = shift
        self.signals = WorkerSignals()

    def run(self):
        signal_buffer = (self.y - self.c) * self.scaler + 50 * self.shift
        spectral_buffer = rfft(self.y)
        self.signals.result.emit(self.shift, signal_buffer, spectral_buffer)

class CrossButtonsWidget(QtWidgets.QWidget):
    names = ['left', 'right', 'up', 'down']
    symbols = ['<', '>', '^', 'v']
    positions = [(1, 0), (1, 2), (0, 1), (2, 1)]

    def __init__(self, parent=None):
        super(CrossButtonsWidget, self).__init__(parent)
        buttons = dict([(name, QtWidgets.QPushButton(symbol)) for name, symbol in zip(self.names, self.symbols)])
        layout = QtWidgets.QGridLayout(self)
        self.clicked_dict = {}
        for name, pos in zip(self.names, self.positions):
            buttons[name].setAutoRepeat(True)
            buttons[name].setAutoRepeatDelay(100)
            layout.addWidget(buttons[name], *pos)
            self.clicked_dict[name] = buttons[name].clicked


class LSLPlotDataItem(pg.PlotDataItem):
    def getData(self):
        x, y = super(LSLPlotDataItem, self).getData()
        if self.opts['fftMode']:
            return x, y / (max(y) - min(y) + 1e-20) * 0.75
        return x, y


class PopupWidget(QPushButton):
    def __init__(self, label, elements, checklist, parent=None):
        super(PopupWidget, self).__init__(parent)
        self.parent = parent
        self.setText(label)
        self.menu = QMenu(self)
        self.checklist = checklist
        for i in range(0, len(elements)):
            Action = QAction(elements[i], self.menu)
            Action.triggered.connect(lambda chk, item=i: self.open_spectral(item))
            self.menu.addAction(Action)

    def mouseReleaseEvent(self, event):
        self.menu.exec_(event.globalPos())

    def open_spectral(self, index):
        self.parent.open_spectral(index)

    def getChecklist(self):
        mlen = len(self.menu.actions())
        checklist = np.zeros(mlen)
        for i in range(0, mlen):
            checklist[i] = self.menu.actions()[i].isChecked()
        return checklist

class SpectralWindow(QWidget):
    def __init__(self, name, n, freq):
        super().__init__()
        layout = QHBoxLayout()
        self.setWindowTitle(name)

        color = np.ones((n, 4), dtype=np.float32)
        color[:, 0] = np.linspace(0, 1, n)
        color[:, 1] = color[::-1, 0]
        canvas = scene.SceneCanvas(keys='interactive', show=True)
        grid = canvas.central_widget.add_grid(spacing=0)
        self.viewbox = grid.add_view(row=0, col=1, camera='panzoom')
        x_axis = scene.AxisWidget(orientation='bottom')
        x_axis.stretch = (1, 0.1)
        grid.add_widget(x_axis, row=1, col=1)
        x_axis.link_view(self.viewbox)
        y_axis = scene.AxisWidget(orientation='left')
        y_axis.stretch = (0.1, 1)
        grid.add_widget(y_axis, row=0, col=0)
        y_axis.link_view(self.viewbox)

        self.pos = np.zeros((n, 2), dtype=np.float32)
        self.pos[:, 0] = rfftfreq(n, 1/freq)
        self.line = scene.Line(self.pos, color, parent=self.viewbox.scene)

        self.viewbox.camera.set_range()
        layout.addWidget(canvas.native)
        self.setLayout(layout)

    def set_data(self, y):
        self.pos[:, 1] = y
        self.line.set_data(pos=self.pos)

class RhythmsWindow(QWidget):
    def __init__(self, n, freq):
        super().__init__()
        layout = QHBoxLayout()
        self.setWindowTitle("Rhythms indicators")
        self.n = n
        self.freq = freq

        self.bar1 = pg.BarGraphItem(x=[1], height=1, width=0.2, brush='g')
        self.bar2 = pg.BarGraphItem(x=[2], height=2, width=0.2, brush='w')
        self.bar3 = pg.BarGraphItem(x=[3], height=3, width=0.2, brush='y')
        self.plot = pg.PlotWidget()
        self.plot.addItem(self.bar1)
        self.plot.addItem(self.bar2)
        self.plot.addItem(self.bar3)
        layout.addWidget(self.plot)
        self.setLayout(layout)

        self.update_counter = 1001

    def get_indicator(self, data, num, f, k2):
        x = rfftfreq(self.n, 1/self.freq)
        buffer = rfft(data[:, num])
        sy0 = 0
        sy1 = 0
        sy2 = 0
        sx = 0
        B = []

        for i in range(0, len(x), 1):
            if f[0] < x[i] < f[3]:
                B.append(np.abs(buffer)[i])

        k1 = len(B)-1 if len(B)%2 == 0 else len(B)
        fb = savgol_filter(B, k1, k2)
        #pg.plot(fb)
        kx = len(fb)/(f[3]-f[0])
        for i in range(0, int(kx*(f[1]-f[0])), 1):
            sy0 += fb[i]
        for i in range(int(kx*(f[1]-f[0])), int(kx*(f[2]-f[0])), 1):
            sy1 += fb[i]
        for i in range(int(kx*(f[2]-f[0])), len(fb), 1):
            sy2 += fb[i]

        # print(kx)
        # print(sy1)
        # print(sy0)
        # print(sy2)

        if sy1:
            sx = 1 - (sy0/sy1+sy2/sy1)/2
        if sx < -1:
            sx = -1
        return sx

    def set_data(self, data):
        self.update_counter += 1
        if self.update_counter <= 10:
            return
        else:
            self.update_counter = 0

        ai = self.get_indicator(data[0], 19, [7, 9, 11, 13], 6) # OZ, alpha
        ti = self.get_indicator(data[1], 3, [3, 5, 7, 9], 6) # F7, theta
        di = self.get_indicator(data[2], 10, [0.4, 1.4, 2.4, 3.4], 7) # CZ, delta
        self.bar1.setOpts(height=ai)
        self.bar2.setOpts(height=ti)
        self.bar3.setOpts(height=di)

class RawViewer(QWidget):
    def __init__(self, fs, channels_labels, parent=None, overlap=False, mode=0, toolbar=None, freqbar=None, widgets=None):
        super(RawViewer, self).__init__(parent)
        n_channels = len(channels_labels)
        #print(channels_labels)
        self.n_channels = n_channels
        self.calib_values = np.zeros(self.n_channels)
        self.calib_flag = False
        self.visible_flags = np.zeros(self.n_channels)
        self.channelsanalyze_flags = np.zeros(self.n_channels)
        self.spectraldata = []
        self.tw = None
        self.sw = [QWidget]*n_channels
        for i in range(0, n_channels):
            #self.sw.append(None)
            self.visible_flags[i] = True
            self.channelsanalyze_flags[i] = False

        print("Signal painter started")

        self.channels_labels = channels_labels
        self.toolbar = toolbar
        #self.freqbar = freqbar
        self.widgets = widgets
        self.fs = fs
        self.current_pos = 0
        self.std = None
        self.raw_buffer = None
        self.filtered_buffer1 = None
        self.filtered_buffer2 = None
        self.filtered_buffer3 = None
        self.scaler = 0.0041
        self.step = 0.0001
        self.curves = []
        self.lines = []
        self.x_mesh = None
        #self.setYRange(0, n_channels)
        self.mode = mode
        self.n_samples_to_display = 0
        #self.notch_filter = NotchFilter(0, fs, n_channels)
        #self.butter_filter = ButterFilter((0.5, 45), fs, n_channels)

        print("FS: "+str(fs))
        self.pool = QThreadPool.globalInstance()

        if self.mode == 0:
            buffer_time_sec = 10
            self.n_samples = int(buffer_time_sec*fs)
            self.n_samples_to_display = self.n_samples
            self.raw_buffer = np.zeros((self.n_samples, self.n_channels))
            self.x_mesh = np.linspace(0, self.n_samples_to_display / self.fs, self.n_samples_to_display)
            #self.setXRange(0, self.x_mesh[self.n_samples_to_display - 1])
            self.create_surface()
            self.rw = RhythmsWindow(int(self.n_samples/2), self.fs)

        self.saveAction = QAction(self)
        self.saveAction.setText("Save data...")
        self.recalibAction = QAction(self)
        self.recalibAction.setText("Recalibrate")
        self.scaleupAction = QAction(self)
        self.scaleupAction.setText(" ^ ")
        self.scaledownAction = QAction(self)
        self.scaledownAction.setText(" v ")
        self.saveAction.triggered.connect(lambda: self.save_data())
        self.recalibAction.triggered.connect(lambda: self.calibrate())
        self.scaleupAction.triggered.connect(lambda: self.update_scaler(increase=True))
        self.scaledownAction.triggered.connect(lambda: self.update_scaler(increase=False))
        self.toolbar.addAction(self.saveAction)
        self.toolbar.addAction(self.recalibAction)
        self.toolbar.addAction(self.scaleupAction)
        self.toolbar.addAction(self.scaledownAction)
        self.stepSpinBox = QDoubleSpinBox()
        self.ampLabel = QLabel()
        #font = self.stepSpinBox.font()
        #font.setPointSizeF(20.25)
        #self.stepSpinBox.setFont(font)

        #self.stepSpinBox.setFloatDomain(True)
        self.stepSpinBox.setDecimals(5)
        self.stepSpinBox.setMaximum(1000)
        self.stepSpinBox.setValue(self.step)
        self.stepSpinBox.setFixedWidth(100)
        self.ampLabel.setText("Amplitude scale: x"+str(self.scaler))
        self.ampLabel.setFixedWidth(150)
        self.toolbar.addWidget(self.ampLabel)
        self.toolbar.addWidget(self.stepSpinBox)

        # self.visMenu = QMenu("menu")
        # self.visMenu.addAction(QAction('50%', self.visMenu, checkable=True))
        # self.visMenu.addAction(QAction('50%', self.visMenu, checkable=True))
        # self.visMenu.addAction(QAction('50%', self.visMenu, checkable=True))
        # self.visMenu.addAction(QAction('50%', self.visMenu, checkable=True))

        #self.vis_widget = PopupWidget('Channel visibility', channels_labels, self.visible_flags, parent=self)
        #self.toolbar.addWidget(self.vis_widget)

        self.model = QtGui.QStandardItemModel()
        self.listView = QtWidgets.QListView()
        self.alglist = QtWidgets.QListView()
        self.algmodel = QtGui.QStandardItemModel()

        if channels_labels is not None:
            for i in range(len(channels_labels)):
                item = QtGui.QStandardItem(channels_labels[i])
                item.setCheckable(True)
                #check = QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked
                item.setCheckState(QtCore.Qt.Checked)

                self.model.appendRow(item)
        self.listView.setModel(self.model)
        algnames = []
        # algnames = ["[preprocessing] Butterworth filter", "[analyze] SVM"]
        for i in range(len(algnames)):
            item = QtGui.QStandardItem(algnames[i])
            item.setCheckable(True)
            item.setCheckState(QtCore.Qt.Checked)
            self.algmodel.appendRow(item)

        self.alglist.setModel(self.algmodel)

        #vbox.addWidget(self.listView)
        #vbox.addStretch(1)
        self.listView.clicked.connect(self.process_vischeck)

        self.listwidget = QWidget()
        self.listlayout = QVBoxLayout(self)
        self.listlayout.addWidget(self.listView)
        self.listlayout.addWidget(self.alglist)
        ab1 = QPushButton()
        ab1.setText("Add")
        ab2 = QPushButton()
        ab2.setText("Edit")
        ab3 = QPushButton()
        ab3.setText("Delete")
        self.listlayout.addWidget(ab1)
        self.listlayout.addWidget(ab2)
        self.listlayout.addWidget(ab3)
        self.listwidget.setLayout(self.listlayout)
        self.widgets.setWidget(self.listwidget)


        self.channelsfilter_widget = PopupWidget('Spectral plots', channels_labels, self.channelsanalyze_flags, parent=self)
        self.toolbar.addWidget(self.channelsfilter_widget)

        self.rwopenAction = QAction(self)
        self.rwopenAction.setText("Rhythms")
        self.rwopenAction.triggered.connect(lambda: self.rw.show())
        self.toolbar.addAction(self.rwopenAction)

        self.twopenAction = QAction(self)
        self.twopenAction.setText("Topography")
        self.twopenAction.triggered.connect(lambda: self.tw.showMaximized())
        self.toolbar.addAction(self.twopenAction)

        #self.setLayout(vbox)

        #self.channelsfilter_widget = PopupWidget('Channels to filter', channels_labels, self.channelsfilter_flags, parent=self)
        #self.toolbar.addWidget(self.channelsfilter_widget)

        self.show_levels_flag = False
        self.overlap = overlap
        if overlap:
            self.show_levels_flag = True
            self.show_levels()

    def create_surface(self):
        N = self.n_samples_to_display
        color = np.ones((N, 4), dtype=np.float32)
        color[:, 0] = np.linspace(0, 1, N)
        color[:, 1] = color[::-1, 0]
        canvas = scene.SceneCanvas(keys='interactive', show=True)
        grid = canvas.central_widget.add_grid(spacing=0)
        self.viewbox = grid.add_view(row=0, col=1, camera='panzoom')

        x_axis = scene.AxisWidget(orientation='bottom')
        x_axis.stretch = (1, 0.1)
        grid.add_widget(x_axis, row=1, col=1)
        x_axis.link_view(self.viewbox)
        y_axis = scene.AxisWidget(orientation='left')
        y_axis.stretch = (0.1, 1)
        grid.add_widget(y_axis, row=0, col=0)
        y_axis.link_view(self.viewbox)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(canvas.native)
        for i in range(0, self.n_channels):
            pos = np.zeros((N, 2), dtype=np.float32)
            pos[:, 0] = self.x_mesh[:self.n_samples_to_display]
            if i:
                pos[:, 1] = pos[:, 1] + 50*i
            c = scene.Line(pos, color, parent=self.viewbox.scene)
            self.curves.append(pos)
            self.lines.append(c)
        self.viewbox.camera.set_range()

    def open_spectral(self, index):
        if not self.channelsanalyze_flags[index]:
            self.sw[index] = SpectralWindow(self.channels_labels[index], int(self.n_samples / 2), self.fs)
            self.channelsanalyze_flags[index] = True
        self.sw[index].set_data(np.abs(self.spectraldata)[:int(self.n_samples/2)])
        self.sw[index].show()

    def process_vischeck(self, index):
        self.visible_flags[index.row()] = self.listView.model().itemData(index)[10]
        if not self.visible_flags[index.row()]:
            self.curves[index.row()][:, 1] = np.ones(self.n_samples_to_display) + 50 * index.row()

        self.update_data()

    def update_std(self, chunk):
        if self.std is None:
            self.std = np.std(chunk)
        else:
            self.std = 0.5 * np.std(chunk) + 0.5 * self.std

    def set_chunk(self, signal, chunk):
        n_samples = len(chunk)

        if self.mode == 1:
            self.raw_buffer = np.zeros((n_samples, self.n_channels))
            self.n_samples_to_display += n_samples
            self.x_mesh = np.linspace(0, self.n_samples_to_display / self.fs, self.n_samples_to_display)
            self.n_samples = n_samples
            self.create_surface()
            # for i in range(0, self.n_channels):
            #     self.sw.append(SpectralWindow(self.channels_labels[i], int(self.n_samples/2), self.fs))
            self.rw = RhythmsWindow(int(self.n_samples/2), self.fs)

        self.raw_buffer[:-n_samples] = self.raw_buffer[n_samples:]
        self.raw_buffer[-n_samples:] = chunk
        if not self.calib_flag:
            self.calibrate()

        if signal is not None:
            # runica(self.raw_buffer, self.fs, self.channels_labels)
            #montage = Montage(ch_names32)
            # ch_names32 = ['EEG FP1', 'EEG FPZ', 'EEG FP2', 'EEG  F7', 'EEG  F3', 'EEG  FZ', 'EEG  F4', 'EEG  F8', 'EEG  T3',
            #               'EEG  C3', 'EEG  CZ', 'EEG C4', 'EEG T4', 'EEG  T5', 'EEG P3', 'EEG PZ', 'EEG P4', 'EEG T6', 'EEG O1',
            #               'EEG OZ', 'EEG O2', 'EEG PBH', 'EEG NOS', 'EEG SHL', 'EEG SHH', 'EEG A1', 'EEG A2', '', '', '', '', '']
            #montage = Montage(ch_names32)
            #print(self.channels_labels)
            #a = ch_names27.conjugate
            #print(np.shape(self.channels_labels))
            #print(np.shape(ch_names27))
            #b = self.channels_labels.conjugate

            #print(montage.get_names())
            # spatial, topo = runica(np.random.normal(size=(100000, 32)), 1000, montage.get_names(), mode='csp')
            if False:
                self.tw = runica(self.raw_buffer, self.fs, self.channels_labels, mode='csp')
                topocanvas = self.tw.table.topographies_items[0].topo
                width, height = topocanvas.get_width_height()
                print("canvas pixels size "+str(width)+"x"+str(height))
                for x in range(0, width):
                    for y in range(0, height):
                        pixel = QImage(topocanvas.buffer_rgba(), width, height, QImage.Format_ARGB32).pixel(x, y)
                        #print(QtGui.QColor(pixel).getRgb()[:-1])
                #print("topography data: ", self.tw.topographies)
                # plot_topomap(spatial, montage.get_pos())
                # plot_topomap(topo, montage.get_pos())

            self.filtered_buffer1 = self.raw_buffer.copy()
            self.filtered_buffer2 = self.raw_buffer.copy()
            self.filtered_buffer3 = self.raw_buffer.copy()
            signal.fit_model(self.filtered_buffer1, [8, 13], self.channels_labels)
            self.filtered_buffer1 = signal.apply(self.filtered_buffer1)

            signal.fit_model(self.filtered_buffer2, [4, 8], self.channels_labels)
            self.filtered_buffer2 = signal.apply(self.filtered_buffer2)

            signal.fit_model(self.filtered_buffer3, [1, 4], self.channels_labels)
            self.filtered_buffer3 = signal.apply(self.filtered_buffer3)
        self.update_data()

    def set_data(self, i, data, spectraldata):
        if self.visible_flags[i]:
            self.curves[i][:, 1] = data
            self.lines[i].set_data(pos=self.curves[i])
        self.spectraldata = spectraldata
        if self.mode == 1 and self.channelsanalyze_flags[i]:
            self.sw[i].set_data(np.abs(spectraldata)[:int(self.n_samples/2)])

    def update_data(self):
        for i in range(0, self.n_channels, 1):
            runnable = DPHandler(self.filtered_buffer3[:, i], self.calib_values[i], self.scaler, i)
            runnable.signals.result.connect(self.set_data)
            self.pool.start(runnable)
        self.rw.set_data([self.filtered_buffer1, self.filtered_buffer2, self.filtered_buffer3])
        self.ampLabel.setText("Amplitude scale: x" + str(self.scaler))

    def update_scaler(self, increase=False):
        self.step = self.stepSpinBox.value()
        self.scaler += self.step if increase else -self.step
        if self.scaler < 0:
            self.scaler = 0
        self.update_data()
        self.update_levels()

    def update_n_samples_to_display(self, increase=False):
        step = int(self.fs * 0.5)
        self.n_samples_to_display += step if increase else -step
        if self.n_samples_to_display < 10:
            self.n_samples_to_display = 10
        elif self.n_samples_to_display > self.n_samples:
            self.n_samples_to_display = self.n_samples
        self.update_data()
        self.update_levels()

    def show_levels(self):
        self.levels = {'zero': [], 'p1': [], 'm1': []}
        for i in (range(self.n_channels) if not self.overlap else [0]):
            for level, val in zip(['zero', 'p1', 'm1'], [0, 1, -1]):
                c = LSLPlotDataItem()
                c.setPen(pg.mkPen('g'))
                #self.addItem(c)
                if not self.overlap:
                    c.setPos(0, i + 1)
                #self.levels[level].append(c)
        self.update_levels()

    def update_levels(self):
        if self.show_levels_flag:
            for i in (range(self.n_channels) if not self.overlap else [0]):
                for level, val in zip(['zero', 'p1', 'm1'], [0, 1, -1]):
                    self.levels[level][i].setData(self.x_mesh[:self.n_samples_to_display],
                              self.x_mesh[:self.n_samples_to_display]*0 + val * self.scaler / (self.std or 1))

    def save_data(self):
        info = mne.create_info(ch_names=self.channels_labels,
                               ch_types=['misc'] * len(self.channels_labels),
                               sfreq=self.fs)
        raw_buffer_t = np.zeros((len(self.channels_labels), self.n_samples_to_display))
        for i in range(0, self.raw_buffer.shape[1]):
            raw_buffer_t[i, :] = self.raw_buffer[:, i]
        raw = mne.io.RawArray(raw_buffer_t, info)
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d-%H_%M_%S")
        print("Timestamp: ", dt_string)
        edf_saver.write_mne_edf(raw, os.path.realpath(os.path.dirname(os.path.realpath(__file__)))+"\\..\\..\\saved_eeg\\"+dt_string+".edf", overwrite=True)

    def calibrate(self):
        valnum = len(self.raw_buffer)
        for i in range(0, self.n_channels):
            av = 0
            for j in range(0, valnum):
                av += self.raw_buffer[j][i]
            self.calib_values[i] = av / valnum
        self.calib_flag = True
        self.scaler = 9e-1 / abs(self.raw_buffer[-1, 0]) if self.raw_buffer[-1, 0] else 1
        self.step = self.scaler / 10
        self.stepSpinBox.setValue(self.step)
        print("Calibration data:")
        print(self.calib_values)
        #print(self.scaler)

if __name__ == '__main__':
    a = QtWidgets.QApplication([])
    plot_widget = RawViewer(250, ['ef', 'sf', 'qwr']*3)

    plot_widget.set_chunk(np.sin(np.arange(20)).reshape(20, 1).dot(np.ones((1, 9))))
    plot_widget.show()
    a.exec_()
