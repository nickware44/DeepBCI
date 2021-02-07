import os
import numpy as np
import pyqtgraph as pg
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction, QSpinBox, QLabel, QDoubleSpinBox, QMenu, QTableWidget, QPushButton
from scipy import signal
from PyQt5 import QtCore, QtGui, QtWidgets
from datetime import datetime
import mne
from pynfb.widgets import edf_saver

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
            Action = QAction(elements[i], self.menu, checkable=True)
            Action.setChecked(checklist[i])
            Action.triggered.connect(lambda: self.doUpdate())
            self.menu.addAction(Action)

    def mouseReleaseEvent(self, event):
        self.menu.exec_(event.globalPos())

    def doUpdate(self):
        self.parent.update()

    def getChecklist(self):
        mlen = len(self.menu.actions())
        checklist = np.zeros(mlen)
        for i in range(0, mlen):
            checklist[i] = self.menu.actions()[i].isChecked()
        return checklist

class RawViewer(pg.PlotWidget):
    def __init__(self, fs, channels_labels, parent=None, overlap=False, mode=0, toolbar=None):
        super(RawViewer, self).__init__(parent, background=pg.mkColor(255, 255, 255, 255))
        # cross
        #cross = CrossButtonsWidget(self)
        #cross.setGeometry(50, 0, 100, 100)
        #cross.clicked_dict['up'].connect(lambda: self.update_scaler(increase=True))
        #cross.clicked_dict['down'].connect(lambda: self.update_scaler(increase=False))
        #cross.clicked_dict['right'].connect(lambda: self.update_n_samples_to_display(increase=True))
        #cross.clicked_dict['left'].connect(lambda: self.update_n_samples_to_display(increase=False))

        self.getPlotItem().setMouseEnabled(x=True, y=False)
        self.getPlotItem().showGrid(True, True, 0.7)

        n_channels = len(channels_labels)
        print(channels_labels)
        self.n_channels = n_channels
        self.calib_values = np.zeros(self.n_channels)
        self.calib_flag = False
        self.visible_flags = np.zeros(self.n_channels)
        for i in range(0, n_channels):
            self.visible_flags[i] = True
        self.channels_labels = channels_labels
        self.toolbar = toolbar
        self.fs = fs
        self.current_pos = 0
        self.std = None
        self.raw_buffer = None
        self.scaler = 2000.
        self.step = 500.
        self.curves = []
        self.x_mesh = None
        self.setYRange(0, n_channels)
        self.mode = mode
        self.n_samples_to_display = 0

        if self.mode == 0:
            buffer_time_sec = 20
            self.n_samples = int(buffer_time_sec * fs)
            self.n_samples_to_display = self.n_samples
            self.raw_buffer = np.zeros((self.n_samples, self.n_channels))
            self.x_mesh = np.linspace(0, self.n_samples_to_display / self.fs, self.n_samples_to_display)
            self.setXRange(0, self.x_mesh[self.n_samples_to_display - 1])

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
        self.stepSpinBox.setDecimals(3)
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

        self.vis_widget = PopupWidget('Channel visibility', channels_labels, self.visible_flags, parent=self)
        self.toolbar.addWidget(self.vis_widget)

        self.getPlotItem().showAxis('right')
        if not overlap:
            self.getPlotItem().getAxis('right').setTicks(
                [[(val, tick) for val, tick in zip(range(1, n_channels + 1, 2), range(1, n_channels + 1, 2))],
                 [(val, tick) for val, tick in zip(range(1, n_channels + 1), range(1, n_channels + 1))]])
            self.getPlotItem().getAxis('left').setTicks(
                [[(val, tick) for val, tick in zip(range(1, n_channels + 1), channels_labels)]])
        else:
            self.getPlotItem().addLegend(offset=(-30, 30))
        for i in range(n_channels):
            c = LSLPlotDataItem(pen=(i, n_channels * 1.3), name=channels_labels[i])
            self.addItem(c)
            if not overlap:
                c.setPos(0, i + 1)
            self.curves.append(c)
        self.show_levels_flag = False
        self.overlap = overlap
        if overlap:
            self.show_levels_flag = True
            self.show_levels()

    def update_std(self, chunk):
        if self.std is None:
            self.std = np.std(chunk)
        else:
            self.std = 0.5 * np.std(chunk) + 0.5 * self.std

    def set_chunk(self, chunk):
        n_samples = len(chunk)
        #print("samples got = "+str(n_samples))

        if self.mode == 1:
            self.raw_buffer = np.zeros((n_samples, self.n_channels))
            self.n_samples_to_display += n_samples
            self.x_mesh = np.linspace(0, self.n_samples_to_display / self.fs, self.n_samples_to_display)
            self.n_samples = n_samples
            self.setXRange(0, self.x_mesh[self.n_samples_to_display - 1])

        self.raw_buffer[:-n_samples] = self.raw_buffer[n_samples:]
        self.raw_buffer[-n_samples:] = chunk
        if not self.calib_flag:
            self.calibrate()
        #print(self.raw_buffer[0])
        self.update()

    def update(self):
        std = 1 if self.std is None or self.std == 0 else self.std
        vislist = self.vis_widget.getChecklist()
        for i in range(0, self.n_channels, 1):
            self.curves[i].setPen(pg.mkPen(pg.mkColor(0, 0, 0)))
            if vislist[i]:
                self.curves[i].setData(self.x_mesh[:self.n_samples_to_display], (self.raw_buffer[-self.n_samples_to_display:, i] - self.calib_values[i]) * self.scaler / std) #- self.calib_values[i]
            else:
                self.curves[i].setData(self.x_mesh[:self.n_samples_to_display], np.zeros(self.n_samples_to_display))
        #self.setXRange(0, self.x_mesh[self.n_samples_to_display-1])
        #print(vislist)
        self.ampLabel.setText("Amplitude scale: x" + str(self.scaler))

    def update_scaler(self, increase=False):
        self.step = self.stepSpinBox.value()
        self.scaler += self.step if increase else -self.step
        if self.scaler < 0:
            self.scaler = 0
        self.update()
        self.update_levels()

    def update_n_samples_to_display(self, increase=False):
        step = int(self.fs * 0.5)
        self.n_samples_to_display += step if increase else -step
        if self.n_samples_to_display < 10:
            self.n_samples_to_display = 10
        elif self.n_samples_to_display > self.n_samples:
            self.n_samples_to_display = self.n_samples
        self.update()
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
        print("Calibration data:")
        print(self.calib_values)

if __name__ == '__main__':
    a = QtWidgets.QApplication([])
    plot_widget = RawViewer(250, ['ef', 'sf', 'qwr']*3)

    plot_widget.set_chunk(np.sin(np.arange(20)).reshape(20, 1).dot(np.ones((1, 9))))
    plot_widget.show()
    a.exec_()
