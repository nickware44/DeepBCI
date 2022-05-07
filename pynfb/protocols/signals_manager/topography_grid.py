import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from PyQt5.QtGui import QImage
from pyqtgraph import PlotWidget

from ...generators import ch_names
from ...protocols.ssd.topomap_canvas import TopographicMapCanvas


class BarLabelWidget(QtWidgets.QWidget):
    def __init__(self, value, max_value, min_value=0):
        super(BarLabelWidget, self).__init__()
        self.max_value = max_value
        self.min_value = min_value
        self.value = value

    def set_values(self, value, max_value, min_value=0):
        self.max_value = max_value
        self.min_value = min_value
        self.value = value

    def paintEvent(self, e):
        qp = QtGui.QPainter()
        qp.begin(self)
        self.draw_value(e, qp)
        qp.end()

    def draw_value(self, event, qp):
        size = self.size()
        qp.setPen(QtCore.Qt.white)
        qp.setBrush(QtGui.QColor(51, 152, 188, 50))
        padding = 50 if 50 < size.height() else 0
        qp.drawRect(0, 0 + padding,
                    int(size.width() * (self.value - self.min_value) / (self.max_value - self.min_value)) - 1,
                    size.height() - 2 * padding - 1)
        qp.setPen(QtCore.Qt.black)
        qp.drawText(1, size.height() // 2 + 1, str(round(self.value, 5)))


class TopoFilterCavas(QtWidgets.QWidget):
    def __init__(self, parent, names, topo, filter, size):
        super(TopoFilterCavas, self).__init__(parent)

        # topography layout
        topo_canvas = TopographicMapCanvas()
        topo_canvas.setMaximumWidth(size)
        topo_canvas.setMaximumHeight(size)
        topo_canvas.update_figure(topo, names=names, show_names=[], show_colorbar=False)

        # filter layout
        filter_canvas = TopographicMapCanvas()
        filter_canvas.setMaximumWidth(size)
        filter_canvas.setMaximumHeight(size)
        filter_canvas.update_figure(filter, names=names, show_names=[], show_colorbar=False)
        #filter_canvas.setHidden(True)

        # layout
        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(topo_canvas)
        layout.addWidget(filter_canvas)

        # attr
        self.show_filter = False
        self.topo = topo_canvas
        self.filter = filter_canvas
        self.names = names

    def switch(self):
        self.show_filter = not self.show_filter
        self.filter.setHidden(not self.show_filter)
        self.topo.setHidden(self.show_filter)

    def update_data(self, topo, filter):
        self.filter.update_figure(filter, names=self.names, show_names=[], show_colorbar=False)
        self.topo.update_figure(topo, names=self.names, show_names=[], show_colorbar=False)


class TopographyGrid(QtWidgets.QScrollArea):
    # one_selected = QtCore.pyqtSignal()
    # more_one_selected = QtCore.pyqtSignal()
    # no_one_selected = QtCore.pyqtSignal()

    def __init__(self, time_series, topographies, filters, channel_names, fs, scores, scores_name='Mutual info',
                 marks=None, *args):
        super(TopographyGrid, self).__init__(*args)

        # attributes
        self.row_items_max_height = 125
        self.time_series = time_series
        self.marks = marks
        self.channel_names = channel_names
        self.fs = fs

        # Container Widget
        # Layout of Container Widget
        self.widget = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.widget)
        self.setWidgetResizable(True)
        self.setWidget(self.widget)

        # Scroll Area Layer add
        # vLayout = QtWidgets.QVBoxLayout(self)
        # vLayout.addLayout(topBox)
        # vLayout.addWidget(scroll)
        # vLayout.addLayout(bottomBox)
        #
        # self.setLayout(vLayout)
        #self.repopulate()

        # set size and names
        # self.columns = ['Selection', scores_name, 'Topography', 'Time series (push to switch mode)']
        # self.setColumnCount(len(self.columns))
        # self.setRowCount(time_series.shape[1])
        # self.setHorizontalHeaderLabels(self.columns)

        # columns widgets
        self.checkboxes = []
        self.topographies_items = []
        self.plot_items = []
        self.scores = []
        _previous_plot_link = None
        c = 0
        r = 0
        w = 5
        ch = list()
        for ind in range(len(self.channel_names)):
            ch.append(ch_names[ind])
        for ind in range(len(self.channel_names)):
            # topographies and filters
            topo_filter = TopoFilterCavas(self, ch, topographies[:, ind], filters[:, ind],
                                          self.row_items_max_height)
            self.topographies_items.append(topo_filter)

            gwidget = QtWidgets.QGroupBox(self.channel_names[ind])
            #gwidget.setWidgetResizable(True)
            panel = QtWidgets.QHBoxLayout(gwidget)
            panel.addWidget(topo_filter)
            self.grid.addWidget(gwidget, c, r)

            c = c + 1
            if c > w:
                c = 0
                r = r + 1

        # formatting
        #self.current_row = None
        #self.horizontalHeader().setStretchLastSection(True)
        #self.resizeColumnsToContents()
        #self.resizeRowsToContents()

        # clickable 3 column header
        #self.horizontalHeader().sectionClicked.connect(self.handle_header_click)
        #self.is_spectrum_mode = False

        # reorder
        #self.order = np.argsort(scores)
        # self.reorder()

        # checkbox signals
        # for checkbox in self.checkboxes:
        #     checkbox.stateChanged.connect(self.checkboxes_state_changed)

        # selection context menu
        # header = self.horizontalHeader()
        # header.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        # header.customContextMenuRequested.connect(self.handle_header_menu)

        # ctrl+a short cut
        # QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_A), self).activated.connect(
        #     self.ctrl_plus_a_event)

        # checkbox cell clicked
        #self.cellClicked.connect(self.cell_was_clicked)
