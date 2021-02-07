from PyQt5 import QtGui, QtWidgets

filter_types = ['f1', 'f2', 'f3', 'f4']


class FilterSettingsWidget(QtWidgets.QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setContentsMargins(0, 0, 0, 0)
        self.combo = QtWidgets.QComboBox()
        self.combo.addItem('raw')
        self.combo.addItem('butter+exp')
        self.combo.addItem('butter+sg')
        self.combo.addItem('fft+exp')
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.combo)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.combo.currentIndexChanged.connect(self.combo_changed_event)
        self.combo.setCurrentIndex(filter_types.index(self.parent().params['sFilterType']))
        self.combo_changed_event()

    def combo_changed_event(self):
        self.parent().params['sFilterType'] = filter_types[self.combo.currentIndex()]

    def reset(self):
        self.combo.setCurrentIndex(filter_types.index(self.parent().params['sFilterType']))
        self.combo_changed_event()

class EventsFilterSettingsWidget(QtWidgets.QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setContentsMargins(0, 0, 0, 0)
        self.use_events = QtWidgets.QCheckBox('Use events LSL Stream')
        self.use_events.stateChanged.connect(self.use_events_changed_action)
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.textChanged.connect(self.name_changed_action)
        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.use_events)
        layout.addWidget(self.name_edit)
        layout.setContentsMargins(0, 0, 0, 0)
        self.reset()

    def use_events_changed_action(self):
        self.name_edit.setEnabled(self.use_events.isChecked())
        self.parent().params['sEventsStreamName'] = ''

    def get_name(self):
        return self.name_edit.text() if self.use_events.isChecked() else ''

    def reset(self):
        name = self.parent().params['sEventsStreamName']
        self.name_edit.setText(name)
        if len(name) == 0:
            self.use_events.setChecked(False)
        self.use_events_changed_action()

    def name_changed_action(self):
        self.parent().params['sEventsStreamName'] = self.name_edit.text()


if __name__ == '__main__':
    a = QtWidgets.QApplication([])
    w = EventsFilterSettingsWidget()
    w.show()

    a.exec_()
