import pylab
from pyface.qt import QtCore, QtGui
import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure

class Data(QtCore.QObject):
    data_changed = QtCore.Signal()
    read_error = QtCore.Signal()
    properties_changed = QtCore.Signal()
    def __init__(self):
        super(Data, self).__init__()

class Viewer(QtCore.QObject):
    def __init__(self):
        super(Viewer, self).__init__()
        self._widget = QtGui.QWidget()

    def get_widget(self):
        return self._widget

class Controll(QtCore.QObject):
    def __init__(self, common_controll, viewer, data):
        super(Controll, self).__init__()
        self._common_controll = common_controll
        self._viewer = viewer
        self._data = data
        self._widget = QtGui.QWidget()
        self._active = True

    def get_viewer(self):
        return self._viewer

    def get_data(self):
        return self._data

    def get_widget(self):
        return self._widget
        
    def set_active(self, state):
        self._active = bool(state)

    def draw_hard(self):
        pass
        
    def draw(self):
        if self._active:
            self.draw_hard()

class Plugin(QtCore.QObject):
    def __init__(self, common_controll):
        super(Plugin, self).__init__()
        self._common_controll = common_controll
        self._viewer = None
        self._data = None
        self._controll = None

    def get_viewer(self):
        return self._viewer

    def get_data(self):
        return self._data

    def get_controll(self):
        return self._controll

    def get_common_controll(self):
        return self._common_controll
