import pylab
from pyface.qt import QtCore, QtGui
import module_template
import embedded_matplotlib

class FitData(module_template.Data):
    def __init__(self):
        super(FitData, self).__init__()
        self.read_fit()

    def get_fit(self, reload=False):
        if reload:
            self.read_fit()
        return self._fit_data

    def get_fit_best_rot(self, reload=False):
        if reload:
            self.read_fit()
        return self._fit_data_best_rot
        
    def read_fit(self):
        sucess = False
        try:
            raw_data = pylab.loadtxt('output/fit.data')
            if len(raw_data.shape) == 1:
                raw_data = raw_data.reshape(1, len(raw_data))
            self._fit_data = raw_data.mean(axis=1)
            sucess = True
        except IOError:
            print "ioerror"
            self.read_error.emit()
        try:
            self._fit_data_best_rot = pylab.loadtxt('output/fit_best_rot.data').mean(axis=1)
            sucess = True
        except IOError:
            print "ioerror"
            self.read_error.emit()
        if sucess:
            self.data_changed.emit()

class FitViewer(module_template.Viewer):
    def __init__(self):
        super(FitViewer, self).__init__()

        self._widget, self._fig, self._canvas, self._mpl_toolbar = embedded_matplotlib.get_matplotlib_widget()
        self._axes = self._fig.add_subplot(111)

    def plot_fit(self, likelihood, likelihood_best_rot=None, iteration=None):
        self._axes.clear()
        self._fit_plot = self._axes.plot(likelihood, color='black', lw=2, label='Average fit')
        self._fit_best_rot_plot = self._axes.plot(likelihood, color='red', lw=2, label='Best rotation fit')
        self._axes.set_ylim((0., 1.))
        self._axes.legend()
        if iteration != None:
            limits = self._axes.get_ylim()
            self._iteration_plot = self._axes.plot([iteration]*2, limits, color='green', lw=2)[0]
        self._canvas.draw()

    def set_iteration(self, iteration):
        self._iteration_plot.set_xdata([iteration]*2)
        self._canvas.draw()

class FitControll(module_template.Controll):
    def __init__(self, common_controll, viewer, data):
        super(FitControll, self).__init__(common_controll, viewer, data)

    def draw_hard(self):
        self._viewer.plot_fit(self._data.get_fit(reload=True), self._data.get_fit_best_rot(reload=True), self._common_controll.get_iteration())
        

class Plugin(module_template.Plugin):
    def __init__(self, common_controll, parent=None):
        super(Plugin, self).__init__(common_controll)
        self._viewer = FitViewer()
        self._data = FitData()
        self._controll = FitControll(common_controll, self._viewer, self._data)

