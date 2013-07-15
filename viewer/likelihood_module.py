import pylab
from pyface.qt import QtCore, QtGui
import module_template
import embedded_matplotlib

class LikelihoodData(module_template.Data):
    def __init__(self):
        super(LikelihoodData, self).__init__()
        self.read_likelihood()

    def get_likelihood(self, reload=False):
        if reload:
            self.read_likelihood()
        return self._likelihood_data
        
    def read_likelihood(self):
        try:
            self._likelihood_data = pylab.loadtxt('output/likelihood.data')
        except IOError:
            try:
                self._likelihood_data = pylab.loadtxt('likelihood.data')
            except IOError:
                self.read_error.emit()
            self.read_error.emit()
        self.data_changed.emit()

class LikelihoodViewer(module_template.Viewer):
    def __init__(self):
        super(LikelihoodViewer, self).__init__()

        self._widget, self._fig, self._canvas, self._mpl_toolbar = embedded_matplotlib.get_matplotlib_widget()
        self._axes = self._fig.add_subplot(111)

    def plot_likelihood(self, likelihood, iteration=None):
        self._axes.clear()
        self._likelihood_plot = self._axes.plot(likelihood, color='black', lw=2)
        if iteration != None:
            limits = self._axes.get_ylim()
            self._iteration_plot = self._axes.plot([iteration]*2, limits, color='green', lw=2)[0]
        self._canvas.draw()

    def set_iteration(self, iteration):
        self._iteration_plot.set_xdata([iteration]*2)
        self._canvas.draw()

class LikelihoodControll(module_template.Controll):
    def __init__(self, common_controll, viewer, data):
        super(LikelihoodControll, self).__init__(common_controll, viewer, data)
        #self._setup_gui()
        # self.load_and_draw()
        # self._data.data_changed.connect(self.load_and_draw)

    # def _setup_gui(self):
    #     reload_button = QtGui.QPushButton("Reload")
    #     reload_button.pressed.connect(self._data.read_likelihood)
    #     layout = QtGui.QVBoxLayout()
    #     layout.addWidget(reload_button)
    #     # layout.addStretch()
    #     self.setLayout(layout)

    # def load_and_draw(self):
    #     self._viewer.plot_likelihood(self._data.get_likelihood(), self._common_controll.get_iteration())

    def draw_hard(self):
        self._viewer.plot_likelihood(self._data.get_likelihood(reload=True), self._common_controll.get_iteration())
        #self._viewer.plot_likelihood(self._data.get_likelihood(), self._common_controll.get_iteration())
        #self._viewer.set_iteration(self._common_controll.get_iteration())
        

class Plugin(module_template.Plugin):
    def __init__(self, common_controll, parent=None):
        super(Plugin, self).__init__(common_controll)
        self._viewer = LikelihoodViewer()
        self._data = LikelihoodData()
        self._controll = LikelihoodControll(common_controll, self._viewer, self._data)

