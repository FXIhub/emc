from pyface.qt import QtCore, QtGui
import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure

def get_matplotlib_widget():
    widget = QtGui.QWidget()
    fig = Figure((5., 4.))
    canvas = FigureCanvas(fig)
    canvas.setParent(widget)
    mpl_toolbar = NavigationToolbar(canvas, widget)
    layout = QtGui.QVBoxLayout()
    layout.addWidget(canvas)
    layout.addWidget(mpl_toolbar)
    widget.setLayout(layout)
    return widget, fig, canvas, mpl_toolbar
