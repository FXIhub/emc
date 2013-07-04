from pylab import *
from pyface.qt import QtCore, QtGui
import module_template
import modelmap_module

class Plugin(module_template.Plugin):
    def __init__(self, common_controll, parent=None):
        super(Plugin, self).__init__(common_controll)
        self._viewer = modelmap_module.ModelmapViewer()
        self._data = modelmap_module.ModelmapData("output/weight")
        self._controll = modelmap_module.ModelmapControll(self._common_controll, self._viewer, self._data)
