from pyface.qt import QtCore, QtGui

from traits.api import HasTraits, Instance, on_trait_change, Int, Dict
from traitsui.api import View, Item
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor

class MlabVisualization(HasTraits):
    """I don't really understand this class, it was stolen from an example for qt-embedded mayavi"""
    scene = Instance(MlabSceneModel, ())

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene), height=600, width=600, show_label=False),
                resizable=True)

class MlabWidget(QtGui.QWidget):
    """This returns an embedded mayavi object throug the get_mlab function
    I don't really understand this class, it was stolen from an example for qt-embedded mayavi"""
    def __init__(self, parent=None):
        i = 0
        QtGui.QWidget.__init__(self, parent)
        layout = QtGui.QHBoxLayout(self)

        self._vis = MlabVisualization()
        self._ui = self._vis.edit_traits(parent=self, kind='subpanel').control
        layout.addWidget(self._ui)
        self.setLayout(layout)
        self._ui.setParent(self)

        # policy = QtGui.QSizePolicy()
        # # policy.setVerticalStretch(1)
        # # policy.setHorizontalStretch(1)
        # policy.setHorizontalPolicy(QtGui.QSizePolicy.Expanding)
        # policy.setHorizontalPolicy(QtGui.QSizePolicy.Expanding)
        # self._ui.setSizePolicy(policy)
        # #self.get_widget().setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)



    def get_mlab(self):
        return self._vis.scene.mlab
