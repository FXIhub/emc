"""Small widget to embedd mayavi visualization in qt."""
from QtVersions import QtCore, QtGui

from traits.api import HasTraits, Instance
from traitsui.api import View, Item
try:
    from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
except RuntimeError:
    pass
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor

class MlabVisualization(HasTraits):
    """I don't really understand this class, it was stolen from an example for qt-embedded mayavi"""
    scene = Instance(MlabSceneModel, ())

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=600, width=600, show_label=False),
                resizable=True)

class MlabWidget(QtGui.QWidget):
    """This returns an embedded mayavi object through the get_mlab function
    I don't really understand this class, it was stolen from an example for qt-embedded mayavi"""
    def __init__(self, parent=None):
        super(MlabWidget, self).__init__(parent)
        layout = QtGui.QHBoxLayout(self)

        self._vis = MlabVisualization()
        self._ui = self._vis.edit_traits(parent=self, kind='subpanel').control
        layout.addWidget(self._ui)
        self.setLayout(layout)
        self._ui.setParent(self)

        self.get_scene().background = (1., 1., 1.)

    def get_mlab(self):
        """Returns an mlab object that can be used to call mayavi functions"""
        return self._vis.scene.mlab

    def get_scene(self):
        """Scene object which mlab calls are rendered into."""
        return self._vis.scene

    def save_image(self, file_name):
        """Save the current view."""
        #self.get_mlab().savefig(file_name, figure=self.get_mlab().figure(1), magnification=1)
        #self._vis.scene.save_png(file_name)
        self.get_scene().save(file_name)
