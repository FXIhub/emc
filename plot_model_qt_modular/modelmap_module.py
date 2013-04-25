from pylab import *
import h5py
from functools import partial
import sphelper
import module_template
from pyface.qt import QtCore, QtGui
import embedded_mayavi

INIT_SURFACE_LEVEL = 0.5

def enum(*enums):
    return type('Enum', (), dict(zip(enums, range(len(enums)))))

VIEW_TYPE = enum('surface', 'slice')

class ModelmapData(module_template.Data):
    def __init__(self, file_prefix):
        super(ModelmapData, self).__init__()
        self._file_prefix = file_prefix
        self._side = None

    def get_map(self, iteration):
        try:
            if iteration >= 0:
                modelmap = sphelper.import_spimage('%s_%.4d.h5' % (self._file_prefix, iteration), ['image'])
            elif iteration == -1:
                modelmap = sphelper.import_spimage('%s_init.h5' % (self._file_prefix), ['image'])
            elif iteration == -2:
                modelmap = sphelper.import_spimage('%s_final.h5' % (self._file_prefix), ['image'])
        except IOError:
            self.read_error.emit()
            return zeros((self._side, )*3)
        if self._side != modelmap.shape[0]:
            self.properties_changed.emit()
        self._side = modelmap.shape[0]
        return modelmap

    def get_mask(self, iteraiton):
        try:
            if iteration >= 0:
                mask = sphelper.import_spimage('%s_%.4d.h5' % (self._file_prefix, iteration), ['mask'])
            elif iteration == -1:
                mask = sphelper.import_spimage('%s_init.h5' % (self._file_prefix), ['mask'])
            elif iteration == -2:
                mask = sphelper.import_spimage('%s_final.h5' % (self._file_prefix), ['mask'])
        except IOError:
            self.read_error.emit()
            return
        if self._side != mask.shape[0]:
            properties_changed.emit()
        self._side = modelmap.shape[0]
        return mask

    def get_side(self):
        return self._side

class ModelmapViewer(module_template.Viewer):
    def __init__(self, parent=None):
        super(ModelmapViewer, self).__init__()
        self._mlab_widget = embedded_mayavi.MlabWidget()
        self._surface_level = INIT_SURFACE_LEVEL

    def get_mlab(self):
        return self._mlab_widget.get_mlab()

    def get_widget(self):
        return self._mlab_widget

    def plot_map(self, modelmap):
        self._scalar_field.scalar_data = modelmap
        self._scalar_field_max = self._scalar_field.scalar_data.max()
        self._surface_plot.contour.contours[0] = self._surface_level*self._scalar_field_max

    def plot_map_init(self, modelmap):
        self._scalar_field = self.get_mlab().pipeline.scalar_field(modelmap)
        self._scalar_field_max = self._scalar_field.scalar_data.max()
        self._surface_plot = self.get_mlab().pipeline.iso_surface(self._scalar_field, contours=[self._surface_level*self._scalar_field_max])
        self._slice_plot = [
            self.get_mlab().pipeline.image_plane_widget(self._scalar_field, plane_orientation='x_axes', slice_index=modelmap.shape[0]/2),
            self.get_mlab().pipeline.image_plane_widget(self._scalar_field, plane_orientation='y_axes', slice_index=modelmap.shape[1]/2)]
        self._set_slice_visibility(False)

    def set_view_type(self, view_type):
        if view_type == VIEW_TYPE.surface:
            self._set_surface_visibility(True)
            self._set_slice_visibility(False)
        elif view_type == VIEW_TYPE.slice:
            self._set_slice_visibility(True)
            self._set_surface_visibility(False)
        else:
            raise TypeError("set_view_type takes only VIEW_TYPE object")

    def set_surface_level(self, value):
        if value < 0 or value > 1.:
            raise ValueError("Surface value must be in [0.,1.], was %g\n", value)
        self._surface_level = value
        self._surface_plot.contour.contours[0] = self._surface_level*self._scalar_field_max

    def get_surface_level(self):
        return self._surface_level

    def _set_surface_visibility(self, state):
        self._surface_plot.visible = state

    def _set_slice_visibility(self, state):
        for s in self._slice_plot:
            s.visible = state

class ModelmapControll(module_template.Controll):
    class State(object):
        """This class contains the current state of the plot to separate them from the rest of the class"""
        def __init__(self):
            self.view_type = VIEW_TYPE.surface
            self.log_scale = False
            
    def __init__(self, common_controll, viewer, data):
        super(ModelmapControll, self).__init__(common_controll, viewer, data)
        self._SLIDER_LENGTH = 1000
        self._state = self.State()
        self._viewer.plot_map_init(self._data.get_map(self._common_controll.get_iteration()))
        self._setup_gui()

    def _setup_gui(self):
        view_type_radio_surface = QtGui.QRadioButton("Isosurface plot")
        view_type_radio_slice = QtGui.QRadioButton("Slice plot")
        view_type_radio_surface.setChecked(True)
        view_type_layout = QtGui.QVBoxLayout()
        view_type_layout.addWidget(view_type_radio_surface)
        view_type_layout.addWidget(view_type_radio_slice)
        view_type_radio_surface.clicked.connect(partial(self.set_view_type, VIEW_TYPE.surface))
        view_type_radio_slice.clicked.connect(partial(self.set_view_type, VIEW_TYPE.slice))

        #surface controll setup
        self._surface_level_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self._surface_level_slider.setTracking(True)
        self._surface_level_slider.setRange(1, self._SLIDER_LENGTH)
        def on_slider_changed(cls, slider_level):
            #cls._viewer.set_surface_level(slider_level/float(cls._SLIDER_LENGTH))
            cls._viewer.set_surface_level(1./(10.*float(slider_level)))
        self._surface_level_slider.valueChanged.connect(partial(on_slider_changed, self))
        self._surface_level_slider.setSliderPosition(self._SLIDER_LENGTH*INIT_SURFACE_LEVEL)

        # slice controll widget setup
        log_scale_box = QtGui.QCheckBox()
        log_scale_label = QtGui.QLabel('Log Scale')
        log_scale_layout = QtGui.QHBoxLayout()
        log_scale_layout.addWidget(log_scale_label)
        log_scale_layout.addWidget(log_scale_box)
        log_scale_layout.addStretch()
        log_scale_box.stateChanged.connect(self.set_log_scale)
        self._log_scale_widget = QtGui.QWidget()
        self._log_scale_widget.setLayout(log_scale_layout)
        self._log_scale_widget.hide()

        layout = QtGui.QVBoxLayout()
        layout.addLayout(view_type_layout)
        layout.addWidget(self._surface_level_slider)
        layout.addWidget(self._log_scale_widget)
        #layout.addStretch()
        self._widget.setLayout(layout)

    def draw_hard(self):
        if (self._state.view_type == VIEW_TYPE.slice) and self._state.log_scale:
            self._viewer.plot_map(log(1.+self._data.get_map(self._common_controll.get_iteration())))
        else:
            self._viewer.plot_map(self._data.get_map(self._common_controll.get_iteration()))
            
    def set_view_type(self, view_type):
        self._state.view_type = view_type
        self._viewer.set_view_type(view_type)
        if view_type == VIEW_TYPE.surface:
            self._surface_level_slider.show()
            self._log_scale_widget.hide()
        elif view_type == VIEW_TYPE.slice:
            self._surface_level_slider.hide()
            self._log_scale_widget.show()
        self.draw()

    def set_log_scale(self, state):
        self._state.log_scale = bool(state)
        print self._state.log_scale
        self.draw()
        
class Plugin(module_template.Plugin):
    def __init__(self, common_controll, parent=None):
        super(Plugin, self).__init__(common_controll)
        self._viewer = ModelmapViewer()
        self._data = ModelmapData("output/model")
        self._controll = ModelmapControll(self._common_controll, self._viewer, self._data)

