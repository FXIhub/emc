"""Plugin for the emc viewer.
Plot the compressed model."""
#from pylab import *
#import pylab
import numpy
from functools import partial
import sphelper
import module_template
from pyface.qt import QtCore, QtGui
import embedded_mayavi

INIT_SURFACE_LEVEL = 0.5

def enum(*enums):
    """Gives enumerate functionality."""
    return type('Enum', (), dict(zip(enums, range(len(enums)))))

VIEW_TYPE = enum('surface', 'slice')

class ModelmapData(module_template.Data):
    """Reads data as requested."""
    def __init__(self, file_prefix):
        super(ModelmapData, self).__init__()
        self._file_prefix = file_prefix
        self._side = None

    def get_map(self, iteration):
        """Return the 3D model as an array for the requested iteration."""
        try:
            if iteration >= 0:
                modelmap = sphelper.import_spimage('%s_%.4d.h5' % (self._file_prefix, iteration), ['image'])
            elif iteration == -1:
                modelmap = sphelper.import_spimage('%s_init.h5' % (self._file_prefix), ['image'])
            elif iteration == -2:
                modelmap = sphelper.import_spimage('%s_final.h5' % (self._file_prefix), ['image'])
        except (IOError, KeyError):
            self.read_error.emit()
            if self._side:
                return numpy.zeros((self._side, )*3)
            else:
                return numpy.zeros((10, )*3)
        if self._side != modelmap.shape[0]:
            self._side = modelmap.shape[0]
            self.properties_changed.emit()
        return modelmap

    def get_mask(self, iteration):
        """Return the mask of the 3D model as an array for the requested iteration."""
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
            self._side = mask.shape[0]
            properties_changed.emit()
        return mask

    def get_side(self):
        """Return the number fo pixels in the side of the 3D model side.
        (The model is always cubic)"""
        return self._side

class ModelmapViewer(module_template.Viewer):
    """Uses mayavi to display the model in 3D as an isosurface or by two slices.
    This is not a widget but contains one, accessible with get_widget()"""
    def __init__(self, parent=None):
        super(ModelmapViewer, self).__init__(parent)
        self._surface_level = INIT_SURFACE_LEVEL
        self._scalar_field = None
        self._scalar_field_max = None
        self._slice_plot = None
        self._surface_plot = None
        self._mlab_widget = embedded_mayavi.MlabWidget()

    def get_mlab(self):
        """Return an mlab object that can be used to access mayavi functions."""
        return self._mlab_widget.get_mlab()

    def get_widget(self):
        """Return the widget containing the view."""
        return self._mlab_widget

    def plot_map(self, modelmap):
        """Update the viwer to show the provided map."""
        self._scalar_field.scalar_data = modelmap
        self._scalar_field_max = self._scalar_field.scalar_data.max()
        self._surface_plot.contour.contours[0] = self._surface_level*self._scalar_field_max

    def plot_map_init(self, modelmap):
        """As opposed to plot_map() this function accepts maps of different side than
        the active one."""
        if self._scalar_field == None:
            self._scalar_field = self.get_mlab().pipeline.scalar_field(modelmap)
            self._scalar_field_max = self._scalar_field.scalar_data.max()
            surface_contours = [self._surface_level*self._scalar_field_max]
            self._surface_plot = self.get_mlab().pipeline.iso_surface(self._scalar_field,
                                                                      contours=surface_contours)
            self._slice_plot = [
                self.get_mlab().pipeline.image_plane_widget(self._scalar_field, plane_orientation='x_axes',
                                                            slice_index=modelmap.shape[0]/2),
                self.get_mlab().pipeline.image_plane_widget(self._scalar_field, plane_orientation='y_axes',
                                                            slice_index=modelmap.shape[1]/2)]
            self._set_slice_visibility(False)
        else:
            self._scalar_field.mlab_source.reset(s=modelmap)
            self._scalar_field.update_pipeline()

    def set_view_type(self, view_type):
        """Select view type, currently supports surface and slice."""
        if view_type == VIEW_TYPE.surface:
            self._set_surface_visibility(True)
            self._set_slice_visibility(False)
        elif view_type == VIEW_TYPE.slice:
            self._set_slice_visibility(True)
            self._set_surface_visibility(False)
        else:
            raise TypeError("set_view_type takes only VIEW_TYPE object")

    def set_surface_level(self, value):
        """Change the level of the surface plot."""
        if value < 0 or value > 1.:
            raise ValueError("Surface value must be in [0.,1.], was %g\n", value)
        self._surface_level = value
        self._surface_plot.contour.contours[0] = self._surface_level*self._scalar_field_max

    def get_surface_level(self):
        """Get the current level of the surface plot."""
        return self._surface_level

    def _set_surface_visibility(self, state):
        """Set wether the isosurface plot is visible or not"""
        self._surface_plot.visible = state

    def _set_slice_visibility(self, state):
        """Set wether the slice plot is visible or not"""
        for this_slice in self._slice_plot:
            this_slice.visible = state

    def save_image(self, filename):
        """Output the current view to file."""
        self._mlab_widget.save_image(filename)


class ModelmapControll(module_template.Controll):
    """Provides a widget for controlling the module and handles the calls to get the
    data and viewer classes."""
    class State(object):
        """This class contains the current state of the plot to separate them from the rest of the class"""
        def __init__(self):
            self.view_type = VIEW_TYPE.surface
            self.log_scale = False

    def __init__(self, common_controll, viewer, data):
        super(ModelmapControll, self).__init__(common_controll, viewer, data)
        self._slider_length = 1000
        self._surface_level_slider = None
        self._log_scale_widget = None

        self._state = self.State()
        self._viewer.plot_map_init(self._data.get_map(self._common_controll.get_iteration()))
        self._setup_gui()
        self._data.properties_changed.connect(self._setup_viewer)

    def _setup_viewer(self):
        """Provedes an empty map to the viewer to initialize it."""
        self._viewer.plot_map_init(numpy.zeros((self._data.get_side(), )*3))

    def _setup_gui(self):
        """Create the gui for the widget."""
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
        self._surface_level_slider.setRange(1, self._slider_length)
        def on_slider_changed(self, slider_level):
            """Handles the signal from a surface level slider change."""
            # surface_level = slider_level/float(cls._slider_length)
            # surface_level = self._slider_length/(float(slider_level))
            surface_level = ((numpy.exp(float(slider_level)/float(self._slider_length))-1.) /
                             (numpy.exp(1.)-1.))
            self._viewer.set_surface_level(surface_level)
        self._surface_level_slider.valueChanged.connect(partial(on_slider_changed, self))
        self._surface_level_slider.setSliderPosition(self._slider_length*INIT_SURFACE_LEVEL)

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
        """Draw the scene. Don't call this function directly. The draw() function calls this one
        if the module is visible."""
        if (self._state.view_type == VIEW_TYPE.slice) and self._state.log_scale:
            self._viewer.plot_map(numpy.log(1.+self._data.get_map(self._common_controll.get_iteration())))
        else:
            self._viewer.plot_map(self._data.get_map(self._common_controll.get_iteration()))

    def set_view_type(self, view_type):
        """Select between isosurface and slice plot."""
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
        """Set wether log scale is used for the slice plot."""
        self._state.log_scale = bool(state)
        print self._state.log_scale
        self.draw()

class Plugin(module_template.Plugin):
    """Collects all parts of the plugin."""
    def __init__(self, common_controll, parent=None):
        super(Plugin, self).__init__(common_controll, parent)
        self._viewer = ModelmapViewer()
        self._data = ModelmapData("output/model")
        self._controll = ModelmapControll(self._common_controll, self._viewer, self._data)

