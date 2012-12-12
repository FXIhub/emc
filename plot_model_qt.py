#! /usr/local/bin/python-32
from pylab import *
#import spimage
import sphelper
import sys
import numpy
import time
from enthought.mayavi import mlab
from optparse import OptionParser
from PySide import QtCore, QtGui

SLIDER_LENGTH = 100

class Model(QtCore.QObject):
    image_changed = QtCore.Signal(int)
    def __init__(self, iteration_number):
        super(Model, self).__init__()
        self._current_iteration = iteration_number
        self._image_type = 0 # 0 = image, 1 = weight
        self._read_image()

    def _image_changed(self):
        if self._image_type == 0:
            self._read_image()
        elif self._image_type == 1:
            self._read_weight()
        self.image_changed.emit(self._current_iteration)

    def _read_image(self):
        self._image, self._mask = sphelper.import_spimage('output/model_%.4d.h5' % self._current_iteration, ['image', 'mask'])

    def _read_weight(self):
        self._image, self._mask = sphelper.import_spimage('output/weight_%.4d.h5' % self._current_iteration, ['image', 'mask'])

    def next_image(self):
        self._current_iteration += 1
        self._image_changed()

    def previous_image(self):
        print "previous image"
        if self._current_iteration >= 0:
            self._current_iteration -= 1
            self._image_changed()

    def set_iteration(self, iteration):
        if iteration >= 0:
            self._current_iteration = iteration
            self._image_changed()

    def get_image(self):
        return self._image

    def get_mask(self):
        return self._mask

    def set_display_image(self):
        self._image_type = 0
        self._image_changed()

    def set_display_weight(self):
        self._image_type = 1
        self._image_changed()
        
        
    
class Viewer(QtCore.QObject):
    def __init__(self, model):
        super(Viewer, self).__init__()
        self._model = model
        #QtCore.QObject.connect(self._model, QtCore.SIGNAL('image_changed(int)'), self._update_scalar_field)
        self._model.image_changed.connect(self._update_scalar_field)
        self._initialize_scalar_field()
        self._surface = None
        self._slices = None
        self._plot_surface()
        self._log_scale = False
        self._mode = 0
        self._surface_level = 0.5
        
    def _initialize_scalar_field(self):
        self._scalar_field = mlab.pipeline.scalar_field(self._model.get_image())
        self._scalar_field_max = self._scalar_field.scalar_data.max()

    def _update_scalar_field(self, dummy=0):
        if self._mode == 1:
            if self._log_scale:
                self._scalar_field.scalar_data = log10(self._model.get_image() + self._model.get_image().max()*0.001)
            else:
                self._scalar_field.scalar_data = self._model.get_image()
        elif self._mode == 0:
            self._scalar_field.scalar_data = self._model.get_image()
            self._surface.contour.contours[0] = self._surface_level*self._scalar_field_max
            self._scalar_field_max = self._scalar_field.scalar_data.max()
            # old_surface_level = self.get_surface_level()
            # self._scalar_field_max = self._scalar_field.scalar_data.max()
            # self.set_surface_level(old_surface_level)


    def _plot_surface(self):
        self._surface = mlab.pipeline.iso_surface(self._scalar_field, contours=[0.5*self._scalar_field_max])
        #mlab.show()

    def _plot_slices(self):
        print "plot_slices"
        self._slices = []
        self._slices.append(mlab.pipeline.image_plane_widget(self._scalar_field, plane_orientation='x_axes',
                                                             slice_index=shape(self._model.get_image())[0]/2))
        self._slices.append(mlab.pipeline.image_plane_widget(self._scalar_field, plane_orientation='y_axes',
                                                             slice_index=shape(self._model.get_image())[1]/2))
        #mlab.show()

    def set_surface_level(self, new_level):
        self._surface_level = new_level
        self._surface.contour.contours[0] = self._surface_level*self._scalar_field_max

    def get_surface_level(self):
        return self._surface.contour.contours[0]/self._scalar_field_max

    def set_log_scale(self, state):
        print "log scale to %d" % state
        self._log_scale = state
        self._update_scalar_field()
        
    def set_surface_mode(self):
        if self._slices:
            self._slices[0].visible = False
            self._slices[1].visible = False
        if not self._surface:
            self._plot_surface()
        else:
            self._surface.visible = True
        self._mode = 0
        self._update_scalar_field()

    def set_slice_mode(self):
        if self._surface:
            self._surface.visible = False
        if not self._slices:
            self._plot_slices()
        else:
            self._slices[0].visible = True
            self._slices[1].visible = True
        self._mode = 1
        self._update_scalar_field()


class StartMain(QtGui.QMainWindow):
    def __init__(self, model, viewer, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self._model = model
        self._viewer = viewer
        self._setup_controller_window()

        # //Set actors and mappers, then instead of creating a renderwindowinteractor,
        # // use the self.ui.meshDisplayWidget to display the mesh. Also define picker, and
        # // set data source (for code about displaying a mesh from coordinates, as
        #                     // an unstructured grid.

    def _setup_controller_window(self):
        self._controller_widget = QtGui.QWidget()

        self._mode_button = QtGui.QPushButton("Mode")
        self._mode_menu = QtGui.QMenu()
        self._surface_action = QtGui.QAction("Surface", self)
        self._surface_action.triggered.connect(self._set_surface_mode)
        self._slice_action = QtGui.QAction("Slice", self)
        self._slice_action.triggered.connect(self._set_slice_mode)
        self._mode_menu.addAction(self._surface_action)
        self._mode_menu.addAction(self._slice_action)
        self._mode_button.setMenu(self._mode_menu)
        
        self._previous_button = QtGui.QPushButton("Previous")
        self._previous_button.pressed.connect(self._model.previous_image)
        
        self._next_button = QtGui.QPushButton("Next")
        self._next_button.pressed.connect(self._model.next_image)

        self._iteration_box = QtGui.QSpinBox()
        self._iteration_box.setRange(-1, 10000)
        #self._iteration_box.valueChanged.connect(self._model.set_iteration)
        self._iteration_box.editingFinished.connect(self._spinbox_value_changed)
        self._model.image_changed.connect(self._on_model_image_changed)

        self._next_previous_layout = QtGui.QHBoxLayout()
        self._next_previous_layout.addWidget(self._previous_button)
        self._next_previous_layout.addWidget(self._next_button)

        self._iteration_layout = QtGui.QVBoxLayout()
        self._iteration_layout.addLayout(self._next_previous_layout)
        self._iteration_layout.addWidget(self._iteration_box)

        #surface controll widget setup
        self._surface_controll_widget = QtGui.QWidget()

        self._surface_level_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self._surface_level_slider.setTracking(True)
        self._surface_level_slider.setRange(1, SLIDER_LENGTH)
        self._surface_level_slider.valueChanged.connect(self._slider_changed)
        self._surface_level_slider.setSliderPosition(SLIDER_LENGTH/2)


        self._surface_controll_layout = QtGui.QVBoxLayout()
        self._surface_controll_layout.addWidget(self._surface_level_slider)
        self._surface_controll_widget.setLayout(self._surface_controll_layout)

        #slice controll widget setup
        self._slice_controll_widget = QtGui.QWidget()
        self._slice_log_scale_box = QtGui.QCheckBox()
        self._slice_controll_layout = QtGui.QVBoxLayout()
        self._slice_controll_layout.addWidget(self._slice_log_scale_box)
        self._slice_log_scale_box.stateChanged.connect(viewer.set_log_scale)
        self._slice_controll_widget.setLayout(self._slice_controll_layout)

        #image type selector setup
        self._model_radio = QtGui.QRadioButton("Model")
        self._model_radio.setChecked(True)
        self._weight_radio = QtGui.QRadioButton("Weight")
        self._image_type_layout = QtGui.QVBoxLayout()
        self._image_type_layout.addWidget(self._model_radio)
        self._image_type_layout.addWidget(self._weight_radio)
        self._model_radio.clicked.connect(self._model.set_display_image)
        self._weight_radio.clicked.connect(self._model.set_display_weight)

        self._main_layout = QtGui.QVBoxLayout()
        self._main_layout.addWidget(self._mode_button)
        self._main_layout.addLayout(self._iteration_layout)
        self._main_layout.addLayout(self._image_type_layout)
        self._main_layout.addWidget(self._surface_controll_widget)
        self._main_layout.addWidget(self._slice_controll_widget)
        self._slice_controll_widget.hide()
    
        self._controller_widget.setLayout(self._main_layout)
        self.setCentralWidget(self._controller_widget)

    def _set_surface_mode(self):
        print "surface mode"
        self._slice_controll_widget.hide()
        self._surface_controll_widget.show()
        self._viewer.set_surface_mode()

    def _set_slice_mode(self):
        print "slice mode"
        self._surface_controll_widget.hide()
        self._slice_controll_widget.show()
        self._viewer.set_slice_mode()

    def _on_model_image_changed(self, iteration):
        self._iteration_box.setValue(iteration)

    def _spinbox_value_changed(self):
        self._model.set_iteration(self._iteration_box.value())

    def _slider_changed(self, value):
        self._viewer.set_surface_level((value/float(SLIDER_LENGTH))**2)

        
if __name__ == "__main__":
    parser = OptionParser(usage="%prog <3d_file.h5>")
    parser.add_option("-s", action="store_true", dest="surface", help="Plot image as isosurface.")
    parser.add_option("-l", action="store_true", dest="log", help="Plot in log scale.")
    parser.add_option("-m", action="store_true", dest="mask", help="Plot mask.")
    (options, args) = parser.parse_args()

    model = Model(0)
    viewer = Viewer(model)
    #viewer.plot_slices()
    
    app = QtGui.QApplication(['Controll window'])
    program = StartMain(model, viewer)
    program.show()
    sys.exit(app.exec_())
