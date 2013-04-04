from pylab import *
import sphelper
import sys
import numpy
import time
import h5py
import os
import rotations
import icosahedral_sphere
from optparse import OptionParser

from pyface.qt import QtCore, QtGui
try:
    from mayavi import mlab
except ImportError:
    print "fallback on enthought"
    from enthought.mayavi import mlab

from traits.api import HasTraits, Instance, on_trait_change, Int, Dict
from traitsui.api import View, Item
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor

#import kernprof

SLIDER_LENGTH = 100

class Model(QtCore.QObject):
    image_changed = QtCore.Signal(int)
    read_error = QtCore.Signal()
    def __init__(self, iteration_number):
        super(Model, self).__init__()
        self._current_iteration = iteration_number
        self._image_type = 0 # 0 = image, 1 = weight, 2 = rotations
        self._rotation_type = 0 # 0 = average, 1 = single
        self._rotation_image_number = 0
        self._setup_rotations()
        self._read_image()


    def _setup_rotations(self):
        def closest_coordinate(coordinate, points_list):
            return (((points_list - coordinate)**2).sum(axis=1)).argmax()
        
        self._rotations = loadtxt('output/rotations.data')
        self._euler_angles = array([rotations.quaternion_to_euler_angle(rotation) for rotation in self._rotations])
        self._coordinates = transpose(array([sin(self._euler_angles[:, 2])*cos(self._euler_angles[:, 1]),
                                             cos(self._euler_angles[:, 2])*cos(self._euler_angles[:, 1]),
                                             sin(self._euler_angles[:, 1])]))

        number_of_rotations = len(self._rotations)
        self._rotation_sphere_coordinates = array(icosahedral_sphere.sphere_sampling(rotations.rots_to_n(number_of_rotations)))
        number_of_bins = len(self._rotation_sphere_coordinates)
        self._rotation_sphere_weights = zeros(number_of_bins)
        self._rotation_sphere_bins = zeros(number_of_bins)
        self._rotation_mapping_table = zeros(number_of_rotations, dtype='int32')
        for i, c in enumerate(self._coordinates):
            index = closest_coordinate(c, self._rotation_sphere_coordinates)
            self._rotation_sphere_weights[index] += 1.
            self._rotation_mapping_table[i] = index
        self._rotation_sphere_good_indices = self._rotation_sphere_weights > 0.


    def _image_changed(self):
        if self._image_type == 0:
            self._read_image()
        elif self._image_type == 1:
            self._read_weight()
        self.image_changed.emit(self._current_iteration)

    def _read_image(self):
        try:
            if self._current_iteration >= 0:
                self._image, self._mask = sphelper.import_spimage('output/model_%.4d.h5' % self._current_iteration, ['image', 'mask'])
            elif self._current_iteration == -1:
                self._image, self._mask = sphelper.import_spimage('output/model_init.h5', ['image', 'mask'])
            elif self._current_iteration == -2:
                self._image, self._mask = sphelper.import_spimage('output/model_final.h5', ['image', 'mask'])
        except IOError:
            self.read_error.emit()

    def _read_weight(self):
        try:
            if self._current_iteration >= 0:
                self._image, self._mask = sphelper.import_spimage('output/weight_%.4d.h5' % self._current_iteration, ['image', 'mask'])
            elif self._current_iteration == -1:
                self._image, self._mask = sphelper.import_spimage('output/model_init.h5', ['image', 'mask'])
        except IOError:
            self.read_error.emit()

    def next_image(self):
        self._current_iteration += 1
        self._image_changed()

    def previous_image(self):
        if self._current_iteration >= 0:
            self._current_iteration -= 1
            self._image_changed()

    def set_iteration(self, iteration):
        if iteration >= -2:
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

    def get_rotation_coordinates(self):
        # best_index = list(int32(loadtxt('output/best_rot.data')[self._current_iteration]))
        # return self._rotation_sphere_coordinates
        return self._rotation_sphere_coordinates

    def get_rotation_values(self):
        try:
            if self._rotation_type == 0:
                average_resp = loadtxt('output/average_resp_%.4d.data' % self._current_iteration)
                for i, r in enumerate(average_resp):
                    self._rotation_sphere_bins[self._rotation_mapping_table[i]] += r
            elif self._rotation_type == 1:
                resp_handle = h5py.File('output/responsabilities_%.4d.h5' % self._current_iteration)
                resp = resp_handle['data'][self._rotation_image_number,:]
                resp_handle.close()
                for i, r in enumerate(resp):
                    self._rotation_sphere_bins[self._rotation_mapping_table[i]] += r
            self._rotation_sphere_bins[self._rotation_sphere_good_indices] /= self._rotation_sphere_weights[self._rotation_sphere_good_indices]
        except IOError:
            self.read_error.emit()
        return self._rotation_sphere_bins

    def get_image_side(self):
        return shape(self._image)[0]

    def set_rotation_type_average(self):
        self._rotation_type = 0
        self.image_changed.emit(self._current_iteration)

    def set_rotation_type_single(self):
        self._rotation_type = 1
        self.image_changed.emit(self._current_iteration)

    def set_rotation_image_number(self, image_number):
        self._rotation_image_number = image_number
        self.image_changed.emit(self._current_iteration)
            
    def reload_image(self):
        self._image_changed()

class MlabVisualization(HasTraits):
    scene = Instance(MlabSceneModel, ())

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene), height=600, width=600, show_label=False),
                resizable=True)


class MlabWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        i = 0
        QtGui.QWidget.__init__(self, parent)
        layout = QtGui.QVBoxLayout(self)
        # layout.setMargin(0)
        # print "%d" % i; i+=1
        # layout.setSpacing(0)
        # print "%d" % i; i+=1

        self._vis = MlabVisualization()
        self._ui = self._vis.edit_traits(parent=self, kind='subpanel').control
        layout.addWidget(self._ui)
        self._ui.setParent(self)

        # self._scene = MlabSceneModel()
        # view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene), height=600, width=600, show_label=False),
        #             resizable=True)
        # self.ui = self._scene.control
        # layout.addWidget(self.ui)
        # self.ui.setParent(self)
        

    def get_mlab(self):
        return self._vis.scene.mlab
    
class Viewer(QtCore.QObject):
    def __init__(self, model, mlab_widget, parent=None):
        super(Viewer, self).__init__()
        self._model = model
        self._mlab_widget = mlab_widget
        #QtCore.QObject.connect(self._model, QtCore.SIGNAL('image_changed(int)'), self._update_scalar_field)
        self._model.image_changed.connect(self._data_changed)
        self._initialize_scalar_field()
        self._surface = None
        self._slices = None
        self._points = None
        self._plot_surface()
        self._log_scale = False
        self._mode = 0
        self._surface_level = 0.5
        
    def _initialize_scalar_field(self):
        self._scalar_field = self._mlab_widget.get_mlab().pipeline.scalar_field(self._model.get_image())
        self._scalar_field_max = self._scalar_field.scalar_data.max()

    def _data_changed(self, dummy=0):
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
        elif self._mode == 2:
            #self._points.glyph.glyph.input.points = self._model.get_rotation_coordinates()
            values = self._model.get_rotation_values()
            self._points.glyph.glyph.input.point_data.scalars = values
            self._points.glyph.glyph.input.point_data.modified()
            self._points.actor.mapper.lookup_table.range = (0., values.max())
            self._points.glyph.glyph.modified()
            self._points.update_pipeline()
            self._points.actor.render()
            self._mlab_widget.get_mlab().draw()
            

    def _plot_surface(self):
        self._surface = self._mlab_widget.get_mlab().pipeline.iso_surface(self._scalar_field, contours=[0.5*self._scalar_field_max])
        #mlab.show()

    def _plot_slices(self):
        self._slices = []
        self._slices.append(self._mlab_widget.get_mlab().pipeline.image_plane_widget(self._scalar_field, plane_orientation='x_axes',
                                                             slice_index=shape(self._model.get_image())[0]/2))
        self._slices.append(mlab.pipeline.image_plane_widget(self._scalar_field, plane_orientation='y_axes',
                                                             slice_index=shape(self._model.get_image())[1]/2))
        #mlab.show()

    def _plot_rotations(self):
        coordinates = (self._model.get_rotation_coordinates()*self._model.get_image_side()/4.)+self._model.get_image_side()/2.
        values = self._model.get_rotation_values()
        values /= values.max()
        self._points = mlab.points3d(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], values, scale_mode='none')
        self._points.glyph.glyph.modified()
        self._points.actor.mapper.lookup_table.range = (0., values.max())
        self._points.module_manager.scalar_lut_manager.data_name = "Resp"
        self._points.update_pipeline()
        self._points.actor.render()

    def set_surface_level(self, new_level):
        self._surface_level = new_level
        self._surface.contour.contours[0] = self._surface_level*self._scalar_field_max

    def get_surface_level(self):
        return self._surface.contour.contours[0]/self._scalar_field_max

    def set_log_scale(self, state):
        self._log_scale = state
        self._data_changed()
        
    def set_surface_mode(self):
        if self._slices:
            self._slices[0].visible = False
            self._slices[1].visible = False
        if self._points:
            self._points.visible = False
            self._points.module_manager.scalar_lut_manager.show_scalar_bar = False
        if not self._surface:
            self._plot_surface()
        else:
            self._surface.visible = True
        self._mode = 0
        self._data_changed()

    def set_slice_mode(self):
        if self._surface:
            self._surface.visible = False
        if self._points:
            self._points.visible = False
            self._points.module_manager.scalar_lut_manager.show_scalar_bar = False
        if not self._slices:
            self._plot_slices()
        else:
            self._slices[0].visible = True
            self._slices[1].visible = True
        self._mode = 1
        self._data_changed()

    def set_rotations_mode(self):
        if self._surface:
            self._surface.visible = False
        if self._slices:
            self._slices[0].visible = False
            self._slices[1].visible = False
        if not self._points:
            self._plot_rotations()
        else:
            self._points.visible = True
        self._points.module_manager.scalar_lut_manager.show_scalar_bar = True
        self._mode = 2
        self._data_changed()

class StartMain(QtGui.QMainWindow):
    def __init__(self, model, viewer, mlab_widget, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self._model = model
        self._viewer = viewer
        self._mlab_widget = mlab_widget
        self._setup_controller_window()

        # //Set actors and mappers, then instead of creating a renderwindowinteractor,
        # // use the self.ui.meshDisplayWidget to display the mesh. Also define picker, and
        # // set data source (for code about displaying a mesh from coordinates, as
        #                     // an unstructured grid.

    def _setup_controller_window(self):
        self._model.read_error.connect(self._on_read_error)
        
        self._controller_widget = QtGui.QWidget()

        self._mode_button = QtGui.QPushButton("Mode")
        self._mode_menu = QtGui.QMenu()
        self._surface_action = QtGui.QAction("Surface", self)
        self._surface_action.triggered.connect(self._set_surface_mode)
        self._slice_action = QtGui.QAction("Slice", self)
        self._slice_action.triggered.connect(self._set_slice_mode)
        self._rotations_action = QtGui.QAction("Best Rotations", self)
        self._rotations_action.triggered.connect(self._set_rotations_mode)
        self._mode_menu.addAction(self._surface_action)
        self._mode_menu.addAction(self._slice_action)
        self._mode_menu.addAction(self._rotations_action)
        self._mode_button.setMenu(self._mode_menu)

        #file_system_model = QtGui.QFileSystemModel()
        work_dir_label = QtGui.QLabel("Dir:")
        file_system_completer = QtGui.QCompleter()
        self._file_system_model = QtGui.QFileSystemModel(file_system_completer)
        #file_system_model.setRootPath('/')
        self._file_system_model.setFilter(QtCore.QDir.Dirs | QtCore.QDir.Hidden)
        file_system_completer.setModel(self._file_system_model)
        self._work_dir_edit = QtGui.QLineEdit(os.getcwd())
        self._work_dir_edit.setCompleter(file_system_completer)
        self._work_dir_edit.editingFinished.connect(self._on_work_dir_changed)
        self._work_dir_layout = QtGui.QHBoxLayout()
        self._work_dir_layout.addWidget(work_dir_label)
        self._work_dir_layout.addWidget(self._work_dir_edit)
        
        self._previous_button = QtGui.QPushButton("Previous")
        self._previous_button.pressed.connect(self._model.previous_image)
        
        self._next_button = QtGui.QPushButton("Next")
        self._next_button.pressed.connect(self._model.next_image)

        self._iteration_box = QtGui.QSpinBox()
        self._iteration_box.setRange(-2, 10000)
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
        self._slice_log_scale_label = QtGui.QLabel('Log scale')
        self._slice_controll_layout = QtGui.QHBoxLayout()
        self._slice_controll_layout.addWidget(self._slice_log_scale_box)
        self._slice_controll_layout.addWidget(self._slice_log_scale_label)
        self._slice_controll_layout.addStretch()
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

        #rotations controll widget setup
        # show average or single
        self._show_average_rotation_radio = QtGui.QRadioButton("Average rot")
        self._show_average_rotation_radio.setChecked(True)
        self._show_single_rotation_radio = QtGui.QRadioButton("Single rot")

        self._rotations_controll_layout = QtGui.QVBoxLayout()
        self._rotations_controll_layout.addWidget(self._show_average_rotation_radio)
        self._rotations_controll_layout.addWidget(self._show_single_rotation_radio)
        self._show_average_rotation_radio.clicked.connect(self._rotation_type_average)
        self._show_single_rotation_radio.clicked.connect(self._rotation_type_single)

        self._rotation_image_number_box = QtGui.QSpinBox()
        self._rotation_image_number_box.setRange(-1, 10000)
        self._rotation_image_number_box.editingFinished.connect(self._image_number_changed)
        self._rotations_controll_layout.addWidget(self._rotation_image_number_box)
        self._rotation_image_number_box.hide()

        self._rotations_controll_widget = QtGui.QWidget()
        self._rotations_controll_widget.setLayout(self._rotations_controll_layout)

        #setup main layout and widget
        self._main_layout = QtGui.QVBoxLayout()
        self._main_layout.addWidget(self._mlab_widget)
        self._main_layout.addLayout(self._work_dir_layout)
        self._main_layout.addWidget(self._mode_button)
        self._main_layout.addLayout(self._iteration_layout)
        self._main_layout.addLayout(self._image_type_layout)
        self._main_layout.addWidget(self._surface_controll_widget)
        self._main_layout.addWidget(self._slice_controll_widget)
        self._main_layout.addWidget(self._rotations_controll_widget)
        self._slice_controll_widget.hide()
        self._rotations_controll_widget.hide()
    
        self._controller_widget.setLayout(self._main_layout)
        self.setCentralWidget(self._controller_widget)


    def _set_surface_mode(self):
        self._slice_controll_widget.hide()
        self._rotations_controll_widget.hide()
        self._surface_controll_widget.show()
        self._viewer.set_surface_mode()

    def _set_slice_mode(self):
        self._surface_controll_widget.hide()
        self._rotations_controll_widget.hide()
        self._slice_controll_widget.show()
        self._viewer.set_slice_mode()

    def _set_rotations_mode(self):
        self._surface_controll_widget.hide()
        self._slice_controll_widget.hide()
        self._rotations_controll_widget.show()
        self._viewer.set_rotations_mode()

    def _on_work_dir_changed(self):
        new_dir = self._work_dir_edit.text()
        self._file_system_model.setRootPath(new_dir)
        if not os.path.isdir(new_dir):
            #self._work_dir_edit.setTextBackgroundColor(QtGui.QColor(255, 0, 0, 127))
            #self._work_dir_edit.setStyleSheet("QLineEdit(background: red;")
            palette = self._work_dir_edit.palette()
            #palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Base, QtGui.QColor(255, 50, 50))
            self._work_dir_edit.setPalette(palette)
            return
        os.chdir(self._work_dir_edit.text())
        palette = self._work_dir_edit.palette()
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor(255, 255, 255))
        self._work_dir_edit.setPalette(palette)
        #self._work_dir_edit.setTextBackgroundColor(QtGui.QColor(255, 255, 255, 127))
        #self._work_dir_edit.setStyleSheet("QLineEdit(background: white;")
        self._model.reload_image()

    def _on_read_error(self):
        palette = self._work_dir_edit.palette()
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor(255, 250, 50))
        self._work_dir_edit.setPalette(palette)

    def _on_model_image_changed(self, iteration):
        self._iteration_box.setValue(iteration)

    def _spinbox_value_changed(self):
        self._model.set_iteration(self._iteration_box.value())

    def _slider_changed(self, value):
        self._viewer.set_surface_level((value/float(SLIDER_LENGTH))**2)

    def _image_number_changed(self):
        #connect to model, probably without this function
        self._model.set_rotation_image_number(self._rotation_image_number_box.value())

    def _rotation_type_average(self):
        # change visibility
        self._rotation_image_number_box.hide()
        # connect to model
        self._model.set_rotation_type_average()

    def _rotation_type_single(self):
        # change visibility
        self._rotation_image_number_box.show()
        # connect to model
        self._model.set_rotation_type_single()

if __name__ == "__main__":
    parser = OptionParser(usage="%prog <3d_file.h5>")
    parser.add_option("-s", action="store_true", dest="surface", help="Plot image as isosurface.")
    parser.add_option("-l", action="store_true", dest="log", help="Plot in log scale.")
    parser.add_option("-m", action="store_true", dest="mask", help="Plot mask.")
    (options, args) = parser.parse_args()

    #app = QtGui.QApplication(['Controll window'])
    app = QtGui.QApplication.instance()

    model = Model(0)
    mlab_widget = MlabWidget()
    viewer = Viewer(model, mlab_widget)
    #viewer.plot_slices()
    
    program = StartMain(model, viewer, mlab_widget)
    program.show()
    sys.exit(app.exec_())
