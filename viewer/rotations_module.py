from pylab import *
import h5py
from functools import partial
import rotations
import icosahedral_sphere
import module_template
from pyface.qt import QtCore, QtGui
import embedded_mayavi
import convenient_widgets
import os

def enum(*enums):
    return type('Enum', (), dict(zip(enums, range(len(enums)))))

ROTATION_TYPE = enum('single', 'average')

class RotationData(module_template.Data):
    def __init__(self):
        super(RotationData, self).__init__()
        self._rotation_type = ROTATION_TYPE.single # 0 = average, 1 = single
        self._number_of_rotations = None
        #self._setup_rotations()
    
    # def setup_rotations(self):
    #     """Create an index of the plotting bins for the rotations."""
    #     def closest_coordinate(coordinate, points_list):
    #         return (((points_list - coordinate)**2).sum(axis=1)).argmax()
        
    #     self._rotations = loadtxt('output/rotations.data')
    #     # self._euler_angles = array([rotations.quaternion_to_euler_angle(rotation) for rotation in self._rotations])
    #     # self._coordinates = transpose(array([sin(self._euler_angles[:, 2])*cos(self._euler_angles[:, 1]),
    #     #                                      cos(self._euler_angles[:, 2])*cos(self._euler_angles[:, 1]),
    #     #                                      sin(self._euler_angles[:, 1])]))
    #     self._coordinates = array([rotations.rotate(rotation, [1., 0., 0.]) for rotation in self._rotations])
    #     self._number_of_rotations = len(self._rotations)
    #     self._rotation_sphere_coordinates = array(icosahedral_sphere.sphere_sampling(rotations.rots_to_n(self._number_of_rotations)))
    #     number_of_bins = len(self._rotation_sphere_coordinates)
    #     self._rotation_sphere_weights = zeros(number_of_bins)
    #     self._rotation_sphere_bins = zeros(number_of_bins)
    #     self._rotation_mapping_table = zeros(self._number_of_rotations, dtype='int32')
    #     for i, c in enumerate(self._coordinates):
    #         index = closest_coordinate(c, self._rotation_sphere_coordinates)
    #         self._rotation_sphere_weights[index] += 1.
    #         self._rotation_mapping_table[i] = index
    #     self._rotation_sphere_good_indices = self._rotation_sphere_weights > 0.

    def _read_average_resp(self, iteration):
        if os.path.isfile("output/average_resp_%.4d.h5" % (iteration)):
            with h5py.File("output/average_resp_%.4d.h5" % (iteration), "r") as f:
                average_resp = f["data"][...]
        else:
            average_resp = loadtxt("output/average_resp_%.4d.data" % (iteration))
        return average_resp

    def setup_rotations(self):
        self._number_of_rotations = len(self._read_average_resp(0))
        n = rotations.rots_to_n(self._number_of_rotations)

        self._rotation_sphere_coordinates = array(icosahedral_sphere.sphere_sampling(n))

        with h5py.File("%s/data/rotations_table_n%d.h5" % (os.path.split(__file__)[0], n)) as f:
            self._rotation_sphere_weights = f['weights'][...]
            self._rotation_mapping_table = f['table'][...]

        self._rotation_sphere_good_indices = self._rotation_sphere_weights > 0.

        self._number_of_bins = len(self._rotation_sphere_weights)
        self._rotation_sphere_bins = zeros(self._number_of_bins)


    def get_rotation_coordinates(self):
        # best_index = list(int32(loadtxt('output/best_rot.data')[self._current_iteration]))
        # return self._rotation_sphere_coordinates
        return self._rotation_sphere_coordinates

    def get_average_rotation_values(self, iteration):
        """Get the average rotation distribution of all images"""
        try:
            #average_resp = loadtxt('output/average_resp_%.4d.data' % iteration)
            average_resp = self._read_average_resp(iteration)
        except IOError:
            self.read_error.emit()
            return self._rotation_sphere_bins
        if len(average_resp) != self._number_of_rotations:
            self.properties_changed.emit()
        self._rotation_sphere_bins[:] = 0.
        for i, r in enumerate(average_resp):
            self._rotation_sphere_bins[self._rotation_mapping_table[i]] += r
        self._rotation_sphere_bins[self._rotation_sphere_good_indices] /= self._rotation_sphere_weights[self._rotation_sphere_good_indices]
        #return float32(self._rotation_sphere_weights) # debug line
        return self._rotation_sphere_bins

    def get_single_rotation_values(self, iteration, image_number):
        """Get the rotation distribution of a single image"""
        try:
            resp_handle = h5py.File('output/responsabilities_%.4d.h5' % iteration)
            resp = resp_handle['data'][image_number, :]
            resp_handle.close()
        except IOError:
            self.read_error.emit()
        if len(resp) != self._number_of_rotations:
            self.properties_changed.emit()
        self._rotation_sphere_bins[:] = 0.
        for i, r in enumerate(resp):
            self._rotation_sphere_bins[self._rotation_mapping_table[i]] += r
        self._rotation_sphere_bins[self._rotation_sphere_good_indices] /= self._rotation_sphere_weights[self._rotation_sphere_good_indices]
        return self._rotation_sphere_bins

class RotationViewer(module_template.Viewer):
    def __init__(self, parent=None):
        super(RotationViewer, self).__init__()
        self._mlab_widget = embedded_mayavi.MlabWidget()
        self._points = None

    def get_mlab(self):
        return self._mlab_widget.get_mlab()

    def get_widget(self):
        return self._mlab_widget

    def plot_rotations(self, values):
        self._points.glyph.glyph.input.point_data.scalars = values
        self._points.glyph.glyph.input.point_data.modified()
        self._points.actor.mapper.lookup_table.range = (0., values.max())
        self._points.glyph.glyph.modified()
        self._points.update_pipeline()
        self._points.actor.render()
        self.get_mlab().draw()

    def plot_rotations_init(self, coordinates):
        if self._points != None:
            #self._points.remove()
            # self._points.mlab_source.m_data.remove()
            # self._points.update_pipeline()
            self._points.mlab_source.reset(x=coordinates[:,0], y=coordinates[:,1], z=coordinates[:,2], scalars=ones(len(coordinates)))
            self._points.update_pipeline()
        else:
            self._points = self.get_mlab().points3d(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], ones(len(coordinates)), scale_mode='none')
            self._points.glyph.glyph.modified()
            self._points.actor.mapper.lookup_table.range = (0., 1.)
            self._points.module_manager.scalar_lut_manager.data_name = "Resp"
            self._points.module_manager.scalar_lut_manager.show_scalar_bar = True
            color_black = (0., 0., 0.)
            self._points.module_manager.scalar_lut_manager.scalar_bar.label_text_property.color = color_black #scalbar text is black
            self._points.module_manager.scalar_lut_manager.scalar_bar.title_text_property.color = color_black #scalbar text is black
            self._points.update_pipeline()
            self._points.actor.render()

    def save_image(self, filename):
        self._mlab_widget.save_image(filename)

class RotationControll(module_template.Controll):
    class State(object):
        """This class contains the current state of the plot to separate them from the rest of the class"""
        def __init__(self):
            self.image_number = 0
            self.rotation_type = ROTATION_TYPE.average
            
    def __init__(self, common_controll, viewer, data):
        super(RotationControll, self).__init__(common_controll, viewer, data)
        self._state = self.State()
        self._setup_gui()
        self._setup_rotation_view()
        self._data.properties_changed.connect(self._setup_rotation_view)

    def _setup_rotation_view(self):
        self._data.setup_rotations()
        self._viewer.plot_rotations_init(self._data.get_rotation_coordinates())

    def _setup_gui(self):
        self._show_average_rotation_radio = QtGui.QRadioButton("Average rot")
        self._show_average_rotation_radio.setChecked(True)
        self._show_single_rotation_radio = QtGui.QRadioButton("Single rot")

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self._show_average_rotation_radio)
        layout.addWidget(self._show_single_rotation_radio)
        self._show_average_rotation_radio.clicked.connect(partial(self.set_rotation_type, rotation_type=ROTATION_TYPE.average))
        self._show_single_rotation_radio.clicked.connect(partial(self.set_rotation_type, rotation_type=ROTATION_TYPE.single))

        self._rotation_image_number_box = QtGui.QSpinBox()
        self._rotation_image_number_box.setRange(-1, 10000)

        self._single_slice_widget = convenient_widgets.IntegerControll(0)
        self._single_slice_widget.valueChanged.connect(self.set_image_number)
        
        single_slice_layout = QtGui.QHBoxLayout()
        # def on_editing_finished(cls):
        #     cls.set_image_number(cls._rotation_image_number_box.value())
        # self._rotation_image_number_box.editingFinished.connect(partial(on_editing_finished, self))
        # single_slice_layout.addWidget(self._rotation_image_number_box)
        
        # def change_spinbox_value(cls, diff):
        #     cls._rotation_image_number_box.setValue(cls._rotation_image_number_box.value()+diff)

        # prev_image_button = QtGui.QPushButton("Previous")
        # prev_image_button.clicked.connect(partial(change_spinbox_value, self, -1))
        # single_slice_layout.addWidget(prev_image_button)
        # next_image_button = QtGui.QPushButton("Next")
        # next_image_button.clicked.connect(partial(change_spinbox_value, self, 1))
        # single_slice_layout.addWidget(next_image_button)

        # #layout.addStretch()

        # # self._rotations_controll_widget = QtGui.QWidget()
        # # self._rotations_controll_widget.setLayout(layout)

        # self._single_slice_widget = QtGui.QWidget()
        # self._single_slice_widget.setLayout(single_slice_layout)

        layout.addWidget(self._single_slice_widget)

        self._widget.setLayout(layout)        
        #self._rotation_image_number_box.hide()
        self._single_slice_widget.hide()

    def draw_hard(self):
        if self._state.rotation_type == ROTATION_TYPE.average:
            self._viewer.plot_rotations(self._data.get_average_rotation_values(self._common_controll.get_iteration()))
            #self._rotation_image_number_box.hide()
            self._single_slice_widget.hide()
        elif self._state.rotation_type == ROTATION_TYPE.single:
            self._viewer.plot_rotations(self._data.get_single_rotation_values(self._common_controll.get_iteration(), self._state.image_number))
            #self._rotation_image_number_box.show()
            self._single_slice_widget.show()

    def set_rotation_type(self, rotation_type, image_number=None):
        self._state.rotation_type = rotation_type
        self.draw()
        
    def set_image_number(self, new_image_number):
        if new_image_number >= 0:
            self._state.image_number = new_image_number
            self.draw()

class Plugin(module_template.Plugin):
    def __init__(self, common_controll, parent=None):
        super(Plugin, self).__init__(common_controll)
        self._viewer = RotationViewer()
        self._data = RotationData()
        self._controll = RotationControll(self._common_controll, self._viewer, self._data)