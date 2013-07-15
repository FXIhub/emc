import os
os.environ['ETS_TOOLKIT'] = 'qt4'
from pylab import *
import sphelper
import sys
import numpy
import time
import h5py
import tools
from functools import partial
from optparse import OptionParser
from pyface.qt import QtCore, QtGui
import convenient_widgets

SLIDER_LENGTH = 100
#state_variables = tools.enum(iteration=1)
#state_variables = ["iteration", "number_of_images"]
state_variables = ["iteration"]

class ModelData(QtCore.QObject):
    data_changed = QtCore.Signal(int)
    read_error = QtCore.Signal()
    def __init__(self):
        super(Model, self).__init__()
        # self._image_type = 0 # 0 = image, 1 = weight, 
        # self._rotation_image_number = 0
        self._current_image = 0
        self._current_type = 0
        self._read_image()

    def _read_image(self, iteration):
        try:
            if iteration >= 0:
                self._image, self._mask = sphelper.import_spimage('output/model_%.4d.h5' % self._current_iteration, ['image', 'mask'])
            elif iteration == -1:
                self._image, self._mask = sphelper.import_spimage('output/model_init.h5', ['image', 'mask'])
            elif iteration == -2:
                self._image, self._mask = sphelper.import_spimage('output/model_final.h5', ['image', 'mask'])
        except IOError:
            self.read_error.emit()

    def _read_weight(self, iteration):
        try:
            if iteration >= 0:
                self._image, self._mask = sphelper.import_spimage('output/weight_%.4d.h5' % self._current_iteration, ['image', 'mask'])
            elif iteration == -1:
                self._image, self._mask = sphelper.import_spimage('output/weight_init.h5', ['image', 'mask'])
        except IOError:
            self.read_error.emit()

    def get_model(self, iteration):
        self._read_image()
        return self._image

    def get_weight(self, iteration):
        self._read_weight()
        return self._weight

    def get_mask(self):
        return self._mask

    def get_model_side(self):
        return shape(self._image)[0]

class ModelViewer(QtCore.QObject):
    def __init__(self, data, mlab_widget, parent=None):
        super(ModelViewer, self).__init__()
        self._data = data
        self._mlab_widget = mlab_widget
        self._initialize_scalar_field()

        self._surface = None
        self._slices = None
        self._plot_surface()
        self._log_scale = False
        self._mode = 0
        self._surface_level = 0.5

    def plot_surface(self, iteration):
        self._scalar_field.scalar_data = self._data.get_model(iteration)
        self._surface.contour.contours[0] = self._surface_level*self._scalar_field_max
        self._scalar_field_max = self._scalar_field.scalar_data.max()

    def plot_slices(self, iteration):
        if self._log_scale:
            self._scalar_field.scalar_data = log10(self._data.get_model(iteration) + self._data.get_model(iteration).max()*0.001)
        else:
            self._scalar_field.scalar_data = self._data.get_model(iteration)

    # def replot(self):
    #     if (

    def _initialize_scalar_field(self):
        self._scalar_field = self._mlab_widget.get_mlab().pipeline.scalar_field(self._data.get_image())
        self._scalar_field_max = self._scalar_field.scalar_data.max()

    def _plot_surface_init(self):
        self._surface = self._mlab_widget.get_mlab().pipeline.iso_surface(self._scalar_field, contours=[0.5*self._scalar_field_max])
        #mlab.show()

    def _plot_slices_init(self):
        self._slices = []
        self._slices.append(self._mlab_widget.get_mlab().pipeline.image_plane_widget(self._scalar_field, plane_orientation='x_axes',
                                                             slice_index=shape(self._data.get_image())[0]/2))
        self._slices.append(self._mlab_widget.get_mlab().pipeline.image_plane_widget(self._scalar_field, plane_orientation='y_axes',
                                                             slice_index=shape(self._data.get_image())[1]/2))

    def set_surface_level(self, new_level):
        self._surface_level = new_level
        self._surface.contour.contours[0] = self._surface_level*self._scalar_field_max

    def get_surface_level(self):
        return self._surface.contour.contours[0]/self._scalar_field_max

    def set_log_scale(self, state):
        self._log_scale = state
        self._replot()
        
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

class Viewer(QtCore.QObject):
    def __init__(self, model, mlab_widget, parent=None):
        super(Viewer, self).__init__()
        self._model = model
        self._mlab_widget = mlab_widget
        #QtCore.QObject.connect(self._model, QtCore.SIGNAL('image_changed(int)'), self._update_scalar_field)
        self._model.data_changed.connect(self._data_changed)
        self._initialize_scalar_field()
        self._surface = None
        self._slices = None
        self._points = None
        self._plot_surface()
        self._log_scale = False
        self._mode = 0
        self._surface_level = 0.5
        

class ModelPlugin(QtCore.QObject):
    def __init__(self, common_controll, parent=None):
        super(ModelPlugin, self).__init__()
        self.viewer = ModelViewer()
        self.data = ModelData()
        self.controll = ModelControll(common_controll, self.viewer, self.data)

# class State(QtCore.QObject):
#     def __init__(self, file_handle, parent=None):
#         super(QtCore.QObject, self).__init__()
#         variable_names = {state_variables.iteration : "iteration"}
#         self._values = {}
#         for v in dir(state_variables):
#             if not v.startswith("__"):
#                 self._values[v] = file_handle[variable_names[v]][...]

#     def get_value(self, variable):
#         return self._values[variable]

#     def diff(self, other_state):
#         variables_that_changed = []
#         for v in dir(state_variables):
#             if not v.startswith("__"):
#                 if self.get_value(v) != other_state.get_value(v):
#                     variables_that_changed.append(v)
#         return variabels_that_changed

class State(QtCore.QObject):
    def __init__(self, file_handle, parent=None):
        super(State, self).__init__()
        self._values = {}
        for v in state_variables:
            self._values[v] = file_handle[v][...]

    def get_value(self, variable):
        return self._values[variable]

    def diff(self, other_state):
        variables_that_changed = []
        for v in state_variables:
            if self.get_value(v) != other_state.get_value(v):
                variables_that_changed.append(v)
        return variables_that_changed

class FileWatcher(QtCore.QThread):
    fileChanged = QtCore.Signal()
    def __init__(self, file_name):
        super(FileWatcher, self).__init__()
        self._file_name = file_name
        self._check_interval = 1. #seconds
        self._last_mtime = os.stat(self._file_name).st_mtime
    
    def __del__(self):
        self.wait()

    def run(self):
        while True:
            time.sleep(self._check_interval)
            mtime = os.stat(self._file_name).st_mtime
            if mtime != self._last_mtime:
                self.fileChanged.emit()
                
                
    def set_file(self, new_file_name):
        if os.path.isfile(new_file_name):
            self._file_name = new_file_name
            self.fileChanged.emit()

class StateWatcher(QtCore.QObject):
    iterationChanged = QtCore.Signal(int)
    def __init__(self, file_name, parent=None):
        super(StateWatcher, self).__init__()
        print file_name
        # self._watcher = QtCore.QFileSystemWatcher(self)
        # self._watcher.addPath(file_name)
        self._watcher = FileWatcher(file_name)
        self._file_handle = h5py.File(file_name)
        self._state = State(self._file_handle)
        self._watcher.fileChanged.connect(self._on_file_change)
        self._watcher.start()

    def __del__(self):
        self._file_handle.close()

    def get_value(self, variable):
        return self._state.get_value(variable)

    def set_file(self, new_file_name):
        # print "start watching %s" % new_file_name
        if os.path.isfile(new_file_name):
            self._watcher.set_file(new_file_name)
            self._file_handle.close()
            self._file_handle = h5py.File(new_file_name)
            
    def set_base_dir(self, new_dir):
        self.set_file("%s/output/state.h5" % (new_dir))

    def _on_file_change(self):
        old_state = self._state
        self._state = State(self._file_handle)
        diff = old_state.diff(self._state)
        for v in diff:
            if v == "iteration": self.iterationChanged.emit(self.get_value('iteration'))

class CommonControll(QtGui.QWidget):
    changed = QtCore.Signal()
    dirChanged = QtCore.Signal(str)
    class State(object):
        def __init__(self):
            self.iteration = 0
    def __init__(self, parent=None):
        super(CommonControll, self).__init__()
        self._state = self.State()
        self._setup_gui()

    def _setup_gui(self):
        layout = QtGui.QVBoxLayout()
        layout.addLayout(self._setup_dir_chooser())
        layout.addWidget(self._setup_iteration_chooser())
        self.setLayout(layout)

    def _setup_dir_chooser(self):
        work_dir_label = QtGui.QLabel("Dir:")
        file_system_completer = QtGui.QCompleter()
        self._file_system_model = QtGui.QFileSystemModel(file_system_completer)
        #file_system_model.setRootPath('/')
        self._file_system_model.setFilter(QtCore.QDir.Dirs | QtCore.QDir.Hidden)
        file_system_completer.setModel(self._file_system_model)
        self._work_dir_edit = QtGui.QLineEdit(os.getcwd())
        self._work_dir_edit.setCompleter(file_system_completer)
        self._work_dir_edit.editingFinished.connect(self._on_work_dir_changed)
        work_dir_layout = QtGui.QHBoxLayout()
        work_dir_layout.addWidget(work_dir_label)
        work_dir_layout.addWidget(self._work_dir_edit)
        return work_dir_layout

    def _setup_iteration_chooser(self):
        self._iteration_chooser = convenient_widgets.IntegerControll(-2)
        self._iteration_chooser.valueChanged.connect(self.set_iteration)
        return self._iteration_chooser

        # button_layout = QtGui.QHBoxLayout()
        # previous_button = QtGui.QPushButton("Previous")
        # previous_button.pressed.connect(partial(self.shift_iteration, -1))
        # button_layout.addWidget(previous_button)
        # next_button = QtGui.QPushButton("Next")
        # next_button.pressed.connect(partial(self.shift_iteration, 1))
        # button_layout.addWidget(next_button)

        # layout = QtGui.QVBoxLayout()
        # layout.addLayout(button_layout)
        # self._iteration_box = QtGui.QSpinBox()
        # self._iteration_box.setRange(-2, 10000)
        # self._iteration_box.editingFinished.connect(self.set_iteration)
        # layout.addWidget(self._iteration_box)
        # return layout

    def _on_work_dir_changed(self):
        new_dir = self._work_dir_edit.text()
        self._file_system_model.setRootPath(new_dir)
        if not os.path.isdir(new_dir):
            palette = self._work_dir_edit.palette()
            palette.setColor(QtGui.QPalette.Base, QtGui.QColor(255, 50, 50))
            self._work_dir_edit.setPalette(palette)
            return
        new_dir = self._work_dir_edit.text()
        os.chdir(new_dir)
        palette = self._work_dir_edit.palette()
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor(255, 255, 255))
        self._work_dir_edit.setPalette(palette)
        self.dirChanged.emit(new_dir)

    def _on_read_error(self):
        palette = self._work_dir_edit.palette()
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor(255, 250, 50))
        self._work_dir_edit.setPalette(palette)

    def set_iteration(self, iteration=None):
        if iteration == None:
            iteration = self._iteration_chooser.get_value()
        if iteration >= -2:
            self._state.iteration = iteration
            #self._iteration_box.setValue(self._state.iteration)
            self.changed.emit()

    def set_max_iterations(self, max_iterations):
        self._iteration_chooser.set_max(max_iterations)

    def shift_iteration(self, iteration_delta):
        if (self._state.iteration + iteration_delta) >= -2:
            self.set_iteration(self._state.iteration + iteration_delta)
        else:
            self.set_iteration(-2)
            # self._state.iteration += iteration_delta
            # self._iteration_box.setValue(self._state.iteration)
        self.changed.emit()

    def get_iteration(self):
        return self._state.iteration
                    
class StartMain(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self._common_controll = CommonControll()
        self._plugins = []
        self._setup_gui()
        self._setup_actions()
        self._setup_menus()

        self._load_module('modelmap_module')
        self._load_module('rotations_module')
        self._load_module('weightmap_module')
        self._load_module('likelihood_module')
        self._load_module('fit_module')
        self._load_module('image_module')
        self._load_module('slice_module')

        self._active_module_index = 0

        #setup watcher of the description of the current rec.
        self._state_watcher = StateWatcher("output/state.h5")
        self._common_controll.set_max_iterations(self._state_watcher.get_value('iteration'))
        self._state_watcher.iterationChanged.connect(self._common_controll.set_max_iterations)
        self._common_controll.dirChanged.connect(self._state_watcher.set_base_dir)

    def _setup_actions(self):
        self._actions = {}

        #exit
        self._actions["exit"] = QtGui.QAction("Exit", self)
        self._actions["exit"].setShortcut("Ctrl+Q")
        self._actions["exit"].triggered.connect(exit)
        
        #save image
        self._actions["save image"] = QtGui.QAction("Save image", self)
        self._actions["save image"].triggered.connect(self._on_save_image)

    def _setup_menus(self):
        self._menus = {}
        self._menus["file"] = self.menuBar().addMenu("&File")
        self._menus["file"].addAction(self._actions["save image"])
        self._menus["file"].addAction(self._actions["exit"])
    
    def _on_save_image(self):
        file_name = QtGui.QFileDialog.getSaveFileName(self, "Save file")
        if file_name:
            self._active_plugin().get_viewer().save_image(file_name)

    def _active_plugin(self):
        return self._plugins[self._active_module_index][1]
                                                      
    def _load_module(self, module_name):
        print "Loading module: %s" % module_name
        module = __import__(module_name)
        #plugin = module.Plugin(self._common_controll, self._view_stack)
        plugin = module.Plugin(self._common_controll)
        self._plugins.append([module_name, plugin])
        plugin.get_data().read_error.connect(self._common_controll._on_read_error)
        self._common_controll.changed.connect(plugin.get_controll().draw)
        self._common_controll.dirChanged.connect(plugin.get_controll().on_dir_change)
        self._view_stack.addWidget(plugin.get_viewer().get_widget())
        self._controll_stack.addWidget(plugin.get_controll().get_widget())
        self._module_box.addItem(module_name)
        plugin.get_controll().draw()

    def _setup_gui(self):
        self._view_stack = QtGui.QStackedWidget(self)
        self._controll_stack = QtGui.QStackedWidget(self)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self._view_stack)
        layout.addLayout(self._setup_module_select())
        
        common_controll_group = QtGui.QGroupBox("Common controlls")
        common_controll_dummy_layout = QtGui.QHBoxLayout()
        common_controll_dummy_layout.addWidget(self._common_controll)
        common_controll_group.setLayout(common_controll_dummy_layout)
        layout.addWidget(common_controll_group)

        module_controll_group = QtGui.QGroupBox("Module controlls")
        module_controll_dummy_layout = QtGui.QHBoxLayout()
        module_controll_dummy_layout.addWidget(self._controll_stack)
        module_controll_group.setLayout(module_controll_dummy_layout)
        layout.addWidget(module_controll_group)

        size_policy_large = self._view_stack.sizePolicy()
        size_policy_large.setVerticalStretch(1)
        self._view_stack.setSizePolicy(size_policy_large)


        # module_controll_group = QtGui.QGroupBox("Module controlls")
        # module_controll_group.setWidget(self._common_controll)
        # layout.addWidget(self.module_cocntroll_group)

        self._central_widget = QtGui.QWidget()
        self._central_widget.setLayout(layout)
        self.setCentralWidget(self._central_widget)

    def _setup_module_select(self):
        self._module_box = QtGui.QComboBox()
        self._module_box.activated.connect(self._select_module)
        # self._module_box.activated.connect(self._view_stack.setCurrentIndex)
        # self._module_box.activated.connect(self._controll_stack.setCurrentIndex)

        label = QtGui.QLabel("Display module: ")
        layout = QtGui.QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self._module_box)
        layout.addStretch()
        return layout

    def _select_module(self, index):
        self._active_module_index = index
        self._view_stack.setCurrentIndex(index)
        self._controll_stack.setCurrentIndex(index)
        for name,plugin in self._plugins:
            plugin.get_controll().set_active(False)
        self._plugins[index][1].get_controll().set_active(True)
        self._plugins[index][1].get_controll().draw()

if __name__ == "__main__":
    parser = OptionParser(usage="%prog <3d_file.h5>")
    parser.add_option("-s", action="store_true", dest="surface", help="Plot image as isosurface.")
    parser.add_option("-l", action="store_true", dest="log", help="Plot in log scale.")
    parser.add_option("-m", action="store_true", dest="mask", help="Plot mask.")
    (options, args) = parser.parse_args()

    app = QtGui.QApplication(['Controll window'])
    #app = QtGui.QApplication.instance()

    program = StartMain()
    program.show()
    sys.exit(app.exec_())

    
