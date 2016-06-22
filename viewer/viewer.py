"""This is a viewer of the output of EMC. Run this program in the
directory containing the runs output and debug directories."""
import os
os.environ['ETS_TOOLKIT'] = 'qt4'
#from pylab import *
import sys
import time
import h5py
from eke.QtVersions import QtCore, QtGui
import convenient_widgets

#state_variables = tools.enum(iteration=1)
#state_variables = ["iteration", "number_of_images"]
STATE_VARIABLES = ["iteration"]
PROGRAM_NAME = "EMC Viewer"

class State(QtCore.QObject):
    """Class used by the state watcher too  indicate the current state"""
    def __init__(self, file_handle, parent=None):
        super(State, self).__init__(parent)
        self._values = {}
        for variable in STATE_VARIABLES:
            self._values[variable] = file_handle[variable][...]

    def get_value(self, variable):
        """Return the value of a variable"""
        return self._values[variable]

    def diff(self, other_state):
        """Return a list of all the variables different between
        this state and other_state"""
        variables_that_changed = []
        for variable in STATE_VARIABLES:
            if self.get_value(variable) != other_state.get_value(variable):
                variables_that_changed.append(variable)
        return variables_that_changed

class FileWatcher(QtCore.QThread):
    """Reports on changes of a file by emitting the fileChanged signal"""
    fileChanged = QtCore.Signal()
    def __init__(self, file_name):
        super(FileWatcher, self).__init__()
        self._file_name = file_name
        self._check_interval = 5. #seconds
        self._last_mtime = os.stat(self._file_name).st_mtime

    def __del__(self):
        self.wait()

    def run(self):
        """Start watching"""
        while True:
            time.sleep(self._check_interval)
            mtime = os.stat(self._file_name).st_mtime
            if mtime != self._last_mtime:
                self._last_mtime = mtime
                self.fileChanged.emit()

    def set_file(self, new_file_name):
        """Change the file being watched"""
        if os.path.isfile(new_file_name):
            self._file_name = new_file_name
            self.fileChanged.emit()

class StateWatcher(QtCore.QObject):
    """Trackes changes in a state file, reporting any change by
    emiting a signal."""
    iterationChanged = QtCore.Signal(int)
    def __init__(self, file_name, parent=None):
        super(StateWatcher, self).__init__(parent)
        #print file_name
        # self._watcher = QtCore.QFileSystemWatcher(self)
        # self._watcher.addPath(file_name)
        self._watcher = FileWatcher(file_name)
        self._file_handle = h5py.File(file_name)
        if not self._file_handle:
            print "Error reading state file. Make sure you are in an EMC output dir."
            exit(1)
        self._state = State(self._file_handle)
        self._watcher.fileChanged.connect(self._on_file_change)
        self._watcher.start()

    def __del__(self):
        self._file_handle.close()

    def get_value(self, variable):
        """Return the current value of the variable"""
        return self._state.get_value(variable)

    def set_file(self, new_file_name):
        """Change what file is being watched (full path)."""
        # print "start watching %s" % new_file_name
        if os.path.isfile(new_file_name):
            self._watcher.set_file(new_file_name)
            self._file_handle.close()
            self._file_handle = h5py.File(new_file_name)

    def set_base_dir(self, new_dir):
        """Change the base dir being watched, assuming the file ends
        with state.h5"""
        self.set_file("%s/state.h5" % (new_dir))

    def _on_file_change(self):
        """Handles file changed calles from the watcher"""
        old_state = self._state
        self._state = State(self._file_handle)
        diff = old_state.diff(self._state)
        for variable in diff:
            if variable == "iteration":
                self.iterationChanged.emit(self.get_value('iteration'))

class CommonControll(QtGui.QWidget):
    """This is a widget that gives the user controll of things that are
    common to all plugins, such as iteration number and active directory."""
    changed = QtCore.Signal()
    dirChanged = QtCore.Signal(str)
    # class State(object):
    #     """Container for the variables keept by the common controll."""
    #     def __init__(self):
    #         self.iteration = 0
    #         self.max_iterations = 0
    def __init__(self, parent=None):
        super(CommonControll, self).__init__(parent)
        #self._state = self.State()
        self._state = {"iteration": 0,
                       "max_iterations": 0}
        self._run_info = {"compact_output": False,
                          "number_of_images": 0,
                          "random_seed": 0}
        self._file_system_model = None
        self._work_dir_edit = None
        self._iteration_chooser = None
        self._setup_gui()

    def _setup_gui(self):
        """Create the widget gui."""
        layout = QtGui.QVBoxLayout()
        layout.addLayout(self._setup_dir_chooser())
        layout.addWidget(self._setup_iteration_chooser())
        layout.addWidget(self._setup_message_board())
        self.setLayout(layout)

    def _setup_dir_chooser(self):
        """Return a directory cooser layout."""
        work_dir_label = QtGui.QLabel("Dir:")
        file_system_completer = QtGui.QCompleter()
        self._file_system_model = QtGui.QFileSystemModel(file_system_completer)
        #file_system_model.setRootPath('/')
        self._file_system_model.setFilter(QtCore.QDir.Dirs | QtCore.QDir.Hidden)
        file_system_completer.setModel(self._file_system_model)
        self._work_dir_edit = QtGui.QLineEdit(os.getcwd())
        self._work_dir_edit.setCompleter(file_system_completer)
        self._work_dir_edit.editingFinished.connect(self.on_work_dir_changed)
        work_dir_layout = QtGui.QHBoxLayout()
        work_dir_layout.addWidget(work_dir_label)
        work_dir_layout.addWidget(self._work_dir_edit)
        return work_dir_layout

    def _setup_iteration_chooser(self):
        """Return an iteration cooser widget."""
        self._iteration_chooser.valueChanged.connect(self.set_iteration)
        self._iteration_chooser = convenient_widgets.IntegerControll(-1)
        return self._iteration_chooser

    def _setup_message_board(self):
        self._message_board = QtGui.QLabel("Empty board")
        self._message_board.setWordWrap(True)
        self._message_board.setFrameStyle(QtGui.QFrame.Panel)
        return self._message_board

    def post_message(self, message):
        self._message_board.setText(message)

    def on_work_dir_changed(self):
        """Handle an attempt to select a new work dir in the
        _work_dir_edit text editor"""
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
        self._read_run_info()
        self.dirChanged.emit(new_dir)
        self.post_message("Read new directory {0}".format(new_dir))

    def on_read_error(self):
        """Handle read error emits"""
        palette = self._work_dir_edit.palette()
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor(255, 250, 50))
        self._work_dir_edit.setPalette(palette)

    def set_iteration(self, iteration=None):
        """Change the current iteration"""
        if iteration == None:
            iteration = self._iteration_chooser.get_value()
        if iteration >= -2:
            #self._state.iteration = iteration
            self._state["iteration"] = iteration
            #self._iteration_box.setValue(self._state.iteration)
            self.changed.emit()

    def set_max_iterations(self, max_iterations):
        """Change the latest iteration number."""
        #self._state.max_iterations = max_iterations
        self._state["max_iterations"] = max_iterations
        self._iteration_chooser.set_max(max_iterations)

    def get_max_iterations(self):
        """Return the current latest iteration."""
        #return self._state.max_iterations
        return self._state["max_iterations"]

    def shift_iteration(self, iteration_delta):
        """Change iteration by shifting the current one."""
        #if (self._state.iteration + iteration_delta) >= -2:
        if (self._state["iteration"] + iteration_delta) >= -2:
            self.set_iteration(self._state["iteration"] + iteration_delta)
            #self.set_iteration(self._state.iteration + iteration_delta)
        else:
            self.set_iteration(-2)
            # self._state.iteration += iteration_delta
            # self._iteration_box.setValue(self._state.iteration)
        self.changed.emit()

    def get_iteration(self):
        """Sets the current iteration to a specified value."""
        #return self._state.iteration
        return self._state["iteration"]

    def _read_run_info(self):
        try:
            with h5py.File("run_info.h5", "r") as file_handle:
                self._run_info["compact_output"] = bool(file_handle["compact_output"][...])
                self._run_info["number_of_images"] = int(file_handle["number_of_images"][...])
                self._run_info["random_seed"] = int(file_handle["random_seed"][...])
        except IOError:
            self._run_info["compact_output"] = False
            with h5py.File("state.h5", "r") as file_handle:
                self._run_info["number_of_images"] = file_handle["number_of_images"][...]
            self._run_info["random_seed"] = 0

    def output_is_compact(self):
        return self._run_info["compact_output"]
    
    def number_of_images(self):
        return self._run_info["number_of_images"]

    def random_seed(self):
        return self._run_info["random_seed"]
    

class StartMain(QtGui.QMainWindow):
    """The program."""
    #pylint: disable=too-many-instance-attributes
    #I think the code is still clear
    def __init__(self, parent=None):
        super(StartMain, self).__init__(parent)
        self._module_box = None
        self._menus = None
        self._actions = None
        self._view_stack = None
        self._controll_stack = None
        self._central_widget = None

        self.setWindowTitle(PROGRAM_NAME)
        self._common_controll = CommonControll()
        self._plugins = []
        self._setup_gui()
        self._setup_actions()
        self._setup_menus()

        #setup watcher of the description of the current rec.
        self._state_watcher = StateWatcher("state.h5")
        self._common_controll.set_max_iterations(self._state_watcher.get_value('iteration'))
        self._state_watcher.iterationChanged.connect(self._common_controll.set_max_iterations)
        self._common_controll.dirChanged.connect(self._state_watcher.set_base_dir)

        self._load_module('modelmap_module') #done
        self._load_module('weightmap_module')
        self._load_module('rotations_module') #done
        self._load_module('slice_module') #done
        self._load_module('likelihood_module')
        self._load_module('fit_module')
        self._load_module('image_module')
        self._load_module('scaling_module')
        
        #self._common_controll.dirChanged.emit(os.getcwd())
        self._common_controll.on_work_dir_changed()
        
        self._active_module_index = 0

    def initialize(self):
        """Call this function after the window is made (for example through show()).
        Some plugins need to some processing after the window to draw into
        becomes available. Calling this function executes that code in the plugins.
        It also does the initial draw of the plugin."""
        for plugin in self._plugins:
            print "Intializing module: {0}".format(plugin[0])
            plugin[1].initialize()
            #plugin[1].get_controll().draw()
        self._select_module(self._active_module_index)

    def _setup_actions(self):
        """Setup actions to be used in menues etc."""
        self._actions = {}

        #exit
        self._actions["exit"] = QtGui.QAction("Exit", self)
        self._actions["exit"].setShortcut("Ctrl+Q")
        self._actions["exit"].triggered.connect(exit)

        #save image
        self._actions["save image"] = QtGui.QAction("Save image", self)
        self._actions["save image"].triggered.connect(self._on_save_image)

    def _setup_menus(self):
        """Setup the menues. Must be called after _setup_actions"""
        self._menus = {}
        self._menus["file"] = self.menuBar().addMenu("&File")
        self._menus["file"].addAction(self._actions["save image"])
        self._menus["file"].addAction(self._actions["exit"])

    def _on_save_image(self):
        """Called when the save image action is triggered."""
        file_name = QtGui.QFileDialog.getSaveFileName(self, "Save file")
        if file_name:
            self._active_plugin().get_viewer().save_image(file_name)

    def _active_plugin(self):
        """Returns the active plugin."""
        return self._plugins[self._active_module_index][1]

    def _load_module(self, module_name):
        """Load a module, takes the name as a string as input."""
        print "Loading module: {0}".format(module_name)
        module = __import__(module_name)
        #plugin = module.Plugin(self._common_controll, self._view_stack)
        plugin = module.Plugin(self._common_controll)
        self._plugins.append([module_name, plugin])
        plugin.get_data().read_error.connect(self._common_controll.on_read_error)
        self._common_controll.changed.connect(plugin.get_controll().draw)
        self._common_controll.dirChanged.connect(plugin.get_controll().on_dir_change)
        self._view_stack.addWidget(plugin.get_viewer().get_widget())
        self._controll_stack.addWidget(plugin.get_controll().get_widget())
        self._module_box.addItem(module_name)
        #plugin.get_controll().draw()

    def _setup_gui(self):
        """Setup the entire gui of the program"""
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
        """Create a drop down menu for selecting modules. Returns a layout containing it."""
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
        """Handles the change of module."""
        self._active_module_index = index
        self._view_stack.setCurrentIndex(index)
        self._controll_stack.setCurrentIndex(index)
        for plugin in self._plugins:
            module = plugin[1]
            module.get_controll().set_active(False)
        self._plugins[index][1].get_controll().set_active(True)
        self._plugins[index][1].get_controll().draw()

def main():
    """Launch program"""
    app = QtGui.QApplication(['Controll window'])
    icon_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], "resources/icon_slices.png")
    app.setWindowIcon(QtGui.QIcon(icon_path))
    app.setApplicationName(PROGRAM_NAME)

    #app = QtGui.QApplication.instance()

    program = StartMain()
    program.show()
    program.initialize()
    #app.setActiveWindow(program)
    program.raise_()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
