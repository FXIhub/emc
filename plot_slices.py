"""
Program to plot diffraction slices

We generally use the indexing [x,y] and [x,y,z].
"""
from pylab import *
import sys
from PySide import QtCore, QtGui
from vtk import *
import h5py
from optparse import OptionParser

class SimulatedData(object):
    def __init__(self):
        self._data = self._read_data('simulated_images.p')
        #self._side = self._data.images[0].shape[0]
        self._side = self._data['images'][0].shape[0]

    @classmethod
    def _read_data(cls, filename):
        pickle_module = __import__('pickle')
        with open(filename, 'rb') as file_handle:
            data = pickle_module.load(file_handle)
        return data
    
    def get_image(self, index):
        #return log10(self._data.images[index]), self._data.rotations[index]
        return log10(self._data['images'][index]), self._data['rotations'][index]
        #return log10(self._data.images[index] - self.min())

    def curvature(self):
        #return self._data.curvature
        return self._data['curvature']

    def side(self):
        return self._side

    def number_of_images(self):
        #return len(self._data.images)
        return len(self._data['images'])

    def max(self):
        #return log10(max([image.max() for image in self._data.images]))
        return log10(max([image.max() for image in self._data['images']]))

    def min(self):
        #return log10(min([image.min() for image in self._data.images]))
        return log10(min([image.min() for image in self._data['images']]))


class FakeData(object):
    def __init__(self):
        self._rotations_module = __import__('rotations')
        self._side = 128
        self._number_of_images = 10
        self._radius = 100.
        
        x = arange(self._side) - self._side/2. + 0.5
        self._x, self._y = meshgrid(x,x)
        self._z = self._radius - sqrt(self._radius**2 - self._x**2 - self._y**2)

        self._rotations = [self._rotations_module.random_quaternion() for i in range(self._number_of_images)]

    def get_image(self, index):
        rot_mat = self._rotations_module.quaternion_to_matrix(self._rotations[index])
        x = rot_mat[0, 0]*self._x + rot_mat[0, 1]*self._y + rot_mat[0, 2]*self._z
        y = rot_mat[1, 0]*self._x + rot_mat[1, 1]*self._y + rot_mat[1, 2]*self._z
        z = rot_mat[2, 0]*self._x + rot_mat[2, 1]*self._y + rot_mat[2, 2]*self._z
        image = 1.0*(z > 0.)
        return image, self._rotations[index]

    def curvature(self):
        return self._radius

    def side(self):
        return self._side

    def number_of_images(self):
        return self._number_of_images

    def max(self):
        return 1.

    def min(self):
        return 0.


class ImageData(object):
    def __init__(self, image_prefix, rotations_file, max_number_of_images=inf, curvature=inf, transpose=False):
        self._image_prefix = image_prefix
        self._rotations_file = rotations_file
        self._image_side = 128
        self._image_stride = 2
        self._image_curvature = curvature
        self._transpose = transpose
        
        self._rotations = self._read_rotations(rotations_file)
        self._number_of_images = min(len(self._rotations), max_number_of_images)
        self._images = self._read_images(self._image_prefix)
            
        #self._scaling = self._read_scaling('images/scaling.data')
        #self._scaling = 1./self._scaling
        #self._scaling = ones(len(self._scaling))

    def _read_rotations(self, filename):
        return loadtxt(filename)

    def _read_scaling(self, filename):
        return loadtxt(filename)[-1]

    # def _process_image(self, image, scaling):
    #     center = image.shape[0]/2
    #     width = self._image_side*self._image_stride/2
    #     cropped_image = self._normalize_image(abs(image[(center-width):(center+width):self._image_stride,
    #                                                     (center-width):(center+width):self._image_stride])) / scaling
    #     return log10(cropped_image + 0.01*average(cropped_image))
    #     #self._normalize_image(processeimage) / scaling
    def _process_image(self, image):
        center = image.shape[0]/2
        width = self._image_side*self._image_stride/2
        cropped_image = self._normalize_image(abs(image[(center-width):(center+width):self._image_stride,
                                                        (center-width):(center+width):self._image_stride]))
        #return log10(cropped_image + 0.01*average(cropped_image))
        return log10(cropped_image)
        #return cropped_image
        #self._normalize_image(processeimage) / scaling
            
    @classmethod
    def _normalize_image(self, image):
        return image / image.sum()

    def _read_images(self, dirname, max_number_of_images=inf):
        images = []
        for i in range(min(self._number_of_images, max_number_of_images)):
            file_handle = h5py.File('%s%.4d.h5' % (dirname, i), 'r')
            image = squeeze(file_handle['real'][...])
            file_handle.close()
            #processed_image = self._process_image(image, self._scaling[i])
            processed_image = self._process_image(image)
            images.append(processed_image)
        return images

    def get_image(self, index):
        if index >= self._number_of_images:
            raise ValueError("Index out of range: %d >= %d." % (index, self._number_of_images))
        if index < 0:
            raise ValueError("Index out of range: %d < 0." % (index))
        #return self._images[index]*self._scaling[index], self._rotations[index]
        if self._transpose:
            return_image = transpose(self._images[index])
        else:
            return_image = self._images[index]
        return return_image, self._rotations[index]

    def curvature(self):
        return self._image_curvature

    def side(self):
        return self._image_side

    def number_of_images(self):
        return self._number_of_images

    def max(self):
        #return max([image[(-isinf(image))*(-isnan(image))].max()*scaling for image, scaling in zip(self._images, self._scaling)])
        return max([image[(-isinf(image))*(-isnan(image))].max() for image in self._images])

    def min(self):
        #return min([image[(-isinf(image))*(-isnan(image))].min()*scaling for image, scaling in zip(self._images, self._scaling)])
        return min([image[(-isinf(image))*(-isnan(image))].min() for image in self._images])

class SliceGenerator(object):
    def __init__(self, side, radius):
        self._side = side
        self._radius = radius
        x_array_single = arange(self._side) - self._side/2. + 0.5
        y_array_single = arange(self._side) - self._side/2. + 0.5
        self._x_array, self._y_array = meshgrid(x_array_single, y_array_single)
        if isinf(radius):
            self._z_array = zeros(self._x_array.shape)
        else:
            self._z_array = self._radius - sqrt(self._radius**2 - self._x_array**2 - self._y_array**2)

        self._image_values = vtkFloatArray()
        self._image_values.SetNumberOfComponents(1)
        self._image_values.SetName("Intensity")
        
        self._points = vtkPoints()
        point_indices = -ones((self._side)*(self._side), dtype='int32')

        for i in range(self._side):
            for j in range(self._side):
                #self._points.InsertNextPoint(self._x_array[i,j], self._y_array[i,j], self._z_array[i,j])
                self._points.InsertNextPoint(self._z_array[i,j], self._y_array[i,j], self._x_array[i,j])
                self._image_values.InsertNextTuple1(0.)

        self._circular_slice()
        self._template_poly_data = vtkPolyData()
        self._template_poly_data.SetPoints(self._points)
        self._template_poly_data.GetPointData().SetScalars(self._image_values)
        self._template_poly_data.SetPolys(self._polygons)

    def _square_slice(self):
        self._polygons = vtkCellArray()    
        for i in range(self._side-1):
            for j in range(self._side-1):
                corners = [(i,j), (i+1, j), (i+1, j+1), (i, j+1)]
                polygon = vtkPolygon()
                polygon.GetPointIds().SetNumberOfIds(4)
                for index, c in enumerate(corners):
                    polygon.GetPointIds().SetId(index, c[0]*self._side+c[1])
                self._polygons.InsertNextCell(polygon)

        self._template_poly_data = vtkPolyData()
        self._template_poly_data.SetPoints(self._points)
        self._template_poly_data.GetPointData().SetScalars(self._image_values)
        self._template_poly_data.SetPolys(self._polygons)

    def _circular_slice(self):
        self._polygons = vtkCellArray()    
        for i in range(self._side-1):
            for j in range(self._side-1):
                corners = [(i,j), (i+1, j), (i+1, j+1), (i, j+1)]
                radius = max([sqrt((c[0] - self._side/2. + 0.5)**2 + (c[1] - self._side/2. + 0.5)**2) for c in corners])
                if radius < self._side/2.:
                    polygon = vtkPolygon()
                    polygon.GetPointIds().SetNumberOfIds(4)
                    for index, c in enumerate(corners):
                        polygon.GetPointIds().SetId(index, c[0]*self._side+c[1])
                    self._polygons.InsertNextCell(polygon)

    def _texture_transform(self, point):
        return (self._side - 1 - point[0], point[1])

    def _get_rotated_coordinates(self, rot):
        z_array, y_array, x_array = rotations.rotate_array(rot, self._z_array.flatten(),
                                                           self._y_array.flatten(),
                                                           self._x_array.flatten())
        x_array = x_array.reshape((self._side, self._side))
        y_array = y_array.reshape((self._side, self._side))
        z_array = z_array.reshape((self._side, self._side))
        return z_array, y_array, x_array

    def insert_slice(self, image, rotation):
        # this_poly_data = vtkPolyData()
        # this_poly_data.DeepCopy(self._template_poly_data)

        rotation_degrees = rotation.copy()
        rotation_degrees[0] = 2.*arccos(rotation[0])*180./pi
        transformation = vtkTransform()
        #transformation.RotateWXYZ(*rotation_degrees)
        transformation.RotateWXYZ(rotation_degrees[0], rotation_degrees[1], rotation_degrees[2], rotation_degrees[3])
        #transformation.RotateWXYZ(-rotation_degrees[0], rotation_degrees[1], rotation_degrees[2], rotation_degrees[3]) #this one worked together with propagator with inverted quaternions and transpose on.
        #transformation.RotateWXYZ(rotation_degrees[0], rotation_degrees[3], rotation_degrees[2], rotation_degrees[1])
        transform_filter = vtkTransformFilter()
        transform_filter.SetInput(self._template_poly_data)
        transform_filter.SetTransform(transformation)
        transform_filter.Update()
        this_poly_data = transform_filter.GetOutput()

        #try
        # polys = this_poly_data.GetPolys()
        # points = this_poly_data.GetPoints().GetData()
        #end try
        
        scalars = this_poly_data.GetPointData().GetScalars()
        for i in range(self._side):
            for j in range(self._side):
                point_coord = this_poly_data.GetPoint(self._side*i + j)
                #if point_coord[0] > 0.:
                scalars.SetTuple1(i*self._side+j, image[i, j])
                #scalars.SetTuple1(i*self._side+j, image[j, i])
                # else:
                #     #polys.GetData().SetTuple4(i*self._side+j, 0., 0., 0., 0.)
                #     scalars.SetTuple1(i*self._side+j, nan)
        this_poly_data.Modified()
        return this_poly_data
    

class VTKRenderer(object):
    def __init__(self, data):
        self._data = data
        self._renderer = vtkRenderer()
        self._renderer.SetBackground(1, 1, 1)
        self._render_window = vtkRenderWindow()
        self._render_window.SetSize((800, 800))
        self._render_window.AddRenderer(self._renderer)

        self._camera = vtkCamera()
        self._camera.SetPosition(130, 130, 170)
        self._camera.SetFocalPoint(0., 0., 0.)
        self._camera.SetViewUp(0, 0, 1)
        self._renderer.SetActiveCamera(self._camera)

        self._lut = vtkLookupTable()
        #self._lut.SetTableRange(-4., -2.)
        #self._lut.SetTableRange(self._data.min() - 0.001*(self._data.max() - self._data.min()), self._data.max())
        self._lut.SetTableRange(self._data.min(), self._data.max())
        #self._lut.SetTableRange(self._data.min(), self._data.min() + (self._data.max()-self._data.min())*.2)
        self._lut.SetHueRange(0., 1.)
        #self._lut.SetValueRange(0.0, 1.)
        self._lut.SetValueRange(.5, 1.)
        self._lut.SetAlphaRange(1.0, 1.0)
        # self._lut.SetNanColor(0., 0., 0.)
        # self._lut.SetNumberOfTableValues(3)
        # self._lut.SetTableRange(-4., -1.)
        # self._lut.SetTableValue(0, 0., 0., 0., 0.1)
        # self._lut.SetTableValue(1, 0., 1., 0., 1.)
        # self._lut.SetTableValue(2, 0., 0., 1., 1.)
        self._lut.SetNumberOfColors(1000)
        self._lut.Build()
        self._lut.SetTableValue(0, 0., 0., 0., 0.)

        self._actors = {}

    def make_interactive(self):
        self._render_window_interactor = vtkRenderWindowInteractor()
        self._render_window_interactor.SetRenderWindow(self._render_window)
        self._render_window_interactor.Start()

    def add_poly_data(self, this_poly_data, id):
        if id in self._actors:
            raise ValueError("Actor with id %d is already plotted" % id)
        mapper = vtkPolyDataMapper()
        mapper.SetInput(this_poly_data)
        mapper.SetScalarModeToUsePointData()
        mapper.UseLookupTableScalarRangeOn()
        mapper.SetLookupTable(self._lut)

        actor = vtkActor()
        actor.SetMapper(mapper)
        #actor.GetProperty().SetOpacity(0.999999)
        self._actors[id] = actor
        self._actors[id] = actor
        
        self._renderer.AddActor(actor)
        #self._renderer.UseDepthPeelingOn()
        
        self._render_window.Render()

    def remove_poly_data(self, id):
        if not id in self._actors:
            raise ValueError("Trying to remove actor with id %d that doesn't exist." % id)
        self._renderer.RemoveActor(self._actors[id])
        del self._actors[id]
        self._render_window.Render()

class StartMain(QtGui.QMainWindow):
    def __init__(self, data, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.resize(200, 800)
        self._data = data
        #self._data = SimulatedData()
        self._vtk_window = VTKRenderer(self._data)
        self._slice_generator = SliceGenerator(self._data.side(), self._data.curvature())
        self._create_main_frame()
        self._vtk_window.make_interactive()

        image, rot = self._data.get_image(0)
        self._vtk_window.add_poly_data(self._slice_generator.insert_slice(image, rot), 0)
        self._counter = 1

    def _create_main_frame(self):
        self._main_frame = QtGui.QWidget()

        stupid_button = QtGui.QPushButton("Press me")
        stupid_button.pressed.connect(self._open_image_dialog)
        #self.connect(stupid_button, QtCore.SIGNAL('pressed()'), self._on_list_item_changed)

        self._pattern_list = self._create_pattern_list()

        #self.connect(self._pattern_list, QtCore.SIGNAL('itemChanged(QListWidgetItem)'), self._on_list_item_changed)
        self._pattern_list.itemChanged.connect(self._on_list_item_changed)
        
        vertical_layout = QtGui.QVBoxLayout()
        vertical_layout.addWidget(stupid_button)
        vertical_layout.addWidget(self._pattern_list)
        
        self._main_frame.setLayout(vertical_layout)
        self.setCentralWidget(self._main_frame)

    def _on_list_item_changed(self, item):
        index = self._pattern_list.indexFromItem(item).row()
        if item.checkState() == QtCore.Qt.Checked:
            image, rot = self._data.get_image(index)
            self._vtk_window.add_poly_data(self._slice_generator.insert_slice(image, rot), index)
        else:
            #remove slice
            self._vtk_window.remove_poly_data(index)

    def _create_pattern_list(self):
        pattern_list = QtGui.QListWidget()
        #pattern_list = QtGui.QCheckList()
        for i in range(self._data.number_of_images()):
            this_item = QtGui.QListWidgetItem(str(i), pattern_list)
            this_item.setFlags(this_item.flags() | QtCore.Qt.ItemIsUserCheckable)# | QtCore.Qt.ItemIsEnabled)
            this_item.setData(QtCore.Qt.CheckStateRole, QtCore.Qt.Checked)
            if i > 0:
                this_item.setCheckState(QtCore.Qt.Unchecked)
        return pattern_list

    def _open_image_dialog(self):
        this_dialog = ImageDialog(self)
        this_dialog.exec_()

class ImageDialog(QtGui.QDialog):
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)
        stupid_button = QtGui.QPushButton("I don't to nothing")
        horizontal_layout = QtGui.QVBoxLayout()
        horizontal_layout.addWidget(stupid_button)
        self.setLayout(horizontal_layout)
        stupid_button.pressed.connect(self._on_press)

    def _on_press(self):
        print "Dialog button pressed"

def get_curvature(pixel_size, detector_distance):
    if not (pixel_size and detector_distance):
        return inf
    else:
        return detector_distance/pixel_size

if __name__ == "__main__":
    parser = OptionParser(usage="%prog [-n NUMBER_OF_IMAGES] IMAGE_DIR ROTFILE")
    parser.add_option("-n", type='int', action='store', dest='number_of_images', default=inf)
    # we don't actually need the wavelength to get the curvature
    # parser.add_option("-w", type='float', action='store', dest='wavelength', default=None)
    parser.add_option("-d", type='float', action='store', dest='pixel_size', default=None)
    parser.add_option("-D", type='float', action='store', dest='detector_distance', default=None)
    options, args = parser.parse_args()

    data = ImageData(args[0], args[1], options.number_of_images, get_curvature(options.pixel_size, options.detector_distance), transpose=False)
    #data = SimulatedData()
    
    app = QtGui.QApplication(['QVTKRenderWindowInteractor'])
    program = StartMain(data)
    program.show()
    sys.exit(app.exec_())
