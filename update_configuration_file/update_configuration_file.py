__author__ = 'ekeberg'

import pylibconfig2
import sys

class ValueUpdate(object):
    def __init__(self, new_key, old_key=None, default_value=None, function=None):
        self._new_key = new_key
        if not isinstance(old_key, (list, tuple)):
            old_key = (old_key, )
        if self._new_key in old_key:
            self._acceptable_keys = tuple(old_key)
        else:
            self._acceptable_keys = (new_key, ) + tuple(old_key)

        self._default_value = default_value
        if function == None:
            self._function = lambda x: x
        else:
            self._function = function

    def key(self):
        return self._new_key

    def acceptable_keys(self):
        return self._acceptable_keys

    def function(self):
        return self._function

    def default_value(self):
        return self._default_value


def update_initial_model_value(old_value):
    value_dict = {0: "uniform", 1: "random orientations", 2: "file",
                  3: "radial average", 4: "given orientations"}

    return value_dict[old_value]

value_updates = [
    ValueUpdate("image_prefix"),
    ValueUpdate("number_of_images", "N_images"),
    ValueUpdate("mask_file"),

    ValueUpdate("output_dir", default_value="output"),

    ValueUpdate("model_side"),
    ValueUpdate("image_binning", "read_stride"),

    ValueUpdate("wavelength"),
    ValueUpdate("pixel_size"),
    ValueUpdate("detector_size"),
    ValueUpdate("detector_distance"),

    ValueUpdate("diff_type", default_value="poisson"),
    ValueUpdate("sigma_start"),
    ValueUpdate("sigma_final"),
    ValueUpdate("sigma_half_life", default_value=30),

    ValueUpdate("compact_output", default_value=True),
    ValueUpdate("chunk_size", "slice_chunk", default_value=5000),
    ValueUpdate("random_seed", default_value=0),
    ValueUpdate("rotations_file"),

    ValueUpdate("number_of_iterations", "max_iterations", default_value=100),

    ValueUpdate("blur_image", default_value=False),
    ValueUpdate("blur_image_sigma", "blur_sigma", default_value=0.),

    ValueUpdate("recover_scaling", "known_intensity", function=lambda x: not x),
    ValueUpdate("normalize_images", default_value=False),

    ValueUpdate("initial_model", "model_input", default_value="radial average", function=update_initial_model_value),
    ValueUpdate("initial_model_noise", default_value=0.1),
    ValueUpdate("initial_model_file", "model_file", default_value="not used"),
    ValueUpdate("initial_rotations_file", "init_rotations", default_value="not used"),

    ValueUpdate("exclude_images", default_value=False),
    ValueUpdate("exclude_images_ratio", "exclude_ratio", default_value=0.),

    ValueUpdate("blur_model", "model_blur", default_value=False, function=lambda x: bool(x)),
    ValueUpdate("blur_model_sigma", default_value=0.2)
]

def change_name_and_write(value_update, old_conf, new_conf_file_handle):
    new_conf = pylibconfig2.Config()
    found_key = False
    for this_old_key in value_update.acceptable_keys():
        if old_conf.get(this_old_key) != None:
            old_value = old_conf.lookup(this_old_key)
            if this_old_key == value_update.key():
                new_value = old_value
            else:
                new_value = value_update.function()(old_value)
            new_conf.set(value_update.key(), new_value)
            found_key = True
            break
    if not found_key:
        if value_update.default_value() != None:
            new_conf.set(value_update.key(), value_update.default_value())
        else:
            raise ValueError("Configuration file must contain {0}".format(value_update.key()))
    new_conf_file_handle.write(str(new_conf)+"\n")

input_file = sys.argv[1]
output_file = sys.argv[2]
#input_file = "emc.conf"
#output_file = "foo.conf"

with open(input_file) as file_handle:
    old_conf = pylibconfig2.Config(file_handle.read())

with open(output_file, "wp") as file_handle:
    for value_update in value_updates:
        change_name_and_write(value_update, old_conf, file_handle)