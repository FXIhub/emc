#!/usr/bin/env python
from pylab import *
from optparse import OptionParser
import h5py
import assemble_slices
import sphelper
import spimage
import time_tools
import sys
from mpi4py import MPI
import parse_libconfig
import os.path

# parser = OptionParser(usage="%prog EMC_CONF ITERATION")
# parser.add_option("-o", action="store", type="string", dest="output_prefix", default="./split")
# options, args = parser.parse_args()

#emc_configuration_file = args[0]
#iteration = int(args[1])

emc_configuration_file = "/home/ekeberg/Work/programs/emc/my_emc_runs/mimi/multi_run4/run_41/emc.conf"
iteration = 99
output_prefix = "/scratch/fhgfs/ekeberg/emc/mimi/multiple_run4/split_41/"

file_parser = parse_libconfig.Parser(emc_configuration_file)
emc_output_dir = file_parser.get_option("output_dir")
if not os.path.isabs(emc_output_dir):
    emc_output_dir = os.path.dirname(emc_configuration_file)+"/"+emc_output_dir
image_dir = emc_output_dir
resp_filename = emc_output_dir+"/responsabilities_%.4d.h5" % iteration
scaling_filename = emc_output_dir+"/scaling_%.4d.h5" % iteration
rotations_filename = file_parser.get_option("rotations_file")


#output_prefix = options.output_prefix

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

def read_images(directory, number_of_images):
    images = []
    masks = []
    for i in range(number_of_images):
        try:
            image, mask = sphelper.import_spimage("%s/image_%.4d.h5" % (directory, i), ['image', 'mask'])
        except IOError:
            #print "Problem reading file %s" % ("%s/image_%.4d.h5" % (directory, i))
            raise IOError("Problem reading file %s" % ("%s/image_%.4d.h5" % (directory, i)))
        mask = bool8(mask)
        image[-mask] = 0.
        images.append(image)
        masks.append(mask)
    return array(images), bool8(array(masks))

with h5py.File(resp_filename, 'r') as resp_file_handle, h5py.File(scaling_filename, 'r') as scaling_file_handle, h5py.File(rotations_filename, 'r') as rotations_file_handle:
    # setup
    resp = resp_file_handle['data']
    scaling = scaling_file_handle['data']
    rotations = rotations_file_handle['rotations']
    number_of_images = resp.shape[0]
    number_of_rotations = resp.shape[1]
    images, masks = read_images(image_dir, number_of_images)
    image_side = images.shape[1]

    model_1 = assemble_slices.Model(image_side, inf)
    model_2 = assemble_slices.Model(image_side, inf)

    all_indices = arange(number_of_images)
    shuffle(all_indices)
    indices_1 = all_indices[:number_of_images/2]
    indices_2 = all_indices[number_of_images/2:]
    indices_1.sort()
    indices_2.sort()
    #indices_1 = arange(number_of_images, dtype='int32')

    # test
    # indices_1 = arange(number_of_images)
    # indices_2 = array([0, 1])

    images_1 = images[indices_1]
    masks_1 = masks[indices_1]
    images_2 = images[indices_2]
    masks_2 = masks[indices_2]

    all_slices = arange(number_of_rotations)
    for s in all_slices[rank::nproc]:
        print "%d: model 1 | %d (%d)" % (rank, s, number_of_rotations); sys.stdout.flush()
        #sl = zeros((image_side, )*2)
        this_resps = resp[indices_1, s]
        this_scaling = scaling[s, indices_1]
        slice_sum = (images_1*masks_1*this_resps[:, newaxis, newaxis]*this_scaling[:, newaxis, newaxis]).sum(axis=0)
        slice_weight = (masks_1*this_resps[:, newaxis, newaxis]).sum(axis=0)
        good_pixels = slice_weight > 0
        slice_sum[good_pixels] /= slice_weight[good_pixels]
        slice_sum[-good_pixels] = 0.
        model_1.insert_slice(slice_sum, good_pixels, rotations[s, :4])
        # progress.iteration_completed()

    for s in all_slices[rank::nproc]:
        print "%d: model 2 | %d (%d)" % (rank, s, number_of_rotations); sys.stdout.flush()
        #sl = zeros((image_side, )*2)
        this_resps = resp[indices_2, s]
        this_scaling = scaling[s, indices_2]
        slice_sum = (images_2*masks_2*this_resps[:, newaxis, newaxis]*this_scaling[:, newaxis, newaxis]).sum(axis=0)
        slice_weight = (masks_2*this_resps[:, newaxis, newaxis]).sum(axis=0)
        good_pixels = slice_weight > 0
        slice_sum[good_pixels] /= slice_weight[good_pixels]
        slice_sum[-good_pixels] = 0.
        model_2.insert_slice(slice_sum, good_pixels, rotations[s, :4])
        # progress.iteration_completed()


print "%d: finished calculating" % rank; sys.stdout.flush()

# send_model = zeros(model_1.get_raw_model().shape)
# send_model[:] = model_1.get_raw_model()
# send_weight = zeros(model_1.get_raw_weight().shape)
# send_weight[:] = model_1.get_raw_weight()

send_model_1 = model_1.get_raw_model().copy()
send_weight_1 = model_1.get_raw_weight().copy()
send_model_2 = model_2.get_raw_model().copy()
send_weight_2 = model_2.get_raw_weight().copy()

if rank == 0:
    # print "%d: start reduce" % (rank); sys.stdout.flush()
    final_model_1 = zeros(send_model_1.shape)
    final_weight_1 = zeros(send_weight_1.shape)
    final_model_2 = zeros(send_model_2.shape)
    final_weight_2 = zeros(send_weight_2.shape)
    comm.Reduce(send_model_1, final_model_1, op=MPI.SUM, root=0)
    comm.Reduce(send_weight_1, final_weight_1, op=MPI.SUM, root=0)
    comm.Reduce(send_model_2, final_model_2, op=MPI.SUM, root=0)
    comm.Reduce(send_weight_2, final_weight_2, op=MPI.SUM, root=0)
    print "%d: done reduce" % (rank); sys.stdout.flush()
else:
    print "%d: start reduce" % (rank); sys.stdout.flush()
    comm.Reduce(send_model_1, None, op=MPI.SUM, root=0)
    comm.Reduce(send_weight_1, None, op=MPI.SUM, root=0)
    comm.Reduce(send_model_2, None, op=MPI.SUM, root=0)
    comm.Reduce(send_weight_2, None, op=MPI.SUM, root=0)
    print "%d: done reduce" % (rank); sys.stdout.flush()

if rank == 0:
    good_indices_1 = final_weight_1 > 0
    final_model_1[-good_indices_1] = 0.
    final_model_1[good_indices_1] /= final_weight_1[good_indices_1]
    sphelper.save_spimage(final_model_1, "%s_assembled_model_1.h5" % (output_prefix), mask=int32(good_indices_1))
    # output_sp = spimage.sp_image_alloc(final_model_1.shape[0], final_model_1.shape[1], final_model_1.shape[2])
    # output_sp.image[:, :, :] = final_model_1
    # output_sp.mask[:, :, :] = int32(good_indices)

    good_indices_2 = final_weight_2 > 0
    final_model_2[-good_indices_2] = 0.
    final_model_2[good_indices_2] /= final_weight_2[good_indices_2]
    sphelper.save_spimage(final_model_2, "%s_assembled_model_2.h5" % (output_prefix), mask=int32(good_indices_2))

    savetxt("%s_indices_1.data" % (output_prefix), indices_1, fmt='%d')
    savetxt("%s_indices_2.data" % (output_prefix), indices_2, fmt='%d')

print "%d: exiting" % rank; sys.stdout.flush()
