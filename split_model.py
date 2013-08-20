#!/usr/bin/env python
from pylab import *
from optparse import OptionParser
import h5py
import assemble_slices
import sphelper
import time_tools
import sys
from mpi4py import MPI

parser = OptionParser(usage="%prog IMAGE_DIR RESP_FILE SCALING_FILE")
options, args = parser.parse_args()

# image_dir = args[0]
# resp_filename = args[1]
# scaling_filename = args[2]
# rotations_filename = args[3]
# image_dir = "/scratch/fhgfs/ekeberg/emc/multiple_run2/run_30/debug/"
# resp_filename = "/scratch/fhgfs/ekeberg/emc/multiple_run2/run_30/output/responsabilities_0099.h5"
# scaling_filename = "/scratch/fhgfs/ekeberg/emc/multiple_run2/run_30/output/scaling_0099.h5"
# rotations_filename = "/home/ekeberg/Work/programs/emc/rotations/rotations_20.h5"

image_dir = "/home/ekeberg/Work/programs/emc/my_emc_runs/mimi/multi_run/run_13/debug/"
resp_filename = "/home/ekeberg/Work/programs/emc/my_emc_runs/mimi/multi_run/run_13/output/responsabilities_0099.h5"
scaling_filename = "/home/ekeberg/Work/programs/emc/my_emc_runs/mimi/multi_run/run_13/output/scaling_0099.h5"
rotations_filename = "/home/ekeberg/Work/programs/emc/rotations/rotations_20.h5"

# image_dir = "/home/ekeberg/Work/programs/emc/my_emc_runs/mimi/multi_run/coarse_run/run_5/debug/"
# resp_filename = "/home/ekeberg/Work/programs/emc/my_emc_runs/mimi/multi_run/coarse_run/run_5/output/responsabilities_0099.h5"
# scaling_filename = "/home/ekeberg/Work/programs/emc/my_emc_runs/mimi/multi_run/coarse_run/run_5/output/scaling_0099.h5"
# rotations_filename = "/home/ekeberg/Work/programs/emc/rotations/rotations_10.h5"

output_prefix = "split_model/coarse"

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
    sphelper.save_spimage(final_model_1, "%s_assembled_model_1.h5" % (output_prefix))

    good_indices_2 = final_weight_2 > 0
    final_model_2[-good_indices_2] = 0.
    final_model_2[good_indices_2] /= final_weight_2[good_indices_2]
    sphelper.save_spimage(final_model_2, "%s_assembled_model_2.h5" % (output_prefix))

    savetxt("%s_indices_1.data" % (output_prefix), indices_1, fmt='%d')
    savetxt("%s_indices_2.data" % (output_prefix), indices_2, fmt='%d')

print "%d: exiting" % rank; sys.stdout.flush()
