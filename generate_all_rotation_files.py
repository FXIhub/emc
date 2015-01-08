import os
from optparse import OptionParser

default_executable = '/home/ekeberg/Work/programs/emc/build_davinci/src/utils/generate_rotations'

parser = OptionParser(usage="%prog MAX_N -e EXECUTABLE")
parser.add_option('-e', action="store", type="string", dest="executable", default=default_executable, help="Path to the generate_rotations.")
options, args = parser.parse_args()

n = int(args[0])

for i in range(n+1):
    command = '%s %d' % (options.executable, i)
    print command
    os.system(command)
