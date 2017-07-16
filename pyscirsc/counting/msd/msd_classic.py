import sys
from pmx.model import Model
from pmx.xtc import Trajectory
from matplotlib import pyplot as plt
from pmx.ndx import IndexFile
from pyscirsc.gromacs.index import indexes, separate_indexes
from pyscirsc.gromacs.molecule import correct_pbc
from pyscirsc.gromacs.trajectory import iterate
from pyscirsc.utils import mkdir_p

# TODO: rewrite it!

# You can adjust those variables
pdb_filename    = "odnPOPC_001.pdb"     # pdb file with model
xtc_filename    = "odnPOPC_test.xtc"    # trajectory file
group_resname   = "OG"           # residue name of the selected group to compute msd
begin_frame     = 0              # start frame of the trajectory (in frames, not ps)
end_frame       = -1             # end frame (if -1 then all) (in frames, not ps)
trestart        = 50             # at each t = k * trestart (k in \N) set new reference for all selected atoms (in ps, see trestart in gromacs)
output_dir      = "data/output"  # output in current directory
separators_unit = "nm"
reference_ttl   = 1000

print "###################################################"
if len(sys.argv) >= 2:
    runfile_path = sys.argv[-1]
    exec(open(runfile_path).read(), globals())
    print "Using config from file: ", sys.argv[-1]
else:
    print "Using default config..."

h = ""
h += "###################################################\n"
h += "# Command issued was:\n"
h += "###################################################\n"
h += "#     python " + " ".join(sys.argv) + "\n"
h += "###################################################\n"
h += "# Config file is:\n"
h += "###################################################\n"
h += "# (for meaning of variables see script file) ######\n"
h += "###################################################\n"
h += "# pdb_filename        = \"{0}\"\n".format(pdb_filename)
h += "# xtc_filename        = \"{0}\"\n".format(xtc_filename)
h += "# group_resname       = \"{0}\"\n".format(group_resname)
h += "# begin_frame         = {0}\n".format(begin_frame)
h += "# end_frame           = {0}\n".format(end_frame)
h += "# separators_unit     = \"{0}\"\n".format(separators_unit)
h += "# reference_ttl       = {0}\n".format(reference_ttl)
h += "# trestart            = {0}\n".format(trestart)
h += "# output_dir          = \"{0}\"\n".format(output_dir)
h += "###################################################\n"
h += "# (input file ends here) ##########################\n"
h += "# Remember to uncomment variables if want to use  #\n"
h += "# them in a rerun. They are commented for gnuplot #\n"
h += "# (with default gnuplot comment character '#'     #\n"
h += "###################################################\n"

print h

if separators_unit not in ["nm", "A"]:
    raise Exception("separators_unit should be either nm or A (Angstroem) but is {0}".format(separators_unit))

# make output directory structure
mkdir_p("{0}".format(output_dir))

# read data model
data = Model(pdb_filename)
# load trajectory to iterate
trajectory = Trajectory(xtc_filename)

# prepare useful indexes
for resname, ndx in indexes(data):
    ndx.write("{0}/{1}.ndx".format(output_dir, resname))
r, index_file = separate_indexes(data, group_resname, 2)
index_file.write("{0}/OG_parted.ndx".format(output_dir))

# get the index for selected group resname
# Notice: there is only one group in groups for this index!
# we work on group to be faster
selected_ndx = IndexFile("{0}/{1}.ndx".format(output_dir, group_resname))
group = selected_ndx.groups[0].select_atoms(data)


# helper class to hold restarts
class Reference:
    def __init__(self):
        self.positions = []
        self.next_ref_time = 0.


# global data to store items from msd_step (I know, global = bad coding ;))
reference = Reference()
times = []
prev_pos = {
    'data': [(a.x[0], a.x[1], a.x[2]) for a in group]
}


# it will be passed to iterate to compute custom msd
def msd_step(data, frame, fc):

    # print data.box

    times.append(frame.time)
    if fc % 50 == 0:
        print fc

    box = (10. * data.box[0][0], 10. * data.box[1][1], 10. * data.box[2][2])
    no_pbc = [correct_pbc(o, a.x, box) for o, a in zip(prev_pos['data'], group)]

    if frame.time >= reference.next_ref_time:
        reference.positions.append({
            'ttl': reference_ttl,
            'positions': no_pbc,
            'msd': [],
            'msd_x': [],
            'msd_y': [],
            'msd_z': [],
        })
        reference.next_ref_time = frame.time + trestart

    for element in reference.positions:
        # print element
        if element['ttl'] > 0:
            x, y, z, N = 0.0, 0.0, 0.0, 0.0
            for atom, orig in zip(no_pbc, element['positions']):
                dx = (atom[0]-orig[0])**2
                dy = (atom[1]-orig[1])**2
                dz = (atom[2]-orig[2])**2
                x += dx
                y += dy
                z += dz
                N += 1.0

            x /= 100.  # A -> nm
            y /= 100.  # A -> nm
            z /= 100.  # A -> nm

            element['msd'].append((x + y + z)/N)
            element['msd_x'].append(x/N)
            element['msd_y'].append(y/N)
            element['msd_z'].append(z/N)
            element['ttl'] -= 1

    prev_pos['data'] = no_pbc


# iterate over frames
iterate(data, trajectory, begin_frame, end_frame, msd_step)

print "##############################################"
print "# trajectory iteration finished, saving data #"
print "##############################################"

# test print!
print reference_ttl, map(len, (element['msd'] for element in reference.positions))


def summary_msd(msd_type):
    # put none for places where sequence is shorter - iportant for zip in next line
    msds = [
        element[msd_type] + [None for i in range(reference_ttl - len(element[msd_type]))]
        for element in reference.positions
    ]
    # zip into lists and then remove Nones
    msds = [filter(lambda x: x is not None, list(a)) for a in zip(*msds)]
    # now we can compute true avarages
    msd = map(lambda(x): sum(x) / len(x), msds)
    return msd


msd = summary_msd('msd')
msd_x = summary_msd('msd_x')
msd_y = summary_msd('msd_y')
msd_z = summary_msd('msd_z')

# normalize time to 0
reftime = times[0]
times = map(lambda x: x - reftime, times)[:len(msd)]

# put to file with commments for future use
output_file = file("{0}/msd.dat".format(output_dir), "w")
output_file.write(h)
output_file.write("#\n# data starts here...\n#\n")
output_file.write("# t[ps] msd[nm2] x[nm2] y[nm2] z[nm2]\n")
for t, m, x, y, z in zip(times, msd, msd_x, msd_y, msd_z):
    output_file.write("{0} {1} {2} {3} {4}\n".format(t, m, x, y, z))
output_file.close()

# gnuplot it
plt.plot(times, msd)
plt.show()
