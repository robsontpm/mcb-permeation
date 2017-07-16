import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import six
from pyscirsc.gromacs.molecule import correct_pbc, correct_box
from pyscirsc.counting.msd.msd_common import MSDException
from pyscirsc.gromacs.band import Band
from pmx.model import Model
from pmx.xtc import Trajectory
from pmx.ndx import IndexFile
from pyscirsc.gromacs.index import indexes
from pyscirsc.gromacs.trajectory import iterate
from pyscirsc.utils import mkdir_p


class MSDBandCounter:
    FACTOR = {
        'xyz': 1000.0 / 6.0,
        # convert D in (nm^2/ps -> 10^-5 cm^2/s for 3D diffusion -> same as gromacs!)
        'xy': 1000.0 / 4.0,
        # convert D in (nm^2/ps -> 10^-5 cm^2/s for 2D diffusion -> same as gromacs!)
        'yz': 1000.0 / 4.0,
        # convert D in (nm^2/ps -> 10^-5 cm^2/s for 2D diffusion -> same as gromacs!)
        'xz': 1000.0 / 4.0,
        # convert D in (nm^2/ps -> 10^-5 cm^2/s for 2D diffusion -> same as gromacs!)
        'x': 1000.0 / 2.0,
        # convert D in (nm^2/ps -> 10^-5 cm^2/s for 1D diffusion -> same as gromacs!)
        'y': 1000.0 / 2.0,
        # convert D in (nm^2/ps -> 10^-5 cm^2/s for 1D diffusion -> same as gromacs!)
        'z': 1000.0 / 2.0,
        # convert D in (nm^2/ps -> 10^-5 cm^2/s for 1D diffusion -> same as gromacs!)
    }

    def __init__(self, atom_group, band, restarts, computation_ttl=250,
                 verbose=False):
        """
        :param atom_group: some AtomSelection as in pmx.atomselection
        :param band:       pyscirsc.gromacs.Band object
        :param restarts:   list of frame numbers to start new msd computation
                           (use msd select_starts ... for this purpose)
        :param verbose:    [default:False] print some info to stdout in process
        """
        # input
        self.band = band
        self.group = atom_group
        self.restarts = restarts
        self.reference_ttl = computation_ttl
        if not restarts:
            raise MSDException("No restart points given!")

        # internal
        self.next_start = 0
        self.positions = []

        # for output
        self.frames = []
        self.times = []
        self.traj_len = 0
        self.prev_pos = [(a.x[0], a.x[1], a.x[2]) for a in self.group]

        # for debug
        self.verbose = verbose
        # self.verbose = True             # TODO: for now!

        self.info = {
            "empty_intersection_count": 0,
            "empty_inband_count": 0,
        }

    def hello(self):
        s = ""
        s += "# hello, its MSDBandCounter!\n"
        s += "#    band:          %s\n" % str(self.band)
        s += "#    atom_group:    %s\n" % str([a.x for a in self.group])
        s += "#    restarts:      %s\n" % self.restarts
        s += "#    reference_ttl: %s\n" % self.reference_ttl
        s += "#    next_start:    %s\n" % self.next_start
        s += "#    positions:     %s\n" % self.positions
        s += "#    frames:        %s\n" % self.frames
        s += "#    times:         %s\n" % self.times
        s += "#    traj_len:      %s\n" % self.traj_len
        s += "#    prev_pos:      %s\n" % self.prev_pos
        s += "#    verbose:       %s\n" % self.verbose
        s += "#    info:          %s\n" % self.info
        return s

    def count(self, data, frame, fc):
        """
        Post-trajectory-update hook to be send to
        pyscirsc.gromacs.trajectory.iterate(...)
        """
        self.times.append(frame.time)
        self.frames.append(fc)
        self.traj_len += 1

        if (True or self.verbose) and fc % 100 == 0:  # TODO: usunac ostatecznie!
            print "# at frame: ", fc

        # set inband[i] = true if atom position is inside band borders
        # we test BEFORE removing pbc, as after correct_pbc it may go to
        # other periodic image... (we assume xtc trajectory is confined to BOX)
        inband = self.band.mask(self.group)
        # now correct periodicity (move atoms to their most probable position)
        box = correct_box(data)
        no_pbc = [correct_pbc(o, a.x, box) for o, a in
                  zip(self.prev_pos, self.group)]

        # append new restart position from list if
        # it was selected for this frame.
        if fc == self.restarts[self.next_start]:  # at defined intervals
            self.positions.append({
                'ttl': self.reference_ttl,  # how long it will live
                'positions': no_pbc,  # original positions of the atoms
                'inband': inband,  # mask for the atoms in band
                'x': [],  # x component
                'y': [],  # y component
                'z': [],  # z component
                # full msd is a sum of the above!
            })
            # this guarantees next_start is always a valid index in restarts
            if self.next_start < len(self.restarts) - 1:
                self.next_start += 1

        # for each element in restart positions
        # compute msd to atoms in current frame.
        # we skip atoms which are not present (wandered off the band)
        for element in self.positions:
            if element['ttl'] <= 0:  # restart positions that are already
                continue  # long enough are skipped

            x, y, z, N = 0.0, 0.0, 0.0, 0.0
            common_test = [u and v for u, v in zip(inband, element['inband'])]
            starts = element['positions']
            for atom_no, (atom, orig, common) in enumerate(zip(no_pbc, starts, common_test)):
                if common:  # if atom is both in ref position and current inband
                    dx = (atom[0] - orig[0]) ** 2
                    dy = (atom[1] - orig[1]) ** 2
                    dz = (atom[2] - orig[2]) ** 2
                    x += dx
                    y += dy
                    z += dz
                    N += 1.0

                # TODO: dodano 208.04.2017 zeby sprawdzic, czy wywalanie zadziala
                else:  
                	# is not in current band or in starting band (element), so we can do that!
                	# now, 
                	element['inband'][atom_no] = False
                # TODO: dodano dotad

            x /= 100.  # A^2 -> nm^2
            y /= 100.  # A^2 -> nm^2
            z /= 100.  # A^2 -> nm^2

            if N:
                element['x'].append(x / N)
                element['y'].append(y / N)
                element['z'].append(z / N)
            else:
                if any(inband):
                    if self.verbose:
                        msg = "# EMPTY INTERSECTION at frame={0} time={1}"
                        print msg.format(fc, frame.time)
                    self.info["empty_intersection_count"] += 1
                else:
                    if self.verbose:
                        msg = "# EMPTY INBAND at frame={0} time={1}"
                        print msg.format(fc, frame.time)
                    self.info["empty_inband_count"] += 1

                element['x'].append(None)  # at the end in accumulate
                element['y'].append(None)  # the nones will be removed
                element['z'].append(None)

            element['ttl'] -= 1

        # remember for next frame PBC correction
        self.prev_pos = no_pbc

    def accumulate(self, type, left=0.2, right=0.6):
        """
        after iterate do the accumulation of data

        :param type: string, either x, y, z, xy, xz, yz, xyz
        :param left: int or float, default: 0.2
        :param right: int or float, default: 0.6

        if left or right is float then they define relative position
        of the bound for fitting (w.r.t. msd length), if they are integers
        then they are used 'as is' (but the program check if they are greater
        than the length of the msd fitting data)
        """

        # if left == 0.2 and right == 0.6 and type == 'z':
        #     # default...
        #     # TODO: make it dependent on the direction...
        #     left = 0.05
        #     right = 0.45

        # for errors... StDev
        left = 0.1
        right = 0.6

        # put none for places where sequence is shorter, important for
        # zip in next line.
        sum_or_none = lambda (x): None if any((y is None for y in x)) else sum(x)
        for element in self.positions:
            if not type in element:
                tuples = zip(*[element[letter] for letter in type])
                element[type] = map(sum_or_none, tuples)

        fill = lambda (o): o + [None for i in range(self.reference_ttl - len(o))]
        msds = [fill(element[type]) for element in self.positions]
        # zip into lists and then remove Nones
        msds = [filter(lambda x: x is not None, list(a)) for a in zip(*msds)]
        # now we can compute true averages
        msd = map(lambda (x): sum(x) / len(x) if x else 0., msds)
        # normalize time to 0
        reftime = self.times[0]
        t = map(lambda x: x - reftime, self.times)[:len(msd)]

        # make good bounds for fitting procedure
        bound = lambda (x): min(len(t), max(0, x if isinstance(x, six.integer_types) else int(x * len(t))))
        lo, up = bound(left), bound(right)
        if lo > up:
            lo, up = up, lo

        fit, resids, rnk, s, rc = np.polyfit(t[lo:up], msd[lo:up], 1, full=True)
        fit_fn = np.poly1d(fit)
        diff_coef = fit[0] * self.FACTOR[type]
        error = resids[0] if resids else -1.

        return {
            'D': diff_coef,
            'type': type,
            't': t,
            'msd': msd,
            'fit_fn': fit_fn,
            'fit': fit,
            'error': error,  # -1. if totally bad..., big also if totally bad
        }


################################################################################
# script to be used in msd.py                                                  #
################################################################################


# helper function
def write_result_file(to_file, result_items, preamble=""):
    to_file.write(preamble)
    for key in sorted(result_items.keys()):
        to_file.write(
            "{0} {1}\n".format(
                key,
                " ".join(map(str, result_items[key]))
            )
        )
        to_file.flush()
    to_file.close()


# main function
def main(runfile_path=None):
    # You can adjust those variables
    pdb_filename   = "odnPOPC_001.pdb"  # pdb file with model
    xtc_filename   = "odnPOPC_test.xtc"  # trajectory file
    begin_frame    = 0     # start frame of the trajectory (in frames, not ps!)
    end_frame      = 1000  # end frame (if -1 then all) (in frames, not ps!)
    output_dir     = "data/output"  # output in current directory
    group_resname  = "OG"  # residue name of the selected group to compute msd
    reference_ttl  = 200   # length of msd calculation for each restart (not used at all in this script, but in msd_computation later)
    min_stay       = 5     # restart points only for atoms in band for that many consecutive frames
    min_gap        = 25    # minimal gap between restarts
    max_gap        = 50    # maximal gap between restarts
    band_dir       = 2     # 0 = x, 1 = y, 2 = z, z is default
    # band separators:
    separators = [  # computed externally (in nm!)
        0.0,         0.85879874,  1.51332498,  # as defined by Ela Plesner
        4.34488884,  5.00243812,  7.38666648,  # see Diff_tables.doc
        8.04406502, 10.8815546,  11.5530146,   # first must be 0, last is some
        25.0                                   # big number greater than the box
    ]
    separators = map(lambda x: 10 * x, separators)  # make it angstroems

    starts = {
        0: [15, 69, 97, 123, 164, 200, 229, 279, 329, 381, 408, 440, 540, 576,
            609, 636, 670, 695, 721, 749, 775, 803, 829, 869, 919, 964, 990],
        1: [1, 30, 60, 92, 120, 156, 187, 229, 256, 283, 315, 341, 381, 408,
            440, 467, 494, 520, 549, 575, 609, 634, 667, 692, 724, 749, 775,
            803, 831, 864, 894, 920, 953, 980],
        2: [1, 26, 51, 77, 104, 129, 154, 182, 209, 236, 273, 303, 330, 358,
            384, 409, 436, 461, 490, 515, 540, 576, 609, 635, 662, 694, 723,
            749, 775, 800, 829, 861, 886, 914, 939, 964, 990],
        3: [1, 51, 97, 123, 164, 196, 222, 247, 283, 329, 379, 408, 440, 490,
            540, 576, 609, 636, 670, 695, 745, 775, 803, 853, 886, 936, 964,
            990],
        4: [1, 51, 97, 123, 164, 200, 229, 279, 329, 379, 408, 440, 490, 540,
            576, 609, 636, 670, 695, 745, 775, 803, 853, 886, 936, 964, 990],
        5: [1, 38, 69, 97, 123, 164, 200, 229, 279, 306, 335, 360, 391, 419,
            446, 496, 535, 560, 595, 636, 662, 694, 724, 749, 775, 803, 837,
            863, 893, 921, 957, 983],
        6: [1, 26, 52, 81, 108, 140, 168, 195, 228, 254, 279, 309, 334, 365,
            391, 425, 450, 475, 500, 530, 557, 593, 627, 653, 679, 705, 730,
            755, 785, 811, 837, 864, 889, 914, 940, 965, 990],
        7: [1, 51, 97, 123, 164, 200, 229, 279, 329, 379, 408, 440, 490, 540,
            576, 609, 636, 670, 695, 720, 749, 775, 803, 853, 886, 936, 964,
            990],
        8: [1, 51, 89, 123, 165, 196, 225, 266, 316, 348, 377, 408, 435, 470,
            495, 524, 560, 609, 646, 679, 725, 764, 814, 864, 914, 964, 992],
    }

    # if number then we start from this layer (numbered from 0)
    continue_from = None
    stop_at = None

    # continue_from = 1  # TODO: remove after test

    print "###################################################"
    if runfile_path:
                # this is not nice, but powerfull..
        d = locals()
        exec(open(runfile_path).read(), d, d)
        # TODO: grrrrrrrrrrr
        # TODO: because of the BUG I need to do this this way, grrrrrr....
        # TODO: make it a local config, (e.g. Main class with main() method)
        # TODO: and stroe config in self -> then can automatically assign in loop
        pdb_filename  = d.get('pdb_filename', pdb_filename)
        xtc_filename  = d.get('xtc_filename', xtc_filename)
        begin_frame   = d.get('begin_frame', begin_frame)
        end_frame     = d.get('end_frame', end_frame)
        output_dir    = d.get('output_dir', output_dir)
        group_resname = d.get('group_resname', group_resname)
        reference_ttl = d.get('reference_ttl', reference_ttl)
        min_stay      = d.get('min_stay', min_stay)
        min_gap       = d.get('min_gap', min_gap)
        max_gap       = d.get('max_gap', max_gap)
        band_dir      = d.get('band_dir', band_dir)
        separators    = d.get('separators', separators)
        starts        = d.get('starts', starts)
        continue_from = d.get('continue_from', continue_from)
        stop_at       = d.get('stop_at', stop_at)

        print "# Using config from file: ", sys.argv[-1]
    else:
        print "# Using default (SAMPLE) config..."
        print "# Please review the config and make your own"
        print "# tailored to your needs."
        print "# To use your own file run: "
        print "#"
        print "#     python " + " ".join(sys.argv) + " my_file.conf"
        print "#"
        print "# my_file.conf should be a python source file"
        print "# (see help below). "

    in_conf = ""
    in_conf += "pdb_filename        = \"{0}\"\n".format(pdb_filename)
    in_conf += "xtc_filename        = \"{0}\"\n".format(xtc_filename)
    in_conf += "begin_frame         = {0}\n".format(begin_frame)
    in_conf += "end_frame           = {0}\n".format(end_frame)
    in_conf += "output_dir          = \"{0}\"\n".format(output_dir)
    in_conf += "group_resname       = \"{0}\"\n".format(group_resname)
    in_conf += "reference_ttl       = {0}\n".format(reference_ttl)
    in_conf += "min_stay            = {0}\n".format(min_stay)
    in_conf += "min_gap             = {0}\n".format(min_gap)
    in_conf += "max_gap             = {0}\n".format(max_gap)
    in_conf += "band_dir            = {0}\n".format(band_dir)
    in_conf += "separators          = {0}\n".format(separators)
    in_conf += "starts              = {0}\n".format(str(starts))
    in_conf += "continue_from       = {0}\n".format(continue_from)
    in_conf += "stop_at             = {0}\n".format(stop_at)

    help_str = ""
    help_str += "###################################################\n"
    help_str += "# Command issued was:\n"
    help_str += "###################################################\n"
    help_str += "#     python " + " ".join(sys.argv) + "\n"
    help_str += "###################################################\n"
    help_str += "# Config file used:\n"
    help_str += "###################################################\n"
    help_str += "# (for meaning of variables see script file) ######\n"
    help_str += "###################################################\n"
    for item in in_conf.split('\n'):
        if item:  # no empty lines
            help_str += "# {0}\n".format(item)
    help_str += "###################################################\n"
    help_str += "# (input file ends here) ##########################\n"
    help_str += "# Remember to uncomment variables if want to use  #\n"
    help_str += "# them in a rerun. They are commented for gnuplot #\n"
    help_str += "# (with default gnuplot comment character '#'     #\n"
    help_str += "###################################################\n"

    # print help / summary
    print help_str

    # make output directory structure
    mkdir_p("{0}".format(output_dir))  # TODO: backup?

    types = ['xyz', 'xy', 'x', 'y', 'z']

    continue_from = continue_from or 0
    output_filename = "{0}/msd_band.dat".format(output_dir)

    result_data = {}
    error_data = {}
    if continue_from:
        try:
            result_file = file(output_filename, "r")
        except :
            print "# WARNING: No input file... "
            print "#          Requested file: {0}".format(output_filename)
            print "#          continuing values from scratch. "
        else:
            for line in result_file:
                if line[0] == '#':
                    pass   # skip comments (will be regenerated)
                else:
                    items = line.strip().split()
                    if len(items):
                        try:
                            layer_no = int(items[0])
                        except ValueError:
                            pass  # skip bad lines
                        else:
                            result_data[layer_no] = map(float, items[1:])

        try:
            error_file = file(output_filename + ".errors", "r")
        except :
            print "# WARNING: No errors file... "
            print "#          Requested file: {0}".format(output_filename + ".errors")
            print "#          continuing errors from scratch. "
        else:
            for line in error_file:
                if line[0] == '#':
                    pass   # skip comments (will be regenerated)
                else:
                    items = line.strip().split()
                    if len(items):
                        try:
                            layer_no = int(items[0])
                        except ValueError:
                            pass  # skip bad lines
                        else:
                            error_data[layer_no] = map(float, items[1:])

    result_preamble = ""
    result_preamble += help_str + "# \n"
    result_preamble += "# Headers: \n"
    result_preamble += "# layer "
    result_preamble += " ".join(("D{0}[10-5cm^2/s]".format(tp) for tp in types))
    result_preamble += "\n"

    result_file = file(output_filename, "w")
    write_result_file(result_file, result_data, result_preamble)

    # read data model
    data = Model(pdb_filename)

    # prepare useful indexes
    group_ndx_filename = "{0}/{1}.ndx".format(output_dir, group_resname)
    if not os.path.isfile(group_ndx_filename):
        for resname, ndx in indexes(data):
            ndx.write("{0}/{1}.ndx".format(output_dir, resname))

    # get the index for selected group resname
    # Notice: there is only one group in groups for this index!
    # we work on group to be faster
    selected_ndx = IndexFile(group_ndx_filename)

    stats_str = ""
    left_bound = separators[continue_from]
    for layer_no, right_bound in enumerate(separators[continue_from + 1:]):
        layer_no += continue_from   # make right number

        band = Band(left_bound, right_bound, band_dir)
        msg = '# Started at layer: {0} [{1}, {2}], direction: {3}\n'
        print msg.format(layer_no, left_bound, right_bound, band_dir)
        left_bound = right_bound   # for the next iteration

        try:
            # reload input data!
            initial_model = Model(pdb_filename)
            trajectory = Trajectory(xtc_filename)
            group = selected_ndx.groups[0].select_atoms(initial_model)

            counter = MSDBandCounter(group, band, starts[layer_no], reference_ttl)
            # print counter.hello()
            iterate(initial_model, trajectory, begin_frame, end_frame, counter.count)

            msd = dict()
            together = []
            for type in types:
                msd[type] = counter.accumulate(type)

                times = msd[type]['t']
                data = msd[type]['msd']
                fit = msd[type]['fit_fn']
                # we have columns, but we want rows for output, so we
                # remember data, and we will transpose it later
                if type == types[0]:  # append time only once
                    together.append(times)  # append time only once
                together.append(data)

                # draw data to png
                plt.plot(times, data, '-k', times, fit(times), '-r')
                name_format = "{0}/msd_{1}_layer_{2:03d}.png"
                plt.savefig(name_format.format(output_dir, type, layer_no))
                plt.close()

            # put to file with commments for future use
            out = file("{0}/msd_layer_{1:03d}.dat".format(output_dir, layer_no), "w")
            data = zip(*together)
            out.write(help_str)
            out.write("# data in [nm^2]\n")
            out.write("# t " + " ".join(types) + "\n")
            for o in data:
                out.write(" ".join(map(str, o)) + "\n")
            out.close()

            loc_stats_str  = '# Layer: {0} [{1}, {2}] dir = {3}\n'.format(
                layer_no, left_bound, right_bound, band_dir
            )
            loc_stats_str += '# D_xyz, D_xy:   {0} {1}\n'.format(msd['xyz']['D'], msd['xy']['D'])
            loc_stats_str += '# D_x, D_y, D_z: {0} {1} {2}\n'.format(msd['x']['D'], msd['y']['D'], msd['z']['D'])
            if reference_ttl <= 1000:  # longer trajectories would grow files too much
                loc_stats_str += '# t:          {0}\n'.format(str(msd['xyz']['t']))
                loc_stats_str += '# msd_xyz(t): {0}\n'.format(str(msd['xyz']['msd']))
                loc_stats_str += '# msd_x(t):   {0}\n'.format(str(msd['x']['msd']))
                loc_stats_str += '# msd_y(t):   {0}\n'.format(str(msd['y']['msd']))
                loc_stats_str += '# msd_z(t):   {0}\n'.format(str(msd['z']['msd']))

            print loc_stats_str
            stats_str += loc_stats_str

            # # for safety we iterate over list, that guarantee order
            # result_file.write("{0} {1}\n".format(
            #     layer_no,
            #     " ".join([str(msd[tp]['D']) for tp in types]))
            # )
            # result_file.flush()

            # for safety we iterate over list, that guarantee order
            result_data[layer_no] = [msd[tp]['D'] for tp in types]
            result_file = file(output_filename, "w")
            write_result_file(result_file, result_data, result_preamble)

            error_data[layer_no] = [msd[tp]['error'] for tp in types]
            result_file = file(output_filename + ".errors", "w")
            write_result_file(result_file, error_data, result_preamble)

            if runfile_path:
                cont_file = file(runfile_path + "_{0:03d}".format(layer_no + 1), "w")
                cont_file.write(help_str)
                cont_file.write(in_conf)
                # this will effectively overwrite wariable set in the in_conf.
                # (I do not want to copy-paste in_conf gen code here)
                # so this is workaround.
                cont_file.write("continue_from = " + str(layer_no + 1))

            # TODO: save setup for continue after this step.
            if stop_at and layer_no == stop_at:
                break  # from over layer loop

        except MSDException as e:
            print "# EXCEPTION at LAYER {0}: ".format(layer_no), e

    print "DONE! See {0} for results.".format(output_filename)
    result_file.close()
