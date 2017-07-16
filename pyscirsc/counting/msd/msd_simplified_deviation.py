import os
import sys
import numpy as np
import math
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


class MSDSimplifiedDeviation:

    def __init__(self,
                 atom_group, bands, restarts,
                 computation_ttl=[10,100],
                 verbose=False):
        """
        :param atom_group: some AtomSelection as in pmx.atomselection
        :param band:       pyscirsc.gromacs.Band object
        :param restarts:   list of frame numbers to start new msd computation
                           (use msd select_starts ... for this purpose)
        :param verbose:    [default:False] print some info to stdout in process
        """
        # input
        # for i, b in enumerate(bands):
        #     print i, b.left, b.right
        # for i, r in enumerate(restarts):
        #     print i, r
        self.bands = bands
        self.group = atom_group
        self.restarts = restarts
        self.computation_ttl = computation_ttl
        self.starts = [map(lambda x: x + computation_ttl[0], r) for r in self.restarts]
        # self.finishes = [map(lambda x: x + computation_ttl[1], r) for r in self.restarts]
        self.finishes = [[] for r in self.restarts]
        self.current_restart = [0 for band in self.bands]
        self.current_start = [0 for band in self.bands]
        self.current_finish = [0 for band in self.bands]
        self.xs = [[] for band in self.bands]
        self.ys = [[] for band in self.bands]
        self.zs = [[] for band in self.bands]
        self.xf = [[] for band in self.bands]
        self.yf = [[] for band in self.bands]
        self.zf = [[] for band in self.bands]
        self.ref_positions = [[] for band in self.bands]
        self.times = []
        self.frames = []
        self.traj_len = 0
        self.verbose = True
        self.prev_pos = [(a.x[0], a.x[1], a.x[2]) for a in self.group]
        if not restarts:
            raise MSDException("No restart points given!")

    def hello(self):
        s = ""
        s += "# hello, its MSDSimplifiedDeviation!\n"
        s += "#    band:          %s\n" % str(self.band)
        s += "#    atom_group:    %s\n" % str([a.x for a in self.group])
        s += "#    restarts:      %s\n" % self.restarts
        s += "#    reference_ttl: %s\n" % self.reference_ttl
        return s

    def count(self, data, frame, fc):
        """
        Post-trajectory-update hook to be send to
        pyscirsc.gromacs.trajectory.iterate(...)
        """
        self.times.append(frame.time)
        self.frames.append(fc)
        self.traj_len += 1

        if (True or self.verbose) and fc % 100 == 0:
            print "# at frame: ", fc

        # if fc not in [self.restarts[r] for r in self.current_restart] and \
        #     fc not in [self.finishes[r] for r in self.current_finish]:
        #     return

        inband = []
        for band in self.bands:

            # set inband[i] = true if atom position is inside band borders
            # we test BEFORE removing pbc, as after correct_pbc it may go to
            # other periodic image... (we assume xtc trajectory is confined to BOX)
            inband.append(band.mask(self.group))

        # now correct periodicity (move atoms to their most probable position)
        box = correct_box(data)
        no_pbc = [correct_pbc(o, a.x, box) for o, a in zip(self.prev_pos, self.group)]

        for bandidx, band in enumerate(self.bands):
            # if fc != self.restarts[bandidx][self.current_restart[bandidx]] and \
            #     fc != [self.finishes[bandidx][self.current_finish[bandidx]]]:
            #     continue

            # append new restart position from list if
            # it was selected for this frame.
            if fc == self.restarts[bandidx][self.current_restart[bandidx]]:  # at defined intervals
                self.ref_positions[bandidx].append({
                    'positions': no_pbc,  # original positions of the atoms
                    'inband': inband[bandidx],     # mask for the atoms in band
                })
                # this guarantees is always a valid index in restarts
                if self.current_restart[bandidx] < len(self.restarts[bandidx]) - 1:
                    self.current_restart[bandidx] += 1

            if fc == self.starts[bandidx][self.current_start[bandidx]] or \
                (self.current_finish[bandidx] < len(self.finishes[bandidx]) and fc == self.finishes[bandidx][self.current_finish[bandidx]]):
                # at defined intervals

                if fc == self.starts[bandidx][self.current_start[bandidx]]:
                    ref = self.ref_positions[bandidx][self.current_start[bandidx]]
                else:
                    ref = self.ref_positions[bandidx][self.current_finish[bandidx]]

                x, y, z, N = 0.0, 0.0, 0.0, 0.0
                common_test = [u and v for u, v in zip(inband[bandidx], ref['inband'])]
                starts = ref['positions']
                for atom, orig, common in zip(no_pbc, starts, common_test):
                    if common:  # if atom is both in ref position and current inband
                        dx = (atom[0] - orig[0]) ** 2
                        dy = (atom[1] - orig[1]) ** 2
                        dz = (atom[2] - orig[2]) ** 2
                        x += dx
                        y += dy
                        z += dz
                        N += 1.0

                x /= 100.  # A^2 -> nm^2
                y /= 100.  # A^2 -> nm^2
                z /= 100.  # A^2 -> nm^2

                if N:
                    if fc == self.starts[bandidx][self.current_start[bandidx]]:
                        self.xs[bandidx].append(x / N)
                        self.ys[bandidx].append(y / N)
                        self.zs[bandidx].append(z / N)
                        print "appending start at", fc
                        self.finishes[bandidx].append(fc - self.computation_ttl[0] + self.computation_ttl[1])
                    else:
                        self.xf[bandidx].append(x / N)
                        self.yf[bandidx].append(y / N)
                        self.zf[bandidx].append(z / N)
                        print "finished at", fc

                else:
                    if any(inband[bandidx]):
                        if self.verbose:
                            msg = "# EMPTY INTERSECTION at frame={0} time={1} bandidx={2}"
                            print msg.format(fc, frame.time, bandidx)
                    else:
                        if self.verbose:
                            msg = "# EMPTY INBAND at frame={0} time={1} bandidx={2}"
                            print msg.format(fc, frame.time, bandidx)

                if fc == self.starts[bandidx][self.current_start[bandidx]]:
                    if self.current_start[bandidx] < len(self.starts[bandidx]) - 1:
                        self.current_start[bandidx] += 1
                else:
                    # this guarantees is always a valid index in finishes
                    if self.current_finish[bandidx] < len(self.finishes[bandidx]) - 1:
                        self.current_finish[bandidx] += 1

        # remember for next frame PBC correction
        self.prev_pos = no_pbc


################################################################################
# script to be used in msd.py                                                  #
################################################################################


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

    print "###################################################"
    if runfile_path:
                # this is not nice, but powerfull..
        d = dict()
        # exec (open(runfile_path).read(), d, d)
        exec open(runfile_path).read() in d, d
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
    in_conf += "end_frame           = {0}\n".format(end_frame)

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

    reference_ttl = [10, 100]  # dla moich przykladow...

    # make output directory structure
    mkdir_p("{0}".format(output_dir))  # TODO: backup?

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


    bands = []
    left_bound = separators[0]
    for right_bound in separators[1:]:
        bands.append(Band(left_bound, right_bound, band_dir))
        left_bound = right_bound

    restarts = [[] for i in range(max(starts.keys())+1)]
    for key, r in starts.iteritems():
        restarts[key] = r
    # reload input data!
    initial_model = Model(pdb_filename)
    trajectory = Trajectory(xtc_filename)
    group = selected_ndx.groups[0].select_atoms(initial_model)

    counter = MSDSimplifiedDeviation(group, bands, restarts, reference_ttl)
    iterate(initial_model, trajectory, begin_frame, end_frame, counter.count)

    print "end_frame           = {0}\n".format(end_frame)

    print "AAAAAAAAAA"
    print counter.xs
    print counter.xf
    print counter.ys
    print counter.yf
    print counter.zs
    print counter.zf
    print "BBBBBBBBBB"

    for layer_no, band in enumerate(bands):
        # put to file with commments for future use
        out = file("{0}/deviation_layer_{1:03d}.csv".format(output_dir, layer_no), "w")
        xys = map(lambda a: sum(a), zip(counter.xs[layer_no], counter.ys[layer_no]))
        xyzs = map(lambda a: sum(a), zip(counter.xs[layer_no], counter.ys[layer_no], counter.zs[layer_no]))
        xyf = map(lambda a: sum(a), zip(counter.xf[layer_no], counter.yf[layer_no]))
        xyzf = map(lambda a: sum(a), zip(counter.xf[layer_no], counter.yf[layer_no], counter.zf[layer_no]))

        def simple_fit(t0, x0, t1, x1):
            return (x1 - x0) / float(t1 - t0)

        fit_fn = lambda x: simple_fit(reference_ttl[0], x[0], reference_ttl[1], x[1])
        xy = map(fit_fn, zip(xys, xyf))
        xyz = map(fit_fn, zip(xyzs, xyzf))
        x = map(fit_fn, zip(counter.xs[layer_no], counter.xf[layer_no]))
        y = map(fit_fn, zip(counter.ys[layer_no], counter.yf[layer_no]))
        z = map(fit_fn, zip(counter.zs[layer_no], counter.zf[layer_no]))

        xyz = [a * 1000.0 / 6.0 for a in xyz if a > 0]
        xy = [a * 1000.0 / 4.0 for a in xy if a > 0]
        x = [a * 1000.0 / 2.0 for a in x if a > 0]
        y = [a * 1000.0 / 2.0 for a in y if a > 0]
        z = [a * 1000.0 / 2.0 for a in z if a > 0]
        mlen = min(map(len, [xyz, xy, x, y, z]))
        if mlen:
            xyz = xyz[:mlen]
            xy = xy[:mlen]
            x = x[:mlen]
            y = y[:mlen]
            z = z[:mlen]
        else:
            xyz = []
            xy = []
            x = []
            y = []
            z = []

        def sddev(lst):
            if len(lst):
                av = sum(lst) / len(lst)
                var = sum(map(lambda x: (x - av) ** 2, lst)) / len(lst)
                return math.sqrt(var)
            else:
                return -1.0

        stdevs = map(sddev, [xyz, xy, x, y, z])

        data = zip(xyz, xy, x, y, z)
        # out.write(help_str)
        # out.write("#data in [nm^2]\n")
        out.write(" ".join(map(str, stdevs)) + "\n\n")
        out.write("#xyz xy x y z\n")
        for o in data:
            if all([x > 0 for x in o]):
                out.write(" ".join(map(str, o)) + "\n")

        out.close()

    print "DONE!"
