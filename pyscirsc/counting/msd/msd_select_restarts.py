import os
import sys
from pyscirsc.counting.msd.msd_common import MSDException
from pyscirsc.gromacs.band import Band
from pmx.model import Model
from pmx.xtc import Trajectory
from pmx.ndx import IndexFile
from pyscirsc.gromacs.index import indexes
from pyscirsc.gromacs.trajectory import iterate
from pyscirsc.utils import mkdir_p


class MSDRestartSelector:

    def __init__(self, atom_group, band, min_stay, gap_conf, verbose=False):
        """
        :param atom_group: some AtomSelection as in pmx.atomselection
        :param band:       pyscirsc.gromacs.Band object
        :param min_stay:   atom should stay in band this many frames to be
                           considered as possible restart point.
        :param gap_conf:   minimal/maximal gap between restart points (selector
                           will add additional restart points if possible)
        :param verbose:    [default:False] print some info to stdout in process
        """
        # for both
        self.band = band
        self.group = atom_group
        self.min_stay = min_stay
        self.min_gap = gap_conf[0]
        self.max_gap = gap_conf[1]

        # for output
        self.enters = [[0 for a in atom_group]]  # in enters we store how long
        self.frames = []                         # given atom stayed until given
        self.traj_len = 0                        # frame.
        self.starts = []
        self.times = []

        # for debug
        self.verbose = verbose
        self.verbose = True             # TODO: for now!

    def select(self, data, frame, fc):
        """
        Post-trajectory-update hook to be send to
        pyscirsc.gromacs.trajectory.iterate(...)
        """
        self.times.append(frame.time)
        self.frames.append(fc)
        self.traj_len += 1

        if self.verbose and fc % 100 == 0:
            print "# at frame: ", fc

        inband = self.band.mask(self.group)

        new_enters = []
        for i, flag in enumerate(inband):
            if flag:
                new_enters.append(self.enters[-1][i] + 1)
            else:
                new_enters.append(0)

        self.enters.append(new_enters)

    def filter(self):
        """
        after iterate do the selection of restart points.
        """

        # 1. select those which stays at least min_stay frames
        inband = [False for a in self.group]
        for atom_lists in self.enters:
            for i, o in enumerate(atom_lists):
                if o == self.min_stay:
                    inband[i] = True

        if any(inband):
            self.starts = []
            for i, lst in enumerate(self.enters):
                if not self.starts:
                    # add any fitting to starts
                    if any((x == self.min_stay for x in lst)):
                        self.starts.append(i - self.min_stay)
                else:
                    start = i - self.min_stay
                    if start - self.starts[-1] >= self.max_gap:
                        # if gap too big then add any >= min_stay (may include given
                        # atom many times if he is only one that enters)
                        if any((x >= self.min_stay for x in lst)):
                            self.starts.append(i - self.min_stay)
                    if start - self.starts[-1] >= self.min_gap:
                        # if gap big enough find only those that has == (should
                        # create restart point only for 'new' atoms)
                        if any((x == self.min_stay for x in lst)):
                            self.starts.append(i - self.min_stay)

            # count how many times given atom will be present in computations
            # (it will when it is present at a given restart position)
            occurence = [0 for a in self.group]
            for p in self.starts:
                for i, a in enumerate(self.enters[p]):
                    if inband[i] and a > 0:
                        occurence[i] += 1

            occurence = [(i, o) for i, o in enumerate(occurence) if inband[i]]
            evenness = sum(map(lambda pair: float(abs(pair[0]-pair[1])), [(a[1], b[1]) for a in occurence for b in occurence])) / float(len(occurence) ** 2)
            representativeness = 1.0 - sum((1. for i, o in occurence if o == 0)) / float(len(occurence))

            # convert starts (internal indices) to frames
            self.starts = [self.frames[p] for p in self.starts]

            return self.starts, {
                'len': len(self.starts),
                'coverage': float(len(self.starts))/float(self.traj_len),
                'exp_coverage': 2.0 / float(self.max_gap + self.min_gap),
                'occurence': occurence,
                'evenness': evenness,
                'representativeness': representativeness,
            }
        else:
            msg = "# No atom in group entered the band! (Band: [{0}, {1}], {2})"
            raise MSDException(msg.format(self.band.left, self.band.right, self.band.direction))


################################################################################
# computation script starts here                                               #
################################################################################

def main(runfile_path=None):

    # You can adjust those variables
    pdb_filename    = "odnPOPC_001.pdb"     # pdb file with model
    xtc_filename    = "odnPOPC_test.xtc"    # trajectory file
    begin_frame     = 0              # start frame of the trajectory (in frames, not ps!)
    end_frame       = 1000           # end frame (if -1 then all) (in frames, not ps!)
    output_dir      = "data/output"  # output in current directory
    group_resname   = "OG"           # residue name of the selected group to compute msd
    reference_ttl   = 200            # length of msd calculation for each restart (not used at all in this script, but in msd_computation later)
    min_stay        = 5              # restart points only for atoms in band for that many consecutive frames
    min_gap         = 25             # minimal gap between restarts
    max_gap         = 50             # maximal gap between restarts
    band_dir        = 2              # 0 = x, 1 = y, 2 = z, z is default
    # band separators:
    separators = [                                # computed externally (in nm!)
        0.0,         0.85879874,     1.51332498,  # as defined by Ela Plesner
        4.34488884,  5.00243812,     7.38666648,  # see Diff_tables.doc
        8.04406502, 10.8815546,     11.5530146,   # first must be 0, last is some
        25.0                                      # big number greater than the box
    ]
    separators = map(lambda x: 10*x, separators)  # make it angstroems

    starts           = None         # output dictionary, if None then do from start, if exists, then start from layer len(starts)

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

    output_filename = "{0}/msd_band.setup".format(output_dir)

    # make output directory structure
    mkdir_p("{0}".format(output_dir))       # TODO: backup?

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
    group = selected_ndx.groups[0].select_atoms(data)

    stats_str = ""
    starts_per_layer = starts or {}
    continue_from = len(starts_per_layer)
    left_bound = separators[continue_from]
    for layer_no, right_bound in enumerate(separators[continue_from+1:]):
        layer_no += continue_from

        band = Band(left_bound, right_bound, band_dir)
        msg = '# Started at layer: {0} [{1}, {2}], direction: {3}\n'
        print msg.format(layer_no, left_bound, right_bound, band_dir)
        left_bound = right_bound   # for the next iteration

        try:
            # reload input data!
            data = Model(pdb_filename)
            trajectory = Trajectory(xtc_filename)
            group = selected_ndx.groups[0].select_atoms(data)

            selector = MSDRestartSelector(group, band, min_stay, (min_gap, max_gap))
            iterate(data, trajectory, begin_frame, end_frame, selector.select)

            starts, stats = selector.filter()

            loc_stats_str  = '# Layer: {0} [{1}, {2}] dir = {3}\n'.format(
                layer_no, left_bound, right_bound, band_dir
            )
            loc_stats_str += '# number of restarts: {0}\n'.format(stats['len'])
            loc_stats_str += '# coverage:           {0} (should be close to {1})\n'.format(stats['coverage'], stats['exp_coverage'])
            loc_stats_str += '# evennes:            {0} (shoud be small)\n'.format(stats['evenness'])
            loc_stats_str += '# representativeness: {0} (shoud be close to 1.0)\n'.format(stats['representativeness'])
            loc_stats_str += "# occurences:\n#     {0}\n".format(stats['occurence'])
            loc_stats_str += "# starts:\n#     {0}\n".format(starts)

            print loc_stats_str
            stats_str += loc_stats_str

            starts_per_layer[layer_no] = starts

            # everything went good - rewrite the output file
            result_file = file(output_filename, "w")
            result_file.write(help_str)
            result_file.write("###################################################\n")
            result_file.write("# Config for msd in band starts here. \n")
            result_file.write("# For next step (compute msd) you can run: \n")
            result_file.write("#\n")
            result_file.write("#    python msd.py band {0}\n".format(output_filename))
            result_file.write("#\n")
            result_file.write("# To continue computation (i.e. failure) you can: \n")
            result_file.write("#\n")
            result_file.write("#    python msd.py select_restarts {0}\n".format(output_filename))
            result_file.write("#\n")
            result_file.write("###################################################\n")
            result_file.write(in_conf)
            result_file.write("starts = {\n")
            for key, items in starts_per_layer.iteritems():
                result_file.write("    {0}: {1}, \n".format(key, str(items)))
            result_file.write("}\n")
            result_file.write("###################################################\n")
            result_file.write(stats_str)
            result_file.close()
        except MSDException as e:
            print "# EXCEPTION at LAYER {0}: ".format(layer_no), e

    print "DONE! See {0} for results.".format(output_filename)
