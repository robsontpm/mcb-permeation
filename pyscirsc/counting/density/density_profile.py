import copy
import json  # to save/load
import sys
from matplotlib import pyplot as plt
from pmx.model import Model
from pmx.xtc import Trajectory
import six
from pyscirsc.gromacs.trajectory import iterate
from pyscirsc.utils import mkdir_p
from pyscirsc.gromacs.index import indexes


class DensityProfile:
    """
    A class to compute density profile across z axis in pmx.Model (gromacs)
    """

    def __init__(self, bin_count, data, count_data=False):
        """
        :param bin_count: int, how many bins it should use
        :param data:      pmx.model.Model, with initial configurations
                          used to determine resnames
        """
        self.bin_count = bin_count
        self.counts = dict(
            (name, [0 for i in xrange(self.bin_count)]) for name in
            set(r.resname for r in data.residues)
        )
        self.bins = [i for i in xrange(self.bin_count)]
        if count_data:
            self.count(data)

        self.history = []  # for redoing normalization
        self.frames = 0

        self.global_volumes = dict(
            (name, [0.0 for i in xrange(self.bin_count)]) for name in
            set(r.resname for r in data.residues)
        )

    def count(self, data):
        """
        Put data from a frame to bins based on z-position
        :param data: pmx.model.Model, with current positions
                    (after trajectory.update)
        """
        # do profile for system stored in data pmx.Model
        # we assume that box is cuboid in standard base
        box_z = data.box[2][2] * 10.0  # !!! BOX is in nm, while
        bin_width = box_z / self.bin_count  # !!! positions are in Angtroems :P
        bin_volume = data.box[0][0] * data.box[1][1] * data.box[2][2]
        bin_volume /= float(self.bin_count)  # in nm^3
        local_volumes = dict(
            (name, [0 for i in xrange(self.bin_count)]) for name in
            set(r.resname for r in data.residues)
        )
        for a in data.atoms:
            z = a.x[2]
            # shift atom back to box if necessary
            # because of the way int(z / box_z) works, we need to handle case
            # z < 0 separately, hence add -1 if (z < 0)
            z -= box_z * (int(z / box_z) - (z < 0))
            # now count each resname separately to resp. bin
            # TODO: think about using index files here (refactor)
            self.counts[a.resname][int(z / bin_width)] += 1
            local_volumes[a.resname][int(z / bin_width)] += 1.0

        n = float(self.frames)
        for key, items in local_volumes.iteritems():
            items = map(lambda x: x / bin_volume, items)
            self.global_volumes[key] = map(
                # x is current, y is last mean
                lambda (x, y): (x + n * y) / (1.0 + n),
                zip(items, self.global_volumes[key])
            )

        self.frames += 1

    def aggregate_volume(self, resnames, direct=True):
        """
        see agregate
        this is for count / volumes [#/nm^3]
        """
        aggregates = [
            self.global_volumes[r] for r in self.global_volumes
            if direct == (r in resnames)
        ]
        return map(sum, zip(*aggregates))

    def aggregate_volume_other(self, resnames):
        """
        see agregate_other
        this is for count / volumes [#/nm^3]
        """
        return self.aggregate_volume(resnames, direct=False)

    def aggregate(self, resnames, direct=True):
        """
        concatenate bins for given resnames and return result
        :param resnames:    list of resnames to concatenate bins
        :return:            list of bins (counts)
        """
        aggregates = [
            self.counts[r] for r in self.counts
            if direct == (r in resnames)
        ]
        return map(sum, zip(*aggregates))

    def aggregate_other(self, resnames):
        """
        concatenate bins that ARE NOT in resnames list
        same as self.agregate(resnames, False)
        :param resnames: list of resnames to exclude
        :return:         list of bins (counts)
        """
        return self.aggregate(resnames, direct=False)

    def normalize(self, factors=None):
        """
        Call only when you have counted everything with count()
        Normalizes counts to sum to 1.0
        """
        self.history.append(copy.deepcopy(self.counts))
        # sum resp. bins for each resname if no factor supplied by user
        # (we normalize to have sum = 1.0 in each bin)
        if factors is None: factors = map(sum, zip(*self.counts.itervalues()))
        if isinstance(factors, six.integer_types):
            val = factors
            factors = [val for i in range(len(self.bins))]
        # test if factors are properly defined if supplied by user
        msg = "Factors should have same length as bins. Was: {0}, expected: {1}"
        assert len(factors) == len(self.bins), msg.format(len(factors), len(self.bins))
        # do normalization
        for r in self.counts:
            # for each resname make list of pairs bin[i], factor[i] (by zip)
            # and then apply by map function bin[i]/factor[i] and assign
            # this again to counts
            self.counts[r] = map(
                lambda x:
                float(x[0]) / float(x[1]) if float(x[1]) > 0 else float(x[0]),
                zip(self.counts[r], factors)
            )

    def denormalize(self):
        """
        Call after normalize to restore previous counts!
        """
        self.counts = self.history[-1]
        self.history.pop()

    def store(self, f):
        if not (isinstance(f, file)):
            f = file(f, "w")
        json.dump(self.__dict__, f, indent=4)

    def load(self, f):
        if not (isinstance(f, file)):
            f = file(f, "r")
        d = json.load(f)
        for key in d:
            self.__dict__[key] = d[key]

    def output_profiler(self,
                        output_prefix,
                        groups=None,
                        extra_header="",
                        out_volumes=False):
        table = self.global_volumes if out_volumes else self.counts
        aggreg_fn = self.aggregate_volume if out_volumes else self.aggregate

        groups = groups or zip(table.keys(), table.keys())
        data_list = [self.bins]
        headers = ["z"]
        for label, group in groups.iteritems():
            headers.append(label)
            counts = aggreg_fn(group)
            data_list.append(counts)

            plt.plot(self.bins, counts, label=label)

        plt.legend()
        plt.savefig('{0}.png'.format(output_prefix))
        plt.close()

        # print "Data lengths (should be all equal):", map(len, data_list)

        outfile = file('{0}.dat'.format(output_prefix), "w")
        line = " ".join(headers)
        outfile.write(extra_header)
        outfile.write("#{0}\n".format(line))
        data_list = zip(*data_list)
        for row in data_list:
            line = " ".join(map(str, row))
            outfile.write("{0}\n".format(line))
        outfile.close()

        # groups = groups or zip(self.counts.keys(), self.counts.keys())
        # data_list = [self.bins]
        # headers = ["z"]
        # for label, group in groups.iteritems():
        #     headers.append(label)
        #     counts = self.aggregate(group)
        #     data_list.append(counts)
        #
        #     plt.plot(self.bins, counts, label=label)
        #
        # plt.legend()
        # plt.savefig('{0}.png'.format(output_prefix))
        # plt.close()
        #
        # # print "Data lengths (should be all equal):", map(len, data_list)
        #
        # outfile = file('{0}.dat'.format(output_prefix), "w")
        # line = " ".join(headers)
        # outfile.write(extra_header)
        # outfile.write("# {0}\n".format(line))
        # data_list = zip(*data_list)
        # for row in data_list:
        #     line = " ".join(map(str, row))
        #     outfile.write("{0}\n".format(line))
        # outfile.close()

# class Profile ends here


def intersections(sequenceA, sequenceB):
    """
    computes intersections of two sequences representing functions over the same
    (unknown in the context of intersections()) interval
    """
    assert len(sequenceA) == len(sequenceB), "Not the same length in sequences"
    count = len(sequenceA)
    result = []
    for i in xrange(count - 1):
        c0 = (sequenceA[i] - sequenceB[i])
        c1 = (sequenceA[i + 1] - sequenceB[i + 1])
        if c0 * c1 < 0:
            # first value (a) is interpolated point between i and i+1 bin
            # that is, if w is the width of the bin then x0 + i * w + a * w
            # is interpolated point of intersection
            a = -c0 / (c1 - c0)
            result.append((a, i, i + 1))
    return result


################################################################################
# computation script starts here                                               #
################################################################################

def main(runfile_path=None):
    pdb_filename = "odnPOPC_test.pdb"
    xtc_filename = "odnPOPC_test.xtc"
    begin_frame = 0
    end_frame = 1000
    bin_count = 133
    groups = {
        "SOL": ["SOL"],
        "SOL+OG": ["SOL", "OG"],
        "OG": ["OG"],
        "PCH": ["PCH"],
        "OLE": ["OLE"],
        "PAL": ["PAL"],
        "POPC": ["PCH", "OLE", "PAL"],
    }
    output_dir = "odnPOPC_test_OUTPUT"

    if runfile_path:
        # this is not nice, but powerfull..
        d = locals()
        exec open(runfile_path).read() in d, d
        # TODO: grrrrrrrrrrr
        # TODO: because of the BUG I need to do this this way, grrrrrr....
        # TODO: make it a local config, (e.g. Main class with main() method)
        # TODO: and stroe config in self -> then can automatically assign in loop
        pdb_filename = d.get('pdb_filename', pdb_filename)
        xtc_filename = d.get('xtc_filename', xtc_filename)
        begin_frame = d.get('begin_frame', begin_frame)
        end_frame = d.get('end_frame', end_frame)
        bin_count = d.get('bin_count', bin_count)
        groups = d.get('groups', groups)
        output_dir = d.get('output_dir', output_dir)

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
    in_conf += "pdb_filename = \"{0}\"\n".format(pdb_filename)
    in_conf += "xtc_filename = \"{0}\"\n".format(xtc_filename)
    in_conf += "begin_frame  = {0}\n".format(begin_frame)
    in_conf += "end_frame    = {0}\n".format(end_frame)
    in_conf += "bin_count    = {0}\n".format(bin_count)
    in_conf += "groups       = {0}\n".format(str(groups))
    in_conf += "output_dir   = \"{0}\"\n".format(output_dir)

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
    # file to save data
    output_filename = "{0}/densities_count.dat".format(output_dir)

    # read data and prepare useful collections
    data = Model(pdb_filename)
    # prepeare usefull indexes
    for resname, ndx in indexes(data):
        ndx.write("{0}/{1}.ndx".format(output_dir, resname))

    # load trajectory to iterate
    trajectory = Trajectory(xtc_filename)
    profiler = DensityProfile(bin_count, data)

    # define post_update function for iterate
    def post_update(data, frame, fc):
        profiler.count(data)
        if fc % 100 == 0:
            print "Frame count:", fc

    print "Starting iteration..."
    iterate(data, trajectory, begin_frame, end_frame, post_update)
    print "Iteration finished!"

    out_filename = "{0}/atom_count_per_volume".format(output_dir)
    profiler.output_profiler(out_filename, groups, help_str, True)

    out_filename = "{0}/atom_count_full".format(output_dir)
    profiler.output_profiler(out_filename, groups, help_str)

    profiler.normalize()
    out_filename = "{0}/normalized_atom_count_profile".format(output_dir)
    profiler.output_profiler(out_filename, groups, help_str)
    profiler.denormalize()

    profiler.normalize(profiler.frames)
    out_filename = "{0}/atom_count_profile_per_frame".format(output_dir)
    profiler.output_profiler(out_filename, groups, help_str)
    profiler.denormalize()

