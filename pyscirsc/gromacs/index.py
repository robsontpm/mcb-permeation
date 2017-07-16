from pmx.ndx import IndexFile, IndexGroup


def indexes(data):
    """
    recreate Index Files for separate groups and writes it to the file under
    filename if specified and return list of pmx.IndexFile object
    :return: list of pairs (resname, IndexFile) objects corresponding to all
             residues defined by resnames in data
    """
    result = []
    for r in set(r.resname for r in data.residues):
        group = IndexGroup(r, atoms=[a for a in data.atoms if a.resname == r])
        index_file = IndexFile()
        index_file.add_group(group)
        result.append((r, index_file))
    return result


def separate_indexes(data, resname, atoms_per_residue):
    """
    Takes all atoms that satisfy atom.resname in resname and separates it into
    groups of atoms_per_residue groups. It is useful for separating single
    molecules of a given type into indexes. For examle, if you have 200
    molecules of O_2 under resname OG. Each of them of course consist of 2 atoms.
    You can create indexes with labels OG_{i} in a IndexFile object if you call

        ndx = separate_indexes(atoms, "OG", 2)

    then ndx will have 200 groups, each containing single oxygen molecule.

    Warning: it assumes that each molecule is presented in the file in the same
    order and together, eg:

        122 R A_1
        123 R A_2
        124 S A_3
        125 R B_1
        126 R B_2
        127 S B_3

    A and B are separate molecules consisting of 3 atoms and they are divided
    into residues R and S. Then separate_indexes(atoms, ["R", "S"], 3)
    can be used to separate the molecules. I have used this procedure to
    separate trajectories consisting of lipid bilayers, where this scenario
    most often happens.

    :param data: pmx.model.Model
    :param resname: string or list, residue name(s) to choose
    :param atoms_per_residue: int >= 1
    :return: resname, pmx.ndx.IndexFile, where index file contains all atoms
             with given resname separated into molecules
    """
    assert int(atoms_per_residue) >= 1
    if isinstance(resname, basestring): resname = [resname]
    index_file = IndexFile()
    og = [a for a in data.atoms if a.resname in resname]
    shifts = [og[i::atoms_per_residue] for i in xrange(atoms_per_residue)]
    for i, atoms in enumerate(zip(*shifts)):
        group = IndexGroup("{0}_{1}".format(resname, i+1), atoms=list(atoms))
        index_file.add_group(group)
    return resname, index_file
