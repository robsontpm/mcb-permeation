from pmx.model import Atom


class PositionAtom:
    """
    Faster for computations of distances. It contains
    only position of the Atom (x). Thus, it can be used
    with true mpx.models.Atom:

    A = mpx.models.Atom(x=(1,2,3))  # requires many additional operations
    B = PositionAtom(x=(1,2,3))     # it only stores x!
    A.dist(B)                       # it works!
    """
    def __init__(self, x, unity="A"):
        self.x = (x[0], x[1], x[2])
        self.unity = unity


def pbc_positions(atom, box):
    """
    :param atom: pmx.models.Atom or PositionAtom
    :return: list of PositionAtom (all 27 periodic images)
    """
    x = atom.x
    return [
        PositionAtom(x=(x[0] + d[0], x[1] + d[1], x[2] + d[2]))
        for d in [
            (a, b, c)
            for a in (-box[0], 0, +box[0])
            for b in (-box[1], 0, +box[1])
            for c in (-box[2], 0, +box[2])
        ]
    ]


def pbc_images(atom, box):
    """
    It returns copies of the original Atom object, thus is slower
    than pbc_positions.

    :param atom: pmx.models.Atom or PositionAtom
    :return: list of pmx.models.Atom (all 27 periodic images)
    """
    res = []
    images = [
        (a, b, c)
        for a in (-box[0], 0, +box[0])
        for b in (-box[1], 0, +box[1])
        for c in (-box[2], 0, +box[2])
    ]
    for d in images:
        a = atom.copy()
        a.translate(d)
        res.append(a)
    return res


def center_of_mass(atoms):
    """
    Helper functions, compute center of a group of Atom objects from pmx.model
    Warning: this returns geometric center for centers of atoms, for center of
    mass it should be weighted with real atomic mass
    :param atoms: list of Atom instances
    :return: Atom instance with x set to geometric center of this group
    """
    no_atoms = len(atoms)
    return Atom(x=[sum((a.x[j] for a in atoms)) / no_atoms for j in xrange(3)])


def closest_distance(atom1, atom2_list):
    """
    Most likely distance between atoms is the smallest distance betwen atom
    and all the periodic images of atom2 stored in atom2_list.
    :param atom1: pmx.model.Atom
    :param atom2_list: collection of pmx.model.Atom or PositionAtom
    """
    return min([atom1.dist(pos) for pos in atom2_list])


def most_likely_distance(atom1, atom2, box):
    """
    Most likely distance between atoms is the smallest distance betwen atom
    and all the periodic images of atom2 with respect to a given box.
    :param atom1: pmx.model.Atom
    :param atom2: pmx.model.Atom
    :param box: a tuple
    :return: most likely distance from atom1 to atom2 w.r.t. box
    """
    return closest_distance(atom1, pbc_positions(atom2, box))


def closest_distance2(atom1, atom2_list):
    """
    Same as most_likely_distance, but distance^2 is computed (faster).
    :param atom1: pmx.model.Atom
    :param atom2_list: collection of pmx.model.Atom or PositionAtom
    :return: most likely distance from atom1 to atom2 w.r.t. box
    """
    return min([atom1.dist2(pos) for pos in atom2_list])


def most_likely_distance2(atom1, atom2, box):
    """
    Same as most_likely_distance, but distance^2 is computed (faster).
    :param atom1: pmx.model.Atom
    :param atom2: pmx.model.Atom
    :param box: a tuple
    :return: most likely distance from atom1 to atom2 w.r.t. box
    """
    return closest_distance2(atom1, pbc_positions(atom2, box))


def remove_pbc_direction(old_w, new_w, box_w):
    hbox_w = box_w * 0.5
    while new_w - old_w <= -hbox_w:
        new_w += box_w
    while new_w - old_w > hbox_w:
        new_w -= box_w
    return new_w


def correct_pbc(old_pos, new_pos, box):
    return (
        remove_pbc_direction(old_pos[0], new_pos[0], box[0]),
        remove_pbc_direction(old_pos[1], new_pos[1], box[1]),
        remove_pbc_direction(old_pos[2], new_pos[2], box[2]),
    )


def correct_box(databox):
    if hasattr(databox, 'box'):
        return (10.0 * databox.box[0][0],
                10.0 * databox.box[1][1],
                10.0 * databox.box[2][2])
    else:
        return (10.0 * databox[0],
                10.0 * databox[1],
                10.0 * databox[2])

