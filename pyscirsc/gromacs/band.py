
class Band:
    def __init__(self, left_bound, right_bound, direction):
        """
        :param left_bound:   float, in angstroems!
        :param right_bound:  float, in angstroems!
        :param direction:     int, 0 = x, 1 = y, 2 = z
        """
        self.left = left_bound
        self.right = right_bound
        self.direction = direction

    def inband(self, atom_position):
        return self.left <= atom_position[self.direction] < self.right

    def select(self, atom_list, select_pos=lambda x: x.x):
        return [
            (i, a)
            for i, a in enumerate(atom_list)
            if self.inband(select_pos(a))
        ]

    def mask(self, atom_list, select_pos=lambda x: x.x):
        return [self.inband(select_pos(a)) for a in atom_list]
