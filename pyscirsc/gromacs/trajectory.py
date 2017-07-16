
def noop_update(data, frame, frame_count):
    """
    Function that does nothing :)
    Used as a default value for function parameters in iterate
    :return: None
    """
    pass


def iterate(data,
            trajectory,
            begin_frame=0,
            end_frame=-1,
            post_update=noop_update,
            pre_update=noop_update):
    """
    Iterate over trajcetory, updates atoms, and allows to insert
    computations by using functions of the signature:

        def fun(data, frame, frame_count)

    It skips begin_frame frames at the beginning and stops at end_frame, unsless
    end_frame < 0, in which case it iterates over all frames in trajectory. At
    the end returns number of last frame.
    :param data:        pmx.model.Model
    :param trajectory:  pmx.trj.Trajectory
    :param begin_frame: int, default: 0
    :param end_frame:   int, default: ALL FRAMES (-1)
    :param post_update: function, default: no operations (noop_update)
    :param pre_update:  function, default: no operations (noop_update)
    :return:            number of last frame reached (counted from 0)
    """
    fc = 0
    for frame in trajectory:
        # some bookkeeping to control frames
        fc += 1
        if fc < begin_frame:         # by default we start at 0
            continue                 # ignore rest of computations, just update
        if 0 < end_frame < fc:       # stop at end_frame-1
            break                    # if end_frame < 0 then never happens

        pre_update(data, frame, fc)     # do some pre calculations
        frame.update(data)              # update atom positions
        post_update(data, frame, fc)    # do some post calculations

    return fc       # return final frame reached
