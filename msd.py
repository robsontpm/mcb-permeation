import sys
from pyscirsc.counting.msd import msd_for_band
from pyscirsc.counting.msd import msd_select_restarts
from pyscirsc.counting.msd import msd_simplified_deviation


if len(sys.argv) >= 2:
    script = sys.argv[1]

if len(sys.argv) >= 3:
    runfile_path = sys.argv[2]
else:
    runfile_path = None

if script == "select_restarts":
    msd_select_restarts.main(runfile_path)

if script == "band":
    msd_for_band.main(runfile_path)

if script == "deviation":
    msd_simplified_deviation.main(runfile_path)

