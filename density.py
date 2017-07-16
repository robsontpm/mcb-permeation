import sys
from pyscirsc.counting.density import main

if len(sys.argv) >= 2:
    runfile_path = sys.argv[1]
else:
    runfile_path = None

main(runfile_path)
