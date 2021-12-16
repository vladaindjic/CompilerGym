import pdb
import sys

import hatchet as ht

if __name__ == "__main__":

    if len(sys.argv) != 2:
        exit

    # pdb.set_trace()
    # Path to HPCToolkit database directory.
    dirname = sys.argv[1]
    print(dirname)

    # Use hatchet's ``from_hpctoolkit`` API to read in the HPCToolkit database.
    # The result is stored into Hatchet's GraphFrame.
    gf = ht.GraphFrame.from_hpctoolkit(dirname)

    # Printout the DataFrame component of the GraphFrame.
    print(gf.dataframe)

    # Printout the graph component of the GraphFrame.
    # Use "time (inc)" as the metric column to be displayed
    print(gf.tree(metric_column="time (inc)"))
