from bsb.output import HDF5Formatter
from bsb.core import from_hdf5
from bsb.config import JSONConfig
from bsb.reporting import set_verbosity
import sys, mpi4py.MPI

if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
    config = JSONConfig(file='mouse_cerebellum_cortex_v1.json')
    HDF5Formatter.reconfigure("300x_200z.hdf5", config)

network = from_hdf5("300x_200z.hdf5")
set_verbosity(3)
network.run_simulation()
