pyrat - Python Radio Astronomy Toolkit
==========================================================

pyrat is a package to make handling large, multi-frequency visibility data sets
easier. It's designed to work on datasets that are too large to fit in main
memory, using HDF5 files as buffers for storing data on hard disk when it is
not needed. The data structures have been designed with the application of 
imaging in mind, but the library could just as well be used for other projects
where manipulation of broadband imaging and/or visibility data is needed.

Features include:

  - Buffering of data using H5PY
  - Convenient data structures for handling both imaging and visibility data.
  - Methods for reading visibility data from MeasurementSet files.
  - Helper class for handling parameter sets and parsing parameter set files.
  - Helper class for writing structured log information to screen and a file.

Installation instructions
-------------------------

Prerequisites for pyrat include

  - python
  - numpy
  - [pyrap](https://code.google.com/p/pyrap/)
  - h5py

To install, simply type

   > python setup.py install

The pyrat modules should then be importable from your python code, e.g.

   import pyrat.Messenger as M

About pyrat
-----------

pyrat is licensed under the [GPLv3](http://www.gnu.org/licenses/gpl.html).

pyrat has been developed at the Max Planck Institute for Astrophysics and 
within the framework of the DFG Forschergruppe 1254, "Magnetisation of 
Interstellar and Intergalactic Media: The Prospects of Low-Frequency Radio 
Observations."
