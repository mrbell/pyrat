"""
Data.py

*******************************************************************************

Copyright 2012 Michael Bell

This file is part of pyrat, the Python Radio Astronomy Toolkit.

pyrat is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

pyrat is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with pyrat.  If not, see <http://www.gnu.org/licenses/>.

*******************************************************************************

Module containing objects useful for dealing with interferometric visibility
data.
"""

from pyrap import tables as pt
import h5py
import numpy as np
import Messenger as M
from BaseRAData import BaseData
import os
from Constants import *

GROUP_BASE_STRING = 'SPW'
DATASET_STRING = 'data'
COORDSET_STRING = 'coords'
FREQSET_STRING = 'freqs'


class Coordinates(object):
    """
    Desc.

    Attributes:

    """
    _initialized = False
    dtype = np.dtype('float64')
    fn = None

    def __init__(self, fn):
        """
        Desc.

        Args:

        Returns:

        """

        self.fn = fn

        if os.path.isfile(fn):
            self._initialized = True

        self.f = _create_file(fn)

#        if self._initialized:
#            self._load_freqs_from_file()

#    def _load_freqs_from_file(self):
#        """
#        Desc.
#
#        Args:
#            None
#
#        Returns:
#            Nothing.
#
#        """
#        self.freqs = []
#
#        for i in self.f.iterkeys():
#            self.freqs.append(self.f[i][FREQSET_STRING][:])

    def init_subgroup(self, spw, freqs, nrecs):
        """
        Desc.

        Args:

        Returns:

        """
        nchan = len(freqs)
        g = self.f.require_group(GROUP_BASE_STRING + str(spw))
        g.require_dataset(COORDSET_STRING, shape=(nchan, nrecs, 2),
                          dtype=self.dtype)
        g.require_dataset(FREQSET_STRING, shape=(nchan,),
                          dtype=self.dtype)
        g[FREQSET_STRING][:] = freqs

    def put_coords(self, u, v, spw, chan):
        """
        Desc.

        Args:

        Returns:

        """
        if isinstance(spw, str) or isinstance(spw, unicode):
            g = self.f[spw]
        else:
            g = self.f[GROUP_BASE_STRING + str(spw)]
        d = g[COORDSET_STRING]
        d[chan, :, 0] = u
        d[chan, :, 1] = v

    def get_coords(self, spw, chan):
        """
        Get a numpy array from an HDF5 file containing all data from a single
        channel.

        Args:

        Returns:

        """
        if isinstance(spw, str) or isinstance(spw, unicode):
            g = self.f[spw]
        else:
            g = self.f[GROUP_BASE_STRING + str(spw)]
        d = g[COORDSET_STRING]
        u = d[chan, :, 0]
        v = d[chan, :, 1]

        return u, v

    def get_min_freq(self):
        """
        Get the minimum frequency in the dataset.

        Args:
            Nothing.

        Returns:
            The minimum frequency.
        """

        l2min = None

        for i in self.f.iterkeys():
            freqs = self.get_freqs(i)
            if l2min is None or min(freqs) < l2min:
                l2min = min(freqs)

        return l2min

    def get_freqs(self, spw):
        """
        Get an array of the channel central frequencies for the given spw.

        Args:

        Returns:

        """
        if isinstance(spw, str) or isinstance(spw, unicode):
            g = self.f[spw]
        else:
            g = self.f[GROUP_BASE_STRING + str(spw)]

        return g[FREQSET_STRING][:]


class Data(BaseData):
    """
    Desc.

    Attributes:

    """
    _initialized = False
    nrecs = 0
    dtype = None
    fn = None
    coords = None

    def __init__(self, fn, dtype, coords=None, template=None, m=None):
        """
        Initialize the Data object. Creates a file (or opens an existing file
        to overwrite). It will optionally also create the file structure based
        on the structure given in an existing Data object. Otherwise, this
        object must be initialized by reading data from an external data
        source. Use one of the helper functions in this method to load data
        from the external file.

        Args:
            fn: File name string in which to store the data on disk
            dtype: Data type describing the data
            coords: A Coordinates object defining the coordinates of
                the data records
            template: An existing Data object to use as a template
                for creating this Data object. If not given, the object is
                not really initialized. This must be done using a data reader
                routine like read_data_from_ms. (optional)

        Returns:
            Nothing
        """
        if m is None:
            self.m = M.Messenger(0)
        else:
            self.m = m

        self.dtype = dtype
        self.fn = fn
        if coords is None:
            coordsfn = _make_fn(fn, 'coords')
            self.coords = Coordinates(coordsfn)
        else:
            self.coords = coords

        if os.path.isfile(fn):
            self._open_existing_file(fn, template)
            self._initialized = True
        else:
            self.f = _create_file(fn)

            if template is not None:
                self._init_file_from_existing(template)
                self._initialized = True

    def _open_existing_file(self, fn, template=None):
        """
        Opens an existing file and checks to make sure that the coordinates
        object matches. Also checks to make sure that the template matches, if
        given.
        """
        self.f = _create_file(fn)

        # Check to make sure the coordinates have the correct shape
        for spw in self.f.iterkeys():
            d = self.f[spw][DATASET_STRING]
            c = self.coords.f[spw][COORDSET_STRING]

            if d.shape != c.shape[:-1]:
                raise Exception('Data and Coordinates have different shapes!')

            if template is not None:
                t = template.f[spw][DATASET_STRING]

                if d.shape != t.shape:
                    raise Exception('Dataset and template have ' +
                                    'different shapes!')

    def _init_file_from_existing(self, skeleton):
        """
        Initializes a file using the same structure as in an existing Data
        object. Also checks to see if the coordinates object is OK.

        Input:
            skeleton - [Data] - Existing Data object to use as a template for
                initializing the file for this object.

        Returns:
            Nothing
        """

        for i in skeleton.f.iterkeys():
            g = self.f.create_group(i)
            g.create_dataset(DATASET_STRING,
                             shape=skeleton.f[i][DATASET_STRING].shape,
                             dtype=self.dtype)

    def _looped_operation(self, other, target, func):
        """
        Desc.

        Args:

        Returns:

        """
        DO = 0
        NP = 1
        SC = 2

        if isinstance(other, Data):
            othertype = DO
        elif np.isscalar(other):
            othertype = SC
        elif isinstance(other, np.ndarray):
            if other.ndim != 1:
                raise Exception("Operation with invalid data type.")
            othertype = NP
        else:
            raise Exception("Operation with invalid data type.")

        for spw in self.f.iterkeys():
            ds = self.f[spw][DATASET_STRING]
            if othertype == DO:
                do = other.f[spw][DATASET_STRING]
            dt = target.f[spw][DATASET_STRING]

            for chan in range(ds.len()):
                rs = ds[chan]
                if othertype == DO:
                    ro = do[chan]
                elif othertype == SC:
                    ro = other
                elif othertype == NP:
                    if len(other) != len(rs):
                        raise Exception("Array size does not match number " +
                                        "of records in the Data object!")
                    ro = other
                rt = dt[chan]

                rt = func(rs, ro)

                dt[chan, :] = rt

    def copy_data_to(self, other):
        """
        Desc.

        Args:

        Returns:
            Nothing.
        """
        func = lambda x, y: y

        self._looped_operation(self, other, func)

    def init_subgroup(self, spw, freqs, nrecs):
        """
        Create a file group.

        Input:
            spw: Number of the subband
            chan_array: Array containing a list of the frequencies that
                correspond to the first index of the data array.
            nrecs: Number of records per channel.

        Returns:
            Nothing.
        """

        nchan = len(freqs)
        g = self.f.create_group(GROUP_BASE_STRING + str(spw))
        g.create_dataset(DATASET_STRING, shape=(nchan, nrecs),
                         dtype=self.dtype)

        self.coords.init_subgroup(spw, freqs, nrecs)

    def transform(self, image):
        """
        A function that transforms the Data object into an Image object. There
        is no default transform function. This must be overwritten by a derived
        class. An error is thrown if this method is called.

        Args:
            image: An Image object on which to transform the data.

        Returns:
            Nothing
        """

        raise Exception("No transform function defined for PolData object!")

    def store_records(self, a, spw, chan):
        """
        Store all data within numpy array a into the file.

        Args:
            a: A numpy array containing a chunk of data to be stored into the
                Data object.
            spw: The spectral window number in which the data chunk belongs.
            chan: The channel number in which to store the array.

        Returns:
            Nothing
        """
        if isinstance(spw, str) or isinstance(spw, unicode):
            g = self.f[spw]
        else:
            g = self.f[GROUP_BASE_STRING + str(spw)]

        d = g[DATASET_STRING]
        d[chan, :] = a

    def get_records(self, spw, chan):
        """
        Get a numpy array from the file containing all data from a single
        channel.

        Args:
            spw: The spectral window number from which to retrieve the data
            chan: The channel number from which read the array.

        Returns:
            A numpy array containing all data from a single channel.
        """
        if isinstance(spw, str) or isinstance(spw, unicode):
            g = self.f[spw]
        else:
            g = self.f[GROUP_BASE_STRING + str(spw)]
        d = g[DATASET_STRING]
        a = d[chan, :]

        return a

    def addto(self, other, target=None):
        """
        Desc.

        Args:

        Returns:
            Nothing
        """

        if target is None:
            target = self

        func = lambda a, b: a + b
        self._looped_operation(other, target, func)

    def subtractoff(self, other, target=None):
        """
        Desc.

        Args:
        Returns:
        """
        if target is None:
            target = self
        func = lambda a, b: a - b
        self._looped_operation(other, target, func)

    def multiplywith(self, other, target=None):
        """
        Desc.

        Args:
        Returns:
        """
        if target is None:
            target = self
        func = lambda a, b: a * b
        self._looped_operation(other, target, func)

    def divideby(self, other, target=None):
        """
        Desc.

        Args:
        Returns:
        """
        if target is None:
            target = self
        func = lambda a, b: a / b
        self._looped_operation(other, target, func)

    def iterkeys(self):
        """
        Desc.

        Args:

        Returns:

        """
        return self.f.iterkeys()


class PolData(BaseData):
    """
    Desc.

    Attributes:

    """

    _initialized = False
    dtype = None

    def __init__(self, fn, dtype=np.dtype('complex128'), coords=None,
                 template=None, m=None):
        """
        Desc.

        Args:

        Returns:

        """

        if m is None:
            self.m = M.Messenger()
        else:
            self.m = m

        Qfn = _make_fn(fn, "Q")
        Ufn = _make_fn(fn, "U")

        self.dtype = dtype

        if template is None:
            Qt = None
            Ut = None
        else:
            Qt = template.Q
            Ut = template.U

        self.Q = Data(Qfn, self.dtype, coords, Qt)
        self.U = Data(Ufn, self.dtype, self.Q.coords, Ut)

        self.coords = self.Q.coords

        if self.Q._initialized and self.U._initialized:
            self._initialized = True

    def iterkeys(self):
        """
        Desc.

        Args:

        Returns:

        """
        return self.Q.f.iterkeys()

    def put_coords(self, u, v, spw, chan):
        """
        Desc.

        Args:

        Returns:

        """

        self.Q.put_coords(u, v, spw, chan)

    def init_subgroup(self, spw, nchan, nrecs):
        """
        Desc.

        Args:

        Returns:

        """
        self.Q.init_subgroup(spw, nchan, nrecs)
        self.U.init_subgroup(spw, nchan, nrecs)

    def transform(self, image):
        """
        A function that transforms the Data object into an Image object. There
        is no default transform function. This must be overwritten by a derived
        class. An error is thrown if this method is called.

        Args:
            image: An Image object on which to transform the data.
        Returns:
            Nothing
        """

        raise Exception("No transform function defined for PolData object!")

    def store_records(self, a, spw, chan):
        """
        Store all data within numpy array a into the file.

        Args:
            a: A numpy array containing a chunk of data to be stored into the
                Data object.
            spw: The spectral window number in which the data chunk belongs.
            chan: The channel number in which to store the array.

        Returns:
            Nothing
        """

        if not type(a) is list and len(a) != 2:
            raise TypeError("Must pass a length 2 list of arrays to store" +
                            " in PolData object.")

        self.Q.store_records(a[0], spw, chan)
        self.U.store_records(a[1], spw, chan)

    def get_records(self, spw, chan):
        """
        Get a numpy array from the file containing all data from a single
        channel.

        Args:
            spw: The spectral window number from which to retrieve the data
            chan: The channel number from which read the array.

        Returns:
            A numpy array containing all data from a single channel.
        """

        qrec = self.Q.get_records(spw, chan)
        urec = self.U.get_records(spw, chan)

        return qrec, urec

    def copy_data_to(self, other):
        """
        Desc.

        Args:
        Returns:
        """

        if not isinstance(other, PolData):
            raise Exception("Illegal target object!")

        self.Q.copy_data_to(other.Q)
        self.U.copy_data_to(other.U)

    def addto(self, other, target=None):
        """
        Desc.

        Args:
        Returns:
        """
        if target is None:
            target = self
        if not isinstance(target, PolData):
            raise Exception("Illegal target object!")

        [oQ, oU] = self._validate_other(other)

        self.Q.addto(oQ, target.Q)
        self.U.addto(oU, target.U)

    def subtractoff(self, other, target=None):
        """
        Desc.

        Args:
        Returns:
        """
        if target is None:
            target = self
        if not isinstance(target, PolData):
            raise Exception("Illegal target object!")

        [oQ, oU] = self._validate_other(other)

        self.Q.subtractoff(oQ, target.Q)
        self.U.subtractoff(oU, target.U)

    def _validate_other(self, other):
        """
        Desc.

        Args:

        Returns:

        """

        if isinstance(other, PolData):
            oQ = other.Q
            oU = other.U
        else:
            oQ = other
            oU = other

        return oQ, oU

    def multiplywith(self, other, target=None):
        """
        Desc.

        Args:
        Returns:
        """
        if target is None:
            target = self
        if not isinstance(target, PolData):
            raise Exception("Illegal target object!")

        [oQ, oU] = self._validate_other(other)

        self.Q.multiplywith(oQ, target.Q)
        self.U.multiplywith(oU, target.U)

    def divideby(self, other, target=None):
        """
        Desc.

        Args:
        Returns:
        """
        if target is None:
            target = self
        if not isinstance(target, PolData):
            raise Exception("Illegal target object!")

        [oQ, oU] = self._validate_other(other)

        self.Q.divideby(oQ, target.Q)
        self.U.divideby(oU, target.U)


def _make_fn(fn, modstr):
    """
    Creates a new file name by modifying a given one. The given file is
    modified according to the following examples (assume the modstr is "mod")
        filename.txt -> filename_mod.txt
        filename -> filename_mod
        file.name.txt -> file.name_mod.txt
    """
    sfn = fn.split('.')

    if len(sfn) > 2:
        sfn = ['.'.join(sfn[:-1]), sfn[-1]]

    if len(sfn) == 1:
        newfn = sfn[0] + '_' + modstr
    else:
        sfn[0] = sfn[0] + '_' + modstr
        newfn = '.'.join(sfn)

    return newfn


def _create_file(fn):
    """
    Creates a file in which to store data or coordinates.
    """

    return h5py.File(fn)


def read_data_from_ms(msfn, vis, noise, viscol="DATA", noisecol='SIGMA',
                      mode='pol'):
    """
    Reads polarization or total intensity data into a visibility and a
    noise data class from a MeasurementSet file.

    Args:
        msfn: Name of the MeasurementSet file from which to read the data
        coords: A Coordinate object to be filled from the MS file.
        vis: The PolData object corresponding to the visibilities to be read.
        noise: The Data object corresponding to the noise or weights to be
            read.
        viscol: A string with the name of the MS column from which to read the
            data [DATASET_STRING]
        noisecol: A string with the name of the MS column from which to read
            the noise or weights ['SIGMA']
        mode: Flag to set whether the function should read in
            polarization data ('pol') or total intensity data ('tot')

    Returns:
        Nothing.
    """

    # TODO: Put in a check to make sure that the MS file contains data for
    #       only one source and or just one pointing direction (no mosaic).

    # Messenger object for displaying messages
    m = vis.m

    if vis._initialized and noise._initialized:
        m.warn("Requested data objects already exist. Using the " +
               "previously parsed data.")
        return

    if mode == 'pol':
        m.header2("Reading polarization data from the MeasurementSet...")
    if mode == 'tot':
        m.header2("Reading total intensity data from the MeasurementSet...")

    # number of rows to read in total. if zero, reads them all
    nrows = 0

    if pt.tableexists(msfn) is False:
        raise IOError('No Measurement Set with the given name')

    mt = pt.table(msfn)  # main table, in read only mode
    swt = pt.table(msfn + '/SPECTRAL_WINDOW/')  # spectral window table
    # table row interface, we only need channel frequencies and the widths
    # from this table.
    swr = swt.row(['CHAN_FREQ', 'CHAN_WIDTH'])

    # polarization table, only used to get the corr_type
    polt = pt.table(msfn + '/POLARIZATION/')
    corr = polt.getcol('CORR_TYPE')  # correlation type (XX-YY, RR-LL, etc.)

    if len(corr) > 1:
        raise ValueError('Unexpected number of polarization configurations')

    # the Q,U OR the I part of the S Jones matrix (hence Spart)
    # from the Stokes enumeration defined in the casa core libraries
    # http://www.astron.nl/casacore/trunk/casacore/doc/html \
        #/classcasa_1_1Stokes.html#e3cb0ef26262eb3fdfbef8273c455e0c
    # this defines which polarization type the data columns correspond to

    corr_announce = "Correlation type detected to be "

    ii = complex(0, 1)

    if mode == 'pol':
        if corr[0, 0] == 5:  # RR, RL, LR, LL
            Spart = np.array([[0, 0.5, 0.5, 0], [0, -0.5 * ii, 0.5 * ii, 0]])
            corr_announce += "RR, RL, LR, LL"
        elif corr[0, 0] == 1:  # I, Q, U, V
            Spart = np.array([[0, 1., 0, 0], [0, 0, 1., 0]])
            corr_announce += "I, Q, U, V"
        elif corr[0, 0] == 9:  # XX, XY, YX, YY
            Spart = np.array([[0.5, 0, 0, -0.5], [0, 0.5, 0.5, 0]])
            corr_announce += "XX, XY, YX, YY"
    if mode == 'tot':
        if corr[0, 0] == 5:  # RR, RL, LR, LL
            Spart = np.array([0.5, 0, 0, 0.5])
            corr_announce += "RR, RL, LR, LL"
        elif corr[0, 0] == 1:  # I, Q, U, V
            Spart = np.array([1., 0, 0, 0])
            corr_announce += "I, Q, U, V"
        elif corr[0, 0] == 9:  # XX, XY, YX, YY
            Spart = np.array([0.5, 0, 0, 0.5])
            corr_announce += "XX, XY, YX, YY"

    m.message(corr_announce, 2)

    # total number of rows to read. Each row has nchan records, so there are
    # a total of nrows*nchan records
    if nrows == 0:
        nrows = mt.nrows()

    nspw = swt.nrows()  # Number of spectral windows (aka subbands)

    # retrieve the list of l2 or frequency values that are sampled
    freqs = list()
    if mode == 'pol':

        for i in range(nspw):
            chan_freqs = swr.get(i).get('CHAN_FREQ')
            chan_widths = swr.get(i).get('CHAN_WIDTH')

            nchan = len(chan_freqs)
            l2_vec = np.zeros(nchan)

            for j in range(nchan):
                templ2 = (0.5 * C2 *
                          ((chan_freqs[j] - chan_widths[j] * 0.5) ** -2 +
                           (chan_freqs[j] + chan_widths[j] * 0.5) ** -2))
                l2_vec[j] = templ2 / PI
            freqs.append(l2_vec)

#           vis.coords.freqs = freqs
    if mode == 'tot':

        lambs = list()
        for i in range(nspw):
            chan_freqs = swr.get(i).get('CHAN_FREQ')
            chan_widths = swr.get(i).get('CHAN_WIDTH')

            nchan = len(chan_freqs)
            freq_vec = np.zeros(nchan)

            for j in range(nchan):
                tempfreq = (chan_freqs[j] - chan_widths[j] * 0.5) \
                    + (chan_freqs[j] + chan_widths[j] * 0.5)
                freq_vec[j] = tempfreq
            freqs.append(freq_vec)
            lambs.append(C / freq_vec)

    nfreqs = 0.  # Total number of frequencies
    for i in range(len(freqs)):
        nfreqs += len(freqs[i])

    nstokes = 4  # Number of Stokes parameters. Must be 4

    m.message("MeasurementSet information", 1)
    m.message("Number of rows: " + str(nrows), 1)
    m.message("Number of spectral windows: " + str(nspw), 1)
    m.message("Number of frequencies: " + str(nfreqs), 1)

    for i in range(nspw):
        m.message("Reading spectral window number " + str(i) + "...", 2)

        stab = pt.taql("SELECT FROM $mt WHERE DATA_DESC_ID == $i")

        nrecs_stab = stab.nrows()

        uvw = stab.getcol('UVW')  # u,v,w coords (in meters)
        nchan = len(freqs[i])

        vis.init_subgroup(i, freqs[i], nrecs_stab)
        noise.init_subgroup(i, freqs[i], nrecs_stab)

        # one noise per SPW (also per cross-corr), applies to all channels
        noise_recs = stab.getcol(noisecol.upper())

        for j in range(nchan):
            m.message(".   Reading channel " + str(freqs[i][j]) +
                      " which has " + str(nrecs_stab) + " records ...", 3)

            if mode == 'pol':
                Qvis_array = np.zeros(nrecs_stab, dtype=vis.dtype)
                Uvis_array = np.zeros(nrecs_stab, dtype=vis.dtype)
            if mode == 'tot':
                vis_array = np.zeros(nrecs_stab, dtype=vis.dtype)
            noise_array = np.zeros(nrecs_stab, dtype=noise.dtype)
            u_array = np.zeros(nrecs_stab)
            v_array = np.zeros(nrecs_stab)

            data_recs = stab.getcolslice(viscol.upper(), [j, 0], [j, 3])
            flag_recs = stab.getcolslice('FLAG', [j, 0], [j, 3])

            data_recs = data_recs.reshape(nrecs_stab, nstokes)
            flag_recs = flag_recs.reshape(nrecs_stab, nstokes)

            #read in data
            if mode == 'pol':
                for k in range(nrecs_stab):
                    data_rec = data_recs[k] * (1. - flag_recs[k])
                    Qvis_array[k] = np.dot(Spart[0], data_rec)
                    Uvis_array[k] = np.dot(Spart[1], data_rec)

                    if True not in flag_recs[k]:
                        # WARNING! Strong flagging...
                        # if any correlation is flagged then flag them all
                        qnoise = np.dot(Spart[0], noise_recs[k])
                        unoise = np.dot(Spart[1], noise_recs[k])
                        noise_array[k] = np.sqrt(qnoise * qnoise.conjugate() +
                                             unoise * unoise.conjugate()).real

                    u_array[k] = uvw[k, 0] / np.sqrt(PI * freqs[i][j])
                    v_array[k] = uvw[k, 1] / np.sqrt(PI * freqs[i][j])

                vis.store_records([Qvis_array, Uvis_array], i, j)

            if mode == 'tot':
                print 'total intensities'
                for k in range(nrecs_stab):
                    data_rec = data_recs[k] * (1. - flag_recs[k])
                    vis_array[k] = np.dot(Spart, data_rec)
                    if np.dot(Spart, 1. - flag_recs[k]) != 0.:
                        noise_array[k] = np.dot(Spart, noise_recs[k])

                    u_array[k] = uvw[k, 0] / lambs[i][j]
                    v_array[k] = uvw[k, 1] / lambs[i][j]

                vis.store_records(vis_array, i, j)

            vis.coords.put_coords(u_array, v_array, i, j)
            noise.store_records(noise_array, i, j)

        stab.close()
        m.message("Done!", 2)

    mt.close()
    swt.close()
    polt.close()

    vis._initialized = True
    vis.coords._initialized = True
    noise._initialized = True

    m.success("Finished reading data from the MeasurementSet!", 0)
