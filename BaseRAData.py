"""
BaseRAData.py

Abstract class definition defining the required interface for pyrat Data
objects.

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
"""

import abc


class BaseData(object):
    """
    Abstract class definition defining the required interface for pyrat Data
    objects.

    Attributes:

    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def init_subgroup(self, spw, nchan, nrecs):
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

    @abc.abstractmethod
    def transform(self, image):
        """
        A function that transforms the Data object into an Image object. There
        is no default transform function. This must be overwritten by a derived
        class.

        Args:
            image: An Image object on which to transform the data.
        """
        return

    @abc.abstractmethod
    def store_records(self, a, spw, chan):
        """
        Store all data within numpy array a into the file.

        Args:
            a: A numpy array containing a chunk of data to be stored into the
                Data object.
            spw: The spectral window number or subgroup name in which the data
                chunk belongs.
            chan: The channel number in which to store the array.

        Returns:
            Nothing
        """

    @abc.abstractmethod
    def get_records(self, spw, chan):
        """
        Get a numpy array from the file containing all data from a single
        channel.

        Args:
            spw: The spectral window number or subgroup name from which to
                retrieve the data
            chan: The channel number from which read the array.

        Returns:
            A numpy array containing all data from a single channel.
        """

    @abc.abstractmethod
    def addto(self, other, target):
        """
        Add the current data set to another.

        Args:
            other: The data set to add to the current one.
            target: The data object into which the sum will be placed
        Returns:
            None
        """

        return

    @abc.abstractmethod
    def subtractoff(self, other, target):
        """
        Subtract another Data object from the current one.

        Args:
            other: The data set that will be subtracted from the current one.
            target: The data object into which the result will be placed
        Returns:
            None
        """

        return

    @abc.abstractmethod
    def multiplywith(self, other, target):
        """
        Multiply the entries of the current data object with those in another.

        Args:
            other: The data object to multiply with.
            target: The data object into which the result will be placed
        Returns:
            None
        """

        return

    @abc.abstractmethod
    def divideby(self, other, target):
        """
        Divide the entries of the current data object by those in another.

        Args:
            other: The data object to divide by.
            target: The data object into which the result will be placed
        Returns:
            None
        """

        return

    @abc.abstractmethod
    def copy_data_to(self, other):
        """
        Copy the data from the current object to another one.

        Args:
            other: The data object into which the result will be placed
        Returns:
            None
        """

        return

    @abc.abstractmethod
    def iterkeys(self):
        """
        Desc.

        Args:
        Returns:
        """

        return
