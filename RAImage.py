"""
Image.py

**********************************************************************************

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

**********************************************************************************

Module containing objects useful for dealing with interferometric imaging data.

"""


import h5py, os
import numpy as np
import Messenger as M

DATASET_STRING = 'data'

class GridParams(object):
    """
    A simple container for gridding parameters W (the size of the grid convolution
    function) and alpha (the oversampling ratio).
    
    We currently use the gridding scheme with minimal oversampling discussed in 
    Beatty et al. (2005).
    
    Attributes:
        W: The width of the GCF in number of pixels (the same for all axes)
        alpha: The oversampling ratio (ratio between the size of the oversized 
            image used during gridding to the requested image size.)
        
    """
    W = 6
    alpha = 1.5
    
    def __init__(self, alpha=1.5, W=6):
        """
        Initializes the gridding parameters.
        
        Args:
            alpha: The oversampling ratio (ratio between the size of the oversized 
                image used during gridding to the requested image size.)
            W: The width of the GCF in number of pixels (the same for all axes)
        
        Returns:
            Nothing
        """
        self.W = W
        self.alpha = alpha

class Image(object):
    """
    The base image object to be used for radio interferometric images that are
    too large to fit into main memory. The images may either be 2- or 3-D. 
    Basic operations are done by looping over the image, only reading parts into 
    memory at a time and storing what is not used on disk. 
    
    Attributes:
        fn: File name string indicating the file in which the image object should
            be stored.
        dtype: The data type of the image object.
        shape: The 
    """
    
    fn = None
    dtype = None
    shape = None
    deltas = None
    f = None
    im = None
    
    def __init__(self, fn, dtype, coords, grid_params=None, m=None, \
        grid_dtype=None):
            """
            Desc.
            
            Args:
                
            Returns:
                
            """
            
            if m is None:
                self.m = M.Messenger(0)
            else:
                self.m = m
            
            if grid_params is None:
                grid_params = GridParams()
                
            if grid_dtype is None:
                grid_dtype = dtype
    
            self.W = grid_params.W
            self.alpha = grid_params.alpha
    
            self.osim = OversizedImage(fn, dtype, coords, grid_params)
            self.fourier_grid = FourierGrid(fn, grid_dtype, coords, grid_params)
            
            self._parse_coords(coords)
            
            self.fn = fn
            self.dtype = dtype
            
            self._create_file()
    
    def _create_file(self):
        """
        Desc.
        
        Args:
        
        Returns:
            
        """
        create_dataset = True
        if os.path.isfile(self.fn):
            create_dataset = False
            
        self.f = h5py.File(self.fn)
        
        if create_dataset:
            self.im = self.f.create_dataset(DATASET_STRING, self.shape, \
                self.dtype)
        else:
            self.im = self.f[DATASET_STRING]
            self.dtype = self.im.dtype
            if self.im.shape != self.shape:
                raise Exception("Existing Image file does not have the "+\
                    "requested shape!")
    
    def _parse_coords(self, c):
        """
        Desc.
        
        Args:
        
        Returns:
            
        """
        
        self.shape = ()
        self.deltas = ()
        self.mincoords = []
        
        if type(c) is not list:
            raise TypeError("Image cube coordinates must be given as a list.")
        
#        if len(c) != 2 and len(c) != 3:
#            raise Exception("Invalid number of coordinate axes.")
        
        for i in range(len(c)):
            if type(c[i]) is tuple:
                self.deltas += (c[i][0], )
                self.shape += (c[i][1], )
                # zero centered by default
                self.mincoords += [-0.5*c[i][1]*c[i][0]]
            elif type(c[i]) is np.ndarray:
                self.deltas += (c[i], )
                self.shape += (len(c[i]), )
                self.mincoords += [None]
            else:
                raise Exception("Image cube coordinate axes must be defined as"+\
                    " either a tuple containing (dx, Nx) or an array of values.")
                    
    def find_max(self, func=None):
        """
        Desc.
        
        Args:
            func: A function to apply to the image before looking for the maximum.
        
        Returns:
            Maximum value in the image.
        
        """

        if func is None:
            func = lambda x: x
        
        maxval = None
        
        for i in range(self.im.shape[0]):
            im_slice = self.im[i]
            slice_max = np.max(func(im_slice.flatten()))
            if maxval is None or slice_max > maxval:
                maxval = slice_max
        
        return maxval
    
    def find_argmax(self, func=None):
        """
        Desc.
        
        Args:
            func: A function to apply to the image before looking for the maximum.
        
        Returns:
            Array indices of the maximum value in the image.
        
        """

        if func is None:
            func = lambda x: x
            
        maxval = None
        maxargs = ()
        
        for i in range(self.im.shape[0]):
            im_slice = self.im[i]
            slice_max = np.max(func(im_slice.flatten()))
            if maxval is None or slice_max > maxval:
                maxargs = (i,)+np.unravel_index(\
                    np.argmax(func(im_slice.flatten())), im_slice.shape)
                maxval = slice_max
        
        return maxargs
        
        
    def find_min(self, func=None):
        """
        Desc.
        
        Args:
            func: A function to apply to the image before looking for the maximum.
        
        Returns:
            Minimum value in the image.
        
        """
        
        if func is None:
            func = lambda x: x
            
        minval = None
        
        for i in range(self.im.shape[0]):
            im_slice = self.im[i]
            slice_min = np.min(func(im_slice.flatten()))
            if minval is None or slice_min < minval:
                minval = slice_min
        
        return minval
    
    def find_argmin(self, func=None):
        """
        Desc.
        
        Args:
            func: A function to apply to the image before looking for the maximum.
        
        Returns:
            Array indices of the minimum value in the image.
        
        """
        if func is None:
            func = lambda x: x
        
        minval = None
        minargs = ()
        
        for i in range(self.im.shape[0]):
            im_slice = self.im[i]
            slice_min = np.min(func(im_slice.flatten()))
            if minval is None or slice_min > minval:
                minargs = (i,)+np.argmin(func(im_slice.flatten()))
                minval = slice_min
        
        return minargs
        
    
    def copy_to(self, target):
        """
        Desc.
        
        Args:
        
        Returns:
            
        """
        func = lambda x,y: y
        self._looped_operation(self, target, func)

    def init_with_scalar(self, s):
        """
        Desc.
        
        Args:
        
        Returns:
            
        """

        slice_shape = self.f[DATASET_STRING].shape[1:]
        
        func = lambda x, y: np.ones(slice_shape, dtype=self.dtype)*y
        self._looped_operation(s, self, func)
      
    def set_mincoord(self, num, minval):
        """
        Desc.
        
        Args:
        
        Returns:
            
        """
        
        self.mincoords[num] = minval
        
    def _looped_operation(self, other, target, func):
        """
        Desc.
        
        Args:
            
        Returns:
            
        """
        IO = 0
        NP = 1
        SC = 2
        
    
        if isinstance(other, Image):
            othertype = IO
        elif np.isscalar(other):
            othertype = SC
        elif isinstance(other, np.ndarray):
            if other.ndim != self.ndim - 1:                                    
                raise Exception("Operation with invalid data type.")
            othertype = NP
        else:
            raise Exception("Operation with invalid data type.")
        
        for i in range(self.im.shape[0]):    
            rs = self.im[i,:]
            if othertype == IO:
                ro = other.im[i,:]
            elif othertype == SC:
                ro = other
            elif othertype == NP:
                if other.shape != rs.shape:
                    raise Exception("Array size does not match size of a slice"+\
                        " in the Data object!")
                ro = other
            rt = target.im[i,:]
            
            rt = func(rs, ro)
            
            target.im[i,:] = rt
    
    def addto(self, other, target=None):
        """
        Desc.
        
        Args:
        
        Returns:
            
        """
        
        if target is None:
            target = self
        
        func = lambda x,y: x+y
        self._looped_operation(other, target, func)
        
    def subtractoff(self, other, target=None):
        """
        Desc.
        
        Args:
        
        Returns:
            
        """
        
        if target is None:
            target = self
        
        func = lambda x,y: x-y
        self._looped_operation(other, target, func)
        
    def multiplywith(self, other, target=None):
        """
        Desc.
        
        Args:
        
        Returns:
            
        """
        
        if target is None:
            target = self
        
        func = lambda x,y: x*y
        self._looped_operation(other, target, func)
    
    def divideby(self, other, target=None):
        """
        Desc.
        
        Args:
        
        Returns:
            
        """
        
        if target is None:
            target = self
        
        func = lambda x,y: x/y
        self._looped_operation(other, target, func)
    
    def get_axis(self, num):
        """
        Desc.
        
        Args:
        
        Returns:
            
        """
        if type(self.deltas[num]) is np.ndarray:
            return self.deltas[num]
        else:
            return self.mincoords[num] + \
                np.arange(self.shape[num])*self.deltas[num]
        
    def get_axes(self):
        """
        Desc.
        
        Args:
        
        Returns:
            
        """
        a = []
        for i in range(len(self.im.shape)):
            a += [self.get_axis(i)]
            
        return a
        
    def transform(self, data):
        """
        A function that transforms the image onto the given Data object. No
        default transformation is defined. This function must be overloaded.
        
        Args:
            data: A BaseData object on which to transform the image.
        
        Returns:
            Nothing.
        """
        
        raise Exception("No transform defined!")
    
    def crop_osimage(self):
        """
        A function that clips the osim and transfers the result to the im file.
        
        Args:
            
        Returns:
            Nothing
        """
        [phindx, decndx, randx] = self.osim.crop_indices
        
        for i in range(self.shape[0]):
            self.im[i] = self.osim.im[i+phindx, decndx:decndx+self.shape[1],\
                randx:randx+self.shape[2]]
    
    def zeropad_image(self):
        """
        A function that transfers the current image to the oversized image.
        
        Args:
            None
        
        Returns:
            Nothing.

        """
        
        self.osim.multiplywith(0.)
        
        [phindx, decndx, randx] = self.osim.crop_indices
        
        for i in range(self.shape[0]):
            self.osim.im[i+phindx, decndx:decndx+self.shape[1],\
                randx:randx+self.shape[2]] = self.im[i]
        

class OversizedImage(Image):
    
    def __init__(self, fn, dtype, coords, grid_params):
        """
        Desc.
        
        Args:
        
        Returns:
            
        """
        
        if grid_params is None:
            grid_params = GridParams()

        self.W = grid_params.W
        self.alpha = grid_params.alpha
        
        self.dtype = dtype
        
        self.fn = _make_fn(fn, 'os')

        self._parse_coords(coords)

        self._create_file()
        

    def _parse_coords(self, c):
        """
        Desc.
        
        Args:
            
        Returns:
            Nothing
        """
        self.shape = ()
        self.deltas = ()
        self.mincoords = []
        self.crop_indices = ()
        
        if type(c) is not list:
            raise TypeError("Image cube coordinates must be given as a list.")
        
        if len(c) != 2 and len(c) != 3:
            raise Exception("Invalid number of coordinate axes.")
        
        for i in range(len(c)):
            if type(c[i]) is tuple:
                self.deltas += (c[i][0], )
                self.shape += (int(c[i][1]*self.alpha), )
                # Zero centered by default
                self.mincoords += [-0.5*int(c[i][1]*self.alpha)*c[i][0]]
                self.crop_indices += (int(0.5*c[i][1]*(self.alpha-1.)), )
            elif type(c[i]) is np.ndarray:
                self.deltas += (c[i], )
                self.shape += (len(c[i]), )
                self.mincoords += [None]
                self.crop_indices += (None, )
            else:
                raise Exception("Image cube coordinate axes must be defined as"+\
                    " either a tuple containing (dx, Nx) or an array of values.")
                    
        
class FourierGrid(Image):
    
    def __init__(self, fn, dtype, coords, grid_params):
        """
        Desc.
        
        Args:
        
        Returns:
            
        """
        
        if grid_params is None:
            grid_params = GridParams()

        self.W = grid_params.W
        self.alpha = grid_params.alpha
        
        self.fn = _make_fn(fn, 'fg')
        self.dtype = dtype
        
        self._parse_coords(coords)

        self._create_file()
    
    def _parse_coords(self, c):
        """
        Desc.
        
        Args:
            
        Returns:
            Nothing
        """
        
        self.shape = ()
        self.deltas = ()
        self.mincoords = []
        
        if type(c) is not list:
            raise TypeError("Image cube coordinates must be given as a list.")
        
        if len(c) != 2 and len(c) != 3:
            raise Exception("Invalid number of coordinate axes.")
        
        for i in range(len(c)):
            if type(c[i]) is tuple:
                self.deltas += (1./c[i][0]/c[i][1]/self.alpha, )
                self.shape += (int(c[i][1]*self.alpha), )
                # Zero centered by default
                self.mincoords += [-0.5/c[i][0], ]
            elif type(c[i]) is np.ndarray:
                self.deltas += (None, )
                self.shape += (len(c[i]), )
                self.mincoords += [None, ]
            else:
                raise Exception("Image cube coordinate axes must be defined as"+\
                    " either a tuple containing (dx, Nx) or an array of values.")
        
def _make_fn(fn, modstr):
    """
    Creates a new file name by modifying a given one. The given file is modified 
    according to the following examples (assume the modstr is "mod")
        filename.txt -> filename_mod.txt
        filename -> filename_mod
        file.name.txt -> file.name_mod.txt
        
    Args:
        
    Returns:
        A string containing the modified file name.
    """
    sfn = fn.split('.')
    
    if len(sfn) > 2:
        sfn = ['.'.join(sfn[:-1]), sfn[-1]]
    
    if len(sfn) == 1:
        newfn = sfn[0]+'_'+modstr
    else:
        sfn[0] = sfn[0]+'_'+modstr
        newfn = '.'.join(sfn)
    
    return newfn