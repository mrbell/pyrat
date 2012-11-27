import abc

class BaseData(object):
    """
    Desc.
    
    Attributes:
        
    """
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def init_subgroup(self, spw, nchan, nrecs):
        """
        Create a file group.
        
        Input:
            spw: Number of the subband
            chan_array: Array containing a list of the frequencies that correspond
                to the first index of the data array.
            nrecs: Number of records per channel.
            
        Returns: 
            Nothing.
        """
        
    @abc.abstractmethod
    def transform(self, image):
        """
        A function that transforms the Data object into an Image object. There is
        no default transform function. This must be overwritten by a derived
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
            spw: The spectral window number or subgroup name in which the data \
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
        Desc.
        
        Args:
        Returns:
        """
        
        return
    
    @abc.abstractmethod    
    def subtractoff(self, other, target):
        """
        Desc.
        
        Args:
        Returns:
        """
        
        return
    
    @abc.abstractmethod
    def multiplywith(self, other, target):
        """
        Desc.
        
        Args:
        Returns:
        """
        
        return
    
    @abc.abstractmethod
    def divideby(self, other, target):
        """
        Desc.
        
        Args:
        Returns:
        """
        
        return
    
    @abc.abstractmethod
    def copy_data_to(self, other):
        """
        Desc.
        
        Args:
        Returns:
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
