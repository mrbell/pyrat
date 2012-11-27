"""
ParsetManager.py

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

An abstract class to read, parse, and manage parset files. Each application
should extend this class to define parameters that are relevant for the
application.

General parset syntax:
    
    One parameter per line
    Seperated by the delimiter character (defined in the base class)
    Empty lines are OK
    Commented lines must begin with the comment character
"""

VERSION = "0.1.0.0"

class ParsetManager(object):
    
    COMMENT_CHAR = "#"
    DELIMITER_CHAR = "="
    KEYLEN = 12
    parset_def = None
    parset = None
    
    def __init__(self):
        """
        Read in the parset file given in the file fn. 
        """
        
        if self.parset_def is None:
            raise Exception("No parameters have been defined!")
        
        self.parset = dict()
            
    def parse_file(self, fn):
        f = open(fn)
        
        parsed_file = dict()        
        
        for line in f:
            if line[0] == self.COMMENT_CHAR:
                continue
            
            sline = line.split(self.DELIMITER_CHAR)
            
            if len(sline) != 2:
                continue
                
            key = sline[0].strip().lower()
            value = sline[1].strip()
            
            if key[0] == self.COMMENT_CHAR:
                continue
            
            parsed_file[key] = value
        
        f.close()
        
        for i in self.parset_def.iterkeys():            
            if self.parset_def[i][1] is None and not parsed_file.has_key(i):
                raise Exception("Required parameter "+str(i)\
                    +" not included in the parset file!")
            
            if not parsed_file.has_key(i):
                value = self.parset_def[i][0](self.parset_def[i][1])
            else:
                value = self.parset_def[i][0](parsed_file[i])
            
            self.parset[i] = value
            
    def print_parset(self):
        """
        Prints the current parset.
        """
        if self.parset is None and self.parset_def is None:
            raise Exception("No parset or parset_def available to print.")
            
        if self.parset is None:
            print "No parset loaded! Printing parset definition."
            self.print_help()
        else:    
            for i in self.parset.iterkeys():
                if len(str(i)) <= self.KEYLEN:
                    print '    {0:{width}}: {1}'.format(i, \
                        str(self.parset[i]), width=self.KEYLEN)
                else:
                    print "    {0}:".format(str(i))
                    print '                  {0}'.format(self.parset[i])
            
    
    def print_help(self):
        """
        Prints the list of parameters and their description.
        """

        if self.parset_def is None:
            raise Exception("No parameters have been defined!")
        
        print "ParsetManager v."+VERSION
        print "*****************************"
        print "ParameterSet file definition "
        print "*****************************"
        print "File syntax:"
        print "    + A valid parset file consists of one pair of parameter "+\
            "name and parameter value per line,"
        print "      separated by a "+self.DELIMITER_CHAR+" character, like so"
        print ""
        print "      parameter_name "+self.DELIMITER_CHAR+" value"
        print ""
        print "    + Lines beginning with "+self.COMMENT_CHAR+ " are skipped."
        print "    + Lines containing invalid parameters are ignored."
        print "    + White space and empty lines are ignored."
        print "    + Parameters with a default value of None are required."
        print "    + Boolean flags should be specified as 1 for True "+\
            "and 0 for False."
        print ""
        print "Parameters:"
        for i in self.parset_def.iterkeys():
            if len(str(i)) <= self.KEYLEN:
                print '    {0:{width}}: {1}'.format(str(i), \
                    self.parset_def[i][2], width=self.KEYLEN)
            else:
                print '    {0}:'.format(str(i))
                print '                  {0}'.format(self.parset_def[i][2])