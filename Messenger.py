"""
Messenger.py

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

A class to assist in printing messages to stdout. Knows about verbosity and 
handles fancy formatting.
"""

import datetime, math, sys

VERSION = "0.1.0.0"

class Messenger(object):
    
    structure_string = '.   '
    
    BOLD    = "\033[1m"
    ENDC    = '\033[0m'
    
    def __init__(self, verbosity=0, use_color=True, use_structure=False, \
        add_timestamp=True):
            """
            Initialize the messenger class. Set overall verbosity level, 
            use_color, use_structure flags, and add_timestamp flags.
            """
            self.verbosity = verbosity
            if use_color:
                self.enable_color()
            else:
                self.disable_color()
            
            self.use_structure = use_structure
            self.add_timestamp = add_timestamp
        
    def set_verbosity(self, verbosity):
        self.verbosity = verbosity
    
    def get_verbosity(self):
        return self.verbosity
        
    def disable_color(self):
        """
        Turns off all color formatting.
        """
        self.HEADER1 = ''
        self.HEADER2 = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL    = ''
        self.MESSAGE = ''
    
    def enable_color(self):
        """
        Enable all color formatting.
        """
        # Color definitions    
        self.HEADER1 = '\033[95m'
        self.HEADER2 = '\033[94m'
        self.OKGREEN = '\033[92m'
        self.WARNING = '\033[93m'
        self.FAIL    = '\033[91m'
        self.MESSAGE = ''

    def _get_structure_string(self, level):
        """
        Returns a string containing the structural part of the message based
        on the integer level provided in the input.
        """
        
        string = ''
        if self.use_structure:
            for i in range(level):
                string = string+self.structure_string
        return string
        
    def _get_time_string(self):
        """
        Returns a string containing the structural part of the message based
        on the integer level provided in the input.
        """
        
        string = ''
        if self.add_timestamp:
            string = '['+str(datetime.datetime.today())+"] "
        return string
        
    def _make_full_msg(self, msg, verb_level):
        struct_string = self._get_structure_string(verb_level)
        time_string = self._get_time_string()
        return time_string+struct_string+msg
        
    # PRINT COMMANDS ##########################################################
        
    def warn(self, msg, verb_level=0):
        if verb_level <= self.verbosity:
            full_msg = self._make_full_msg(msg, verb_level)
            print self.WARNING+full_msg+self.ENDC
        
    def header1(self, msg, verb_level=0):
        if verb_level <= self.verbosity:
            full_msg = self._make_full_msg(msg, verb_level)
            print self.BOLD+self.HEADER1+full_msg+self.ENDC
        
    def header2(self, msg, verb_level=1):
        if verb_level <= self.verbosity:
            full_msg = self._make_full_msg(msg, verb_level)
            print self.BOLD+self.HEADER2+full_msg+self.ENDC
    
    def success(self, msg, verb_level=1):
        if verb_level <= self.verbosity:
            full_msg = self._make_full_msg(msg, verb_level)
            print self.OKGREEN+full_msg+self.ENDC
    
    def failure(self, msg, verb_level=0):
        if verb_level <= self.verbosity:
            full_msg = self._make_full_msg(msg, verb_level)
            print self.FAIL+full_msg+self.ENDC
    
    def message(self, msg, verb_level=2):
        if verb_level <= self.verbosity:
            full_msg = self._make_full_msg(msg, verb_level)
            print self.MESSAGE+full_msg+self.ENDC
            
            
def progress(width, part, whole):
    percent = part/whole
    marks = math.floor(width * (percent))
    spaces = math.floor(width - marks)
    loader = '[' + ('=' * int(marks)) + (' ' * int(spaces)) + ']'
    #sys.stdout.write("%s %d/%d %d%%\r" % (loader, part, whole, percent*100))
    sys.stdout.write("%s %d%%\r" % (loader, percent*100))
    if percent <= 1:
        sys.stdout.write("\n")
    sys.stdout.flush()