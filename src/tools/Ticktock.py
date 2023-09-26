#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Author: Harvey
Date: 01/27/2019
Including: some common tools we used in godeye python algorithm code
'''

import time

class ticktock(object):
    '''
    Time comsumption helper class
    '''
    def __init__(self):
        '''
        Record the start time when we initilize the class
        '''
        self.start_time = int(round(time.time()*1000))
    
    def resetTime(self):
        '''
        Reset the time record based on current time stamp
        '''
        self.start_time = int(round(time.time()*1000))
    
    def timeDist(self, time_stamp = None):
        '''
        Calculate the time distance from start time to now or a specific time
        '''
        # check whether the given time stamp is valid
        if time_stamp is None:
            result = int(round(time.time()*1000)) - self.start_time
        else:
            if len(str(time_stamp)) != 13:
                raise Exception("Time Stamp invalid, input time stamp should be 13 digitals format!")
            else:
                result = time_stamp - self.start_time
        return result
