#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:37:44 2023

@author: lorenzo
"""
from pynter.testing.core import PynterTest

class JobSettingsTest(PynterTest):
    """
    Provides methods to test job settings dictionary
    """
    @property
    def common_keys(self):
        common_keys = [
            'add_automation',
            'add_stop_array',
            'array_size',
            'filename',
            ]
        return common_keys
    
    @property
    def sbatch_keys(self):
        sbatch_keys =  [
            'ntasks',
            'error',
            'mem-per-cpu',
            'job-name',
            'output',
            'partition',
            'processor',
            'time'
            ]
        return sbatch_keys
    
    def assert_job_settings_equal(self,settings1,settings2):
        """
        Check only common keys to avoid system dependent differences
        """
        for key in self.common_keys:
            actual = settings1[key]
            desired = settings2[key]
            self.assert_object_almost_equal(actual,desired)
        
        for kwarg in self.sbatch_keys: 
            actual = settings1['sbatch_kwargs'][kwarg]
            desired = settings2['sbatch_kwargs'][kwarg]
            self.assert_object_almost_equal(actual,desired)
        