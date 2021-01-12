#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:12:05 2020

@author: lorenzo
"""

import numpy as np
from pymatgen.core.periodic_table import Element
from pynter.phase_diagram.analysis import ChempotAnalysis, Reservoirs

class ChempotExperimental:
    
    def __init__(self,temperature=300,partial_pressure=1):
        """
        Class to extract and organize data regarding experimental chemical potentials

        Parameters
        ----------
        temperature : (float), optional
            Temperature. The default is 300.
        partial_pressure : (float), optional
            Partial pressure (p/p0). The default is 1.
        """
        self.temperature = temperature
        self.partial_pressure = partial_pressure


    def chempot_ideal_gas(self, mu0, temperature=None,partial_pressure=None):
        """
        Get chemical potential at a given temperature and partial pressure. The chemical potential in standard conditions (mu0)
        has to be know.

        Parameters
        ----------
        mu0 : (float)
            Value of chemical potential in standard conditions.
        temperature : (float), optional
            Temperature. The default is None. If None the value initialized with the class is used.
        partial_pressure : (float), optional
            Partial pressure. The default is None. If None the value initialized with the class is used.

        Returns
        -------
        chempot : (float)
            Value of chemical potential at given T and p/p0.
        """
        temperature = temperature if temperature else self.temperature
        partial_pressure = partial_pressure if partial_pressure else self.partial_pressure
        kb = 8.6173324e-5  # eV / K
        chempot = mu0 + 0.5*kb*temperature*np.log(partial_pressure)
        return chempot
        

        
    def oxygen_standard_chempot(self,temperature=None):
        """
        Get value of oxygen delta mu standard (mu_0(T,Po)) at a speficic temperature
        The data is taken from the following work: 
            Reuter and Scheffler, “Composition, Structure, and Stability of RuO 2 ( 110 ) as a Function of Oxygen Pressure.”
        The value at a specific temperature is extracted from a linear fitting of the behaviour of mu_O with Temperature.
        
        Parameters
        ----------
        temperature : (float)

        Returns
        -------
        Chemical potential at standard p0 at given temperature.

        """
        temperature = temperature if temperature else self.temperature
        T = np.array([100,200,300,400,500,600,700,800,900,1000])
        mu_0 = np.array([-0.08, -0.17,-0.27,-0.38,-0.50,-0.61,-0.73,-0.85,-0.98,-1.10])
        
        coef = np.polyfit(T,mu_0,1)
        poly1d_fn = np.poly1d(coef)
        return poly1d_fn(temperature)
    
    
    def oxygen_partial_pressure_range(self,phase_diagram,target_comp,temperature=None,p_range=(-20,10),get_pressures_as_strings=False):
        """
        Generate Reservoirs object with a set of different chemical potentials starting from a range of oxygen partial pressure.
        The code distinguishes between 2-component and 3-component phase diagrams.
        In the case of a 2-comp PD the other chemical potential is calculated directly from the formation energy of the phase
        with target composition.
        In the case of a 3-comp PD the set of chemical potentials is obtained first calculating the boundary phases starting
        from the constant value of muO, then the arithmetic average between this two points in the stability diagram is taken.
        
        Parameters
        ----------
        phase_diagram : (PhaseDiagram)
            Pymatgen PhaseDiagram object.
        target_comp : (Composition)
            Pymatgen Composition object.
        temperature : (float), optional
            Temperature value. If None the value with which the class is initialized is taken. The default is None.
        p_range : (tuple), optional
            Exponential range in which to evaluate the partial pressure . The default is from 1e-20 to 1e10.
        get_pressures_as_strings : (bool), optional
            Get pressure values (keys in the Reservoirs dict) as strings. The default is set to floats.

        Returns
        -------
        reservoirs
            Reservoirs object. The dictionary is organized as {partial_pressure:chempots}.

        """
        chempots = {}
        reservoirs = {}
        pd = phase_diagram
        temperature = temperature if temperature else self.temperature
        partial_pressures = np.logspace(p_range[0],p_range[1],num=50,base=10)
        mu_standard = self.oxygen_standard_chempot(temperature)
        
        for p in partial_pressures:
            muO = self.chempot_ideal_gas(mu_standard,temperature=temperature,partial_pressure=p) #delta value
            fixed_chempot = {Element('O'):muO} 
            
            if len(pd.elements) == 2: # 2-component PD case
                ca = ChempotAnalysis(pd)
                mu = ca.calculate_single_chempot(target_comp, fixed_chempot)
                for el in target_comp.elements:
                    if el is not Element('O'):
                        chempots = fixed_chempot.copy()
                        chempots.update({el:mu})
                    
            elif len(pd.elements) == 3: # 3-component PD case
                ca = ChempotAnalysis(pd)
                res = ca.boundary_analysis(target_comp, fixed_chempot)
                for el in list(res.values())[0]:
                    chempots[el] = np.mean(np.array([mu[el] for mu in res.values()]))
                
            chempots_abs = ca.get_chempots_abs(chempots)
            if get_pressures_as_strings:
                p = "%.1g" % p
                p = str(p)
            reservoirs[p] = chempots_abs
    
        
        return Reservoirs(reservoirs,phase_diagram=pd,are_chempots_delta=False)
                                
        
        
        
        
        
        
        
        
        