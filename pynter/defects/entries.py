#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 14:33:12 2020

@author: villa
"""


import numpy as np
from scipy.optimize import bisect
from pymatgen.core.units import kb
from pymatgen.core.structure import Structure, PeriodicSite, Lattice
from pymatgen.core.periodic_table import Element
from pymatgen.electronic_structure.dos import FermiDos
from pymatgen.analysis.defects.core import *
import matplotlib
import matplotlib.pyplot as plt
from pynter.defects.pmg_dos import FermiDosCarriersInfo
from pynter.defects.utils import *
from monty.json import MontyDecoder, MSONable


def get_defect_entry_from_jobs(job_defect,job_bulk,corrections,defect_structure=None,multiplicity=None):
    """
    Get defect entry from VaspJob objects of defect and bulk calculation.
    The defect_finder method is used to determine if the defect site is single or 
    a list of defect sites (defect complex). In this way the SingleDefectEntry or the 
    DefectComplexEntry is initialized.

    Parameters
    ----------
    job_defect : (VaspJob)
        Defect calculation.
    job_bulk : (VaspJob)
        Bulk calculation.
    corrections : (dict)
        Dict of corrections for defect formation energy. All values will be summed and
        added to the defect formation energy.
    defect_structure : (Structure)
        Structure of the defect. If None the intial structure of job_defect is taken. The default is None. 
    multiplicity : (int), optional
        Multiplicity of defect within the supercell. The default is None.
        If not provided, for single defects it is determined by pymatgen with symmetry analysis,
        for a defect complex is set to 1.

    Returns
    -------
    SingleDefectEntry or DefectComplexEntry

    """
    defect_structure = defect_structure if defect_structure else job_defect.initial_structure
    bulk_structure = job_bulk.final_structure
    defects = defect_finder(defect_structure,bulk_structure)
    if isinstance(defects,list):
        entry = DefectComplexEntry.from_jobs(job_defect,job_bulk,corrections,defect_structure,multiplicity)
    else:
        entry = SingleDefectEntry.from_jobs(job_defect,job_bulk,corrections,defect_structure,multiplicity)
        
    return entry



class SingleDefectEntry:
    
    def __init__(self,defect,bulk_structure,energy_diff,corrections):
        """
        Initializes the data of a single defect. Inspired from the DefectEntry class in pymatgen (pmg).

        Args:
            defect: Pymatgen defect object (Vacancy, Interstitial or Substitution)
            bulk_structure: Pymatgen Structure without any defects
            energy_diff (float): difference btw energy of defect structure and energy of pure structure
            corrections (dict): Dict of corrections for defect formation energy. All values will be summed and
                                added to the defect formation energy.                           
        """        
        self._defect = defect 
        self._bulk_structure = bulk_structure
        self._energy_diff = energy_diff
        self._corrections = corrections if corrections else {}


    @property
    def defect(self):
        return self._defect
        
    @property
    def bulk_structure(self):
        return self._bulk_structure
    
    @property
    def energy_diff(self):
        return self._energy_diff
    
    @property
    def corrections(self):
        return self._corrections

    @property
    def charge(self):
        return self.defect.charge
    
    @property
    def multiplicity(self):
        return self.defect.multiplicity
    
    @multiplicity.setter
    def multiplicity(self,multiplicity):
        self.defect._multiplicity = multiplicity
        return
    
    @property
    def name(self):
        return self.defect.name

    @property
    def delta_atoms(self):
        """
        Dictionary with Element as keys and particle difference between defect structure
        and bulk structure as values.
        """
        comp_defect = self.defect.defect_composition
        comp_bulk = self.bulk_structure.composition
        return get_delta_atoms_from_comp(comp_defect,comp_bulk)

    @property
    def classname(self):
        return self.__class__.__name__


    @staticmethod
    def from_jobs(job_defect, job_bulk, corrections, defect_structure=None,multiplicity=None):
        """
        Generate SingleDefectEntry object from VaspJob objects.

        Parameters
        ----------
        job_defect : (VaspJob)
            Defect calculation.
        job_bulk : (VaspJob)
            Bulk calculation.
        corrections : (dict)
            Dict of corrections for defect formation energy. All values will be summed and
            added to the defect formation energy.
        defect_structure : (Structure)
            Structure of the defect. If None the intial structure of job_defect is taken. The default is None. 
        multiplicity : (int), optional
            Multiplicity of defect within the supercell. The default is None.
            If not provided is calculated by Pymatgen analysing the symmetry of the structure.

        Returns
        -------
        SingleDefectEntry
        """
        defect_structure = defect_structure if defect_structure else job_defect.initial_structure
        bulk_structure = job_bulk.final_structure
        energy_diff = job_defect.final_energy - job_bulk.final_energy
        charge = job_defect.charge
        
        return SingleDefectEntry.from_structures(defect_structure, bulk_structure, energy_diff, corrections,charge,multiplicity)


    @staticmethod
    def from_structures(defect_structure,bulk_structure,energy_diff,corrections,charge=0,multiplicity=None):
        """
        Generate SingleDefectEntry object from Structure objects.

        Parameters
        ----------
        defect_structure : (Structure)
            Defect structure.
        bulk_structure : (Structure)
            Bulk structure.
        energy_diff (float): 
            Difference btw energy of defect structure and energy of pure structure
        corrections (dict): 
            Dict of corrections for defect formation energy. All values will be summed and
            added to the defect formation energy.  
        charge : (int), optional
            Charge of the defect system. The default is 0.
        multiplicity : (int), optional
            multiplicity of defect within the supercell. The default is None.
            If not provided is calculated by Pymatgen analysing the symmetry of the structure.

        Returns
        -------
        SingleDefectEntry
        """
        dsite,dtype = defect_finder(defect_structure, bulk_structure)
        module = importlib.import_module("pymatgen.analysis.defects.core")
        defect_class = getattr(module,dtype)
        defect = defect_class(bulk_structure,dsite,charge,multiplicity=multiplicity)
        
        return SingleDefectEntry(defect, bulk_structure, energy_diff, corrections)


    def __repr__(self):
        output = [
            "SingleDefectEntry :",
            "Defect = %s %s %s" %(self.defect.__class__.__name__,self.defect.site.specie.symbol,self.defect.site.frac_coords.__str__()),
            "Bulk System = %s" %self.bulk_structure.composition,
            "Energy = %.4f" %self.energy_diff,
            "Corrections = %.4f" %sum([v for v in self.corrections.values()]),
            "Charge = %i" %self.charge,
            "Multiplicity = %i" %self.multiplicity,
            "Name = %s" %self.name,
            "\n"
            ]
        return "\n".join(output)
    
    def __str__(self):
        return self.__repr__()


    def as_dict(self):
        """
        Returns:
            Json-serializable dict representation of SignleDefectData
        """
        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "defect": self.defect.as_dict(),
             "bulk_structure": self.bulk_structure.as_dict(),
             "energy_diff": self.energy_diff,
             "corrections": self.corrections
             }
        return d        

    @classmethod
    def from_dict(cls,d):
        """
        Reconstitute a SingleDefectData object from a dict representation created using
        as_dict().

        Args:
            d (dict): dict representation of SingleDefectData.

        Returns:
            SingleDefectData object
        """
        defect = MontyDecoder().process_decoded(d['defect'])
        bulk_structure = Structure.from_dict(d['bulk_structure'])
        energy_diff = d['energy_diff']
        corrections = d['corrections']
        return cls(defect,bulk_structure,energy_diff,corrections)


    def formation_energy(self,vbm,chemical_potentials,fermi_level=0):
        """
        Compute the formation energy for a defect taking into account a given chemical potential and fermi_level
        Args:
            vbm(float): Valence band maximum of pure structure
            chemical_potentials (dict): Dictionary of elemental chemical potential values.
                Keys are Element objects within the defect structure's composition.
                Values are float numbers equal to the atomic chemical potential for that element.
            fermi_level (float):  Value corresponding to the electron chemical potential.
            """
            
        formation_energy = (self.energy_diff + self.charge*(vbm+fermi_level) + 
                       sum([ self.corrections[correction_type]  for correction_type in self.corrections ]) 
                        ) 
        
        if chemical_potentials:
            chempot_correction = -1 * sum([self.delta_atoms[el]*chemical_potentials[el] for el in self.delta_atoms])
        else:
            chempot_correction = 0
            
        formation_energy = formation_energy + chempot_correction
        
        return formation_energy
    
    def defect_concentration(self, vbm, chemical_potentials, temperature=300, fermi_level=0.0):
        """
        Compute the defect concentration for a temperature and Fermi level.
        Args:
            temperature:
                the temperature in K
            fermi_level:
                the fermi level in eV (with respect to the VBM)
        Returns:
            defects concentration in cm^-3
        """
        n = self.multiplicity * 1e24 / self.bulk_structure.volume
        conc = n * np.exp(-1.0 * self.formation_energy(vbm, chemical_potentials, fermi_level=fermi_level) /
                          (kb * temperature))
        return conc
    
    
    
class DefectComplexEntry:
    
    def __init__(self,defect_list,bulk_structure,energy_diff,corrections,charge=0,multiplicity=None):
        """
        Initializes the data of a defect complex. Inspired from the DefectEntry class in pymatgen (pmg).

        Args:
            defect: Pymatgen defect object (Vacancy, Interstitial or Substitution)
            bulk_structure: Pymatgen Structure without any defects
            energy_diff (float): difference btw energy of defect structure and energy of pure structure
            corrections (dict): Dict of corrections for defect formation energy. All values will be summed and
                                added to the defect formation energy.   
            charge : (int), optional
                Charge of the defect system. The default is 0.
                multiplicity : (int), optional
                multiplicity of defect within the supercell. The default is None.
                If not provided is calculated by Pymatgen analysing the symmetry of the structure.
        """        
        self._defect_list = defect_list 
        self._bulk_structure = bulk_structure
        self._energy_diff = energy_diff
        self._corrections = corrections if corrections else {}
        self._charge = int(charge)
        self._multiplicity = multiplicity if multiplicity else 1


    @property
    def defect_list(self):
        return self._defect_list
        
    @property
    def bulk_structure(self):
        return self._bulk_structure
    
    @property
    def energy_diff(self):
        return self._energy_diff
    
    @property
    def corrections(self):
        return self._corrections

    @property
    def charge(self):
        return self._charge
    
    @property
    def multiplicity(self):
        return self._multiplicity
    
    @multiplicity.setter
    def multiplicity(self,multiplicity):
        self._multiplicity = multiplicity
        return
    
    @property
    def name(self):
        name = '-'.join(self.defect_list_names)
        name = name + '_mult%i' %self.multiplicity
        return name
    
    @property
    def defect_list_names(self):
        names = []
        for d in self.defect_list:
            names.append(d.name.split('_mult', 1)[0])
        return names
    
    @property
    def delta_atoms(self):
        comp_bulk = self.bulk_structure.composition
        da_global = None
        for d in self.defect_list:
            da_single = get_delta_atoms_from_comp(d.defect_composition, comp_bulk)
            if da_global is None:
                da_global = da_single.copy()
            else:
                for e in da_single:
                    prec = da_global[e] if e in da_global.keys() else 0
                    da_global[e] = prec + da_single[e]
        
        return da_global


    @property
    def classname(self):
        return self.__class__.__name__
  
    
    @staticmethod
    def from_jobs(job_defect, job_bulk, corrections, defect_structure=None,multiplicity=None):
        """
        Generate DefectComplexEntry object from VaspJob objects.

        Parameters
        ----------
        job_defect : (VaspJob)
            Defect calculation.
        job_bulk : (VaspJob)
            Bulk calculation.
        corrections : (dict)
            Dict of corrections for defect formation energy. All values will be summed and
            added to the defect formation energy.
        defect_structure : (Structure)
            Structure of the defect. If None the intial structure of job_defect is taken. The default is None. 
        multiplicity : (int), optional
            Multiplicity of defect within the supercell. The default is None.
            If not provided is calculated by Pymatgen analysing the symmetry of the structure.

        Returns
        -------
        DefectComplexEntry
        """ 
        defect_structure = defect_structure if defect_structure else job_defect.initial_structure
        bulk_structure = job_bulk.final_structure
        energy_diff = job_defect.final_energy - job_bulk.final_energy
        charge = job_defect.charge
        
        return DefectComplexEntry.from_structures(defect_structure, bulk_structure, energy_diff, corrections,charge,multiplicity)
        
    
    @staticmethod
    def from_structures(defect_structure,bulk_structure,energy_diff,corrections,charge=0,multiplicity=None):
        """
        Generate DefectComplexEntry object from Structure objects.

        Parameters
        ----------
        defect_structure : (Structure)
            Defect structure.
        bulk_structure : (Structure)
            Bulk structure.
        energy_diff (float): 
            Difference btw energy of defect structure and energy of pure structure
        corrections (dict): 
            Dict of corrections for defect formation energy. All values will be summed and
            added to the defect formation energy.  
        charge : (int), optional
            Charge of the defect system. The default is 0.
        multiplicity : (int), optional
            multiplicity of defect within the supercell. The default is None.
            If not provided is calculated by Pymatgen analysing the symmetry of the structure.

        Returns
        -------
        DefectComplexEntry
        """
        defect_list = []
        defects = defect_finder(defect_structure, bulk_structure)
        for dsite,dtype in defects:
            module = importlib.import_module("pynter.defects.pmg_defects_core") #modified pymatgen version with corrected multiplicity bug in super().init for Interstitials
            defect_class = getattr(module,dtype)
            defect = defect_class(bulk_structure,dsite,charge=charge,multiplicity=1)
            defect_list.append(defect)
        
        return DefectComplexEntry(defect_list, bulk_structure, energy_diff, corrections, charge, multiplicity)
    
    
    def __repr__(self):
        defect_list_string = [(d.__class__.__name__,d.site.specie.symbol,d.site.frac_coords.__str__()) for d in self.defect_list]
        output = [
            "DefectComplexEntry :",
            f"Defects =  {defect_list_string}",
            "Bulk System = %s" %self.bulk_structure.composition,
            "Energy = %.4f" %self.energy_diff,
            "Corrections = %.4f" %sum([v for v in self.corrections.values()]),
            "Charge = %i" %self.charge,
            f"Multiplicity = {self.multiplicity}",
            "Name = %s" %self.name,
            "\n"
            ]
        return "\n".join(output)
    
    def __str__(self):
        return self.__repr__()
    
    
    def as_dict(self):
        """
        Returns:
            Json-serializable dict representation of DefectComplexData
        """
        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "defect_list": [d.as_dict() for d in self.defect_list],
             "bulk_structure": self.bulk_structure.as_dict(),
             "energy_diff": self.energy_diff,
             "corrections": self.corrections,
             "charge": self.charge,
             "multiplicity": self.multiplicity
             }
        return d        

    @classmethod
    def from_dict(cls,d):
        """
        Reconstitute a DefectComplexData object from a dict representation created using
        as_dict().

        Args:
            d (dict): dict representation of DefectComplexData.

        Returns:
            DefectComplexData object
        """
        defect_list = [MontyDecoder().process_decoded(d) for d in d['defect_list']]
        bulk_structure = Structure.from_dict(d['bulk_structure'])
        energy_diff = d['energy_diff']
        corrections = d['corrections']
        charge = d['charge']
        multiplicity = d['multiplicity']
        return cls(defect_list,bulk_structure,energy_diff,corrections,charge,multiplicity)
    
    
    def formation_energy(self,vbm,chemical_potentials,fermi_level=0):
        """
        Compute the formation energy for a defect taking into account a given chemical potential and fermi_level
        Args:
            vbm(float): Valence band maximum of pure structure
            chemical_potentials (dict): Dictionary of elemental chemical potential values.
                Keys are Element objects within the defect structure's composition.
                Values are float numbers equal to the atomic chemical potential for that element.
            fermi_level (float):  Value corresponding to the electron chemical potential.
            """
            
        formation_energy = (self.energy_diff + self.charge*(vbm+fermi_level) + 
                       sum([ self.corrections[correction_type]  for correction_type in self.corrections ]) 
                        ) 
        
        if chemical_potentials:
            chempot_correction = -1 * sum([self.delta_atoms[el]*chemical_potentials[el] for el in self.delta_atoms])
        else:
            chempot_correction = 0
            
        formation_energy = formation_energy + chempot_correction
        
        return formation_energy
    
    def defect_concentration(self, vbm, chemical_potentials, temperature=300, fermi_level=0.0):
        """
        Compute the defect concentration for a temperature and Fermi level.
        Args:
            temperature:
                the temperature in K
            fermi_level:
                the fermi level in eV (with respect to the VBM)
        Returns:
            defects concentration in cm^-3
        """
        n = self.multiplicity * 1e24 / self.bulk_structure.volume
        conc = n * np.exp(-1.0 * self.formation_energy(vbm, chemical_potentials, fermi_level=fermi_level) /
                          (kb * temperature))
        return conc    
    
    
    
