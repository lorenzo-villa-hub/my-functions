
import os
import os.path as op
from pynter.data.jobs import VaspJob
from pynter.slurm.interface import HPCInterface
import pandas as pd
import importlib


class Dataset:
    
    def __init__(self,path=None,name=None,jobs=None): 
        """
        Class to store sets of calculations

        Parameters
        ----------
        path : (str), optional
            Path of dataset directory. The default is None. If None the work dir is used.
        name : (str), optional
            Name to assign to dataset. The default is None. If None the folder name is used.
        jobs : (list), optional
            List of Job objects belonging to the dataset. The default is None.
        """
        
        self.path = path if path else os.getcwd()
        self.name = name if name else op.basename(op.abspath(self.path))
        self.jobs = jobs
        if jobs:
            self._group_jobs()

        self._localdir = HPCInterface().localdir
        self._workdir = HPCInterface().workdir
        self._path_relative = op.abspath(self.path).replace(self._localdir,'')
        
        self.path_in_hpc = self._workdir + self._path_relative


    def __str__(self):     
        printout = f'Dataset "{self.name}":\n'
        for j in self.jobs:
            printout += j.__str__() + '\n'
        return printout
            
    def __repr__(self):
        return self.__str__()

    @staticmethod
    def from_directory(path,job_script_filename='job.sh'): 
        """
        Static method to build Dataset object from a directory. Jobs are selected based on where the job bash script
        is present. VaspJobs are selected based on where all input files are present (INCAR,KPOINTS,POSCAR,POTCAR).

        Parameters
        ----------
        path : (str)
            Parent directory of the dataset.
        job_script_filename : (str), optional
            Filename of job bash script. The default is 'job.sh'.
        """
        jobs = []
        for root , dirs, files in os.walk(path):
            if files != [] and job_script_filename in files:
                if ('INCAR' and 'KPOINTS' and 'POSCAR' and 'POTCAR') in files:
                    j = VaspJob.from_directory(root,job_script_filename=job_script_filename)
                    j.job_script_filename = job_script_filename
                    jobs.append(j)
     
        return  Dataset(path=path,jobs=jobs)
        
    @property
    def groups(self):
        """Directory names of the first subdirectories in the dataset path."""
        return next(os.walk(self.path))[1]


    def _group_jobs(self):
        """
        Create "groups" and "nodes" attributes based on the common subpaths of the jobs.
        The first subdirectories of the parent dataset dir will be the groups. All of the following
        subpaths will be nodes
        """
        path = op.abspath(self.path)
        gjobs = {}
        groups = self.groups
        for group in groups:
            gjobs[group] = {}
            group_path = op.join(path,group)
            for job in self.jobs:
                if group in job.path:
                    jpath = op.abspath(job.path)
                    nodes = jpath.replace(op.commonpath([group_path,jpath]),'')
                    gjobs[group][nodes] = job
                    job.group = group
                    job.nodes = nodes   
 

    def create_job(self,job_class,group='',nodes='',inputs=None,job_settings=None,
                   outputs=None,job_script_filename='job.sh',name=None):
        """
        Create Job object and add it to the dataset

        Parameters
        ----------
        job_class : (str) 
            Name of the job class. For now available only 'VaspJob'.
        group : (str), optional
            Name of the group to which the job belongs. The default is ''.
        nodes : (str), optional
            Name of the nodes of the job. The default is ''.
        inputs : (dict), optional
            Dictionary with input data. The default is None.
        job_settings : (dict), optional
            Dictionary with job settings. The default is None. Documentation in ScriptHandler class in slurm.job_script module
        outputs : (dict), optional
            Dictionary with output data. The default is None.
        job_script_filename : (str), optional
            Filename of job script. The default is 'job.sh'.
        name : (str)
            Name of the job. If None the name is searched in the job script.
        """
        
        path = op.join(self.path,group,nodes)
        module = importlib.import_module("pynter.data.jobs")
        jobclass = getattr(module, job_class)
        
        job = jobclass(path=path,inputs=inputs,job_settings=job_settings,outputs=outputs,
                       job_script_filename=job_script_filename,name=name)
        job.group = group
        job.nodes = nodes
        if self.jobs:
            self.jobs.append(job)
        else:
            self.jobs = [job]
        
        return
        

    def get_jobs_inputs(self):
        """Read inputs for all jobs from the data stored in the respective directories"""
        for j in self.jobs:
            j.get_inputs()


    def get_jobs_outputs(self):
        """Read output for all jobs from the data stored in respective directories"""
        for j in self.jobs:
            j.get_outputs()

    
    def jobs_table(self,jobs=[],properties_to_display=[]):
        """
        Create a pandas DataFrame object to display the jobs in the dataset.

        Parameters
        ----------
        jobs : (list), optional
            List of jobs to display in the table. The default is []. If [] the attribute "jobs" is used.
        properties_to_display : (list), optional
            List of kwargs with methods in the Job class. The properties referred to these will be added
            to the table. The default is [].

        Returns
        -------
        df : (DataFrame object)

        """
        jobs = jobs if jobs else self.jobs
        table = []
        index = []
        for j in jobs:
            index.append(j.name)
            d = {}
            d['formula'] = j.formula
            d['group'] = j.group
            d['nodes'] = j.nodes
            d['is_converged'] = j.is_converged
            for feature in properties_to_display:
                d[feature] = getattr(j,feature) ()
            table.append(d)
            
        df = pd.DataFrame(table,index=index)
        df.index.name = 'job_name'
        
        return df
            

    def queue(self,stdouts=False):
        """
        Display queue from HPC. If stdouts is True returns out and err strings.
        
        Returns
        -------
        stdout : (str)
            Output.
        stderr : (str)
            Error.
        """
        hpc = HPCInterface()
        stdout,stderr = hpc.qstat()
        if stdouts:
            return stdout,stderr
        else:
            return


    def select_jobs(self,names=None,groups=None,common_node=None,**kwargs):
        """
        Function to filter jobs based on different selection criteria.
        The priority of the selection criterion follows the order of the input
        parameters. When more than one criterion is present, all of them need to be 
        satisfied for the job to be selected.

        Parameters
        ----------
        names : (str), optional
            Job name. The default is None.
        groups : (list), optional
            List of groups that jobs need to belong to. The default is None.
        common_node : (str), optional
            String that needs to be present in the node. The default is None.
        **kwargs : (dict)
            Properties that the jobs need to satisfy. Keys are referred to methods 
            present in the relative Job class.

        Returns
        -------
        sel_jobs : (list)
            List with filtered jobs. If list has one element only the element is returned.
        """
        sel_jobs = self.jobs.copy()
        jobs = sel_jobs.copy()
        for j in jobs:
            if names:
                if j.name not in names:
                    sel_jobs.remove(j)
        
        jobs = sel_jobs.copy()
        for j in jobs:
            if groups:
                if j.group not in groups:
                    sel_jobs.remove(j)

        jobs = sel_jobs.copy()
        for j in jobs:
            if common_node:
                if common_node not in j.nodes:
                    sel_jobs.remove(j)
        
        for feature in kwargs:
            jobs = sel_jobs.copy()
            for j in jobs:
                job_feature = getattr(j,feature) ()
                if job_feature != kwargs[feature]:
                    sel_jobs.remove(j)
         
        if len(sel_jobs) == 1:
            sel_jobs = sel_jobs[0]
            
        return sel_jobs


    def sync_dataset_from_hpc(self,stdouts=False):
        """
        Sync Dataset project folder from HPC to local machine

        Parameters
        ----------
        stdouts : (bool), optional
            Return output and error strings. The default is False.

        Returns
        -------
        stdout : (str)
            Output.
        stderr : (str)
            Error.
        """
        hpc = HPCInterface()
        abs_path = op.abspath(self.path)
        localdir = abs_path 
        print(localdir,self.path_in_hpc)
        stdout,stderr = hpc.rsync_from_hpc(localdir=localdir,remotedir=self.path_in_hpc)
        if stdouts:
            return stdout,stderr
        else:
            return


    def sync_dataset_to_hpc(self,stdouts=False):
        """
        Sync Dataset project folder from local machine to HPC

        Parameters
        ----------
        stdouts : (bool), optional
            Return output and error strings. The default is False.

        Returns
        -------
        stdout : (str)
            Output.
        stderr : (str)
            Error.
        """
        hpc = HPCInterface()
        abs_path = op.abspath(self.path)
        localdir = abs_path 
        print(localdir,self.path_in_hpc)
        stdout,stderr = hpc.rsync_to_hpc(localdir=localdir,remotedir=self.path_in_hpc)
        if stdouts:
            return stdout,stderr
        else:
            return

           
    def sync_jobs(self):
        """Sync job data from HPC to local machine"""
        for j in self.jobs:
            j.sync_from_hpc()
        self.get_jobs_outputs()
        return
        
                
    def write_jobs_input(self):
        """Write jobs inputs to files"""
        for job in self.jobs:
            job.write_input()
            
        
        