#!/usr/bin/env python3

import invoke # A task execution tool; unused
from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator
# Two of these aren't used yet; B.E. is the parent class of OPENBT
import tempfile # Generate temporary directories/files
from pathlib import Path # To write filepaths
from collections import defaultdict # For setting dictionaries with default values; unused
from scipy.stats import spearmanr # For calculating the spearman coeff
import pickle # For writing to compressed (pickled) test files; unused
import numpy as np # For manipulating arrays, doing math, etc.
import subprocess # For running a process in the command line
from scipy.stats import norm # Similar to pnorm, rnorm, qnorm, etc. in R
import sys; import os # For exit() and checking the config files
import itertools # To makes labels for sobol variable pairs
import pandas as pd # Sobol has column names, so each returned array has to be a pandas df

class OPENBT(BaseEstimator):
    """Class to run openbtcli by using sklearn-like calls"""
    def __init__(self, **kwargs):
        self.ndpost = 1000 # All of these are defaults; overwriting comes later
        self.nskip = 100
        self.nadapt = 1000
        self.power = 2.0
        self.base = .95
        self.tc = 2
        self.pbd = .7
        self.pb = .5
        self.stepwpert = .1
        self.probchv = .1
        self.minnumbot = 5
        self.printevery = 100
        self.numcut = 100
        self.adaptevery = 100
        # Here are the extra parameters that I added, since I wanted to customize them:
        self.overallsd = 1
        self.overallnu = 10
        self.k = 2
        self.ntree = 1
        self.ntreeh = 1
        # I added a few more if statements in _define_params() to make these go smoothly
        self.modelname = "model"
        self.summarystats = "FALSE"
        self.__dict__.update((key, value) for key, value in kwargs.items())
        # print(self.__dict__)
        self._set_model_type()


    def fit(self, X, y):
        """Writes out data and fits model
        """
        self.X_train = np.transpose(X)
        self.f_mean = np.mean(y)
        self.y_train = y - self.f_mean
        self._define_params() # This is where the default variables get overwritten
        print("Writing config file and data")
        self._write_config_file()
        self._write_data()
        print("Running model...")
        self._run_model()
        
        # return attributes to be saved as a separate fit object:
        res = {} # Missing the influence attribute from the R code (what is it??)
        self.minx = int(np.floor(np.min(self.xi[0])))
        self.maxx = int(np.ceil(np.max(self.xi[0])))
        
        for key in self.__dict__.keys():
             res[key] = self.__dict__[key]
        res['minx'] = self.minx; res['maxx'] = self.maxx;
        return res


    def _set_model_type(self):
        models = {"bt": 1,
                  "binomial": 2,
                  "poisson": 3,
                  "bart": 4,
                  "hbart": 5,
                  "probit": 6,
                  "modifiedprobit": 7,
                  "merck_truncated": 8}
        if self.model not in models:
            raise KeyError("Not supported model type")
        self.modeltype = models[self.model]
        if not(isinstance(self.k, (int,float))): # If it wasn't user-inputted
             k_map = {4: 2, 5: 5, 6: 1, 7: 1, 8: 2}
             self.k = k_map[self.modeltype]


    def _update_h_args(self, arg):
        try:
            self.__dict__[arg + "h"] = self.__dict__[arg][1]
            self.__dict__[arg] = self.__dict__[arg][0]
        except:
            self.__dict__[arg + "h"] = self.__dict__[arg]


    def _define_params(self):
        """Set up parameters for the openbtcli
        """
        self.n = self.y_train.shape[0]
        self.p = self.X_train.shape[0]
        # Cutpoins
        if "xicuts" not in self.__dict__:
            self.xi = {}
            maxx = np.ceil(np.max(self.X_train, axis=1))
            minx = np.floor(np.min(self.X_train, axis=1))
            for feat in range(self.p):
                xinc = (maxx[feat] - minx[feat])/(self.numcut+1)
                self.xi[feat] = [
                    np.arange(1, (self.numcut)+1)*xinc + minx[feat]]
        self.rgy = [np.min(self.y_train), np.max(self.y_train)
                    ] if self.modeltype in [1, 4, 5] else [-2, 2]
        self.tau = (self.rgy[1] - self.rgy[0])/(2*np.sqrt(self.ntree)*self.k)
        self.fmeanout = 0 if self.modeltype in [
            1, 4, 5] else norm.ppf(self.f_mean)

        # TODO ntreeh; I think it might be OK to just let the user set it...
        # self.ntreeh = 1    # Removed so the user can set it
        if not(isinstance(self.overallsd, (int,float))): # If it wasn't user-inputted
             if self.modeltype in [1, 4, 5]:
                  self.overallsd = np.std(self.y_train)
             else: self.overallsd = 1
        self.overalllambda = np.square(self.overallsd)
        if not(isinstance(self.overallnu, (int,float))): # If it wasn't user-inputted
             if self.modeltype in [1, 4, 5, 6]:
                  self.overallnu = 10
             else: self.overallnu = 1
        if (self.modeltype == 6) & (isinstance(self.pbd, float)):
            self.pbd = [self.pbd, 0]
        [self._update_h_args(arg) for arg in ["power", "base",
                                              "pbd", "pb", "stepwpert",
                                              "probchv", "minnumbot"]]
        self.xroot = "x"
        self.yroot = "y"
        self.sroot = "s"
        self.chgvroot = "chgv"
        self.xiroot = "xi"
        # Check probit:
        if self.modeltype == 6:
            if self.ntreeh > 1:
                raise ValueError("ntreeh should be 1")
            if self.pbdh > 0:
                raise ValueError("pbdh should be 1")
        # print((self.k, self.overallsd, self.overallnu, self.ntree, self.ntreeh))


    def _write_config_file(self):
        """Create temp directory to write config and data files
        """
        f = tempfile.mkdtemp(prefix="openbtpy_")
        self.fpath = Path(f)
        run_params = [self.modeltype,
                      self.xroot, self.yroot, self.fmeanout,
                      self.ntree, self.ntreeh,
                      self.ndpost, self.nskip,
                      self.nadapt, self.adaptevery,
                      self.tau, self.overalllambda,
                      self.overallnu, self.base,
                      self.power, self.baseh, self.powerh,
                      self.tc, self.sroot, self.chgvroot,
                      self.pbd, self.pb, self.pbdh, self.pbh, self.stepwpert,
                      self.stepwperth,
                      self.probchv, self.probchvh, self.minnumbot,
                      self.minnumboth, self.printevery, "xi", self.modelname,
                      self.summarystats]
        # print(run_params)
        self.configfile = Path(self.fpath / "config")
        with self.configfile.open("w") as tfile:
            for param in run_params:
                tfile.write(str(param)+"\n")
        # print(os.path.abspath(self.configfile))
        # sys.exit('Examining tmp file(s)') # The config file was correct when I looked at it manually.


    def __write_chunks(self, data, no_chunks, var, *args):
        splitted_data = np.array_split(data, no_chunks)
        int_added = 0 if var == "xp" else 1
        for i, ch in enumerate(splitted_data):
            np.savetxt(str(self.fpath / Path(self.__dict__[var+"root"] + str(i+int_added))),
                       ch, fmt=args[0])


    def _write_data(self):
        splits = (self.n - 1) // (self.n/(self.tc))
        self.__write_chunks(self.y_train, splits, "y", '%.7f')
        self.__write_chunks(np.transpose(self.X_train), splits, "x", '%.7f')
        self.__write_chunks(np.ones((self.n), dtype="int"),
                            splits, "s", '%.0f')
        if self.X_train.shape[0] == 1:
             print("1 x variable, so correlation = 1")
             np.savetxt(str(self.fpath / Path(self.chgvroot)), [1], fmt='%.7f')
        else:
             print("2+ x variables")
             np.savetxt(str(self.fpath / Path(self.chgvroot)),
                        [spearmanr(self.X_train, axis=1)[0]], fmt='%.7f')
        for k, v in self.xi.items():
            np.savetxt(
                str(self.fpath / Path(self.xiroot + str(k+1))), v, fmt='%.7f')
        # print(os.path.abspath(self.fpath))
        # sys.exit('Examining tmp file(s)') # The data files were correct:
        # 1 chgv, 1 config, 3 s's, 3 x's, 3 y's, 1 xi (xi had the most data).


    def _run_model(self, train=True):
        cmd = "openbtcli" if train else "openbtpred"
        sp = subprocess.run(["mpirun", "-np", str(self.tc), cmd, str(self.fpath)],
                            stdin=subprocess.DEVNULL, capture_output=True)
        # print(sp)


    def predict(self, X, q_lower=0.025, q_upper=0.975, **kwargs):
        self.p_test = X.shape[1]
        self.n_test = X.shape[0]
        self.q_lower = q_lower; self.q_upper = q_upper
        self.xproot = "xp"
        self.__write_chunks(X, (self.n_test) // (self.n_test/(self.tc)),
                            self.xproot,
                            '%.7f')
        self.configfile = Path(self.fpath / "config.pred")
        pred_params = [self.modelname, self.modeltype,
                       self.xiroot, self.xproot, self.ndpost,
                       self.ntree, self.ntreeh,
                       self.p_test, self.tc, self.f_mean]
        # print(self.ntree); print(self.ntreeh)
        with self.configfile.open("w") as pfile:
            for param in pred_params:
                pfile.write(str(param)+"\n")
        self._run_model(train=False)
        self._read_in_preds()
        res = {}
        res['mdraws'] = self.mdraws; res['sdraws'] = self.sdraws;
        res['mmean'] = self.mmean; res['smean'] = self.smean;
        res['mmeans'] = self.mmeans; res['smeans'] = self.smeans;
        res['msd'] = self.msd; res['ssd'] = self.ssd;
        res['m_5'] = self.m_5; res['s_5'] = self.s_5;
        res['m_lower'] = self.m_lower; res['s_lower'] = self.s_lower;
        res['m_upper'] = self.m_upper; res['s_upper'] = self.s_upper;
        res['q_lower'] = self.q_lower; res['q_upper'] = self.q_upper;
        res['x_test'] = X; res['modeltype'] = self.modeltype
        return res


    def _read_in_preds(self):
        mdraw_files = sorted(list(self.fpath.glob("model.mdraws*")))
        sdraw_files = sorted(list(self.fpath.glob("model.sdraws*")))
        mdraws = []
        for f in mdraw_files:
            mdraws.append(np.loadtxt(f))
        # print(mdraws[0].shape); print(len(mdraws))
        self.mdraws = np.concatenate(mdraws, axis=1) # Got rid of the transpose
        sdraws = []
        for f in sdraw_files:
            sdraws.append(np.loadtxt(f))
        self.sdraws = np.concatenate(sdraws, axis=1) # Got rid of the transpose
        # print(self.sdraws.shape); print(self.mdraws.shape)
        
        # New (added by me), since R returns arrays like these by default:
        # Calculate mmean(s) and smean(s) arrays (non-plural -> 1 number, repeated),
        # and related statistics
        self.mmeans = np.empty(len(self.mdraws[0]))
        self.smeans = np.empty(len(self.sdraws[0]))
        self.msd = np.empty(len(self.mdraws[0]))
        self.ssd = np.empty(len(self.mdraws[0]))
        self.m_5 = np.empty(len(self.mdraws[0]))
        self.s_5 = np.empty(len(self.mdraws[0]))
        self.m_lower = np.empty(len(self.mdraws[0]))
        self.s_lower = np.empty(len(self.sdraws[0]))
        self.m_upper = np.empty(len(self.mdraws[0]))
        self.s_upper = np.empty(len(self.sdraws[0]))
        # print(self.mmeans.shape); print(self.smeans.shape)
        for j in range(len(self.mdraws[0])):
             self.mmeans[j] = np.mean(self.mdraws[:, j])
             self.smeans[j] = np.mean(self.sdraws[:, j])
             self.msd[j] = np.std(self.mdraws[:, j])
             self.ssd[j] = np.std(self.sdraws[:, j])
             self.m_5[j] = np.percentile(self.mdraws[:, j], 0.50)
             self.s_5[j] = np.percentile(self.sdraws[:, j], 0.50)
             self.m_lower[j] = np.percentile(self.mdraws[:, j], self.q_lower)
             self.s_lower[j] = np.percentile(self.sdraws[:, j], self.q_lower)
             self.m_upper[j] = np.percentile(self.mdraws[:, j], self.q_upper)
             self.s_upper[j] = np.percentile(self.sdraws[:, j], self.q_upper)
        self.mmean = np.ones(len(self.mdraws[0]))*np.mean(self.mmeans)
        self.smean = np.ones(len(self.sdraws[0]))*np.mean(self.smeans)


    def clean_model(self):
        subprocess.run(f"rm -rf {str(self.fpath)}", shell=True)
       
          
#-----------------------------------------------------------------------------
# My functions:
    def _read_in_vartivity(self, q_lower, q_upper):
        vdraws_files = sorted(list(self.fpath.glob("model.vdraws")))
        self.vdraws = np.array([])
        for f in vdraws_files:
            self.vdraws = np.append(self.vdraws, np.loadtxt(f))
        # self.vdraws[3] = 0.5 # to test transposing/ normalizing counts
        self.vdraws = self.vdraws.reshape(self.ndpost, self.p)
        # print(self.vdraws.shape); print(self.vdraws)
        # Normalize counts:
        colnorm = np.array([])
        for i in range(len(self.vdraws)): # should = ndpost in most cases
             colnorm = np.append(colnorm, self.vdraws[i].sum())
        idx = np.where(colnorm > 0)[0] # print(idx)
        # print(colnorm); print(idx)
        
        colnorm = colnorm.reshape(self.ndpost, 1) # Will always have 1 column since we summed
        # print(colnorm.shape); print(idx.shape)
        self.vdraws[idx] = self.vdraws[idx] / colnorm[idx]
        # print(self.vdraws[0:60]); print(self.vdraws.shape)
        
        self.mvdraws = np.empty(self.p)
        self.vdraws_sd = np.empty(self.p)
        self.vdraws_5 = np.empty(self.p)
        self.q_lower = q_lower; self.q_upper = q_upper
        self.vdraws_lower = np.empty(self.p)
        self.vdraws_upper = np.empty(self.p)

        for j in range(len(self.vdraws[0])): # (should = self.p)
             self.mvdraws[j] = np.mean(self.vdraws[:, j])
             self.vdraws_sd[j] = np.std(self.vdraws[:, j])
             self.vdraws_5[j] = np.percentile(self.vdraws[:, j], 0.50)
             self.vdraws_lower[j] = np.percentile(self.vdraws[:, j], self.q_lower)
             self.vdraws_upper[j] = np.percentile(self.vdraws[:, j], self.q_upper)
        if (len(self.vdraws[0]) == 1): #  Make the output just a double, not a 2D array
             self.mvdraws = self.mvdraws[0]
             self.vdraws_sd = self.vdraws_sd[0]
             self.vdraws_5 = self.vdraws_5[0]
             self.vdraws_lower = self.vdraws_lower[0]
             self.vdraws_upper = self.vdraws_upper[0]
             
        # Now do everything again for the "h" version of all these quantities: 
        vdrawsh_files = sorted(list(self.fpath.glob("model.vdrawsh")))
        self.vdrawsh = np.array([])
        for f in vdrawsh_files:
            self.vdrawsh = np.append(self.vdrawsh, np.loadtxt(f))
        self.vdrawsh = self.vdrawsh.reshape(self.ndpost, self.p)
        # Normalize counts:
        colnormh = np.array([])
        for i in range(len(self.vdrawsh)): # should = ndpost in most cases
             colnormh = np.append(colnormh, self.vdrawsh[i].sum())
        idxh = np.where(colnormh > 0)[0]
        colnormh = colnormh.reshape(self.ndpost, 1) # Will always have 1 column since we summed
        self.vdrawsh[idxh] = self.vdrawsh[idxh] / colnormh[idxh]
        
        self.mvdrawsh = np.empty(self.p)
        self.vdrawsh_sd = np.empty(self.p)
        self.vdrawsh_5 = np.empty(self.p)
        self.vdrawsh_lower = np.empty(self.p)
        self.vdrawsh_upper = np.empty(self.p)

        for j in range(len(self.vdrawsh[0])): # (should = self.p)
             self.mvdrawsh[j] = np.mean(self.vdrawsh[:, j])
             self.vdrawsh_sd[j] = np.std(self.vdrawsh[:, j])
             self.vdrawsh_5[j] = np.percentile(self.vdrawsh[:, j], 0.50)
             self.vdrawsh_lower[j] = np.percentile(self.vdrawsh[:, j], self.q_lower)
             self.vdrawsh_upper[j] = np.percentile(self.vdrawsh[:, j], self.q_upper)
             
        if (len(self.vdrawsh[0]) == 1): #  Make the output just a double, not a 2D array
             self.mvdrawsh = self.mvdrawsh[0]
             self.vdrawsh_sd = self.vdrawsh_sd[0]
             self.vdrawsh_5 = self.vdrawsh_5[0]
             self.vdrawsh_lower = self.vdrawsh_lower[0]
             self.vdrawsh_upper = self.vdrawsh_upper[0]
         
             
    def vartivity(self, q_lower=0.025, q_upper=0.975):
        """Calculate and return variable activity information
        """
        # params (all are already set, actually)
        # self.p = len(self.xi[0][0])? # This definition is bad, but p is already defined in define_params()
        # Write to config file:
        vartivity_params = [self.modelname, self.ndpost, self.ntree,
                            self.ntreeh, self.p]
        self.configfile = Path(self.fpath / "config.vartivity")
        # print(vartivity_params)
        with self.configfile.open("w") as tfile:
            for param in vartivity_params:
                tfile.write(str(param)+"\n")
        # Run vartivity program  -- it's not actually parallel so no call to mpirun.
        # run_local = os.path.exists("openbtvartivity") # Doesn't matter, since the command is the same either way
        cmd = "openbtvartivity"
        sp = subprocess.run([cmd, str(self.fpath)],
                            stdin=subprocess.DEVNULL, capture_output=True)
        # print(sp)
        # Read in result (and set extra attributes like .5, .lower, .upper, etc.):
        self._read_in_vartivity(q_lower, q_upper)
        
        # Compile all the new attributes into something that will be saved as "fitv" when the function is called:
        res = {}
        res['vdraws'] = self.vdraws; res['vdrawsh'] = self.vdrawsh; 
        res['mvdraws'] = self.mvdraws; res['mvdrawsh'] = self.mvdrawsh; 
        res['vdraws_sd'] = self.vdraws_sd; res['vdrawsh_sd'] = self.vdrawsh_sd; 
        res['vdraws_5'] = self.vdraws_5; res['vdrawsh_5'] = self.vdrawsh_5; 
        res['vdraws_lower'] = self.vdraws_lower; res['vdrawsh_lower'] = self.vdrawsh_lower; 
        res['vdraws_upper'] = self.vdraws_upper; res['vdrawsh_5'] = self.vdrawsh_upper; 
        res['q_lower'] = self.q_lower; res['q_upper'] = self.q_upper;
        res['modeltype'] = self.modeltype
        return res
   
    
    def _read_in_sobol(self, q_lower, q_upper):
        sobol_draws_files = sorted(list(self.fpath.glob("model.sobol*")))
        # print(sobol_draws_files)
        self.so_draws = np.loadtxt(sobol_draws_files[0])
        for i in range(1, self.tc):
             self.so_draws = np.vstack((self.so_draws,
                                           np.loadtxt(sobol_draws_files[i])))   
        # print(self.so_draws.shape); print(self.so_draws[0:10])
        labs_temp = list(itertools.combinations(range(1, self.p + 1), 2))
        labs = np.empty(len(labs_temp), dtype = '<U4')
        for i in range(len(labs_temp)):
             labs[i] =  ', '.join(map(str, labs_temp[i]))
        # print(self.so_draws); print(self.so_draws.shape)
        nrow = self.so_draws.shape[0]; ncol = self.so_draws.shape[1]
        draws = self.so_draws; p = self.p # ^ Shorthand to make the next lines shorter
        self.vidraws = draws[:, 1:p]
        self.vijdraws = draws[:, (p+1):int((p+p*(p-1)/2))]
        self.tvidraws = draws[:, (ncol-p):(ncol-1)]
        self.vdraws = draws[:, ncol]
        self.sidraws = self.vidraws / self.vdraws
        self.sijdraws = self.vijdraws / self.vdraws
        self.tsidraws = self.tvidraws / self.vdraws
        # Compute a ton of sobol statistics:
        self.msi = np.empty(self.p)
        self.msi_sd = np.empty(self.p)
        self.si_5 = np.empty(self.p)
        self.q_lower = q_lower; self.q_upper = q_upper
        self.si_lower = np.empty(self.p)
        self.si_upper = np.empty(self.p)
        for j in range(len(self.sidraws[0])): # (should = self.p?)
             self.msi[j] = np.mean(self.sidraws[:, j])
             self.msi_sd[j] = np.std(self.sidraws[:, j])
             self.si_5[j] = np.percentile(self.sidraws[:, j], 0.50)
             self.si_lower[j] = np.percentile(self.sidraws[:, j], self.q_lower)
             self.si_upper[j] = np.percentile(self.sidraws[:, j], self.q_upper)
             
        if (len(self.sidraws[0]) == 1): #  Make the output just a double, not a 2D array
             self.msi = self.msi[0]
             self.msi_sd = self.msi_sd[0]
             self.si_5 = self.si_5[0]
             self.si_lower = self.si_lower[0]
             self.si_upper = self.si_upper[0]
        """
        names(res$msi)=paste("S",1:p,sep="")
        names(res$msi.sd)=paste("S",1:p,sep="")
        names(res$si.5)=paste("S",1:p,sep="")
        names(res$si.lower)=paste("S",1:p,sep="")
        names(res$si.upper)=paste("S",1:p,sep="") # Again, implement if pandas are implemented
        """
        
        # Do this again for i,j:
        self.msij = np.empty(self.p)
        self.msij_sd = np.empty(self.p)
        self.sij_5 = np.empty(self.p)
        self.sij_lower = np.empty(self.p)
        self.sij_upper = np.empty(self.p)
        for j in range(len(self.sijdraws[0])): # (should = self.p?)
             self.msij[j] = np.mean(self.sijdraws[:, j])
             self.msij_sd[j] = np.std(self.sijdraws[:, j])
             self.sij_5[j] = np.percentile(self.sijdraws[:, j], 0.50)
             self.sij_lower[j] = np.percentile(self.sijdraws[:, j], self.q_lower)
             self.sij_upper[j] = np.percentile(self.sijdraws[:, j], self.q_upper)
             
        if (len(self.sijdraws[0]) == 1): #  Make the output just a double, not a 2D array
             self.msij = self.msij[0]
             self.msij_sd = self.msij_sd[0]
             self.sij_5 = self.sij_5[0]
             self.sij_lower = self.sij_lower[0]
             self.sij_upper = self.sij_upper[0]   
             
        # Do this again for t:
        self.mtsi = np.empty(self.p)
        self.mtsi_sd = np.empty(self.p)
        self.tsi_5 = np.empty(self.p)
        self.tsi_lower = np.empty(self.p)
        self.tsi_upper = np.empty(self.p)
        for j in range(len(self.tsidraws[0])): # (should = self.p?)
             self.mtsi[j] = np.mean(self.tsidraws[:, j])
             self.mtsi_sd[j] = np.std(self.tsidraws[:, j])
             self.tsi_5[j] = np.percentile(self.tsidraws[:, j], 0.50)
             self.tsi_lower[j] = np.percentile(self.tsidraws[:, j], self.q_lower)
             self.tsi_upper[j] = np.percentile(self.tsidraws[:, j], self.q_upper)
             
        if (len(self.tsidraws[0]) == 1): #  Make the output just a double, not a 2D array
             self.mtsi = self.mtsi[0]
             self.mtsi_sd = self.mtsi_sd[0]
             self.tsi_5 = self.tsi_5[0]
             self.tsi_lower = self.tsi_lower[0]
             self.tsi_upper = self.tsi_upper[0] 
             
             
    def sobol(self, cmdopt = 'serial', q_lower=0.025, q_upper=0.975):  
        """Calculate Sobol indices (more accurate than vartivity)
        """
        # Write to config file:
        sobol_params = [self.modelname, self.xiroot, self.ndpost, self.ntree,
                            self.ntreeh, self.p, self.minx, self.maxx, self.tc]
        self.configfile = Path(self.fpath / "config.sobol")
        # print(sobol_params)
        with self.configfile.open("w") as tfile:
            for param in sobol_params:
                tfile.write(str(param)+"\n")
        # Run sobol program: optional to use MPI.
        cmd = "openbtsobol"
        if(cmdopt == 'serial'):
             sp = subprocess.run([cmd, str(self.fpath)],
                            stdin=subprocess.DEVNULL, capture_output=True)
        elif(cmdopt == 'MPI'):
             sp = subprocess.run(["mpirun", "-np", str(self.tc), cmd, str(self.fpath)],
                            stdin=subprocess.DEVNULL, capture_output=True)
        else:
             sys.exit('Invalid cmdopt (command option)')
        # print(sp)
        # Read in result (and set extra attributes like --- etc.):
        self._read_in_sobol(q_lower, q_upper)
        # Compile all the new attributes into something that will be saved as "fits" when the function is called:
        res = {}
        # colnames(res$vidraws)=paste("V",1:p,sep="") # Implement this (and all the other colnames) if you use pandas
        # Set all of the self variables/attributes to res here:
        res['vidraws'] = self.vidraws;
        
        
        
        res['q_lower'] = self.q_lower; res['q_upper'] = self.q_upper;
        res['modeltype'] = self.modeltype; res['so_draws'] = self.so_draws
        return res
   





# Scratch lines:
# os.path.exists("openbtvartivity"); os.path.abspath("openbtvartivity")