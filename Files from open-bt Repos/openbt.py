#!/usr/bin/env python3

import invoke
from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator
import tempfile
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr
import pickle
import numpy as np
import subprocess
from scipy.stats import norm


class OPENBT(BaseEstimator):
    """Class to run openbtcli by using sklearn-like calls"""
    def __init__(self, **kwargs):
        self.ndpost = 1000
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
        self.nadapt = 1000
        self.adaptevery = 100
        self.modelname = "model"
        self.summarystats = "FALSE"
        self.__dict__.update((key, value) for key, value in kwargs.items())
        self._set_model_type()

    def fit(self, X, y):
        """Writes out data and fits model
        """
        self.X_train = np.transpose(X)
        self.f_mean = np.mean(y)
        self.y_train = y - self.f_mean
        self._define_params()
        print("Writing config file and data")
        self._write_config_file()
        self._write_data()
        print("Running model...")
        self._run_model()

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

        # TODO ntreeh
        self.ntreeh = 1
        overallsd = np.std(self.y_train) if self.modeltype in [
            1, 4, 5] else 1
        self.overalllambda = np.square(overallsd)
        self.overallnu = 10 if self.modeltype in [1, 4, 5, 6] else 1
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
        # CHeck probit:
        if self.modeltype == 6:
            if self.ntreeh > 1:
                raise ValueError("ntreeh should be 1")
            if self.pbdh > 0:
                raise ValueError("pbdh should be 1")

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
        self.configfile = Path(self.fpath / "config")
        with self.configfile.open("w") as tfile:
            for param in run_params:
                tfile.write(str(param)+"\n")

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
        np.savetxt(str(self.fpath / Path(self.chgvroot)), spearmanr(self.X_train, axis=1)[0],
                   fmt='%.7f')
        for k, v in self.xi.items():
            np.savetxt(
                str(self.fpath / Path(self.xiroot + str(k+1))), v, fmt='%.7f')

    def _run_model(self, train=True):
        cmd = "openbtcli" if train else "openbtpred"
        sp = subprocess.run(["mpirun", "-np", str(self.tc), cmd, str(self.fpath)],
                            stdin=subprocess.DEVNULL, capture_output=True)
        # print(sp)

    def predict(self, X, **kwargs):
        self.p_test = X.shape[1]
        self.n_test = X.shape[0]
        self.xproot = "xp"
        self.__write_chunks(X, (self.n_test) // (self.n_test/(self.tc)),
                            self.xproot,
                            '%.7f')
        self.configfile = Path(self.fpath / "config.pred")
        pred_params = [self.modelname, self.modeltype,
                       self.xiroot, self.xproot, self.ndpost,
                       self.ntree, self.ntreeh,
                       self.p_test, self.tc, self.f_mean]
        with self.configfile.open("w") as pfile:
            for param in pred_params:
                pfile.write(str(param)+"\n")
        self._run_model(train=False)
        self._read_in_preds()
        ret_mean = kwargs["return_mean"] if "return_mean" in kwargs else True
        if ret_mean:
            return np.mean(self.mpreds, axis=1)
        else:
            return self.mpreds

    def _read_in_preds(self):
        mdraw_files = sorted(list(self.fpath.glob("model.mdraws*")))
        sdraw_files = sorted(list(self.fpath.glob("model.sdraws*")))
        mdraws = []
        for f in mdraw_files:
            mdraws.append(np.loadtxt(f))
        self.mpreds = np.transpose(np.concatenate(mdraws, axis=1))
        sdraws = []
        for f in sdraw_files:
            sdraws.append(np.loadtxt(f))
        self.spreds = np.concatenate(sdraws, axis=1)

    def clean_model(self):
        subprocess.run(f"rm -rf {str(self.fpath)}", shell=True)