#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 16:41:02 2021

@author: clark
"""
import numpy as np
def summarize_fitp(fitp):
     print(fitp['mdraws'][0, 0:5]); print(np.mean(fitp['mdraws']))
     print(fitp['sdraws'][0, 0:5]); print(np.mean(fitp['sdraws']))
     # print(fitp['mmean'][0:5]); print(np.mean(fitp['mmean']))
     # print(fitp['smean'][0:5]); print(np.mean(fitp['smean']))
     print(fitp['msd'][0:5]); print(np.mean(fitp['msd']))
     print(fitp['ssd'][0:5]); print(np.mean(fitp['ssd']))
     print(np.mean(fitp['m_5'])); print(np.mean(fitp['m_lower'])); print(np.mean(fitp['m_upper']))
     print(np.mean(fitp['s_5'])); print(np.mean(fitp['s_lower'])); print(np.mean(fitp['s_upper']))
def summarize_fitv(fitv):
     print(fitv['vdraws'][0:3, :]); print(np.mean(fitv['vdraws']))
     print(fitv['vdrawsh'][0:3, :]); print(np.mean(fitv['vdrawsh']))
     print(fitv['mvdraws']); print(fitv['mvdrawsh'])
     print(fitv['vdraws_sd']); print(fitv['vdrawsh_sd']); print(fitv['vdraws_5'])
     print(fitv['vdrawsh_5']); print(fitv['vdraws_lower']); print(fitv['vdraws_upper'])
     print(fitv['vdrawsh_lower']); print(fitv['vdrawsh_upper'])
def summarize_fits(fits):
     print(np.mean(fits['vidraws'])); print(np.mean(fits['vijdraws']))
     print(np.mean(fits['tvidraws'])); print(np.mean(fits['vdraws']))
     print(np.mean(fits['sidraws'])); print(np.mean(fits['sijdraws']))
     print(np.mean(fits['tsidraws']))
     # print(fits['msi']); print(fits['msi_sd']); print(fits['si_5'])
     # print(fits['si_lower']); print(fits['si_upper']); print(fits['msij'])
     # print(fits['sij_sd']); print(fits['sij_5']); print(fits['sij_lower'])
     # print(fits['sij_upper']); print(fits['mtsi']); print(fits['tsi_sd'])
     # print(fits['tsi_5']); print(fits['tsi_lower']); print(fits['tsi_upper'])
def save_fits(fits, fname):
     from pathlib import Path
     to_save = dict(fits) # A "shallow copy"
     for cat in ('vidraws', 'vijdraws', 'tvidraws', 'vdraws', 'sidraws', 'sijdraws', 'tsidraws'):
          del to_save[cat]
     configfile = Path(fname)
     with configfile.open("w") as tfile:
          for cat in to_save:
               tfile.write(str(cat)+": \n")
               tfile.write(str(to_save[cat])+"\n")
     # print(to_save)