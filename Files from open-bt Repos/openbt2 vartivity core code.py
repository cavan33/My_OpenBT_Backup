#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 14:28:20 2021

@author: clark
"""
        vdraws_files = sorted(list(self.fpath.glob("model.vdraws")))
        vdrawsh_files = sorted(list(self.fpath.glob("model.vdrawsh")))
        
        vdraws = np.array([])
        for f in vdraws_files:
            vdraws = np.append(vdraws, np.loadtxt(f))
        vdraws[3] = 0.5 # to test transposing/ bringing it back to 1
        vdraws = vdraws.reshape(self.ndpost, self.p)
        self.vartivity['vdraws'] = vdraws
        print(self.vartivity['vdraws'].shape); print(self.vartivity['vdraws'])
        # Normalize counts:
        colnorm = np.array([])
        for i in range(len(self.vartivity['vdraws'])): # should = ndpost in most cases
             colnorm = np.append(colnorm, self.vartivity['vdraws'][i].sum())
        idx = np.where(colnorm > 0)[0] # print(idx)
        # print(colnorm); print(idx)
        
        colnorm = colnorm.reshape(self.ndpost, 1)
        # print(colnorm.shape); print(idx.shape);
        
        self.vartivity['vdraws'][idx] = self.vartivity['vdraws'][idx] / colnorm[idx]
        print(self.vartivity['vdraws'][0:60])