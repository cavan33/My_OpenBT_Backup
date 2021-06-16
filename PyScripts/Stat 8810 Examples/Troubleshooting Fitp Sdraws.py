#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Fitp for Slides 13 Testing is not having the correct sdraws. This script attempts to figure out why.
"""
import numpy as np
f = "/tmp/openbtpy_wiedmi8s/model."
test0 = np.loadtxt(f+"sdraws0")
print(test0[0])
print(len(test0[0]))
print(test0[:,0])
print(len(test0[:,0]))

test1 = np.loadtxt(f+"sdraws1")
print(test1[0])
print(len(test1[0]))
print(test1[:,0])
print(len(test1[:,0]))

test2 = np.loadtxt(f+"sdraws2")
print(test2[0])
print(len(test2[0]))
print(test2[:,0])
print(len(test2[:,0]))

test3 = np.loadtxt(f+"sdraws3")
print(test3[0])
print(len(test3[0]))
print(test3[:,0])
print(len(test3[:,0]))

print(test0[0]); print(test1[0]); print(test2[0]); print(test3[0])
# The sdraws repeat (even across the file borders) at a 17-8-17-8 pattern! Argggg
# For another run, they repeat in an 11-14-11-14 pattern. Dang
# OR, 9-8-8

"""
test4 = np.loadtxt(f+"mdraws0")
print(test4[0])
print(len(test4[0]))
print(test4[:,0])
print(len(test4[:,0]))

test5 = np.loadtxt(f+"mdraws1")
print(test5[0])
print(len(test5[0]))
print(test5[:,0])
print(len(test5[:,0]))

test6 = np.loadtxt(f+"mdraws2")
print(test6[0])
print(len(test6[0]))
print(test6[:,0])
print(len(test6[:,0]))

test7 = np.loadtxt(f+"mdraws3")
print(test7[0])
print(len(test7[0]))
print(test7[:,0])
print(len(test7[:,0]))

print(test4[0]); print(test5[0]); print(test6[0]); print(test7[0])
"""