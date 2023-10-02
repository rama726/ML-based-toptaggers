#!/usr/bin/env python

import sys

import ROOT

import math
import numpy as np
from array import array
from ROOT import TMVA, TFile, TString
from subprocess import call
from os.path import isfile

# Setup TMVA
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()
reader = TMVA.Reader("Color:!Silent")

mass = array('f', [0.0])
tau21 = array('f', [0.0])
tau32 = array('f', [0.0])
tau43 = array('f', [0.0])
btag = array('f', [0.0])
score = array('f', [0.0])


reader.AddVariable("mass", mass)
reader.AddVariable("tau21", tau21)
reader.AddVariable("tau32", tau32)
reader.AddVariable("tau43", tau43)
reader.AddVariable("btag", btag)
reader.AddVariable("score", score)

reader.BookMVA('BDT', TString('/dataset35/weights/TMVAClassification_BDT.weights.xml'))

file1 = np.load("300_500.npy")

count = np.zeros(41)
effcount = 0
for entry in range(0, file1.shape[0]):
  if file1[entry][0] == 1:
    mass[0]=file1[entry][3]
    tau21[0]=file1[entry][4]
    tau32[0]=file1[entry][5]
    tau43[0]=file1[entry][6]
    btag[0]=file1[entry][7]
    score[0]=file1[entry][2]
    effcount +=1 
        
    for i in range(41):
      d = (i - 20) /20
      if reader.EvaluateMVA('BDT') > d :
        count[i] += 1
   
   
  
file1 = open(r'efft.txt',"w")        
eff = np.zeros(41)
for i in range(41):
  eff[i] = count[i]/effcount
  d = (i - 20) /20
  file1.write('%f\t%f\n' % (d,eff[i]))


file1.close()
