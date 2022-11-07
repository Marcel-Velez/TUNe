# TUNe

This repository contains the code to reproduce the experiments from the paper "Tailed U-Net: Multi-Scale Music Representation Learning" [] of a mix of two repositories from CLMR[] and jukemir



# Code alterations

- changed part of the gtzan.py code from clmr in "clmr/datasets/gtzan.py" because the download link from the gtzan dataset was unavailable at the time of writing.
- added checkpoint saving every 200 epochs to "clmr.XXXX.contrastivelearning.py" line 35/36
