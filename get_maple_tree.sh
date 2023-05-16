#!/usr/bin/env bash

INDIR=$1
INFASTA_NAME=$2

MAPLE_DIR=/Users/berk/Projects/MAPLE


/opt/anaconda3/envs/pypy-env/bin/pypy $MAPLE_DIR/createMapleFile.py --path $INDIR/ --fasta $INFASTA_NAME --output temp_FWsim.txt --overwrite
/opt/anaconda3/envs/pypy-env/bin/pypy3 $MAPLE_DIR/MAPLEv0.3.1.py --input $INDIR/temp_FWsim.txt --output $INDIR/ --overwrite --calculateLKfinalTree --estimateSiteSpecificErrorRate
