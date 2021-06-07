#!/bin/bash

nohup python -u train.py -a -e 100 -it 10 -k 1000 -ua -sb 10 -ual 0.5 -sn 7 -g 1 > train_s7_g1.log &