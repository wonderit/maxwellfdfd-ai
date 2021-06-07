#!/bin/bash

nohup python -u train.py -a -e 100 -it 10 -k 1000 -ua -sb 100 -ual 0.1 -sn 7 -g 3 > train_s7_g3.log &