#!/bin/bash

nohup python -u train.py -a -e 100 -it 10 -k 1000 -ua -sb 1 -ual 0.1 -sn 7 -g 7 > train_s7_g7.log &