#!/bin/bash

nohup python -u train.py -a -e 100 -it 10 -k 1000 -ua -sb 10 -ual 0.9 -sn 7 -g 2 > train_s7_g2.log &