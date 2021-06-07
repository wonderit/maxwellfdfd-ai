#!/bin/bash

nohup python -u train.py -a -e 100 -it 10 -k 1000 -sn 2 -g 1 -rm -w > train_s2_g1.log &