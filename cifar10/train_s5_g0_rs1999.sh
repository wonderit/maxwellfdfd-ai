#!/bin/bash

nohup python -u train.py -a -e 100 -it 10 -k 1000 -rt random -sn 5 -g 0 -rs 1999 > train_s5_g0.log &