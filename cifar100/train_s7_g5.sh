#!/bin/bash

nohup python -u train.py -a -e 100 -it 10 -k 1000 -rt random -sn 7 -g 5 > train_s7_g5.log &