#!/bin/bash

nohup python -u train.py -a -e 100 -it 10 -k 1000 -sn 7 -g 4 -rm -w > train_s7_g4.log &