#!/bin/bash

nohup python -u train_ua.py -a -rt max_diff -e 100 -it 10 -l l1 -r 0.05 > train_ua_max_diff_l1_r05.log &