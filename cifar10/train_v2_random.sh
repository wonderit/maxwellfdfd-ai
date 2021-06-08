#!/bin/bash

nohup python -u train_v2.py -a -e 200 -it 10 -k 1000 -sn 5 -rt random > train_v2_random_s5.log &