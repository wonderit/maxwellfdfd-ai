#!/bin/bash

nohup python -u train.py -a -e 200 -it 5 -k 1000 -ua -sb 10 -sn 7 > train_ua.log &