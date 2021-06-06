#!/bin/bash

nohup python -u train.py -a -e 200 -it 5 -k 1000 -rt random > train_random.log &