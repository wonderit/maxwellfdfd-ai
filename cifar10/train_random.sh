#!/bin/bash

nohup python -u train.py -a -e 100 -it 5 -k 1000 -ar -o adam > train_random.log &