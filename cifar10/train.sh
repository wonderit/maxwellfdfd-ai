#!/bin/bash

nohup python -u train.py -a -e 350 -it 5 -k 1000 -o sgd > train.log &