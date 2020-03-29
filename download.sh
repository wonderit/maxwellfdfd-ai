#!/bin/bash

# Download Data
curl -L -o data.zip https://drive.google.com/uc?id=14-Bl89OzRtLM1MCW2H81Xvivq8EvTrmB
mkdir data
unzip data.zip -d ./data