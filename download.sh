#!/bin/bash

# Download Data
curl -L -o data.zip https://drive.google.com/uc?id=1Gs9Per_unwdmlXufDxmYEgLAve0ep8Xx
mkdir data
unzip data.zip -d ./data
