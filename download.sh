#!/bin/bash

if [ "$1" == "full" ]; then
    wget https://syncandshare.lrz.de/dl/fi7btjPeTr2Ecj8HVgBGvcmf/data_IJGI18.zip
    unzip data_IJGI18.zip
    rm data_IJGI18.zip
elif [ "$1" == "demo" ]; then
    wget https://s3.eu-central-1.amazonaws.com/corupublic/mtlcc/data_IJGI18_demo.zip
    unzip data_IJGI18_demo.zip
    rm data_IJGI18_demo.zip
else
    echo "please provide either 'full' (40GB) or 'demo' (1GB) as arguments"
fi




