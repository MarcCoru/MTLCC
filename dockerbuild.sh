#!/bin/bash

docker build -t ijgi18 .
docker tag ijgi18 marccoru/ijgi18
docker push marccoru/ijgi18
