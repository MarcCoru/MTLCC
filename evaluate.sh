#!/bin/bash

dir=$1

echo ground truth
rm $dir/ground_truth.tif
gdal_merge.py -o $dir/ground_truth.tif $dir/ground_truth/*.tif

echo predictions
rm $dir/prediction.tif
gdal_merge.py -o $dir/prediction.tif $dir/prediction/*.tif

echo loss
rm $dir/loss.tif
gdal_merge.py -o $dir/loss.tif $dir/loss/*.tif

echo mask
rm $dir/mask.tif
gdal_merge.py -o $dir/mask.tif $dir/mask/*.tif

mkdir $dir/confidences
classes=$(find $dir/confidences -maxdepth 1 -type d -printf '%P\n')

for class in $classes; do
    echo $class
    rm $dir/confidences/$class.tif
    gdal_merge.py -o $dir/confidences/$class.tif $dir/confidences/$class/*.tif
done
#gdal_merge.py -o $dir/lpp.tif $dir/lpp/*.tif
