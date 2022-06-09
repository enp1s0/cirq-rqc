#!/bin/sh

mode_list="TF32TC TF32TCEC FP16TC FP16TCEC"

for mode in $mode_list;do
	ybatch job.sh $mode
done
