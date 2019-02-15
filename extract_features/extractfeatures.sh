#!/bin/bash

extractor=../overfeat/bin/linux_64/overfeat
img_dir=../data/processed
feat_dir=../data/features
out_file=features.data

#echo Start

output_file="$feat_dir"/"$out_file"

touch "$output_file"

i=0
for d in "$img_dir"/*
do
	#echo Processing image "$d"
	for f in "$d"/*
	do
		#echo Extracting features of "$f"
		"$extractor" -f "$f" | sed -n 2p >> $output_file
		#echo Features of "$f" extracted
		ff=$(echo "$f" | cut -d"/" -f5)
		echo "$i $ff"
		((i++))
	done
	#echo Image "$d" is done
done

#echo Done
