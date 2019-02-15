#!/bin/bash

img_dir=../data/original
out_dir=../data/processed
w=250

echo Start

n=$(ls -l "$img_dir" | wc -l)
n="$(($n-1))"

echo Directory "$img_dir" has "$n" files

i=1
for f in "$img_dir"/*
do
	img=$(echo ${f##*/} | cut -d"." -f1)
	[ -d "$out_dir"/"$img" ] && rm -r "$out_dir"/"$img"
	mkdir "$out_dir"/"$img"
	convert "$f" -crop "$w"x"$w" +repage "$out_dir"/"$img"/"$img"_"$w"x"$w"_%03d.jpg
	echo "Image $i of $n done"
	i="$(($i+1))"
done

echo Done
