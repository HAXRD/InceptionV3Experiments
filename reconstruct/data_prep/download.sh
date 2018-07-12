#!/bin/bash

rm -r ../data
rm -r ./*.tgz

while read url ; do
    echo "fetching $url"
    wget "$url"
done < files.txt

mkdir ../data

for archive in ./*.tgz ; do
    tar -xvzf "$archive" -C ../data/
done

echo "Done Data Prep!"