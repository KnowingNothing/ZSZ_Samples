#!/bin/bash

echo "Add +x to these files"
for file in $(ls); do
    if [ ./$file != $0 ]; then
        echo $file
        chmod +x $file
    fi
done