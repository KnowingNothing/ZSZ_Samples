#!/bin/bash

echo '$0 is name of shell'
echo "$0"

echo '$# is number of parameters'
echo "$#"

echo '$$ is process id'
echo "$$"

echo '$1 is the first parameter'
echo "$1"

echo '$2 is similar'
echo "$2"

echo '$* is all the parameters'
echo "$*"
echo 'change $IFS'
echo "$IFS"
OLD_IFS="$IFS"
echo 'change to ""'
IFS=""
echo "$*"
echo "recover IFS"
IFS="$OLD_IFS"

echo '$@ is all the parameters'
echo "$@"
echo 'change $IFS'
echo "$IFS"
OLD_IFS="$IFS"
echo 'change to ""'
IFS=""
echo "$@"
IFS="$OLD_IFS"

echo '$PS1'
echo "$PS1"

echo '$PS2'
echo "$PS2"

echo '$PS3'
echo "$PS3"

echo '$HOME'
echo "$HOME"

echo '$PATH'
echo "$PATH"