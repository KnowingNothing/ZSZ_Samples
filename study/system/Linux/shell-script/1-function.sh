#!/bin/bash

foo () {
    echo "Call foo"
}

echo "Script begins"

foo

boo () {
    echo "Call boo"
    echo "The arguments are '$@'"
    return 0
}

if boo these are parameters
then
    echo "boo returns 1"
fi

echo "Script ends"