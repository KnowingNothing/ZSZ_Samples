#!/bin/bash

echo "Please enter yes or no:"
read val

case "$val" in
    [yY][eE][sS]|[yY] ) echo "Yes!";;
    [nN][oO]|[nN] ) echo "No!";;
    * ) echo "Nothing";;
esac