#!/bin/bash

echo "With \n"
echo -n "Without \n "
echo "append something"
echo "not interpret \\ \nIt doesn't work!"
echo -e "interpret \\ \nIt works!"
echo -e "without \c"
echo "append something"

printf "printf "
echo "append something"