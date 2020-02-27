#!/bin/bash

trap 'echo "oops! do not want to kill me!"' INT
echo "Your shell is locked! Give me the password:"
read passwd 
while [ $passwd != "password" ]; do
    echo "Not correct"
    read passwd
done
echo "Right! You can kill me now!"
trap - INT