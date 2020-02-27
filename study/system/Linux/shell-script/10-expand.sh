#!/bin/bash

echo "No expanding"
x=0
i=0
while [ $i -ne 10 ]; do
    i=$(($i+1))
    x=$x+1
done
echo "Result is $x"

echo "With expanding"
x=0
i=0
while [ $i -ne 10 ]; do
    i=$(($i+1))
    x=$(($x+1))
done
echo "Result is $x"

echo "{} expanding"
for i in 1 2; do
    echo ${i}_file
done

unset foo
echo ${foo:-hello}
echo $foo
echo ${foo:=hello}
echo $foo
# echo ${foo:?hello}
# echo ${foo:+hello}

foo=bye
echo ${foo:-hello}

path=/a/b/c/d/e/c/a/c/e 
echo "path=$path"
echo ${path#*c}
echo ${path##*c}
echo ${path%a*}
echo ${path%%a*}