#!/bin/bash

file=1-function.sh

echo "Use $file"
echo "find echo"
grep -i echo $file

echo "count echo"
grep -c echo $file

echo "find line end with 0"
grep 0$ $file

echo "find word end with o"
grep o[[:blank:]] $file

echo "find word ech."
grep ech.[[:blank:]] $file

# [:alnum:]
# [:alpha:]
# [:ascii:]
# [:blank:]
# [:cntrl:]
# [:digit:]
# [:graph:]
# [:lower:]
# [:print:]
# [:punct:]
# [:space:]
# [:upper:]
# [:xdigit:]
# ?
# *
# +
# {n}
# {n,}
# {n,m}

echo "find 4 alpha"
grep -E [a-z]\{4\}[[:blank:]] $file