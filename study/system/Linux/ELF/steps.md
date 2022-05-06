### Compile without link
```
gcc -c SimpleSection.c
```

### Check the results of objdump
```
objdump -h SimpleSection.o
```

### Readelf
```
# summary
readelf -h SimpleSection.o

# sections
readelf -S SimpleSection.o
```

### Elf headers in /usr/include/elf.h