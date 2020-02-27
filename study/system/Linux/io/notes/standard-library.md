## 标准I/O库

在stdio.h提供了许多标准接口，与底层文件描述符对应的概念是流，它被实现为指向结构FILE的指针。程序启动时有三个流是打开的：stdin, stdout, stderr。
接下来学习的库函数包括：
```c
fopen
fclose
fread
fwrite
fflush
fseek
fgetc
getc
getchar
fputc
putc
putchar
fgets
gets
printf
fprintf
sprintf
scanf
fscanf
sscanf
```

### 1. fopen
主要用于文件或终端的输入输出
```c
#include <stdio.h>

FILE *fopen(const char *filename, const char *mode);
```
mode的可选组合为"r"/"rb"/"w"/"wb"/"a"/"ab"/"r+"/"rb+"/"r+b"/"w+"/"wb+"/"w+b"/"a+"/"ab+"/"a+b"。

失败返回NULL，可开启文件数目为FOPEN_MAX，至少为8，通常为16。

### 2. fread
```c
#include <stdio.h>

size_t fread(void *ptr, size_t size, size_t nitems, FILE *stream);
```
从stream里读nitems个数据，每个数据的大小为size字节，存到ptr指向的空间中，返回真正读到的数据个数（与nitems对应）。

### 3. fwrite
```c
#include <stdio.h>

size_t fwrite(const void *ptr, size_t size, size_t nitems, FILE *stream);
```
### 4. fclose
关闭流，把没写出的内容全部写出
```c
#include <stdio.h>

int fclose(FILE *stream);
```

### 4. fflush
清空缓存
```c
#include <stdio.h>

int fflush(FILE *stream);
```

### 5. fseek
为下一次读写指定位置，类比lseek函数
```c
#include <stdio.h>

int fseek (FILE *stream, long int offset, int whence);
```

### 6. fgetc, getc, getchar
fgetc从文件流中获取一个字符并返回，当文件尾或出错时返回EOF，需要用ferror和feof区分这两种情况。getchar是从stdin读。

```c
#include <stdio.h>

int fgetc(FILE *stream);
int getc(FILE *stream);
int getchar();
```

### 7. fputc, putc, putchar
fputc向文件流输出一个字符，返回这个字符值，失败时返回EOF。putchar固定向stdout输出。
```c
#include <stdio.h>

int fputc(int c, FILE *stream);
int putc(int c, FILE *stream);
int putchar(int c);
```

### 8. fgets, gets
读取字符串。fgets读入停止的条件有三种，换行，达到数字n-1，达到文件尾，换行符也会被读进来，最后会加一个\0。出错或到文件尾部，返回空指针，否则返回指向s的指针。gets和fgets的区别在于，从stdin读，不读入换行。gets很不安全，因为它允许任意长度的读入，但是缓冲区有限，会造成溢出。
```c
#include <stdio.h>

char *fgets(char *s, int n, FILE *stream);
char *gets(char *s);
```
