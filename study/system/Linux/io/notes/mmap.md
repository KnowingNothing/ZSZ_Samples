## mmap
UNIX 提供了共享内存的方法，允许建立一段可以被两个或更多程序读写的内存。mmap创建指向一段内存区域的指针。
```c
#include <sys/mman.h>

void *mmap(void *addr, size_t len, int prot, int flags, int fildes, off_t off);
```
off指定了文件中数据起始偏移，fildes是打开的文件描述符，内存段的长度为len，addr可以用来请求某个内存地址，如果是0就自动分配，prot是权限，由以下数值按位或：
```
PROT_READ
PROT_WRITE
PROT_EXEC
PROT_NONE
```
flags控制内存段所受的影响：
```
MAP_PRIVATE: 仅本进程可修改
MAP_SHARED: 修改保存到磁盘
MAP_FIXED: 必须指向addr指定的位置
```

另外一个函数msync用于控制修改：
```c
#include <sys/mman.h>

int msync(void *addr, size_t len, int flags);
```
从addr开始长为len的内存块内按照flags指定的方式修改：
```
MS_ASYNC
MS_SYNC
MS_INVALIDATE: 从文件中读回
```

munmap用于释放内存：
```c
#include <sys/mman.h>

int munmap(void *addr, size_t len);
```