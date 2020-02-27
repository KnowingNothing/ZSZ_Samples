## 底层文件访问

1. 每个进程都有三个自动打开的文件描述符：0（标准输入），1（标准输出），2（标准错误）
2. write 将缓冲区n个字节写入描述符对应的文件，返回实际写入字节数。返回-1表示出错，错误代码在全局变量errno里。
```c
#include <unistd.h>

size_t write(int fildes, const void *buf, size_t nbytes);
```
3. read 从文件描述符对应的文件里读取n个字节放在缓冲区，返回实际读取的字节数，返回0表示文件尾，返回-1表示出错。
```c
#include <unistd.h>

size_t read(int fildes, void *buf, size_t nbytes);
```

4. open 创建新的文件描述符
```c
#include <fcntl.h>
#include <sys/types.h> // 某些系统需要
#include <sys/stat.h> // 某些系统需要

int open(const char *path, int oflags);
int open(const char *path, int oflags, mode_t mode);
```
不同进程中的文件描述符不会重复，即使打开的是同一个文件，两个进程写一个文件会导致相互覆盖。oflags的可选范围为：
必须使用下列之一：
```
O_RDONLY
O_WRONLY
O_RDWR
```
可选下列中的：
```
O_APPEND
O_TRUNC
O_CREAT // 按照mode中的访问模式
O_EXCL 
```
成功返回非负整数，失败返回-1，新的描述符总是未用描述符的最小值

5. creat 调用，相当于oflags标志O_CREAT|O_WRONLY|O_TRUNC

6. 一个运行程序能打开的文件描述符有限，在limitis.h中定义为OPEN_MAX，POSIX规定要>=16，实际多为256.

7. 设置访问权限：在使用O_CREAT标志时，会创建文件，此时要在mode设置访问权限，共九个可以或起来的标志：
```
S_IRUSR
S_IWUSR
S_IXUSR
S_IRGRP
S_IWGRP
S_IXGRP
S_IROTH
S_IWOTH
S_IXOTH
```
但是创建出来的文件的真实权限还要和umask进行运算，将umask作为掩码（将umask取反后与运算）。

8. umask: 一个系统变量，对应一个umask命令可以改这个值，由三个八进制数组成，每个八进制数的三个二进制位代表了rwx，三个八进制数对应了用户、组、其他。注意umask是用来做掩码的，计算时要用取反操作。

9. close 关闭描述符。
```c
#include <unistd.h>

int close(int fildes);
```

10. ioctl 大部分的设备都定义了自己ioctl。
```c
#include <unistd.h>

int ioctl(int fildes, int cmd, ...);
```

10. 其他底层调用：

(1) lseek
```c
#include <unistd.h>
#include <sys/types.h>  // 定义了off_t

off_t lseek(int fildes, off_t offset, int whence);
```
whence可以取以下中：
```
SEEK_SET: 绝对位置
SEEK_CUR: 相对于当前位置的一个位置
SEEK_END: 相对于文件尾部的一个位置
```
返回文件头到移动后指针的偏移值

(2) fstat stat lstat
```c
#include <unistd.h>
#include <sys/stat.h> // 定义与st_mode有关的宏
#include <sys/types.h>

int fstat(int fildes, struct stat *buf);
int stat(const char *path, struct stat *buf);
int lstat(const cahr *path, struct stat *buf); // 与stat几乎一样，但是可以用来返回符号链接本身的信息
```

stat结构体的属性
```
st_mode: 文件权限和类型
st_ino: inode
st_dev: 保存文件的设备
st_uid: 文件属主的uid
st_gid: gid
st_atime: 上一次访问时间
st_ctime: 上一次改变权限、属主、组、内容的时间
st_mtime: 上一次修改内容时间
st_nlink: 硬链接个数
```

st_mode的有关宏
```
S_IFBLK 块设备
S_IFDIR 目录
S_IFCHR 特殊字符设备
S_IFIFO 一个FIFO
S_IFREG 普通文件
S_FLINK 符号链接

S_ISUID 设置了SUID
S_ISGID 设置了SGID

S_IFMT 文件类型掩码
S_IRWXU 属主读写执行权限
S_IRWXG 组
S_IRWXO 其他

//用作比较
S_ISBLK
S_ISCHR
S_ISDIR
S_ISFIFO
S_ISREG
S_ISLNK
```

(3) dup dup2 根据给的描述符复制一个新的描述符，指向同一个文件，dup2要指定新描述符。
```c
#include <unistd.h>

int dup(int fildes);
int dup2(int fildes, int fildes2);
```