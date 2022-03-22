文件除了包含的内容以外，还有一个名字和一些属性，叫“管理信息”，包括文件的创建修改日期和它的权限。这些属性被保存在文件的inode中，它是文件系统中一个特殊的数据块，它同时还包含文件长度和文件在磁盘存放的位置。系统使用的是inode编号。

目录是用于保存其他文件的节点号和名字的文件。可以用`stat`查看inode编号。

删除文件实际上是删除对应的目录项，同时使指向该文件的链接数减一。链接数为零时可以标记为可用空间。

设备也作为文件看待，三个重要的设备文件：/dev/console, /dev/tty, /dev/null。可以通过拷贝/dev/null创建空文件`cp /dev/null empty_file`。

文件操作五个重要命令：open, close, read, write, ioctl

## 文件和目录的维护

### chmod
```c
#include <sys/stat.h>

int chmod(const char *path, mode_t mode);
```

### chown
```c
#include <sys/types.h>
#include <unistd.h>

int chown(const char *path, uid_t owner, gid_t group);
```

### unlink, link, symlink
```c
#include <unistd.h>

int unlink(const char *path);
int link(const char *path1, const char *path2);
int symlink(const char *path1, const char *path2);
```

### mkdir, rmdir
```c
#include <sys/types.h>
#include <sys/stat.h>

int mkdir(const char *path, mode_t mode);
int rmdir(const char *path);
```

### chdir, getcwd
切换目录和确定当前目录
```c
#include <unistd.h>

int chdir(const char *path);
char *getcwd(char *buf, size_t size);
```

## 扫描目录
目录流和文件流相似，定义在dirent.h里声明了DIR和dirent结构。主要使用的函数有:
```sh
opendir
closedir
readdir
telldir
seekdir
closedir
```

### opendir
创建目录流，返回一个指向DIR结构指针。失败时返回空指针。
```c
#include <sys/types.h>
#include <dirent.h>

DIR *opendir(const char *name);
```

### readdir
函数返回一个指针，指向的内容是目录流中下一个目录项，遇到错误或目录尾时返回NULL，如果是目录尾，不改变errno，否则会改变errno。
```c
#include <sys/types.h>
#include <dirent.h>

struct dirent *readdir(DIR *dirp);
```

dirent结构中包含的目录项内容包括：
```sh
ino_t d_ino: 文件inode节点号
char d_name[]: 文件名
```

### telldir
telldir返回目录流的当前位置。
```c
#include <sys/types.h>
#include <dirent.h>

long int telldir(DIR *dirp);
```

### seekdir
设置目录流位置。
```c
#include <sys/types.h>
#include <dirent.h>

void seekdir(DIR *dirp, long int loc);
```

### closedir
关闭目录流
```c
#include <sys/types.h>
#include <dirent.h>

int closedir(DIR *dirp);
```