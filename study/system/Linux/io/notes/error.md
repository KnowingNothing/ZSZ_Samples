## 错误处理

库函数的错误处理都是靠errno完成的。每个函数调用后需要立即查看errno，否则它可能会被覆盖。错误处理相关内容都在errno.h里定义。可能的错误代码有：
```sh
EPERM: 操作不被允许
ENOENT: 文件或目录不存在
EINTR: 系统调用被中断
EIO: I/O错误
EBUSY: 设备或资源忙
EEXIST: 文件存在
EINVAL: 参数无效
EMFILE: 打开文件过多
ENODEV: 设备不存在
EISDIR: 是一个目录
ENOTDIR: 不是一个目录
```

用来查看错误的函数：

### 1. strerror
用字符串说明错误
```c
#include <string.h>

char *strerror(int errnum);
```

### 2. perror
输出错误信息到标准错误，信息前一部分由参数给出，冒号之后是errno对应的错误信息。
```c
#include <stdio.h>

void perror(const char *s);
```