## 用户信息
除了init程序以外，其他程序都是由用户或程序启动的。用户有用户名和密码，有一个唯一的UID。可以修改一个程序，使它们运行看上去似乎是某个用户启动的。UID是uid_t类型的，定义在sys/types.h里，通常是一个小整数，一般大于100.

```c
#include <sys/types.h>
#include <unistd.h>

uid_t getuid(void);
char *getlogin(void);
```
getuid返回程序关联的UID，getlogin返回与当前用户关联的登录名。

/etc/passwd包含了一个用户账号的数据库，每行一个用户，包括用户名，密码，UID，GID，全名，家目录和默认shell。但是现代Linux系统不会这样做，而是放在/etc/shadow中，该文件普通用户不可用。

```c
#include <sys/types.h>
#include <pwd.h>

struct passwd *getpwuid(uid_t uid);
struct passwd *getpwnam(const char *name);
```

passwd的结构包含如下信息：
```
char *pw_name: 登录名
uid_t pw_uid: UID
gid_t pw_gid: GID
char *pw_dir: home
char *pw_gecos: 全名
char *pw_shell: 默认shell
```
