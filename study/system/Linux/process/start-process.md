## 启动新进程

1. 程序内部启动新进程
```c
#include <stdlib.h>

int system(const char *string);
```
运行字符串中的命令并等待结束，返回错误码，如果是shell无法启动命令，返回127，否则错误返回-1
这种方式的缺点是不能很好的控制进程，要么等待结束才继续后面的工作，要么就通过后台运行的方式，提前返回。
并且由于system的原理是启动一个shell，这就对具体的环境有要求，且效率不高。
相比system我们应该优先使用下面的方法。

- 替换进程映像：
使用exec系列函数，它们把当前进程替换为新的进程，新进程启动后原来的进程就不再运行了。
```c
#include <unistd.h>

char **environ;

int execl (const char *path, const char *arg0, ..., (char *)0);
int execlp (const char *file, const char *arg0, ..., (char *)0);
int execle (const char *path, const char *arg0, ..., (char *)0, char *const envp[]);
int execv (const char *path, char *const argv[]);
int execvp (const char *file, char *const argv[]);
int execve (const char *path, char *const argv[], char *const envp[]);
```

以p结尾的函数会自动搜索PATH环境变量找到目标文件
exec执行的新进程和原来的进程有相同的PID, PPID, nice值。exec能传递的参数和环境长度有限。一般也不会返回，除非出错。原来进程打开的文件描述符也会被新进程使用。

- 复制进程映像：
使用fork函数复制当前进程，在进程表中创建新的表项，但是新的进程有自己的数据空间、环境和文件描述符。

```c
#include <sys/types.h>
#include <unistd.h>

pid_t fork(void;
```
fork调用一次返回两次，分别在父进程和子进程里返回，在父进程里返回子进程的PID，在子进程里返回0，通过这一点，代码可以判断自己是在子进程还是父进程里。fork失败时返回-1。一个父进程拥有的子进程数目有限(CHILD_MAX)。一个典型的fork使用：
```c
pid_t new_pid;
new_pid = fork();
switch(new_pid) {
    case -1: 
        break;
    case 0:
        break;
    default:
        break;
}
```

### 1. 等待一个进程
fork后的进程独立运行。父进程可以wait子进程。
```c
#include <sys/types.h>
#include <sys/wait.h>

pid_t wait(int *stat_loc);
```

传入的参数如果不是空指针，就会被写入子进程退出的状态：
```sh
WIFEXITED(stat_loc) 如果子进程正常结束，就取非零值
WEXITSTATUS(stat_loc) 如果上一个非零，它返回子进程退出码
WIFSIGNALED(stat_loc) 子进程因为非捕获的信号而退出
WTERMSIG(stat_loc) 返回上面情况的信号代码
WIFSTOPPED(stat_loc) 子进程意外终止
WSTOPSIG(stat_loc) 返回上面情况的信号代码
```

### 2. 僵尸进程
子进程结束了但是父进程没结束，子进程的内容不会被删除，而是记为僵尸进程。如果父进程异常终止，子进程就会以init进程为父进程，init进程会定期清理僵尸进程，但是进程表越大，这个过程越缓慢。这些僵尸进程将消耗系统资源。
waitpid也是可以用来等待子进程的。
```c
#include <sys/types.h>
#include <sys/wait.h>

pid_t waitpid(pid_t pid, int *stat_loc, int options);
```
pid参数指定要等待的子进程，如果是-1，则等待任意一个子进程。option可以用来改变waitpid的行为，如WNOHANG就是查找是否有子进程，如果没有，程序将继续执行。子进程没有停止或者意外终止，都会返回0，否则返回子进程的PID。调用失败时返回-1。