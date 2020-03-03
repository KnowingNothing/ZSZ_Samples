## 向程序传递参数
main函数的argc和argv，需要由另一个程序传递过来，一般是shell，处理参数的方式多样，导致了不同程序的参数风格各异。为了提供一致的参数接口，提供了getopt等函数用来处理参数。其中getopt用来处理-开头的只含有一个字母的短命令参数，getopt_long处理--开头的长命令参数。

```c
#include <unistd.h>

int getopt(int argc, char *argv[], const char *optstring);
extern char *optarg;
extern int optind, opterr, optopt;
```

optstring用来告诉哪些参数选项可用，每个字符代表一个选项，如果一个字符后面跟一个冒号，表明该选项有一个关联值作为下一个参数。getopt返回argv数组中下一个选项字符，循环调用getopt可以依次得到每个选项，如果遇到了有关联值的选项，optarg指向这个值，如果处理完毕，则返回-1，特殊参数--会使getopt停下来，遇到无法识别的选项，getopt返回一个问号，optopt也存一个问号，如果有关联值的选项没有传入关联值，也会返回一个问号，如果optstring第一个就是:，那在没有得到传入关联值时返回冒号。optind是下一个待处理的参数的索引。
