#include <unistd.h> // 必须在行首，因为POSIX规范许多标志会影响其他文件
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>

int main() {
    char c;
    int in, out;

    in = open("file.in", O_RDONLY);
    out = open("file.out", O_WRONLY|O_CREAT, S_IRUSR|S_IWUSR);
    while (read(in, &c, 1) == 1) {
        write(out, &c, 1);
    }
    return 0;
}

// run it: time ./a.out