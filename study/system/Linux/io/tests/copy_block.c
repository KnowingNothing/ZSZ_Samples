#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>

int main() {
    char block[1024];
    int in, out;
    int nread;

    in = open("file.in", O_RDONLY);
    out = open("file.out", O_CREAT|O_WRONLY, S_IRUSR|S_IWUSR);

    while ((nread = read(in, block, 1024)) > 0) {
        write(out, block, nread);
    }

    return 0;
}