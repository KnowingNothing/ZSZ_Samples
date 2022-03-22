#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>

int main() {
    struct stat statbuf;
    mode_t modes;
    stat("file.in", &statbuf);
    modes = statbuf.st_mode;
    if (!S_ISDIR(modes)) {
        printf("not a directory\n");
        if ((modes & S_IRWXU) == S_IXUSR)
            printf("only execute\n");
        else if ((modes & S_IRUSR))
            printf("can read\n");
    }
    return 0;
}