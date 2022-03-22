#include <stdio.h>

int main() {
    FILE *in, *out;
    in = fopen("file.in", "r");
    out = fopen("file.out", "w");

    char str[1024];

    while (fgets(str, 1024, in) != NULL) {
        fprintf(out, "%s", str);
    }

    return 0;
}