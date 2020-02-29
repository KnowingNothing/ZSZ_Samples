#include <stdio.h>

int main() {
    printf("|----------|\n");
    printf("|%10d|\n", 3);
    printf("|%010d|\n", 3);
    printf("|%10s|\n", "hello");
    printf("|%*s|\n", 10, "hello");
    return 0;
}