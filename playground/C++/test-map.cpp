#include <map>

int main() {
    std::map<int, int> m;
    int tmp = m[1];
    printf("%d\n", tmp);
    printf("Now map doesn't give error when using [] before assigning\n");
    return 0;
}