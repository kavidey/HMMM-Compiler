#include <stdio.h>

int double_func(int x) {
    return x * 2;
}

int quadruple_func(int x) {
    int a = double_func(x);
    return double_func(a);
}

// Test functions calling other functions
int main()
{
    int a;
    scanf("%d", &a);

    int b = quadruple_func(a);

    printf("%d\n", b);

    return 0;
}