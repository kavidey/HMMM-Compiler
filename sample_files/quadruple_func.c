#include <stdio.h>

int double_func(int x) {
    return x * 2;
}

int quadruple_func(int x) {
    return double_func(double_func(x));
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