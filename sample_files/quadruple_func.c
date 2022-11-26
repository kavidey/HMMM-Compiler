#include <stdio.h>

int double_func(int x) {
    return x * 2;
}

int quadruple_func_v1(int x) {
    int a = double_func(x);
    return double_func(a);
}

int quadruple_func_v2(int x) {
    return double_func(double_func(x));
}

// Test functions calling other functions
int main()
{
    int a;
    scanf("%d", &a);

    int b = quadruple_func_v1(a);
    int c = quadruple_func_v2(a);

    printf("%d\n", b);
    printf("%d\n", c);

    return 0;
}