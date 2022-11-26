#include <stdio.h>

int double_func(int x) {
    return x * 2;
}

// Test simple function
int main()
{
    int a;
    scanf("%d", &a);

    int b = double_func(a);
    int c = double_func(3);

    printf("%d", b);
    printf("%d", c);

    return 0;
}