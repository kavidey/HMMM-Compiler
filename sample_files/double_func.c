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

    printf("%d", b);

    return 0;
}