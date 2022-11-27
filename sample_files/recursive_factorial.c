#include <stdio.h>

int factorial(int x) {
    if (x < 1) {
        return 1;
    } else {
        return x * factorial(x - 1);
    }
}

// Test recursive function calling
int main()
{
    int a;
    scanf("%d", &a);

    int b = factorial(a);

    printf("%d\n", b);
    printf("%d\n", factorial(3));

    return 0;
}
