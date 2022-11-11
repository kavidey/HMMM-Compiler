#include <stdio.h>

// Test graph coloring and variables with non rX names
int main()
{
    int a;
    int b;

    scanf("%d", &a);
    scanf("%d", &b);

    int d = 0;
    int e = a;
    while (e > 0)
    {
        d = d + b;
        e = e - 1;
    }

    printf("%d\n", d);

    return 0;
}