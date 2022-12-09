#include <stdio.h>

int divisible_by_v1(int a, int b)
{
    if (a == 0)
    {
        return 1;
    }
    else if (a < b)
    {
        return 0;
    }
    else
    {
        return divisible_by_v1(a - b, b);
    }
}

int divisible_by_v2(int a, int b) {
    if (a % b == 0) {
        return 1;
    } else {
        return 0;
    }
}

// Test recursive function with multiple inputs
int main()
{

    int a, b;
    scanf("%d", &a);
    scanf("%d", &b);

    printf("%d\n", divisible_by_v1(a, b));
    printf("%d\n", divisible_by_v2(a, b));

    return 0;
}
