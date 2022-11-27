#include <stdio.h>

int ackermann(int m, int n)
{
    if (m == 0)
    {
        return n + 1;
    }
    else if (n == 0)
    {
        return ackermann(m - 1, 1);
    }
    else
    {
        return ackermann(m - 1, ackermann(m, n - 1));
    }
}

// Test complex recursion
int main()
{
    int a, b;

    scanf("%d", &a);
    scanf("%d", &b);

    int c = ackermann(a, b);

    printf("%d\n", c);

    return 0;
}
