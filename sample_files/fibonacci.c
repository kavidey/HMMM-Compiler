#include <stdio.h>

// Test for loops
int main()
{
    int num;
    scanf("%d", &num);

    int curr_fib = 0;
    int nex_fib = 1;

    for (int i = 0; i < num; i++)
    {
        curr_fib = curr_fib + nex_fib;
        nex_fib = curr_fib - nex_fib;
        printf("%d\n", curr_fib);
    }

    return 0;
}