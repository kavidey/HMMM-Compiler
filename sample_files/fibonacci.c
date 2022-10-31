#include <stdio.h>

// Test for loops
int main()
{
    int r1;
    scanf("%d", &r1);

    int r2 = 0;
    int r3 = 1;

    for (int r4 = 0; r4 < r1; r4++)
    {
        r2 = r2 + r3;
        r3 = r2 - r3;
        printf("%d\n", r2);
    }

    return 0;
}