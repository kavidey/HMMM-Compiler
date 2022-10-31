#include <stdio.h>

// Test nested control flow, break, and continue
int main()
{
    int r1; // low
    int r2; // high

    scanf("%d", &r1);
    scanf("%d", &r2);

    while (r1 < r2)
    {
        if (r1 < 2)
        {
            r1++;
            continue;
        }

        int r3 = 0; // is prime

        for (int r4 = 2; r4 < r1; r4++)
        {
            if (r1 % r4 == 0)
            {
                r3 = 1;
                break;
            }
        }

        if (r3 == 0)
        {
            printf("%d\n", r1);
        }

        r1++;
    }

    return 0;
}