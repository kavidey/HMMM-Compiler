#include <stdio.h>

// Test nested control flow, break, and continue
int main()
{
    int lower_bound;
    int upper_bound;

    scanf("%d", &lower_bound);
    scanf("%d", &upper_bound);

    while (lower_bound < upper_bound)
    {
        if (lower_bound < 2)
        {
            lower_bound++;
            continue;
        }

        int is_prime = 0;

        for (int i = 2; i < lower_bound; i++)
        {
            if (lower_bound % i == 0)
            {
                is_prime = 1;
                break;
            }
        }

        if (is_prime == 0)
        {
            printf("%d\n", lower_bound);
        }

        lower_bound++;
    }

    return 0;
}