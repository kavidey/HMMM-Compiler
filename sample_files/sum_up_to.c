#include <stdio.h>

// Test while loops
int main()
{
    int sum_to;
    scanf("%d", &sum_to);

    int running_sum = 0;

    while (sum_to > 0)
    {
        running_sum = running_sum + sum_to;
        sum_to--;
    }

    printf("%d\n", running_sum);
    
    return 0;
}