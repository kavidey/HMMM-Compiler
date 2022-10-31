#include <stdio.h>

// Test while loops
int main()
{
    int r1;
    scanf("%d", &r1);

    int r2 = 0;

    while (r1 > 0)
    {
        r2 = r2 + r1;
        r1--;
    }

    printf("%d\n", r2);
    
    return 0;
}