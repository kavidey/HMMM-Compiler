#include <stdio.h>

// Test chain of if/else if/else statements
int main()
{
    int r1;
    
    scanf("%d", &r1);

    if (r1 < 0)
    {
        printf("%d\n", -1);
    } else if (r1 > 0) {
        printf("%d\n", 1);
    } else {
        printf("%d\n", 0);
    }
    
    return 0;
}