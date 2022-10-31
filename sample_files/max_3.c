#include <stdio.h>

// Test nested if/else statements
int main()
{
    int r1;
    int r2;
    int r3;
    
    scanf("%d", &r1);
    scanf("%d", &r2);
    scanf("%d", &r3);

    if (r1 > r2)
    {
        if (r1 > r3) {
            printf("%d\n", r1);
        } else {
            printf("%d\n", r3);
        }
    } else {
        if (r2 > r3) {
            printf("%d\n", r2);
        } else {
            printf("%d\n", r3);
        }
    }
    return 0;
}