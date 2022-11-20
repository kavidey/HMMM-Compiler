#include <stdio.h>

// Test chain of if/else if/else statements
int main()
{
    int input;
    
    scanf("%d", &input);

    if (input < 0)
    {
        printf("%d\n", -1);
    } else if (input > 0) {
        printf("%d\n", 1);
    } else {
        printf("%d\n", 0);
    }
    
    return 0;
}