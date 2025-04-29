// Name: Ben Williams
// CPU random walk. 
// nvcc HW27.cu -o temp

/*
 What to do:
 Create a function that returns a random number that is either -1 or 1.
 Start at 0 and call this function to move you left (-1) or right (1) one step each call.
 Do this 10000 times and print out your final position.
*/

// Include files
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


// Defines
#define N 10000

// Globals
int p = 0; 

// Function prototypes


int main(int, char**);
int move(int);
int move()
{
    float direction = ((float)rand()/(float)RAND_MAX);
    return (direction > .5) ? 1 : -1;
}

int main(int argc, char** argv)
{
    srand(time(NULL));
    p = 0;

    for (int i = 0; i<N; i++)
    {
	p+= move();
    }
    printf("Final position: %d\n", p);
    return 0;
}
