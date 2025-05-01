// Name: Ben Williams 
// CPU random walk. 
// nvcc HW28.cu -o temp

/*
 What to do:
 This is some code that runs a random walk for 10000 steps.
 Use cudaRand and run 10 of these runs at once with diferent seeds on the GPU.
 Print out all 10 final positions.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <curand_kernel.h>

// Defines
#define N 10 // Number of random walks
// Globals
int *P_GPU;
int *P_CPU;
curandState *d_states;
int NumberOfRandomSteps = 10000;
float MidPoint = (float)RAND_MAX/2.0f;
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
// Function prototypes
int main(int, char**);
void setup();
__global__ void initRNG(curandState*, unsigned long);

__global__ void randomwalk(curandState*, int*, int);
/*
 RAND_MAX = 2147483647
 rand() returns a value in [0, 2147483647].
 Because RAND_MAX is odd and we are also using 0 this is an even number.
 Hence there is no middle interger so RAND_MAX/2 will divide the number in half if it is a float.
 You might could do this faster with a clever idea using ints but I'm going to use a float.
 Also I'm not sure how long the string of random numbers is. I'm sure it is longer than 10,000.
 Before you use this as a huge string check this out.
*/
void setup()
{
    BlockSize.x = 10; 
    BlockSize.y = 1;
    BlockSize.z = 1;
    GridSize.x = (N - 1) / BlockSize.x + 1; 
    GridSize.y = 1;
    GridSize.z = 1;
    P_CPU = (int*)malloc(N * sizeof(int));
    cudaMalloc(&P_GPU, N * sizeof(int));
    cudaMalloc(&d_states, N * sizeof(curandState));

}
__global__ void initRNG(curandState *states, unsigned long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &states[tid]);
}

__global__ void randomwalk(curandState *states, int *p, int n)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curandState localState = states[id];
     int tp = 0;

    for (int i = 0; i < n; ++i) {
        float randVal = curand_uniform(&localState); // (0, 1]
        tp = tp + (randVal > 0.5f ? 1.0 : -1.0); // Move left or right

    }
    p[id] = tp; 

}


int main(int argc, char** argv)
{
    setup();
	initRNG<<<GridSize, BlockSize>>>(d_states, time(NULL));
    cudaDeviceSynchronize();
	randomwalk<<<GridSize, BlockSize>>>(d_states, P_GPU, NumberOfRandomSteps);
    cudaDeviceSynchronize();
    cudaMemcpy(P_CPU, P_GPU, N * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++)
    {
        printf("Random walk %d final position = %d \n", i, P_CPU[i]);
    }
	free(P_CPU);
    cudaFree(P_GPU);
    cudaFree(d_states);
    printf("Done\n");
	return 0;
}
 
