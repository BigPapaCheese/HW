// Name:Ben Williams
// nvcc HW6.cu -o temp -lglut -lGL
// glut and GL are openGL libraries.
/*
 What to do:
 This code displays a simple Julia fractal using the CPU.
 Rewrite the code so that it uses the GPU to create the fractal. 
 Keep the window at 1024 by 1024.
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>
#include <cuda_runtime.h>
#include <cuda.h>

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define A  -0.824    //Real part of C
#define B  -0.1711    //Imaginary part of C

// Global variables
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize;

float *pixels;   // Moved the declaration here
float *d_pixels; // Moved the declaration here

// Function prototypes
void cudaErrorCheck(const char*, int);
__global__ void escapeOrNotColor(float *pixels, int width, int height, float XMin, float XMax, float YMin, float YMax);
void setUpDevices();

void cudaErrorCheck(const char *file, int line)
{
    cudaError_t  error;
    error = cudaGetLastError();

    if(error != cudaSuccess)
    {
        printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
        exit(0);
    }
}
void setUpDevices()
{
    BlockSize.x = 32;
    BlockSize.y = 32;
    BlockSize.z = 1;

    GridSize.x = (WindowWidth + BlockSize.x - 1) / BlockSize.x;
    GridSize.y = (WindowHeight + BlockSize.y - 1) / BlockSize.y;
    GridSize.z = 1;
}
__global__ void escapeOrNotColor(float *pixels, int width, int height, float XMin, float XMax, float YMin, float YMax)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    float x = XMin + i * (XMax - XMin) / width;
    float y = YMin + j * (YMax - YMin) / height;

    float mag, tempX;
    int count;

    int maxCount = MAXITERATIONS;
    float maxMag = MAXMAG;

    count = 0;
    mag = sqrt(x * x + y * y);
    while (mag < maxMag && count < maxCount)
    {
        tempX = x; //We will be changing the x but we need its old value to find y.
        x = x * x - y * y + A;
        y = (2.0 * tempX * y) + B;
        mag = sqrt(x * x + y * y);
        count++;
    }
    int k = (j * width + i) * 3;
    pixels[k] = (count < MAXITERATIONS) ? 0.0f : 1.0f; // If they Explode, they are black(0.0f)
    pixels[k + 1] = 0.0f;  // No Green
    pixels[k + 2] = 0.0f;  // No Blue
}



void display(void)
{
    escapeOrNotColor<<<GridSize, BlockSize>>>(d_pixels, WindowWidth, WindowHeight, XMin, XMax, YMin, YMax);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaDeviceSynchronize();



    //Copies results from GPU calculations back to CPU
    cudaMemcpy(pixels, d_pixels, WindowWidth * WindowHeight * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaErrorCheck(__FILE__, __LINE__);

    //Putting pixels on the screen.
    glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixels);
    glFlush();
}

int main(int argc, char** argv)
{
    pixels = (float *)malloc(WindowWidth * WindowHeight * 3 * sizeof(float));
    cudaMalloc((void**)&d_pixels, WindowWidth * WindowHeight * 3 * sizeof(float));

    setUpDevices();
    glutInit(&argc, argv);//this confused me 
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);//single buffer display mode using RGB
    glutInitWindowSize(WindowWidth, WindowHeight);//setss the window size to 1024x1024
    glutCreateWindow("Fractals--Man--Fractals");//makes the window with a title
    glutDisplayFunc(display);//draws the fractal
    glutMainLoop();//OpenGL toolkit making sure eveything executes when needed


    cudaFree(d_pixels);
    free(pixels);
}
