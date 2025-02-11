// Name: Ben Williams
// Not simple Julia Set on the GPU
// nvcc HW7.cu -o temp -lglut -lGL -lX11

/*
 What to do:
 This code displays a simple Julia set fractal using the GPU.
 But it only runs on a window of 1024X1024.
 Extend it so that it can run on any given window size.
 Also, color it to your liking. I will judge you on your artisct flare. 
 Don't cute off your ear or anything but make Vincent wish he had, had a GPU.
*/

// Include files 
#include <stdio.h>
#include <GL/glut.h>
#include <X11/Xlib.h>
#include <iostream>
// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define A  -0.824	//Real part of C
#define B  -0.1711	//Imaginary part of C

// Global variables
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

// Function prototypes
void cudaErrorCheck(const char*, int);
__global__ void colorPixels(float, float, float, float, float);

int getscreenwidth(){
	Display* disp = XOpenDisplay(NULL);
	   Screen* scrn = DefaultScreenOfDisplay(disp);
	   int width = scrn->width;
	   int height = scrn->height;
		  std::cout << "Screen Resolution: " << width << "x" << height << std::endl;
	   return width;
   }
   int getscreenheight(){
	   Display* disp = XOpenDisplay(NULL);
	   Screen* scrn = DefaultScreenOfDisplay(disp);
	   int width = scrn->width;
	   int height = scrn->height;
		  std::cout << "Screen Resolution: " << width << "x" << height << std::endl;
	   return height;
   }

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

__global__ void colorPixels(float *pixels, float xMin, float yMin, float dx, float dy, int width, int height) 
{
	float x,y,mag,tempX;
	int count, id;
	int maxCount = MAXITERATIONS;
	float maxMag = MAXMAG;
	
	
	//Getting the offset into the pixel buffer. 
	//We need the 3 because each pixel has a red, green, and blue value.
	id = 3*(threadIdx.x + blockDim.x*blockIdx.x);
	
	//Asigning each thread its x and y value of its pixel.
	int pixelIdx = threadIdx.x + blockDim.x * blockIdx.x;  // Unique index for each thread
	int pixelX = pixelIdx % width;  
    int pixelY = pixelIdx / width;  
	x = xMin + pixelX * dx;  
    y = yMin + pixelY * dy;
	
	count = 0;
	mag = sqrt(x*x + y*y);;
	while (mag < maxMag && count < maxCount) 
	{
		//We will be changing the x but we need its old value to find y.	
		tempX = x; 
		x = x*x - y*y + A;
		y = (2.0 * tempX * y) + B;
		mag = sqrt(x*x + y*y);
		count++;
	}
	
	//Setting the red value
	if(count < maxCount) //It excaped
	{
		pixels[id]     = 0.0;
		pixels[id + 1] = 0.0;
		pixels[id + 2] = 0.0;
	}
	else //It Stuck around
	{
		pixels[id]     = 1.0;
		pixels[id + 1] = 0.0;
		pixels[id + 2] = 0.0;
	}
	//Setting the green
	pixels[id+1] = 0.0;
	//Setting the blue 
	pixels[id+2] = 0.0;
}

void display(void) 
{ 
	int height = getscreenheight();
	int width = getscreenwidth();
	dim3 blockSize, gridSize;
	float *pixelsCPU, *pixelsGPU; 
	float stepSizeX, stepSizeY;
	
	//We need the 3 because each pixel has a red, green, and blue value.
	pixelsCPU = (float *)malloc(width*height*3*sizeof(float));
	cudaMalloc(&pixelsGPU,width*height*3*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	stepSizeX = (XMax - XMin)/(width);
	stepSizeY = (YMax - YMin)/(height);
	
	blockSize.x = 1024; //max block size;
	blockSize.y = 1;
	blockSize.z = 1;
	
	//Blocks in a grid
	gridSize.x =(((width*height)-1)/1024)+1;
	gridSize.y = 1;
	gridSize.z = 1;
	
	colorPixels<<<gridSize, blockSize>>>(pixelsGPU, XMin, YMin, stepSizeX, stepSizeY, width, height);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Copying the pixels that we just colored back to the CPU.
	cudaMemcpyAsync(pixelsCPU, pixelsGPU, width*height*3*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Putting pixels on the screen.	
	glDrawPixels(width, height, GL_RGB, GL_FLOAT, pixelsCPU); 
	glFlush(); 
}

int main(int argc, char** argv)
{ 
	int width = getscreenwidth(); 
    int height = getscreenheight(); 
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(width, height);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);
   	glutMainLoop();
}



