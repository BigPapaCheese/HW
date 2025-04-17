// Name:
// nBody run on all available GPUs. 
// nvcc HW25.cu -o temp -lglut -lm -lGLU -lGL

/*
 What to do:
 This is some robust N-body code with all the bells and whistles removed. 
 It runs on two GPUs and two GPUs only. Rewrite it so it automatically detects the number of 
 available GPUs on the machine and runs using all of them.
*/

// Include files
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Defines
#define BLOCK_SIZE 128
#define PI 3.14159265359
#define DRAW_RATE 10

// This is to create a Lennard-Jones type function G/(r^p) - H(r^q). (p < q) p has to be less than q.
// In this code we will keep it a p = 2 and q = 4 problem. The diameter of a body is found using the general
// case so it will be more robust but in the code leaving it as a set 2, 4 problem make the coding much easier.
#define G 10.0f
#define H 10.0f
#define LJP  2.0
#define LJQ  4.0

#define DT 0.0001
#define RUN_TIME 1.0

// Globals
int N;
int C;
int NN, NS;
int HalfN; // Half the vector size
int NumberOfGpus;
float3 *P, *V, *F;
float *M; 
float3 *PGPU0, *VGPU0, *FGPU0;
float *MGPU0;
float3 **PG, **VG, **FG;
float3 *PGPU1, *VGPU1, *FGPU1;
float **MG;
float GlobeRadius, Diameter, Radius;
float Damp;
dim3 BlockSize;
dim3 GridSize;
dim3 GridSizeS;

// Function prototypes
void cudaErrorCheck(const char *, int);
void drawPicture();
void setup();
__global__ void getForces(float3 *, float3 *, float3 *, float *, float, float, int, int, int, int);
__global__ void moveBodies(float3 *, float3 *, float3 *, float *, float, float, float, int, int, int, int);
void nBody();
int main(int, char**);

void sync()
{
    for (int i = 0; i< C;i++)
    {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
        cudaErrorCheck(__FILE__, __LINE__);
    }
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

void drawPicture()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	cudaSetDevice(0);
	cudaMemcpyAsync(P, PG[0], N*sizeof(float3), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	glColor3d(1.0,1.0,0.5);
	
	for(int i=0; i<N; i++)
	{
		glPushMatrix();
		glTranslatef(P[i].x, P[i].y, P[i].z);
		glutSolidSphere(Radius,20,20);
		glPopMatrix();
	}
	
	glutSwapBuffers();
}

void setup()
{

    	float randomAngle1, randomAngle2, randomRadius;
    	float d, dx, dy, dz;
    	int test;
	
	N = 101;
	
	cudaGetDeviceCount(&C);
	
    NN = N/C;
    if (C==1)
    {
        NN = N;
    }
    NS = NN + N%C;	
		
		BlockSize.x = 128;
		BlockSize.y = 1;
		BlockSize.z = 1;
		
		GridSize.x = (NN - 1)/BlockSize.x + 1; // This gives us the correct number of blocks.
		GridSize.y = 1;
		GridSize.z = 1;
        GridSizeS.x = (NS - 1)/BlockSize.x + 1; // This gives us the correct number of blocks.
		GridSizeS.y = 1;
		GridSizeS.z = 1;
	
	
    	Damp = 0.5;

    	PG = (float3**)malloc(C * sizeof(float3*));
        VG = (float3**)malloc(C * sizeof(float3*));
        FG = (float3**)malloc(C * sizeof(float3*));
        MG = (float**) malloc(C * sizeof(float*));

    	M = (float*)malloc(N*sizeof(float));
    	P = (float3*)malloc(N*sizeof(float3));
    	V = (float3*)malloc(N*sizeof(float3));
    	F = (float3*)malloc(N*sizeof(float3));
    	
    	// !! Important: Setting the number of bodies a little bigger if it is not even or you will 
    	// get a core dump because you will be copying memory you do not own. This only needs to be
    	// done for positions but I did it for all for completness incase the code gets used for a
    	// more complicated force function.
    	// Device "GPU0" Memory

        for(int i = 0; i<C;i++)
        {
            cudaSetDevice(i);
            cudaMalloc(&MG[i],N*sizeof(float));
	        cudaErrorCheck(__FILE__, __LINE__);
	        cudaMalloc(&PG[i],N*sizeof(float3));
	        cudaErrorCheck(__FILE__, __LINE__);
	        cudaMalloc(&VG[i],N*sizeof(float3));
            cudaErrorCheck(__FILE__, __LINE__);
            cudaMalloc(&FG[i],N*sizeof(float3));
            cudaErrorCheck(__FILE__, __LINE__);
        }

    	
	Diameter = pow(H/G, 1.0/(LJQ - LJP)); // This is the value where the force is zero for the L-J type force.
	Radius = Diameter/2.0;
	
	// Using the radius of a body and a 68% packing ratio to find the radius of a global sphere that should hold all the bodies.
	// Then we double this radius just so we can get all the bodies setup with no problems. 
	float totalVolume = float(N)*(4.0/3.0)*PI*Radius*Radius*Radius;
	totalVolume /= 0.68;
	float totalRadius = pow(3.0*totalVolume/(4.0*PI), 1.0/3.0);
	GlobeRadius = 2.0*totalRadius;
	
	// Randomly setting these bodies in the glaobal sphere and setting the initial velosity, inotial force, and mass.
	for(int i = 0; i < N; i++)
	{
		test = 0;
		while(test == 0)
		{
			// Get random position.
			randomAngle1 = ((float)rand()/(float)RAND_MAX)*2.0*PI;
			randomAngle2 = ((float)rand()/(float)RAND_MAX)*PI;
			randomRadius = ((float)rand()/(float)RAND_MAX)*GlobeRadius;
			P[i].x = randomRadius*cos(randomAngle1)*sin(randomAngle2);
			P[i].y = randomRadius*sin(randomAngle1)*sin(randomAngle2);
			P[i].z = randomRadius*cos(randomAngle2);
			
			// Making sure the bodies' centers are at least a diameter apart.
			// If they are not throw these positions away and try again.
			test = 1;
			for(int j = 0; j < i; j++)
			{
				dx = P[i].x-P[j].x;
				dy = P[i].y-P[j].y;
				dz = P[i].z-P[j].z;
				d = sqrt(dx*dx + dy*dy + dz*dz);
				if(d < Diameter)
				{
					test = 0;
					break;
				}
			}
		}
	
		V[i].x = 0.0;
		V[i].y = 0.0;
		V[i].z = 0.0;
		
		F[i].x = 0.0;
		F[i].y = 0.0;
		F[i].z = 0.0;
		
		M[i] = 1.0;
	}
	
	
	for(int j = 0;j<C-1;j++)
    {
    cudaSetDevice(j);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMemcpyAsync(PG[j], P, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(VG[j], V, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(FG[j], F, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(MG[j], M, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
    }
    if (C>1)
    {
    cudaSetDevice(C-1);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMemcpyAsync(PG[C-1], P, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(VG[C-1], V, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(FG[C-1], F, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(MG[C-1], M, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
    }
	
		
	printf("\n Setup finished.\n");
}

__global__ void getForces(float3 *p, float3 *v, float3 *f, float *m, float g, float h, int myN, int n, int device, int nn)
{
	float dx, dy, dz,d,d2;
	float force_mag;
	int offset = device*nn;
	
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	
	if(i < myN)
	{
        i+= offset;
		f[i].x = 0.0f;
		f[i].y = 0.0f;
		f[i].z = 0.0f;
		
		for(int j = 0; j < n; j++)
		{
			if(i != j)
			{
				dx = p[j].x-p[i].x;
				dy = p[j].y-p[i].y;
				dz = p[j].z-p[i].z;
				d2 = dx*dx + dy*dy + dz*dz;
				d  = sqrt(d2);
				
				force_mag  = (g*m[i]*m[j])/(d2) - (h*m[i]*m[j])/(d2*d2);
				f[i].x += force_mag*dx/d;
				f[i].y += force_mag*dy/d;
				f[i].z += force_mag*dz/d;
			}
		}
	}
}

__global__ void moveBodies(float3 *p, float3 *v, float3 *f, float *m, float damp, float dt, float t, int halfN, int n, int device, int nn)
{
    int offset = device*nn;
	
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	
	if(i < n)
	{
        i+=offset;
		if(t == 0.0f)
		{
			v[i].x += ((f[i].x-damp*v[i].x)/m[i])*dt/2.0f;
			v[i].y += ((f[i].y-damp*v[i].y)/m[i])*dt/2.0f;
			v[i].z += ((f[i].z-damp*v[i].z)/m[i])*dt/2.0f;
		}
		else
		{
			v[i].x += ((f[i].x-damp*v[i].x)/m[i])*dt;
			v[i].y += ((f[i].y-damp*v[i].y)/m[i])*dt;
			v[i].z += ((f[i].z-damp*v[i].z)/m[i])*dt;
		}

		p[i].x += v[i].x*dt;
		p[i].y += v[i].y*dt;
		p[i].z += v[i].z*dt;
	}
}

void nBody()
{
    int offset;
    int size;
	int    drawCount = 0; 
	float  t = 0.0;
	float dt = 0.0001;
    if(C==1)
    {
        while(t < RUN_TIME)
	{
        getForces<<<GridSize,BlockSize>>>(PGPU0, VGPU0, FGPU0, MGPU0, G, H, HalfN, N, 0, NN);
		cudaErrorCheck(__FILE__, __LINE__);
		moveBodies<<<GridSize,BlockSize>>>(PGPU0, VGPU0, FGPU0, MGPU0, Damp, dt, t, HalfN, N, 0, NN);
		cudaErrorCheck(__FILE__, __LINE__);
        cudaDeviceSynchronize();
        if(drawCount == DRAW_RATE) 
		{	
			drawPicture();
			drawCount = 0;
		}
		
		t += dt;
		drawCount++;
    }
    }
    else
    {
	while(t < RUN_TIME)
	{
        for(int k=0;k<C-1;k++)
        {
        cudaSetDevice(k);
        getForces<<<GridSize,BlockSize>>>(PG[k], VG[k], FG[k], MG[k], G, H, NN, N, k, NN);
		cudaErrorCheck(__FILE__, __LINE__);
		moveBodies<<<GridSize,BlockSize>>>(PG[k], VG[k], FG[k], MG[k], Damp, dt, t, NN, N, k, NN);
		cudaErrorCheck(__FILE__, __LINE__);
        }
        cudaSetDevice(C-1);
        getForces<<<GridSizeS,BlockSize>>>(PG[C-1], VG[C-1], FG[C-1], MG[C-1], G, H, NS, N, C-1, NN);
		cudaErrorCheck(__FILE__, __LINE__);
		moveBodies<<<GridSizeS,BlockSize>>>(PG[C-1], VG[C-1], FG[C-1], MG[C-1], Damp, dt, t, NS, N, C-1, NN);
		cudaErrorCheck(__FILE__, __LINE__);
    
        //sync();
        for (int i = 0; i<C;i++)//giver
        {
            offset = i*NN;
            size = (i == C - 1) ? NS : NN;
            for(int j = 0;j<C;j++)//getter
            {
                if(i!=j)
                {
                // bulk of sharing
                        //     getter             giver
                cudaMemcpyPeer(PG[j] + offset, j, PG[i] + offset, i, size * sizeof(float3));
                cudaErrorCheck(__FILE__, __LINE__);
                }
            }
        }

		if(drawCount == DRAW_RATE) 
		{	
			drawPicture();
			drawCount = 0;
		}
		
		t += dt;
		drawCount++;
	}
}
}

int main(int argc, char** argv)
{
	setup();
	if(C==0)
    {
        printf("Get a GPU");
        exit(1);
    }
	int XWindowSize = 1000;
	int YWindowSize = 1000;
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("Nbody Two GPUs");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutDisplayFunc(drawPicture);
	glutIdleFunc(nBody);
	
	float3 eye = {0.0f, 0.0f, 2.0f*GlobeRadius};
	float near = 0.2;
	float far = 5.0*GlobeRadius;
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, near, far);
	glMatrixMode(GL_MODELVIEW);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluLookAt(eye.x, eye.y, eye.z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	
	glutMainLoop();
	return 0;
}
