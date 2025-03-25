// Name:
// Two body problem
// nvcc HW17.cu -o temp -lglut -lGLU -lGL
//To stop hit "control c" in the window you launched it from.

/*
 What to do:
 This is some crude code that moves two bodies around in a box, attracted by gravity and 
 repelled when they hit each other. Take this from a two-body problem to an N-body problem, where 
 NUMBER_OF_SPHERES is a #define that you can change. Also clean it up a bit so it is more user-friendly.
*/

// Include files
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Defines
#define XWindowSize 1000
#define YWindowSize 1000
#define STOP_TIME 10000.0
#define DT        0.00001
#define GRAVITY 0.1 
#define MASS 10.0  	
#define DIAMETER 1.0
#define SPHERE_PUSH_BACK_STRENGTH 50.0
#define PUSH_BACK_REDUCTION 0.1
#define DAMP 0.01
#define DRAW 100
#define LENGTH_OF_BOX 6.0
#define MAX_VELOCITY 5.0
#define NUMBER_OF_SPHERES 20

// Globals
const float XMax = (LENGTH_OF_BOX/2.0);
const float YMax = (LENGTH_OF_BOX/2.0);
const float ZMax = (LENGTH_OF_BOX/2.0);
const float XMin = -(LENGTH_OF_BOX/2.0);
const float YMin = -(LENGTH_OF_BOX/2.0);
const float ZMin = -(LENGTH_OF_BOX/2.0);
float* px = (float*)malloc(NUMBER_OF_SPHERES * sizeof(float));
float* py = (float*)malloc(NUMBER_OF_SPHERES * sizeof(float));
float* pz = (float*)malloc(NUMBER_OF_SPHERES * sizeof(float));
float* vx = (float*)malloc(NUMBER_OF_SPHERES * sizeof(float));
float* vy = (float*)malloc(NUMBER_OF_SPHERES * sizeof(float));
float* vz = (float*)malloc(NUMBER_OF_SPHERES * sizeof(float));
float* fx = (float*)malloc(NUMBER_OF_SPHERES * sizeof(float));
float* fy = (float*)malloc(NUMBER_OF_SPHERES * sizeof(float));
float* fz = (float*)malloc(NUMBER_OF_SPHERES * sizeof(float));
float* mass = (float*)malloc(NUMBER_OF_SPHERES * sizeof(float));

// Function prototypes
void set_initail_conditions();
void Drawwirebox();
void draw_picture();
void keep_in_box();
void get_forces();
void move_bodies(float);
void nbody();
void Display(void);
void reshape(int, int);
int main(int, char**);

void set_initail_conditions()
{ 
	time_t t;
	srand((unsigned) time(&t));
	int yeahBuddy = 0;
	float dx, dy, dz, seperation;

    px[0] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
	py[0] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
	pz[0] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
    while (yeahBuddy == 0)
    {
		yeahBuddy =1;
        for(int m = 1;m<NUMBER_OF_SPHERES;m++)
        {
            px[m] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
	        py[m] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
	        pz[m] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
        }
        for(int i = NUMBER_OF_SPHERES -1 ; i>0 ;i--)
        {
            for(int j=0;j<i;j++)
            {
            dx = px[i] - px[j];
            dy = py[i] - py[j];
            dz = pz[i] - pz[j];
            seperation = sqrt(dx*dx + dy*dy + dz*dz);
            if(seperation < DIAMETER) 
			{
				yeahBuddy = 0;
			}
			}
        }
    
}
	printf("%f %f %f %f", px[0], px[1], px[2] ,px[3]);
    for (int v = 0;v<NUMBER_OF_SPHERES;v++)
    {
	vx[v] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	vy[v] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	vz[v] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
    }
	for (int k = 0;k<NUMBER_OF_SPHERES;k++)
    {
	mass[k]=MASS;
    }
}

void Drawwirebox()
{		
	glColor3f (5.0,1.0,1.0);
	glBegin(GL_LINE_STRIP);
		glVertex3f(XMax,YMax,ZMax);
		glVertex3f(XMax,YMax,ZMin);	
		glVertex3f(XMax,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMax);
		glVertex3f(XMax,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		glVertex3f(XMin,YMax,ZMin);	
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMin,YMax,ZMax);	
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMax,YMin,ZMax);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMin);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMax,ZMin);
		glVertex3f(XMax,YMax,ZMin);		
	glEnd();
	
}

void draw_picture()
{
	float radius = DIAMETER/2.0;
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	Drawwirebox();
	
    for(int i=0;i<NUMBER_OF_SPHERES;i++)
    {
    glColor3d(1.0,0.2*i,1.0);
	glPushMatrix();
	glTranslatef(px[i], py[i], pz[i]);
	glutSolidSphere(radius,20,20);
	glPopMatrix();
    
    }
    glutSwapBuffers();
}

void keep_in_box()
{
	float halfBoxLength = (LENGTH_OF_BOX - DIAMETER)/2.0;
	for(int i=0;i<NUMBER_OF_SPHERES;i++)
    {
	if(px[i] > halfBoxLength)
	{
		px[i] = 2.0*halfBoxLength - px[i];
		vx[i] = - vx[i];
	}
	else if(px[i] < -halfBoxLength)
	{
		px[i] = -2.0*halfBoxLength - px[i];
		vx[i] = - vx[i];
	}
	
	if(py[i] > halfBoxLength)
	{
		py[i] = 2.0*halfBoxLength - py[i];
		vy[i] = - vy[i];
	}
	else if(py[i] < -halfBoxLength)
	{
		py[i] = -2.0*halfBoxLength - py[i];
		vy[i] = - vy[i];
	}
			
	if(pz[i] > halfBoxLength)
	{
		pz[i] = 2.0*halfBoxLength - pz[i];
		vz[i] = - vz[i];
	}
	else if(pz[i] < -halfBoxLength)
	{
		pz[i] = -2.0*halfBoxLength - pz[i];
		vz[i] = - vz[i];
	}
}
}

void get_forces()
{	
	for (int i = 0; i < NUMBER_OF_SPHERES; i++) {
		fx[i] = 0.0;
		fy[i] = 0.0;
		fz[i] = 0.0;
	}
	float dx,dy,dz,r,r2,dvx,dvy,dvz,forceMag,inout;
    float tfx[NUMBER_OF_SPHERES][NUMBER_OF_SPHERES];
    float tfy[NUMBER_OF_SPHERES][NUMBER_OF_SPHERES];
    float tfz[NUMBER_OF_SPHERES][NUMBER_OF_SPHERES];
	for (int i = 0; i < NUMBER_OF_SPHERES; i++) {
		for (int j = 0; j < NUMBER_OF_SPHERES; j++) {
			tfx[i][j] = 0.0;
			tfy[i][j] = 0.0;
			tfz[i][j] = 0.0;
		}
	}
	
for(int i = 0;i<NUMBER_OF_SPHERES;i++)
{
    for(int j = i; j<NUMBER_OF_SPHERES;j++)
    {
        dx = px[j]-px[i];
        dy = py[j]-py[i];
        dz = pz[j]-pz[i];
        r2 = dx*dx + dy*dy + dz*dz;
	r = sqrt(r2);
	forceMag =  MASS*MASS*GRAVITY/r2;
			
	if (r < DIAMETER)
	{
		dvx = vx[j] - vx[i];
		dvy = vy[j] - vy[i];
		dvz = vz[j] - vz[i];
		
		inout = dx*dvx + dy*dvy + dz*dvz;
		if(inout <= 0.0)
		{
			forceMag +=  SPHERE_PUSH_BACK_STRENGTH*(r - DIAMETER);
		}
		else
		{
			forceMag += PUSH_BACK_REDUCTION*SPHERE_PUSH_BACK_STRENGTH*(r - DIAMETER);
		}
	}
	if(i==j)
	{
		tfx[i][j] =0;
		tfy[i][j] =0;
		tfz[i][j] =0;
	}
	else { 
	tfx[i][j] = -forceMag*dx/r;
	tfy[i][j] = -forceMag*dy/r;
	tfz[i][j] = -forceMag*dz/r;
	tfx[j][i] = -tfx[i][j]; // Symmetric forces
	tfy[j][i] = -tfy[i][j];
	tfz[j][i] = -tfz[i][j];

	}
    }
}
for (int k = 0;k<NUMBER_OF_SPHERES;k++)
{
for (int l = 0;l<NUMBER_OF_SPHERES;l++)
{
    fx[k] += tfx[l][k];
    fy[k] += tfy[l][k];
    fz[k] += tfz[l][k];
}

}
}


void move_bodies(float time)
{
    for (int i = 0; i<NUMBER_OF_SPHERES;i++)
    {
        if(time == 0.0)
	{
		vx[i] += 0.5*DT*(fx[i] - DAMP*vx[i])/MASS;
		vy[i] += 0.5*DT*(fy[i] - DAMP*vy[i])/MASS;
		vz[i] += 0.5*DT*(fz[i] - DAMP*vz[i])/MASS;
		
	}
	else
	{
		vx[i] += DT*(fx[i] - DAMP*vx[i])/MASS;
		vy[i] += DT*(fy[i] - DAMP*vy[i])/MASS;
		vz[i] += DT*(fz[i] - DAMP*vz[i])/MASS;
	}

	px[i] += DT*vx[i];
	py[i] += DT*vy[i];
	pz[i] += DT*vz[i];
	
	keep_in_box();
    }
}

void nbody()
{	
	int    tdraw = 0;
	float  time = 0.0;

	set_initail_conditions();
	
	draw_picture();
	
	while(time < STOP_TIME)
	{
		get_forces();
	
		move_bodies(time);
	
		tdraw++;
		if(tdraw == DRAW) 
		{
			draw_picture(); 
			tdraw = 0;
		}
		
		time += DT;
	}
	printf("\n DONE \n");
	while(1);
}

void Display(void)
{
	gluLookAt(0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glutSwapBuffers();
	glFlush();
	nbody();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, 50.0);

	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("2 Body 3D");
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
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;
}
