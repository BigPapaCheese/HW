// Name: Ben Williams
// 2D GPU Random Walk with Trail Visualization
// Compile with:
//   nvcc HW28b.cu -o randomwalk2d -lglut -lGL -lGLU


#include <stdio.h>
#include <time.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <GL/glut.h>

#define N               10
#define NUM_STEPS       10000
#define TRAIL_INTERVAL  100
#define TRAIL_POINTS    (NUM_STEPS / TRAIL_INTERVAL + 1)

// Device and host storage
curandState *d_states;
float2      *d_trails;
float2      *h_trails;

// CUDA grid/block configuration
dim3 BlockSize(N, 1, 1);
dim3 GridSize(1, 1, 1);

// Window size
int winW = 800, winH = 800;

// --- CUDA kernels ---

__global__ void initRNG(curandState *states, unsigned long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed + tid, tid, 0, &states[tid]);
}

__global__ void randomwalk2D(curandState *states, float2 *trails) {
    int id   = threadIdx.x + blockIdx.x * blockDim.x;
    int base = id * TRAIL_POINTS;
    curandState local = states[id];

    float x = 0.0f, y = 0.0f;
    int   idx = 0;
    trails[base + idx] = make_float2(x, y);

    for (int i = 1; i <= NUM_STEPS; ++i) {
        float r = curand_uniform(&local) * 4.0f;
        if      (r < 1.0f) x += 1.0f;
        else if (r < 2.0f) x -= 1.0f;
        else if (r < 3.0f) y += 1.0f;
        else               y -= 1.0f;

        if (i % TRAIL_INTERVAL == 0) {
            ++idx;
            trails[base + idx] = make_float2(x, y);
        }
    }
}

// --- OpenGL display callback ---

void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();

    glLineWidth(2.0f);
    for (int w = 0; w < N; ++w) {
        // pick a color per walk
        float hue = float(w) / float(N);
        glColor3f(0.5f + 0.5f*sinf(6.2831f*hue),
                  0.5f + 0.5f*sinf(6.2831f*(hue+0.33f)),
                  0.5f + 0.5f*sinf(6.2831f*(hue+0.66f)));

        glBegin(GL_LINE_STRIP);
        for (int i = 0; i < TRAIL_POINTS; ++i) {
            float2 p = h_trails[w*TRAIL_POINTS + i];
            glVertex2f(p.x, p.y);
        }
        glEnd();
    }

    glutSwapBuffers();
}

// --- Main function ---

int main(int argc, char** argv) {
    // Allocate CUDA memory
    cudaMalloc(&d_states, N * sizeof(curandState));
    cudaMalloc(&d_trails, N * TRAIL_POINTS * sizeof(float2));
    h_trails = (float2*)malloc(N * TRAIL_POINTS * sizeof(float2));

    // Initialize RNG and perform walks
    initRNG      <<<GridSize, BlockSize>>>(d_states, (unsigned)time(NULL));
    cudaDeviceSynchronize();

    randomwalk2D <<<GridSize, BlockSize>>>(d_states, d_trails);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_trails, d_trails,
               N * TRAIL_POINTS * sizeof(float2),
               cudaMemcpyDeviceToHost);

    // Debug: print start/end of each walk
    for (int w = 0; w < N; ++w) {
        float2 start = h_trails[w*TRAIL_POINTS + 0];
        float2 end   = h_trails[w*TRAIL_POINTS + (TRAIL_POINTS-1)];
        printf("Walk %d start=(%.1f,%.1f) end=(%.1f,%.1f)\n",
               w, start.x, start.y, end.x, end.y);
    }

    // Free CUDA device memory; we only need h_trails for drawing
    cudaFree(d_states);
    cudaFree(d_trails);

    // Setup GLUT window
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(winW, winH);
    glutCreateWindow("2D Random Walk Trails");

    // White background
    glClearColor(1, 1, 1, 1);

    // Orthographic view to cover [-R,R] in both axes
    float R = 150.0f;  // a bit > sqrt(NUM_STEPS)=100
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-R, R, -R, R, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Register display callback and enter loop
    glutDisplayFunc(display);
    glutMainLoop();

    // Cleanup host memory (never reached)
    free(h_trails);
    return 0;
}
