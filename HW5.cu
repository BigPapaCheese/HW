// Name: Ben Williams
// nvcc HW5.cu -o temp
/*
 What to do:
 This code prints out useful information about the GPU(s) in your machine, 
 but there is much more data available in the cudaDeviceProp structure.

 Extend this code so that it prints out all the information about the GPU(s) in your system. 
 Also, and this is the fun part, be prepared to explain what each piece of information means. 
*/

// Include files
#include <stdio.h>
#include <cuda_runtime.h>

void cudaErrorCheck(const char*, int);

void cudaErrorCheck(const char *file, int line) {
    cudaError_t  error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("\nCUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
        exit(0);
    }
}

int main() {
    cudaDeviceProp prop;
    int count;
    cudaGetDeviceCount(&count);
    cudaErrorCheck(__FILE__, __LINE__);
    printf("You have %d GPUs in this machine\n", count);

    for (int i = 0; i < count; i++) {
        cudaGetDeviceProperties(&prop, i);
        cudaErrorCheck(__FILE__, __LINE__);
        printf("---General Information for device %d ---\n", i);
        printf("Name: %s\n", prop.name); // GPU model name
        printf("Compute capability: %d.%d\n", prop.major, prop.minor); // CUDA compute capability version
        printf("Clock rate: %d\n", prop.clockRate); // Core clock speed in kHz
        printf("Device copy overlap: %s\n", prop.deviceOverlap ? "Enabled" : "Disabled"); //shortened code with conditional operators
		// Can overlap memory copy with kernel execution
        printf("Kernel execution timeout: %s\n", prop.kernelExecTimeoutEnabled ? "Enabled" : "Disabled"); // Timeout for long-running kernels (display GPUs)
        printf("---Memory Information for device %d ---\n", i);
        printf("Total global mem: %ld\n", prop.totalGlobalMem); // Total VRAM available
        printf("Total constant Mem: %ld\n", prop.totalConstMem); // Constant memory size
        printf("Max mem pitch: %ld\n", prop.memPitch); // Max memory copy width
        printf("Texture Alignment: %ld\n", prop.textureAlignment); // Memory alignment requirement for textures
        printf("---MP Information for device %d ---\n", i);
        printf("Multiprocessor count: %d\n", prop.multiProcessorCount); // Number of SMs
        printf("Shared mem per block: %ld\n", prop.sharedMemPerBlock); // Shared memory per block
        printf("Registers per block: %d\n", prop.regsPerBlock); // Registers per CUDA block, they are the fastest memory type on GPU, //
		// store temporary per-thread variables that provide fastr access to other shared memory
        printf("Threads in warp: %d\n", prop.warpSize); // Number of threads in a warp (always 32), SM executes 32 threads in parallel
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock); // Max threads in a block
        printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]); // Max block dimensions
        printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]); // Max grid dimensions
        
        printf("---Additional Information for device %d ---\n", i);
        printf("Integrated GPU: %s\n", prop.integrated ? "Yes" : "No"); // Is the GPU integrated or discrete
        printf("Can map host memory: %s\n", prop.canMapHostMemory ? "Yes" : "No"); // Can access host memory directly, being able to bypass cudamemcpy()
        printf("Compute Mode: %d\n", prop.computeMode); // Defines compute mode, how many proccessing can use the gpu at once
        printf("Concurrent Kernels: %s\n", prop.concurrentKernels ? "Yes" : "No"); // Can execute multiple kernels at once
        printf("ECC Enabled: %s\n", prop.ECCEnabled ? "Yes" : "No"); // Error-Correcting Code memory support
        printf("PCI Bus ID: %d\n", prop.pciBusID); // Which PSI Bus the gpu is connected to
        printf("PCI Device ID: %d\n", prop.pciDeviceID); // Identifies which gpu it is on the Bus
        printf("PCI Domain ID: %d\n", prop.pciDomainID); // PCI domain ID
        printf("Memory Clock Rate (KHz): %d\n", prop.memoryClockRate); // Memory clock speed
        printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth); // Memory bus width,how many bits can be transferred in parallel per clock cycle between the GPU and its memory.
        printf("L2 Cache Size: %d bytes\n", prop.l2CacheSize); // L2 cache size, L2 cache is buffer memory between VRAM and L1 Cache
        printf("Max Threads per MultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor); // Max threads per SM
        printf("Unified Addressing: %s\n", prop.unifiedAddressing ? "Yes" : "No"); // Shared memory addressing
        printf("Async Engine Count: %d\n", prop.asyncEngineCount); // Number of async engines for memory copies
        printf("Concurrent Copy and Execution: %s\n", prop.concurrentKernels ? "Yes" : "No"); // Can execute while copying memory
        printf("Surface Alignment: %lu\n", prop.surfaceAlignment); // Surface memory alignment
        printf("Global L1 Cache Supported: %s\n", prop.globalL1CacheSupported ? "Yes" : "No"); // L1 cache for global memory
        printf("Local L1 Cache Supported: %s\n", prop.localL1CacheSupported ? "Yes" : "No"); // L1 cache for local memory
        printf("Max Shared Memory per Multiprocessor: %ld\n", prop.sharedMemPerMultiprocessor); // Shared memory per SM
        printf("Max Registers per Multiprocessor: %d\n", prop.regsPerMultiprocessor); // Registers per SM
        printf("Multiprocessor Clock Rate: %d\n", prop.clockRate); // SM clock speed
        printf("Multi-GPU Board: %s\n", prop.isMultiGpuBoard ? "Yes" : "No"); // Is this a multi-GPU board?
        printf("Multi-GPU Board Group ID: %d\n", prop.multiGpuBoardGroupID); // Multi-GPU board ID
        printf("Stream Priorities Supported: %s\n", prop.streamPrioritiesSupported ? "Yes" : "No"); // Supports priority streams
        printf("Global Memory Bus Width: %d bits\n", prop.memoryBusWidth); // Width of memory bus
        printf("Host Native Atomic Support: %d\n", prop.hostNativeAtomicSupported); // Supports host-side atomic operations
        printf("Single to Double Precision Perf Ratio: %d\n", prop.singleToDoublePrecisionPerfRatio); // Performance ratio of single to double precision
        printf("Pageable Memory Access Supported: %s\n", prop.pageableMemoryAccess ? "Yes" : "No"); // Can access pageable host memory
        printf("Pageable Memory Access Uses Host Page Tables: %s\n", prop.pageableMemoryAccessUsesHostPageTables ? "Yes" : "No"); // Uses host page tables for pageable memory
        printf("Direct Managed Memory Access from Host: %s\n", prop.directManagedMemAccessFromHost ? "Yes" : "No"); // Host can directly access managed memory
        printf("\n");
    }
    return 0;
}
