#include <stdio.h>
#include <cuda_runtime.h>

#include "cuda_devices.h"

int _ConvertSMVer2Cores(int major, int minor) {
	// Defines the number of CUDA cores per multiprocessor for different compute capabilities
	// This mapping may vary depending on the GPU architecture
	struct {
		int major;
		int minor;
		int cores;
	} smCoresMap[] = {
		{ 0, 1, 8 },   // Compute Capability 1.0
		{ 1, 0, 8 },   // Compute Capability 1.1
		{ 1, 1, 8 },   // Compute Capability 1.2
		{ 1, 2, 8 },   // Compute Capability 1.3
		{ 1, 3, 8 },   // Compute Capability 2.0
		{ 2, 0, 32 },  // Compute Capability 2.1
		{ 2, 1, 48 },  // Compute Capability 2.2
		{ 3, 0, 192 }, // Compute Capability 3.0
		{ 3, 5, 192 }, // Compute Capability 3.5
		{ 3, 7, 192 }, // Compute Capability 3.7
		{ 5, 0, 128 }, // Compute Capability 5.0
		{ 5, 2, 128 }, // Compute Capability 5.2
		{ 5, 3, 128 }, // Compute Capability 5.3
		{ 6, 0, 64 },  // Compute Capability 6.0
		{ 6, 1, 128 }, // Compute Capability 6.1
		{ 6, 2, 128 }, // Compute Capability 6.2
		{ 7, 0, 64 },  // Compute Capability 7.0
		{ 7, 2, 64 },  // Compute Capability 7.2
		{ 7, 5, 64 },  // Compute Capability 7.5
		{ -1, -1, -1 } // End of map
	};

	int i = 0;
	while (smCoresMap[i].major != -1) {
		if (smCoresMap[i].major == major && smCoresMap[i].minor == minor) {
			return smCoresMap[i].cores;
		}
		++i;
	}

	return -1; // Unknown compute capability
}

void showGPUDevicesInfo() {
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0) {
		printf("No CUDA devices found.\n");
		return;
	}

	for (int i = 0; i < deviceCount; ++i) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, i);

		printf("Device %d:\n", i);
		printf("  Device Name: %s\n", deviceProp.name);
		printf("  Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
		printf("  Memory Clock Rate: %d kHz\n", deviceProp.memoryClockRate);
		printf("  Memory Bus Width: %d bits\n", deviceProp.memoryBusWidth);
		printf("  Total Global Memory: %lu bytes\n", deviceProp.totalGlobalMem);
		printf("  Total Constant Memory: %lu bytes\n", deviceProp.totalConstMem);
		printf("  Multiprocessors: %d\n", deviceProp.multiProcessorCount);
		printf("  CUDA Cores per Multiprocessor: %d\n", _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
		printf("  Max Threads per Multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
		printf("  Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
		printf("  Warp Size: %d\n", deviceProp.warpSize);
		printf("  Max Grid Size: (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
		printf("  Max Block Size: (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
		printf("\n");
	}
}
