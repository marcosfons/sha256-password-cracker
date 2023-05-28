#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>
#include <sys/time.h>
#include <pthread.h>
#include <locale.h>

#include "sha256.cuh"
#include "hash_entry.cuh"


// #define THREADS 512
#define THREADS 16
// #define THREADS 1500
// #define BLOCKS 32
#define BLOCKS 4
// #define BLOCKS 256
#define GPUS 1

#define THREAD_EXECUTION_ITERATIONS ((MAX_PASSWORD_LENGTH - MIN_PASSWORD_CHECK))

#define CHARSET_LENGTH 68
__constant__ BYTE charset[CHARSET_LENGTH + 1] = {"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890%*$@"};

char* g_solution;

__host__ __device__ void print_hash_entry(hash_entry entry) {
	printf("Hash: ");
	for (size_t i = 0; i < HASH_BYTES_LENGTH; i++) {
		printf("%02x", entry.hash_bytes[i]);
	}
	printf("\nSalt: %.32s\n", entry.salt);
}

__device__ unsigned long deviceRandomGen(unsigned long x) {
  x ^= (x << 21);
  x ^= (x >> 35);
  x ^= (x << 4);
  return x;
}

__global__ void sha256_cuda(hash_entry *entry, unsigned char *blockContainsSolution, unsigned long baseSeed) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long seed = deviceRandomGen(baseSeed + id);

	BYTE input[MAX_PASSWORD_LENGTH];

	SHA256_CTX sha_ctx;
	BYTE digest[32];

	int found;
	
	for (int i = 0; i < MAX_PASSWORD_LENGTH; i++) {
		seed = deviceRandomGen(seed);
		input[i] = charset[seed % CHARSET_LENGTH];
	}

	for (int x = MIN_PASSWORD_CHECK; x < MAX_PASSWORD_LENGTH; x++) {
		sha256_init(&sha_ctx);
		sha256_update(&sha_ctx, entry->salt, SALT_LENGTH);
		sha256_update(&sha_ctx, input, (x % MAX_PASSWORD_LENGTH) + 1);
		sha256_update(&sha_ctx, "\n", 1);
		sha256_final(&sha_ctx, digest);

		found = 1;
		for (int i = 0; i < HASH_BYTES_LENGTH; i++) {
			if (digest[i] != entry->hash_bytes[i]) {
				found = 0;
				break;
			}
		}

		if (found) {
			for (int i = 0; i < x + 1; i++) {
				entry->solution[i] = input[i];
			}
			*blockContainsSolution = 1;
		}
	}
}

void hostRandomGen(unsigned long *x) {
  *x ^= (*x << 21);
  *x ^= (*x >> 35);
  *x ^= (*x << 4);
}

void pre_sha256() {
  cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice);
}

long long timems() {
  struct timeval end;
  gettimeofday(&end, NULL);
  return end.tv_sec * 1000LL + end.tv_usec / 1000;
}

struct HandlerInput {
  int device;
  unsigned long long hashesProcessed;
	hash_entry entry;
};
typedef struct HandlerInput HandlerInput;

pthread_mutex_t solutionLock;

void *launchGPUHandlerThread(void *vargp) {
  HandlerInput *hi = (HandlerInput *) vargp;
  cudaSetDevice(hi->device);

  pre_sha256();

	hash_entry *d_hash_entry;
	cudaMalloc(&d_hash_entry, sizeof(hash_entry));
	cudaMemcpy(d_hash_entry, &(hi->entry), sizeof(hash_entry), cudaMemcpyHostToDevice);

  unsigned char blockContainsSolution = 0;
  unsigned char *d_blockContainsSolution;
  cudaMalloc(&d_blockContainsSolution, sizeof(unsigned char));
	cudaMemcpy(&blockContainsSolution, d_blockContainsSolution, sizeof(unsigned char), cudaMemcpyHostToDevice);

  unsigned long rngSeed = timems();

  // while (1) {
	srand(rngSeed);
  while (1) {
		rngSeed = rand();

    hi->hashesProcessed += ((unsigned long long) (THREADS * BLOCKS * THREAD_EXECUTION_ITERATIONS)) * 8193;
    sha256_cuda<<<THREADS, BLOCKS>>>(d_hash_entry, d_blockContainsSolution, rngSeed);
		for (int i = 0; i < 8192; i++) {
			sha256_cuda<<<THREADS, BLOCKS>>>(d_hash_entry, d_blockContainsSolution, rand());
		}
    cudaDeviceSynchronize();

    cudaMemcpy(&blockContainsSolution, d_blockContainsSolution, sizeof(unsigned char), cudaMemcpyDeviceToHost);

		if (blockContainsSolution == 1) {
			char* solution = (char*) malloc(sizeof(char) * MAX_PASSWORD_LENGTH);
      cudaMemcpy(solution, &(d_hash_entry->solution), sizeof(char) * MAX_PASSWORD_LENGTH, cudaMemcpyDeviceToHost);
			printf("\nSolution: %s\n", solution);
			exit(1);
			break;
		}
  }

  cudaDeviceReset();
  return NULL;
}

void hexToBytes(const char* hex_string, BYTE bytes[HASH_BYTES_LENGTH]) {
  for (unsigned int i = 0; i < HASH_BYTES_LENGTH; i += 1) {
		sscanf(&hex_string[i * 2], "%02x", (unsigned int *) &bytes[i]);
  }
}


int main() {
	setlocale(LC_NUMERIC, "");

	hash_entry line;
	memset(line.solution, 0, MAX_PASSWORD_LENGTH);
	// Admin0
	// hexToBytes("27a575da417e1e4cdbf4fbbe8752579b6e1d65e79731ed773a6886812e2da116", line.hash_bytes);
	// strncpy((char*) line.salt, "3354623a2c1deaed1362f124c75db8a7", SALT_LENGTH);

	// teste
	// hexToBytes("6cea8869d44eefacc4b56d300905b9aa770503b46cca36f7cec9b36c8bb45ded", line.hash_bytes);
	// strncpy((char*) line.salt, "2609ad21084c3cc3e64f0e6777466000", SALT_LENGTH);

	// userC eu acho
	hexToBytes("e57e3489a84e3ad626608378bde5b0873b85c8bef443998d6bb1d3f2a6c7d0bc", line.hash_bytes);
	strncpy((char*) line.salt, "5958d4968fc679958e32c0b22d625856", SALT_LENGTH);

	print_hash_entry(line);

	pthread_mutex_init(&solutionLock, NULL);
	pthread_mutex_lock(&solutionLock);

	unsigned long long **processedPtrs = (unsigned long long **) malloc(sizeof(unsigned long long *) * GPUS);
	pthread_t *tids = (pthread_t *) malloc(sizeof(pthread_t) * GPUS);
	long long start = timems();
	for (int i = 0; i < GPUS; i++) {
    HandlerInput *hi = (HandlerInput *) malloc(sizeof(HandlerInput));
    hi->device = i;
    hi->hashesProcessed = 0;
		hi->entry = line;
    processedPtrs[i] = &hi->hashesProcessed;
    pthread_create(tids + i, NULL, launchGPUHandlerThread, hi);
    usleep(10);
	}

	// while (1) {
	usleep(100000);
	while (1) {
		usleep(10000);
		unsigned long totalProcessed = 0;
		for (int i = 0; i < GPUS; i++) {
			totalProcessed += *(processedPtrs[i]);
		}
		long long elapsed = timems() - start;
		printf("Hashes (%'lu) Seconds (%'f) Hashes/sec (%'lu)\r", totalProcessed, ((float) elapsed) / 1000.0, (unsigned long) ((double) totalProcessed / (double) elapsed) * 1000);
		if (g_solution) {
			break;
		}
	}
	printf("\n");

	pthread_mutex_lock(&solutionLock);
	long long end = timems();
	long long elapsed = end - start;

	for (int i = 0; i < GPUS; i++) {
		pthread_join(tids[i], NULL);
	}

	unsigned long totalProcessed = 0;
	for (int i = 0; i < GPUS; i++) {
		totalProcessed += *(processedPtrs[i]);
	}

	printf("Solution: %s\n", g_solution);
	printf("Hashes processed: %'lu\n", totalProcessed);
	printf("Time: %llu\n", elapsed);
	printf("Hashes/sec: %'lu\n", (unsigned long) ((double) totalProcessed / (double) elapsed) * 1000);

	return 0;
}
