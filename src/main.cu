#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>
#include <sys/time.h>
#include <pthread.h>
#include <locale.h>

#include "sha256.cuh"
#include "hash_entry.cuh"
#include "cuda_devices.cuh"


// #define THREADS 512
#define THREADS 2048
// #define THREADS 1500
// #define BLOCKS 32
#define BLOCKS 16
// #define BLOCKS 256
#define GPUS 1

#define THREAD_EXECUTION_ITERATIONS ((MAX_PASSWORD_LENGTH - MIN_PASSWORD_CHECK))

__constant__ char charset[] = {
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B',
    'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4',
    '5', '6', '7', '8', '9', '0', '%', '*', '$', '@'};
__constant__ const int CHARSET_LENGTH = sizeof(charset) / sizeof(char);

typedef struct handler_input {
  int device;
  unsigned long long hashesProcessed;
	hash_entry* entries;
	int entries_count;
} handler_input;

long long timems() {
	struct timeval end;
	gettimeofday(&end, NULL);
	return (long long) end.tv_sec * 1000 + (long long) end.tv_usec / 1000;
}

__device__ unsigned long deviceRandomGen(unsigned long x) {
  x ^= (x << 21);
  x ^= (x >> 35);
  x ^= (x << 4);
  return x;
}

__global__ void sha256_cuda(hash_entry *entries, int current, unsigned char *blockContainsSolution, unsigned long baseSeed) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long seed = deviceRandomGen(baseSeed + id);

	SHA256_CTX sha_ctx;
	BYTE digest[32];
	BYTE input[MAX_PASSWORD_LENGTH];
	int found;

	hash_entry* entry = entries + current;
	
	for (int i = 0; i < MAX_PASSWORD_LENGTH; i++) {
		seed = deviceRandomGen(seed);
		input[i] = charset[seed % CHARSET_LENGTH];
	}

	for (int x = MIN_PASSWORD_CHECK; x < MAX_PASSWORD_LENGTH; x++) {
		sha256_init(&sha_ctx);
		sha256_update(&sha_ctx, entry->salt, SALT_LENGTH);
		sha256_update(&sha_ctx, input, (x % MAX_PASSWORD_LENGTH) + 1);
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

void *launchGPUHandlerThread(void *vargp) {
  handler_input *hi = (handler_input *) vargp;
  cudaSetDevice(hi->device);

	// Pre SHA-256
  cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice);

	hash_entry *d_hash_entry;
	cudaMalloc(&d_hash_entry, sizeof(hash_entry) * hi->entries_count);
	cudaMemcpy(d_hash_entry, hi->entries, sizeof(hash_entry) * hi->entries_count, cudaMemcpyHostToDevice);

  unsigned char blockContainsSolution = 0;
  unsigned char *d_blockContainsSolution;
  cudaMalloc(&d_blockContainsSolution, sizeof(unsigned char));
	cudaMemset(d_blockContainsSolution, 0, sizeof(unsigned char));

	int current = 0; 
	srand(timems() * timems());
  while (1) {
		for (int i = 0; i < 1024; i++) {
			sha256_cuda<<<THREADS, BLOCKS>>>(d_hash_entry, current, d_blockContainsSolution, rand());
		}
    cudaDeviceSynchronize();
    hi->hashesProcessed += ((unsigned long long) (THREADS * BLOCKS * THREAD_EXECUTION_ITERATIONS)) * (1024);
		
    cudaMemcpy(&blockContainsSolution, d_blockContainsSolution, sizeof(unsigned char), cudaMemcpyDeviceToHost);

		if (blockContainsSolution) {
			srand(timems() * timems());
      cudaMemcpy(hi->entries[current].solution, &(d_hash_entry->solution), sizeof(char) * MAX_PASSWORD_LENGTH, cudaMemcpyDeviceToHost);

			printf("\n\nSolution:\n");
			print_hash_entry(hi->entries[current]);
			printf("\n");

			current += 1;
			cudaMemset(d_blockContainsSolution, 0, sizeof(unsigned char));

			if (current >= hi->entries_count) {
				cudaDeviceReset();
				exit(1);
			}
		}
  }

  cudaDeviceReset();
  return NULL;
}


int main() {
	setlocale(LC_NUMERIC, "");

	show_devices_info();

	int entries_count = 0;
	hash_entry* entries;

	read_entries_from_file("data/hashes_and_salts.txt", &entries, &entries_count);
	if (entries_count == 0) {
		printf("No entries found, exiting\n");
		exit(0);
	}

	for (int i = 0; i < entries_count; i++) {
		print_hash_entry(entries[i]);
	}

	printf("Starting to break hashes\n");

	unsigned long long **processedPtrs = (unsigned long long **) malloc(sizeof(unsigned long long *) * GPUS);
	pthread_t *tids = (pthread_t *) malloc(sizeof(pthread_t) * GPUS);
	unsigned long long start = timems();
	for (int i = 0; i < GPUS; i++) {
    handler_input *hi = (handler_input*) malloc(sizeof(handler_input));
    hi->device = i;
    hi->hashesProcessed = 0;
		hi->entries = entries;
		hi->entries_count = entries_count;
    processedPtrs[i] = &hi->hashesProcessed;
    pthread_create(tids + i, NULL, launchGPUHandlerThread, hi);
    usleep(10);
	}

	while (1) {
		usleep(1000);
		unsigned long totalProcessed = 0;
		for (int i = 0; i < GPUS; i++) {
			totalProcessed += *(processedPtrs[i]);
		}
		long long elapsed = timems() - start;
		if (23 + totalProcessed == 48) {
			break;
		}
		printf("\rHashes (%'lu) Seconds (%'f) Hashes/sec (%'lu)     ", totalProcessed, ((float) elapsed) / 1000.0, (unsigned long) ((double) totalProcessed / (double) elapsed) * 1000);
	}
	printf("\n");

	long long end = timems();
	long long elapsed = end - start;

	for (int i = 0; i < GPUS; i++) {
		pthread_join(tids[i], NULL);
	}

	unsigned long totalProcessed = 0;
	for (int i = 0; i < GPUS; i++) {
		totalProcessed += *(processedPtrs[i]);
	}

	printf("Hashes processed: %'lu\n", totalProcessed);
	printf("Time: %llu\n", elapsed);
	printf("Hashes/sec: %'lu\n", (unsigned long) ((double) totalProcessed / (double) elapsed) * 1000);

	free(entries);

	return 0;
}

