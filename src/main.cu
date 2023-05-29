#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>
#include <stddef.h>
#include <sys/time.h>
#include <pthread.h>
#include <locale.h>

#include "sha256.cuh"
#include "hash_entry.cuh"
#include "cuda_devices.cuh"


// Current stage
#define DEBUG
// #define TEST
// #define RELEASE

#define GPUS 1

#ifdef DEBUG
#define BLOCKS 16
#define THREADS 10
#define RUNS_PER_ITERATION 16
#define PRINT_STATUS_DELAY 10000
#endif

#ifdef TEST
#define BLOCKS 32
#define THREADS 30
#define RUNS_PER_ITERATION 32
#define PRINT_STATUS_DELAY 100000
#endif

#ifdef RELEASE
#define BLOCKS 32
#define THREADS 2048
#define RUNS_PER_ITERATION 32
#define PRINT_STATUS_DELAY 100000
#endif

#define THREAD_EXECUTION_ITERATIONS ((MAX_PASSWORD_LENGTH - MIN_PASSWORD_CHECK))

#ifdef DEBUG
__constant__ char charset[] = {'t', 'e', 's', 'a', 'b', 'c',
                               'd', 'f', 'g', 'h', 'i', 'j'};
#endif

#ifdef TEST
__constant__ char charset[] = {
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B',
};
#endif

#ifdef RELEASE
__constant__ char charset[] = {
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B',
    'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4',
    '5', '6', '7', '8', '9', '0', '%', '*', '$', '@'};
#endif

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

__global__ void sha256_cuda(hash_entry *entries, int entries_count,
                            unsigned char *blockContainsSolution,
                            unsigned long baseSeed) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long seed = deviceRandomGen(baseSeed + id);

	int entry_pos = (blockIdx.x % entries_count);
	hash_entry* entry = entries + entry_pos;

	SHA256_CTX sha_ctx;
	BYTE digest[32];
	BYTE input[MAX_PASSWORD_LENGTH];
	int found;
	
	for (int i = 0; i < MAX_PASSWORD_LENGTH; i++) {
		seed = deviceRandomGen(seed);
		input[i] = charset[seed % CHARSET_LENGTH];
	}

	for (int x = MIN_PASSWORD_CHECK; x < MAX_PASSWORD_LENGTH; x++) {
		sha256_init(&sha_ctx);
		sha256_update(&sha_ctx, entry->salt, SALT_LENGTH);
		sha256_update(&sha_ctx, input, x);
		sha256_final(&sha_ctx, digest);

		found = 1;
		for (int i = 0; i < HASH_BYTES_LENGTH; i++) {
			if (digest[i] != entry->hash_bytes[i]) {
				found = 0;
				break;
			}
		}

		if (found) {
			for (int i = 0; i < x; i++) {
				entry->solution[i] = input[i];
			}
			blockContainsSolution[entry_pos] = 1;
		}
	}
}

void reorganize_not_solved_entries(hash_entry *entries,
                                   unsigned char *contains_solution,
                                   int entries_total, int *current_total,
                                   unsigned char *d_blockContainsSolution,
                                   hash_entry *d_hash_entry) {
	// This will place solved entries into the end of the list
	// It will change the CPU (host) variables in the launchGPUHandlerThread function
	for (int i = 0; i < *current_total; i++) {
		if (contains_solution[i]) {
			// SWAP
			int final_index = (*current_total) - 1;
			hash_entry entry_copy = entries[i];
			unsigned char contains_solution_copy = contains_solution[i];

			entries[i] = entries[final_index];
			contains_solution[i] = contains_solution[final_index];

			entries[final_index] = entry_copy;
			contains_solution[final_index] = contains_solution_copy;

			*current_total = *current_total - 1;
		}
	}

	// Still needs to update the GPU (device) variables to reflect those
	// changes For the blockContainsSolution we can only set all to zero.
	// Because the rest will be not accessed. It sets already entries_count
	// because it easier, and it will happen not frequently
	cudaMemset(d_blockContainsSolution, 0, sizeof(unsigned char) * (entries_total));
	cudaMemcpy(d_hash_entry, entries, sizeof(hash_entry) * (entries_total), cudaMemcpyHostToDevice);
}

void *launchGPUHandlerThread(void *vargp) {
  handler_input *hi = (handler_input *) vargp;
  cudaSetDevice(hi->device);

	// Pre SHA-256
  cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice);

	int current_total = hi->entries_count;
	
  unsigned char *blockContainsSolution = (unsigned char*) malloc(sizeof(unsigned char) * current_total);
  unsigned char *d_blockContainsSolution;
  cudaMalloc(&d_blockContainsSolution, sizeof(unsigned char) * current_total);
	cudaMemset(d_blockContainsSolution, 0, sizeof(unsigned char) * current_total);

	srand(timems() * timems());

	hash_entry *d_hash_entry;
	cudaMalloc(&d_hash_entry, sizeof(hash_entry) * current_total);
	cudaMemcpy(d_hash_entry, hi->entries, sizeof(hash_entry) * current_total, cudaMemcpyHostToDevice);

#ifdef RELEASE
	int j = 1;
#endif

  while (1) {
		for (int i = 0; i < RUNS_PER_ITERATION; i++) {
			sha256_cuda<<<THREADS, BLOCKS>>>(d_hash_entry, current_total, d_blockContainsSolution, rand());
		}
    cudaDeviceSynchronize();
    hi->hashesProcessed += THREADS * BLOCKS * THREAD_EXECUTION_ITERATIONS * RUNS_PER_ITERATION;
		
    cudaMemcpy(blockContainsSolution, d_blockContainsSolution, sizeof(unsigned char) * current_total, cudaMemcpyDeviceToHost);

		int found = 0;
		for (int i = 0; i < current_total; i++) {
			if (blockContainsSolution[i]) {
				srand(timems() * timems());
				if (!found) {
					cudaMemcpy(hi->entries, d_hash_entry, sizeof(hash_entry) * hi->entries_count, cudaMemcpyDeviceToHost);
				}

				printf("\n\nSolution at %d:\n", i);
				print_hash_entry(hi->entries[i]);
				printf("\n");
				found = 1;

				if (current_total <= 0) {
					cudaDeviceReset();
					exit(0);
				}
			}
		}
#ifdef RELEASE
			j++;
			if ((j % 340) == 0) {
				printf("Breaking\n");
				usleep(6000000);
				j = 1;
			}
#endif
		if (found) {
			reorganize_not_solved_entries(
					hi->entries, blockContainsSolution,
					hi->entries_count, &current_total,
					d_blockContainsSolution, d_hash_entry);
		
#ifdef DEBUG
			for (int i = 0; i < hi->entries_count; i++) {
				print_hash_entry(hi->entries[i]);
			}
#endif
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

	printf("\nStarting to break hashes\n");

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
		usleep(PRINT_STATUS_DELAY);
		// usleep(1000);
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

