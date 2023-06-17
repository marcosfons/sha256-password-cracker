#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>
#include <stddef.h>
#include <sys/time.h>
#include <pthread.h>
#include <locale.h>
#include <math.h>

#include "sha256.cuh"
#include "wordlist.cuh"
#include "hash_entry.cuh"
#include "cuda_devices.cuh"


// Current stage
#define DEBUG
// #define TEST
// #define RELEASE

#define TEST_TYPE SEQUENTIAL_WORDLIST
// #define TEST_TYPE SEQUENTIALLY
// #define TEST_TYPE RANDOMLY

#define GPUS 1

#ifdef DEBUG
#define BLOCKS_PER_ENTRY 200
#define THREADS 1024
#define RUNS_PER_ITERATION 1
#define LOOPS_INSIDE_THREAD 66
#define PRINT_STATUS_DELAY 10000
#endif

#ifdef TEST
#define BLOCKS_PER_ENTRY 12
#define THREADS 1024
#define RUNS_PER_ITERATION 1
#define LOOPS_INSIDE_THREAD 128
#define PRINT_STATUS_DELAY 10000
#endif

#ifdef RELEASE
#define THREADS 2048
#define RUNS_PER_ITERATION 128
#define PRINT_STATUS_DELAY 1000000
#endif

#define THREAD_EXECUTION_ITERATIONS ((MAX_PASSWORD_LENGTH - MIN_PASSWORD_CHECK))

const char charset[] = {
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B',
    'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4',
    '5', '6', '7', '8', '9', '0', '%', '*', '$', '@'};

__constant__ char d_charset[sizeof(charset)];

__constant__ const int CHARSET_LENGTH = sizeof(charset) / sizeof(char);

typedef struct handler_input {
	int device;
	unsigned long long hashesProcessed;
	unsigned long long start;
	hash_entry* entries;
	int entries_count;
	unsigned char finished;
	sequential_wordlist *sequential_wordlist;
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

__device__ unsigned char hash_cmp_equal(const unsigned char hash1[HASH_BYTES_LENGTH], const unsigned char hash2[HASH_BYTES_LENGTH]) {
	#pragma unroll
	for (unsigned short i = 0; i < HASH_BYTES_LENGTH; i++) {
		if (hash1[i] != hash2[i]) {
			return 0;
		}
	}
	return 1;
}

__host__ int h_get_input_from_number(unsigned long long current, unsigned char input[8]) {
	for (unsigned short i = 0; i < 8; i++) {
		input[i] = charset[current % CHARSET_LENGTH];
		current /= CHARSET_LENGTH;
		if (current <= 0) {
			input[i+1] = '\0';
			return i + 1;
		}
	}
	return 0;
}

__device__ void get_input_from_number(unsigned long long current, unsigned char input[8]) {
	#pragma unroll
	for (unsigned short i = 0; i < 8; i++) {
		input[i] = d_charset[current % CHARSET_LENGTH];
		current /= CHARSET_LENGTH;
		if (current <= 0) {
			return;
		}
	}
	return;
}

__device__ void sha256_update_from_number(SHA256_CTX *ctx, unsigned long long current) {
	while (current > 0) {
		ctx->data[ctx->datalen] = d_charset[current % CHARSET_LENGTH];
		ctx->datalen++;

		current /= CHARSET_LENGTH;
	}
}

__global__ void sha256_cuda_all_posibilities(
    hash_entry *__restrict__ entries, int entries_count,
    unsigned long long *__restrict__ solutions,
    unsigned long long start) {

	hash_entry* entry = entries + (blockIdx.x / BLOCKS_PER_ENTRY);

	SHA256_CTX sha_ctx;
	u_hash_bytes digest;

	unsigned long long current = start + ( (blockIdx.x % BLOCKS_PER_ENTRY) * (blockDim.x * LOOPS_INSIDE_THREAD) ) + (threadIdx.x * LOOPS_INSIDE_THREAD);

	#pragma unroll
	for (unsigned short j = 0; j < LOOPS_INSIDE_THREAD; j++) {
	// for (unsigned long long j = current; j < current + LOOPS_INSIDE_THREAD; j++) {
		sha256_init(&sha_ctx);
		// sha256_update(&sha_ctx, entry->salt, SALT_LENGTH);
		#pragma unroll
		for (unsigned short i = 0; i < SALT_LENGTH; i++) {
			sha_ctx.data[i] = entry->salt[i];
		}
		sha_ctx.datalen = SALT_LENGTH;

		sha256_update_from_number(&sha_ctx, current);
		sha256_final(&sha_ctx, digest.hash_bytes);

		if (hash_cmp_equal(digest.hash_bytes, entry->hash_bytes.hash_bytes)) {
			solutions[(blockIdx.x / BLOCKS_PER_ENTRY)] = current;
			break;
		}
		current += 1;
	}
}

__global__ void sha256_sequential_wordlist(hash_entry *__restrict__ entries,
                                           int entries_count,
                                           unsigned long long start,
                                           unsigned char *sequential_wordlist) {
	hash_entry* entry = entries + blockIdx.x;

	SHA256_CTX sha_ctx;
	u_hash_bytes digest;

	#pragma unroll
	for (unsigned short j = MIN_PASSWORD_CHECK; j < MAX_PASSWORD_LENGTH; j++) {
		sha256_init(&sha_ctx);
		#pragma unroll
		for (unsigned short i = 0; i < SALT_LENGTH; i++) {
			sha_ctx.data[i] = entry->salt[i];
		}
		sha_ctx.datalen = SALT_LENGTH;

		sha256_update(&sha_ctx, sequential_wordlist + start + (blockIdx.y * THREADS) + threadIdx.x, j);
		sha256_final(&sha_ctx, digest.hash_bytes);

		if (hash_cmp_equal(digest.hash_bytes, entry->hash_bytes.hash_bytes)) {
			for (int i = 0; i < j; i++) {
				entry->solution[i] = sequential_wordlist[start + (blockIdx.y * THREADS) + threadIdx.x + i];
			}
			entry->solution[j] = '\0';
			break;
		}
	}
}

// __global__ void sha256_cuda(hash_entry *entries, int entries_count,
//                             unsigned long long *blockContainsSolution,
//                             unsigned long baseSeed) {
//   int id = blockIdx.x * blockDim.x + threadIdx.x;
// 	int entry_pos = blockIdx.x % entries_count;
//   unsigned long seed = deviceRandomGen(baseSeed + id);
//
// 	hash_entry* entry = entries + entry_pos;
//
// 	SHA256_CTX sha_ctx;
// 	u_hash_bytes digest;
// 	BYTE input[MAX_PASSWORD_LENGTH];
//
// 	#pragma unroll
// 	for (int i = 0; i < MAX_PASSWORD_LENGTH; i++) {
// 		seed = deviceRandomGen(seed);
// 		input[i] = charset[seed % CHARSET_LENGTH];
// 	}
// 	
// 	for (int input_length = MIN_PASSWORD_CHECK; input_length < MAX_PASSWORD_LENGTH; input_length++) {
// 		sha256_init(&sha_ctx);
// 		sha256_update(&sha_ctx, entry->salt, SALT_LENGTH);
// 		sha256_update(&sha_ctx, input, input_length);
// 		sha256_final(&sha_ctx, digest.hash_bytes);
//
// 		if (hash_cmp_equal(digest.hash_bytes, entry->hash_bytes.hash_bytes)) {
// 			for (int i = 0; i < input_length; i++) {
// 				entry->solution[i] = input[i];
// 			}
// 			blockContainsSolution[entry_pos] = 1;
// 		}
// 	}
// }
//

void reorganize_not_solved_entries(hash_entry *entries, int entries_total,
                                   int *current_total,
                                   hash_entry *d_hash_entry) {
	// This will place solved entries into the end of the list
	// It will change the CPU (host) variables in the launch_gpu_handler_thread function
	for (int i = 0; i < *current_total; i++) {
		if (contains_solution_hash_entry(entries + i)) {
			// printf("FOUND: %s\n", (char*) solutions + (i * MAX_PASSWORD_LENGTH));
			// SWAP
			int final_index = (*current_total) - 1;
			hash_entry entry_copy = entries[i];
			entries[i] = entries[final_index];
			entries[final_index] = entry_copy;

			*current_total = *current_total - 1;
			i -= 1;
		}
	}

	cudaMemcpy(d_hash_entry, entries, sizeof(hash_entry) * (entries_total), cudaMemcpyHostToDevice);
}

bool check_if_solution_was_found(hash_entry* entries, int current_total) {
	for (int i = 0; i < current_total; i++) {
		if (contains_solution_hash_entry(entries + i)) {
			return true;
		}
	}

	return false;
}

void process_after_solution_was_found(hash_entry *entries, int entries_total,
                                      int *current_total,
                                      hash_entry *d_hash_entry) {
	reorganize_not_solved_entries(entries, entries_total, current_total, d_hash_entry);

	printf("\n");
	for (int i = 0; i < entries_total; i++) {
		print_hash_entry(entries[i]);
		printf("\n");
	}
}

void *launch_gpu_handler_thread(void *vargp) {
	handler_input *hi = (handler_input *) vargp;
	cudaSetDevice(hi->device);

	cudaMemcpyToSymbol(d_charset, charset, sizeof(charset), 0, cudaMemcpyHostToDevice);

	// Pre SHA-256
	cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice);

	int current_total = hi->entries_count;
	
	hash_entry *d_hash_entry;
	cudaMalloc(&d_hash_entry, sizeof(hash_entry) * current_total);
	cudaMemcpy(d_hash_entry, hi->entries, sizeof(hash_entry) * current_total, cudaMemcpyHostToDevice);


	#if TEST_TYPE == SEQUENTIAL_WORDLIST
	unsigned char* d_wordlist;
	cudaMalloc(&d_wordlist, sizeof(unsigned char) * hi->sequential_wordlist->character_count);
	cudaMemcpy(d_wordlist, hi->sequential_wordlist->words, sizeof(unsigned char) * hi->sequential_wordlist->character_count, cudaMemcpyHostToDevice);
	
	hi->start = 0;


	while(1) {
		dim3 num_blocks(current_total, BLOCKS_PER_ENTRY, 1);
		dim3 num_threads(THREADS, 1, 1);

		sha256_sequential_wordlist<<<num_blocks, num_threads>>>(
				d_hash_entry, current_total, hi->start, d_wordlist
		);
		hi->start += BLOCKS_PER_ENTRY * THREADS;
		cudaDeviceSynchronize();

		cudaMemcpy(hi->entries, d_hash_entry, sizeof(hash_entry) * current_total, cudaMemcpyDeviceToHost);

		if (check_if_solution_was_found(hi->entries, current_total)) {
			printf("\nSTART: %llu %d\n", hi->start, current_total);
			process_after_solution_was_found(hi->entries, hi->entries_count, &current_total, d_hash_entry);
		}

		hi->hashesProcessed += current_total * (MAX_PASSWORD_LENGTH - MIN_PASSWORD_CHECK) * THREADS * BLOCKS_PER_ENTRY;

		if (hi->start > hi->sequential_wordlist->character_count) {
			break;
		}
	}
	hi->finished = 1;

	#elif TEST_TYPE == SEQUENTIALLY
	// hi->start = 305000;
	hi->start = 0;

	while(1) {
		for (int i = 0; i < RUNS_PER_ITERATION; i++) {
			sha256_cuda_all_posibilities<<<current_total * BLOCKS_PER_ENTRY, THREADS>>>(
					d_hash_entry, current_total, d_solutions,
					hi->start + (THREADS * i * LOOPS_INSIDE_THREAD * BLOCKS_PER_ENTRY)
			);
		}
		cudaDeviceSynchronize();

		if (hi->start > 130000000) {
		// if (hi->start > 20000000) {
			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
			printf("\n");
					printf("CUDA Error: %s\n", cudaGetErrorString(error));
			}
		// if (hi->start > 290000000) {
			break;
		}

		cudaMemcpy(solutions, d_solutions, sizeof(unsigned long long) * current_total, cudaMemcpyDeviceToHost);

		if (check_if_solution_was_found(solutions, current_total)) {
			printf("\nSTART: %llu\n", hi->start);
			process_after_solution_was_found(
					hi->entries, solutions,
					hi->entries_count, &current_total,
					d_solutions, d_hash_entry);
		}

		hi->start += RUNS_PER_ITERATION * (THREADS * LOOPS_INSIDE_THREAD) * BLOCKS_PER_ENTRY;
		hi->hashesProcessed += RUNS_PER_ITERATION * (THREADS * LOOPS_INSIDE_THREAD) * (current_total * BLOCKS_PER_ENTRY);
	}
	hi->finished = 1;

	#elif TEST_TYPE == RANDOMLY

	srand(timems() * timems());
	while (1) {
		for (int i = 0; i < RUNS_PER_ITERATION; i++) {
			sha256_cuda<<<THREADS, BLOCKS>>>(d_hash_entry, current_total, d_entries_has_solution, rand());
		}
		cudaDeviceSynchronize();
		hi->hashesProcessed += THREADS * BLOCKS * THREAD_EXECUTION_ITERATIONS * RUNS_PER_ITERATION;
			
		cudaMemcpy(entries_has_solution, d_entries_has_solution, sizeof(unsigned char) * current_total, cudaMemcpyDeviceToHost);

		if (check_if_solution_was_found(&entries_has_solution, d_entries_has_solution, current_total)) {
			process_after_solution_was_found(
					hi->entries, entries_has_solution,
					hi->entries_count, &current_total,
					d_entries_has_solution, d_hash_entry);
		}
	}

	#endif

  cudaDeviceReset();
  return NULL;
}


int main() {
	setlocale(LC_NUMERIC, "");

	show_gpu_devices_info();

	sequential_wordlist wordlist;
	read_sequential_wordlist_from_file("wordlists/rockyou.txt", &wordlist);

	printf("\n");
	printf("WORDS: %lu\n SIZE (bytes): %lu\n", wordlist.words_count, wordlist.character_count * sizeof(char));
	// printf("%s\n", wordlist.words);

	int entries_count = 0;
	hash_entry* entries = (hash_entry*) malloc(1);
	printf("Loading hashes from the file\n\n");

	read_entries_from_file("data/hashes_and_salts.txt", &entries, &entries_count);
	// read_entries_from_file("data/correct.txt", &entries, &entries_count);
	if (entries_count == 0) {
		printf("No entries found, exiting\n");
		exit(0);
	}

	for (int i = 0; i < entries_count; i++) {
		print_hash_entry(entries[i]);
		printf("\n");
	}

	printf("\nStarting to break hashes\n");

	// 6%Fg
	// abcDef
	// passworD
	// p@ssw0rd
	// AbCdEfGh
	// 00000000
	// 12081786
	// 12345678
	// 1@2@3@4@
	// M3t@llic@
	// HarryP0tter

	unsigned long long **processedPtrs = (unsigned long long **) malloc(sizeof(unsigned long long *) * GPUS);
	unsigned long long **singleProcessedPtrs = (unsigned long long **) malloc(sizeof(unsigned long long *) * GPUS);
	unsigned char **finished_ptrs = (unsigned char**) malloc(sizeof(unsigned char*) * GPUS);

	pthread_t *tids = (pthread_t *) malloc(sizeof(pthread_t) * GPUS);
	unsigned long long start = timems();
	for (int i = 0; i < GPUS; i++) {
		handler_input *hi = (handler_input*) malloc(sizeof(handler_input));
		hi->device = i;
		hi->hashesProcessed = 0;
		hi->entries = entries;
		hi->entries_count = entries_count;
		hi->finished = 0;
		hi->sequential_wordlist = &wordlist;
		processedPtrs[i] = &hi->hashesProcessed;
		singleProcessedPtrs[i] = &hi->start;
		finished_ptrs[i] = &hi->finished;
		pthread_create(tids + i, NULL, launch_gpu_handler_thread, hi);
	}

	while (1) {
		usleep(PRINT_STATUS_DELAY);
		unsigned long totalProcessed = 0;
		unsigned long singleProcesseds = 0;
		unsigned char finished = 1;
		for (int i = 0; i < GPUS; i++) {
			singleProcesseds += *(singleProcessedPtrs[i]);
			totalProcessed += *(processedPtrs[i]);
			finished = finished && *(finished_ptrs[i]);
		}

		long long elapsed = timems() - start;
		printf("Total Hashes (%'lu) Hashes per Salt (%'lu) Seconds (%'f) Hashes/sec (%'lu)\r",
					totalProcessed, 
				 singleProcesseds,
				 ((float) elapsed) / 1000.0,
					(unsigned long) ((double) totalProcessed / (double) elapsed) * 1000);

		if (finished) {
			break;
		}
	}

	long long elapsed = timems() - start;

	for (int i = 0; i < GPUS; i++) {
		pthread_join(tids[i], NULL);
	}

	unsigned long totalProcessed = 0;
	for (int i = 0; i < GPUS; i++) {
		totalProcessed += *(processedPtrs[i]);
	}

	printf("\nHashes processed: %'lu\n", totalProcessed);
	printf("Time: %llu\n", elapsed);
	printf("Hashes/sec: %'lu\n", (unsigned long) ((double) totalProcessed / (double) elapsed) * 1000);

	free(entries);

	return 0;
}

