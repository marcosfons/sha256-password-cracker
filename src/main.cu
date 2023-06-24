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

// #define TEST_TYPE SEQUENTIAL_WORDLIST
#define TEST_TYPE SEQUENTIALLY
// #define TEST_TYPE RANDOMLY

#define GPUS 1

#ifdef DEBUG
#define BLOCKS_PER_ENTRY 100
#define THREADS 1024
#define RUNS_PER_ITERATION 1
#define LOOPS_INSIDE_THREAD 66
#define PRINT_STATUS_DELAY 10000
#endif

#define THREAD_EXECUTION_ITERATIONS ((MAX_PASSWORD_LENGTH - MIN_PASSWORD_CHECK))

#define MAX_SEQUENTIAL_WORDLIST_CHARS 600000000

const char charset[] = {
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B',
    'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4',
    '5', '6', '7', '8', '9', '0', '%', '*', '$', '@'};

const int CHARSET_LENGTH = sizeof(charset) / sizeof(char);

typedef struct handler_input {
	int device;
	unsigned long long hashesProcessed;
	unsigned long long start;
	hash_entries entries;
	unsigned char finished;
	sequential_wordlist *sequential_wordlist;
} handler_input;

long long timems() {
	struct timeval end;
	gettimeofday(&end, NULL);
	return (long long) end.tv_sec * 1000 + (long long) end.tv_usec / 1000;
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

__global__ void sha256_wordlist(hash_entry *__restrict__ entries,
                                int entries_count, unsigned long long start,
                                unsigned char *wordlist, unsigned short word_length) {
	hash_entry* entry = entries + blockIdx.x;

	SHA256_CTX sha_ctx;
	u_hash_bytes digest;

	sha256_init(&sha_ctx);
	#pragma unroll
	for (unsigned short i = 0; i < SALT_LENGTH; i++) {
		sha_ctx.data[i] = entry->salt[i];
	}
	sha_ctx.datalen = SALT_LENGTH;

	sha256_update(&sha_ctx, wordlist + start + (blockIdx.y * THREADS) + (threadIdx.x * word_length), word_length);
	sha256_final(&sha_ctx, digest.hash_bytes);

	if (hash_cmp_equal(digest.hash_bytes, entry->hash_bytes.hash_bytes)) {
		for (int i = 0; i < word_length; i++) {
			entry->solution[i] = wordlist[start + (blockIdx.y * THREADS) + (threadIdx.x * word_length) + i];
		}
		entry->solution[word_length] = '\0';
	}
}

void process_after_solution_was_found(hash_entries *entries,
                                      hash_entry *d_hash_entry) {
	reorganize_not_solved_entries(entries);
	cudaMemcpy(d_hash_entry, entries->entries, sizeof(hash_entry) * (entries->entries_count), cudaMemcpyHostToDevice);

	printf("\n");
	print_hash_entries(entries);
}

void *launch_gpu_handler_thread(void *vargp) {
	handler_input *hi = (handler_input *) vargp;
	cudaSetDevice(hi->device);

	// Pre SHA-256
	cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice);
	
	hash_entry *d_hash_entry;
	cudaMalloc(&d_hash_entry, sizeof(hash_entry) * hi->entries.entries_count);
	cudaMemcpy(d_hash_entry, hi->entries.entries, sizeof(hash_entry) * hi->entries.entries_count, cudaMemcpyHostToDevice);


	#if TEST_TYPE == SEQUENTIAL_WORDLIST
	// unsigned char* d_wordlist;
	// cudaMalloc(&d_wordlist, sizeof(unsigned char) * hi->sequential_wordlist->character_count);
	// cudaMemcpy(d_wordlist, hi->sequential_wordlist->words, sizeof(unsigned char) * hi->sequential_wordlist->character_count, cudaMemcpyHostToDevice);
	// 
	// hi->start = 0;
	//
	// while(1) {
	// 	dim3 num_blocks(hi->entries.current_total, BLOCKS_PER_ENTRY, 1);
	// 	dim3 num_threads(THREADS, 1, 1);
	//
	// 	sha256_sequential_wordlist<<<num_blocks, num_threads>>>(
	// 			d_hash_entry, hi->entries.current_total, hi->start, d_wordlist
	// 	);
	// 	hi->start += BLOCKS_PER_ENTRY * THREADS;
	// 	cudaDeviceSynchronize();
	//
	// 	cudaMemcpy(hi->entries.entries, d_hash_entry, sizeof(hash_entry) * hi->entries.current_total, cudaMemcpyDeviceToHost);
	//
	// 	if (contains_new_solution(&hi->entries)) {
	// 		process_after_solution_was_found(&hi->entries, d_hash_entry);
	// 	}
	//
	// 	hi->hashesProcessed += hi->entries.current_total * (MAX_PASSWORD_LENGTH - MIN_PASSWORD_CHECK) * THREADS * BLOCKS_PER_ENTRY;
	//
	// 	if (hi->start > hi->sequential_wordlist->character_count) {
	// 		break;
	// 	}
	// }
	// hi->finished = 1;


	unsigned char* d_wordlist;
	unsigned char* d_wordlist2;
	cudaMalloc(&d_wordlist, sizeof(unsigned char) * hi->sequential_wordlist->character_count);
	cudaMalloc(&d_wordlist2, sizeof(unsigned char) * hi->sequential_wordlist->character_count);

	cudaMemcpy(d_wordlist, hi->sequential_wordlist->words, sizeof(unsigned char) * hi->sequential_wordlist->character_count, cudaMemcpyHostToDevice);
	
	hi->start = 0;

	while(1) {
		dim3 num_blocks(hi->entries.current_total, BLOCKS_PER_ENTRY, 1);
		dim3 num_threads(THREADS, 1, 1);

		sha256_wordlist<<<num_blocks, num_threads>>>(
				d_hash_entry, hi->entries.current_total, hi->start, d_wordlist, hi->sequential_wordlist->word_length
		);
		hi->start += BLOCKS_PER_ENTRY * THREADS * hi->sequential_wordlist->word_length;
		cudaDeviceSynchronize();

		cudaMemcpy(hi->entries.entries, d_hash_entry, sizeof(hash_entry) * hi->entries.current_total, cudaMemcpyDeviceToHost);

		if (contains_new_solution(&hi->entries)) {
			process_after_solution_was_found(&hi->entries, d_hash_entry);
		}

		hi->hashesProcessed += hi->entries.current_total * THREADS * BLOCKS_PER_ENTRY;

		if (hi->start > hi->sequential_wordlist->character_count) {
			bool generated_more_words = generate_sequential_wordlist(
					hi->sequential_wordlist,
					MAX_SEQUENTIAL_WORDLIST_CHARS, charset,
					CHARSET_LENGTH
			);

			if (generated_more_words) {
				cudaMemcpy(d_wordlist, hi->sequential_wordlist->words, sizeof(unsigned char) * hi->sequential_wordlist->character_count, cudaMemcpyHostToDevice);
				hi->start = 0;
			} else {
				break;
			}
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

	sequential_wordlist wordlist;
	create_sequential_wordlist(&wordlist, 8, charset, sizeof(charset), MAX_SEQUENTIAL_WORDLIST_CHARS);

	// for (size_t i = 0; i < wordlist.words_count; i++) {
	// 	for (int j = 0; j < wordlist.word_length; j++) {
	// 		printf("%c", wordlist.words[(i * wordlist.word_length) + j]);
	// 	}
	// 	printf("\n");
	// }


	show_gpu_devices_info();

	// sequential_wordlist wordlist;
	// // read_sequential_wordlist_from_file("wordlists/rockyou_shuf.txt", &wordlist);
	// read_sequential_wordlist_from_file("wordlists/n_crackstation-human-only.txt", &wordlist);
	//
	// printf("\n");
	// printf("WORDS: %lu\n SIZE (bytes): %lu\n", wordlist.words_count, wordlist.character_count * sizeof(char));


	printf("Loading hashes from the file\n\n");
	hash_entries entries;
	read_hash_entries_from_file("data/hashes_and_salts.txt", &entries);
	if (entries.entries_count == 0) {
		printf("No entries found, exiting\n");
		exit(0);
	}

	print_hash_entries(&entries);

	printf("\nStarting to break hashes\n");

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

	return 0;
}

