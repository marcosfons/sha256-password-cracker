#include <sys/time.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <locale.h>
#include <stdio.h>
#include <cuda.h>

#include "sha256.cuh"
#include "wordlist.h"
#include "hash_entry.h"
#include "cuda_devices.h"


// #define TEST_TYPE SEQUENTIAL_WORDLIST
#define TEST_TYPE SEQUENTIALLY
// #define TEST_TYPE RANDOMLY

#define BLOCKS_PER_ENTRY 200
#define THREADS 1024
#define RUNS_PER_ITERATION 1
#define LOOPS_INSIDE_THREAD 66
#define PRINT_STATUS_DELAY 10000

#define THREAD_EXECUTION_ITERATIONS ((MAX_PASSWORD_LENGTH - MIN_PASSWORD_CHECK))

#define MAX_SEQUENTIAL_WORDLIST_CHARS 600000000

// const char CHARSET[] = {
//     'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
//     'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B',
//     'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
//     'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4',
//     '5', '6', '7', '8', '9', '0', '%', '*', '$', '@'};

const char CHARSET[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'};

// const int CHARSET_LENGTH = sizeof(CHARSET) / sizeof(char);

typedef struct HandlerInput {
	unsigned long long hashesProcessed;
	unsigned long long start;
	HashEntries entries;
	bool finished;
	SequentialWordlist *sequentialWordlist;
} HandlerInput;

long long timems() {
	struct timeval end;
	gettimeofday(&end, NULL);
	return (long long) end.tv_sec * 1000 + (long long) end.tv_usec / 1000;
}

__device__ bool hashCompare(const unsigned char hash1[HASH_BYTES_LENGTH],
                            const unsigned char hash2[HASH_BYTES_LENGTH]) {
	#pragma unroll
	for (unsigned short i = 0; i < HASH_BYTES_LENGTH; i++) {
		if (hash1[i] != hash2[i]) {
			return 0;
		}
	}
	return 1;
}

__global__ void sha256SequentialWordlist(HashEntry *__restrict__ entries,
                                         int entriesCount,
                                         unsigned long long start,
                                         unsigned char *sequentialWordlist) {
	HashEntry* entry = entries + blockIdx.x;

	SHA256_CTX shaCtx;
	unsigned char digest[HASH_BYTES_LENGTH];

	#pragma unroll
	for (unsigned short j = MIN_PASSWORD_CHECK; j < MAX_PASSWORD_LENGTH; j++) {
		sha256_init(&shaCtx);
		#pragma unroll
		for (unsigned short i = 0; i < SALT_LENGTH; i++) {
			shaCtx.data[i] = entry->salt[i];
		}
		shaCtx.datalen = SALT_LENGTH;

		sha256_update(&shaCtx, sequentialWordlist + start + (blockIdx.y * THREADS) + threadIdx.x, j);
		sha256_final(&shaCtx, digest);

		if (hashCompare(digest, entry->hashBytes)) {
			for (int i = 0; i < j; i++) {
				entry->solution[i] = sequentialWordlist[start + (blockIdx.y * THREADS) + threadIdx.x + i];
			}
			entry->solution[j] = '\0';
			break;
		}
	}
}

__global__ void sha256Wordlist(HashEntry *__restrict__ entries,
                               int entriesCount, unsigned long long start,
                               unsigned char *wordlist,
                               unsigned short wordLength) {
	HashEntry* entry = entries + blockIdx.x;

	SHA256_CTX shaCtx;
	unsigned char digest[HASH_BYTES_LENGTH];

	sha256_init(&shaCtx);
	#pragma unroll
	for (unsigned short i = 0; i < SALT_LENGTH; i++) {
		shaCtx.data[i] = entry->salt[i];
	}
	shaCtx.datalen = SALT_LENGTH;

	sha256_update(&shaCtx, wordlist + start + (blockIdx.y * THREADS) + (threadIdx.x * wordLength), wordLength);
	sha256_final(&shaCtx, digest);

	if (hashCompare(digest, entry->hashBytes)) {
		for (int i = 0; i < wordLength; i++) {
			entry->solution[i] = wordlist[start + (blockIdx.y * THREADS) + (threadIdx.x * wordLength) + i];
		}
		entry->solution[wordLength] = '\0';
	}
}

void processAfterSolutionWasFound(HashEntries *entries,
                                  HashEntry *d_hashEntry) {
	reorganizeNotSolvedEntries(entries);
	cudaMemcpy(d_hashEntry, entries->entries, sizeof(HashEntry) * (entries->entriesCount), cudaMemcpyHostToDevice);

	printf("\n");
	printHashEntries(entries);
}

void *launchGPUHandlerThread(void *vargp) {
	HandlerInput *hi = (HandlerInput *) vargp;

	// Pre SHA-256
	cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice);
	
	HashEntry *d_hashEntry;
	cudaMalloc(&d_hashEntry, sizeof(HashEntry) * hi->entries.entriesCount);
	cudaMemcpy(d_hashEntry, hi->entries.entries, sizeof(HashEntry) * hi->entries.entriesCount, cudaMemcpyHostToDevice);

	unsigned char* d_wordlist;
	cudaMalloc(&d_wordlist, sizeof(unsigned char) * hi->sequentialWordlist->characterCount);
	cudaMemcpy(d_wordlist, hi->sequentialWordlist->words, sizeof(unsigned char) * hi->sequentialWordlist->characterCount, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	hi->start = 0;

	while(1) {
		dim3 numBlocks(hi->entries.currentTotal, BLOCKS_PER_ENTRY, 1);
		dim3 numThreads(THREADS, 1, 1);

		cudaEventRecord(start);
		sha256SequentialWordlist<<<numBlocks, numThreads>>>(
				d_hashEntry, hi->entries.currentTotal, hi->start, d_wordlist
		);
		cudaEventRecord(stop);
		hi->start += BLOCKS_PER_ENTRY * THREADS;
		// cudaDeviceSynchronize();
		cudaEventSynchronize(stop);

		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);

		int processed = hi->entries.currentTotal * (MAX_PASSWORD_LENGTH - MIN_PASSWORD_CHECK) * THREADS * BLOCKS_PER_ENTRY;
		// printf("Milliseconds elapsed: %f  Milliseconds per execution %f  Processed: %d\r", milliseconds, milliseconds / (processed), processed);

		cudaMemcpy(hi->entries.entries, d_hashEntry, sizeof(HashEntry) * hi->entries.currentTotal, cudaMemcpyDeviceToHost);

		if (containsNewSolution(&hi->entries)) {
			processAfterSolutionWasFound(&hi->entries, d_hashEntry);
		}

		hi->hashesProcessed += hi->entries.currentTotal * (MAX_PASSWORD_LENGTH - MIN_PASSWORD_CHECK) * THREADS * BLOCKS_PER_ENTRY;

		if (hi->start > hi->sequentialWordlist->characterCount) {
			break;
		}
	}
	hi->finished = true;


  cudaDeviceReset();
  return NULL;
}


int main() {
	setlocale(LC_NUMERIC, "");

	SequentialWordlist wordlist;
	createSequentialWordlist(&wordlist, 9, CHARSET, sizeof(CHARSET), MAX_SEQUENTIAL_WORDLIST_CHARS);

	// for (size_t i = 0; i < wordlist.words_count; i++) {
	// 	for (int j = 0; j < wordlist.word_length; j++) {
	// 		printf("%c", wordlist.words[(i * wordlist.word_length) + j]);
	// 	}
	// 	printf("\n");
	// }

	showGPUDevicesInfo();

	// read_sequential_wordlist_from_file("wordlists/rockyou_shuf.txt", &wordlist);
	// readSequentialWordlistFromFile("wordlists/n_crackstation-human-only.txt", &wordlist);
	printf("\nWORDS: %llu\n SIZE (bytes): %llu\n", wordlist.wordsCount, wordlist.characterCount * sizeof(char));


	printf("Loading hashes from the file\n\n");
	HashEntries entries;
	readHashEntriesFromFile("data/hashes_and_salts.txt", &entries);
	if (entries.entriesCount == 0) {
		printf("No entries found, exiting\n");
		exit(0);
	}

	printHashEntries(&entries);

	printf("\nStarting to break hashes\n");

	pthread_t threadId;
	unsigned long long start = timems();

	HandlerInput input;
	input.hashesProcessed = 0;
	input.entries = entries;
	input.finished = false;
	input.sequentialWordlist = &wordlist;

	pthread_create(&threadId, NULL, launchGPUHandlerThread, &input);

	while (1) {
		usleep(PRINT_STATUS_DELAY);
		unsigned long long singleProcesseds = input.start;
		unsigned long long totalProcessed = input.hashesProcessed;

		long long elapsed = timems() - start;
		printf("Total Hashes (%'llu) Hashes per Salt (%'llu) Seconds (%'f) Hashes/sec (%'llu)\r",
					totalProcessed, singleProcesseds, ((double) elapsed) / 1000.0,
					(unsigned long long) ((double) totalProcessed / (double) elapsed) * 1000
		);

		if (input.finished) {
			break;
		}
	}

	long long elapsed = timems() - start;

	pthread_join(threadId, NULL);

	printf("\nHashes processed: %'llu\n", input.hashesProcessed);
	printf("Time: %llu\n", elapsed);
	printf("Hashes/sec: %'lu\n", (unsigned long) ((double) input.hashesProcessed / (double) elapsed) * 1000);

	return 0;
}

