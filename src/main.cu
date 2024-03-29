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


#define BLOCKS_PER_ENTRY 300
#define THREADS 1024
#define PRINT_STATUS_DELAY 10000

const char *CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890%*$@";

typedef struct HandlerInput {
	unsigned long long hashesProcessed;
	unsigned long long start;
	HashEntries entries;
	bool finished;
	SequentialWordlist *sequentialWordlist;
} HandlerInput;

__forceinline__ __device__ bool hashCompare(const u_HashBytes *hash1,
                                            const u_HashBytes *hash2) {
	#pragma unroll
	for (unsigned short i = 0; i < HASH_BYTES_LENGTH; i++) {
		if (hash1->bytes[i] != hash2->bytes[i]) {
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
	u_HashBytes digest;
	unsigned long long step = start + (blockIdx.y * THREADS) + threadIdx.x;

	#pragma unroll
	for (unsigned short j = MIN_PASSWORD_CHECK; j < MAX_PASSWORD_LENGTH + 1; j++) {
		sha256_init(&shaCtx);
		#pragma unroll
		for (unsigned short i = 0; i < SALT_LENGTH; i++) {
			shaCtx.data[i] = entry->salt[i];
		}
		shaCtx.datalen = SALT_LENGTH;

		sha256_update(&shaCtx, sequentialWordlist + step, j);
		sha256_final(&shaCtx, digest.bytes);

		if (hashCompare(&digest, &entry->hashBytes)) {
			for (int i = 0; i < j; i++) {
				entry->solution[i] = sequentialWordlist[step + i];
			}
			return;
		}
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
	hi->sequentialWordlist->copied = true;

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
		hi->start += BLOCKS_PER_ENTRY * THREADS;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		cudaMemcpy(hi->entries.entries, d_hashEntry, sizeof(HashEntry) * hi->entries.currentTotal, cudaMemcpyDeviceToHost);

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			printf("Error cuda sha256 hash: %s \n", cudaGetErrorString(error));
		}

		if (containsNewSolution(&hi->entries)) {
			processAfterSolutionWasFound(&hi->entries, d_hashEntry);
		}

		hi->hashesProcessed += hi->entries.currentTotal * (MAX_PASSWORD_LENGTH - MIN_PASSWORD_CHECK) * THREADS * BLOCKS_PER_ENTRY;

		if (hi->start > hi->sequentialWordlist->characterCount) {
			hi->start = 0;
			cudaMemcpy(d_wordlist, hi->sequentialWordlist->words, sizeof(unsigned char) * hi->sequentialWordlist->characterCount, cudaMemcpyHostToDevice);
			printf("\nCopied to GPU\n");
			hi->sequentialWordlist->copied = true;
			if (hi->sequentialWordlist->finished) {
				break;
			}
		}
	}
	hi->finished = true;

  cudaDeviceReset();
  return NULL;
}

long long timems() {
	struct timeval end;
	gettimeofday(&end, NULL);
	return (long long) end.tv_sec * 1000 + (long long) end.tv_usec / 1000;
}


int main(int argc, char *argv[]) {
	setlocale(LC_NUMERIC, "");

	showGPUDevicesInfo();

	srand(time(NULL));

	printf("Loading hashes from the file\n\n");
	HashEntries entries;
	readHashEntriesFromFile("data/hashes_and_salts.txt", &entries);
	if (entries.entriesCount == 0) {
		printf("No entries found, exiting\n");
		exit(0);
	}

	printHashEntries(&entries);


	SequentialWordlist wordlist;
	createSequentialWordlist(&wordlist, 1500000000);

	unsigned short currentWordlist = 1;
	changeSequentialWordlistFile(&wordlist, argv[currentWordlist]);

	readNextChunkFromSequentialWordlist(&wordlist, CHARSET);

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

		if (wordlist.copied) {
			printf("\nGetting more passwords\n");
			readNextChunkFromSequentialWordlist(&wordlist, CHARSET);
			printf("Got more passwords\n");

			if (wordlist.finished) {
				currentWordlist += 1;
				if (currentWordlist < argc) {
					printf("Using %s\n", argv[currentWordlist]);
					changeSequentialWordlistFile(&wordlist, argv[currentWordlist]);
				}
			}
		}

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

