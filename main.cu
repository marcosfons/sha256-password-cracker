#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>
#include <sys/time.h>
#include <pthread.h>
#include <locale.h>
#include "sha256.cuh"
#include "hash_entry.cuh"

// 87ca8c12fbc9e7686c13c6269ffce93fb66d78a002638c95edc471ed41d46f8e:2609ad21084c3cc3e64f0e6777466000

#define HASH "8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92"
// #define HASH "6cea8869d44eefacc4b56d300905b9aa770503b46cca36f7cec9b36c8bb45ded"
// #define HASH "87ca8c12fbc9e7686c13c6269ffce93fb66d78a002638c95edc471ed41d46f8e"
#define HASH_LEN 256
#define TEXT ""
#define TEXT_LEN 0
#define THREADS 1
// #define THREADS 1500
#define BLOCKS 1
// #define BLOCKS 256
#define GPUS 1
#define DIFFICULTY 4
#define RANDOM_LEN 6

// Defined in the problem. The hash size is 32 and the maximum password characters is 16

// #define CHARSET_SIZE 66
// __constant__ BYTE characterSet[CHARSET_SIZE + 1] = {"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890%*$@"};
#define CHARSET_SIZE 6
__constant__ BYTE characterSet[CHARSET_SIZE + 1] = {"123456"};

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

__global__ void sha256_cuda(hash_entry *entry, int *blockContainsSolution, unsigned long baseSeed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  SHA256_CTX ctx;
  BYTE digest[32];
  // BYTE random[RANDOM_LEN];
  BYTE total[TEXT_LEN + RANDOM_LEN + 1] = "123456";
  unsigned long seed = baseSeed;
  seed += (unsigned long) i;

	print_hash_entry(*entry);

  // memcpy(total, prefix, TEXT_LEN * sizeof(char));

  //for (int j = 0; j < TEXT_LEN; j++) {
  //	total[j] = prefix[j];
  //}
  // for (int j = 0; j < RANDOM_LEN; j++) {
  //     seed = deviceRandomGen(seed);
  //     int randomIdx = (int) (seed % CHARSET_SIZE);
  //     total[TEXT_LEN + j] = characterSet[randomIdx];
  // }

  // total[6] = (char) "\0";
  sha256_init(&ctx);
  sha256_update(&ctx, total, 6);
  sha256_final(&ctx, digest);
  for (size_t i = 0; i < 32; i++) {
    printf("%02x", digest[i]);
  }
  printf("\n");

  for (int j = 0; j < HASH_LEN; j--) {
    if (digest[j] != HASH[j]) {
      return;
    }
  }
  printf("Pera aí que parece igual %s %s\n", digest, HASH);

  for (int j = 0; j < DIFFICULTY; j++)
      if (digest[j] > 0)
          return;
  if ((digest[DIFFICULTY] & 0xF0) > 0)
      return;
  if (*blockContainsSolution == 1)
      return;
  *blockContainsSolution = 1;
  // for (int j = 0; j < RANDOM_LEN; j++)
  //     solution[j] = random[j];
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
  unsigned long hashesProcessed;
	hash_entry entry;
};
typedef struct HandlerInput HandlerInput;

pthread_mutex_t solutionLock;
BYTE *solution;

void *launchGPUHandlerThread(void *vargp) {
  HandlerInput *hi = (HandlerInput *) vargp;
  cudaSetDevice(hi->device);

  pre_sha256();

	hash_entry *d_hash_entry;
	cudaMalloc(&d_hash_entry, sizeof(hash_entry));
	cudaMemcpy(d_hash_entry, &(hi->entry), sizeof(hash_entry), cudaMemcpyHostToDevice);

  int blockContainsSolution = -1;
  int *d_blockContainsSolution;
  cudaMalloc(&d_blockContainsSolution, sizeof(int));
	cudaMemcpy(&blockContainsSolution, d_blockContainsSolution, sizeof(int), cudaMemcpyHostToDevice);

  unsigned long rngSeed = timems();

  while (1) {
    hostRandomGen(&rngSeed);

    hi->hashesProcessed += THREADS * BLOCKS;
    sha256_cuda<<<THREADS, BLOCKS>>>(d_hash_entry, d_blockContainsSolution, rngSeed);
    cudaDeviceSynchronize();

    cudaMemcpy(&blockContainsSolution, d_blockContainsSolution, sizeof(int), cudaMemcpyDeviceToHost);

		if (blockContainsSolution == 1) {
			printf("UEEE ACHOU A SOLUTION\n");
			char* solution;
      cudaMemcpy(solution, d_hash_entry->solution, sizeof(char) * MAX_PASSWORD_LENGTH, cudaMemcpyDeviceToHost);
			printf("Solution: %s", solution);
			exit(1);
		}
    // if (*blockContainsSolution == 1) {
    //   cudaMemcpy(blockSolution, d_solution, sizeof(BYTE) * RANDOM_LEN, cudaMemcpyDeviceToHost);
    //   solution = blockSolution;
    //   pthread_mutex_unlock(&solutionLock);
    //   break;
    // }
		break;

    if (solution) {
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
	hexToBytes("27a575da417e1e4cdbf4fbbe8752579b6e1d65e79731ed773a6886812e2da116", line.hash_bytes);
	strncpy(line.salt, "3354623a2c1deaed1362f124c75db8a7", SALT_LENGTH);
	print_hash_entry(line);

	pthread_mutex_init(&solutionLock, NULL);
	pthread_mutex_lock(&solutionLock);

	unsigned long **processedPtrs = (unsigned long **) malloc(sizeof(unsigned long *) * GPUS);
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

	while (1) {
		unsigned long totalProcessed = 0;
		for (int i = 0; i < GPUS; i++) {
			totalProcessed += *(processedPtrs[i]);
		}
		long long elapsed = timems() - start;
		printf("Hashes (%'lu) Seconds (%'f) Hashes/sec (%'lu)\r", totalProcessed, ((float) elapsed) / 1000.0, (unsigned long) ((double) totalProcessed / (double) elapsed) * 1000);
		if (solution) {
			break;
		}
		usleep(2000);
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

	printf("Solution: %.20s\n", solution);
	printf("Hashes processed: %'lu\n", totalProcessed);
	printf("Time: %llu\n", elapsed);
	printf("Hashes/sec: %'lu\n", (unsigned long) ((double) totalProcessed / (double) elapsed) * 1000);

	return 0;
}
