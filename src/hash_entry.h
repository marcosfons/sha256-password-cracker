#ifndef H_HASH_TYPES_H
#define H_HASH_TYPES_H


#include <stdbool.h>


#define SALT_LENGTH 16

#define MAX_PASSWORD_LENGTH 16

#define MIN_PASSWORD_CHECK 4

#define MAX_INPUT_LENGTH (SALT_LENGTH + MAX_PASSWORD_LENGTH)

#define HASH_BYTES_LENGTH 32

#define HASH_STRING_LENGTH 256

#define USERNAME_LENGTH 5



typedef union u_HashBytes {
	unsigned char bytes[HASH_BYTES_LENGTH];
	unsigned long long hash_llu[4];
	unsigned int hash_du[8];
} u_HashBytes;

typedef struct HashEntry {
	char username[USERNAME_LENGTH];
	u_HashBytes hashBytes;
	unsigned char salt[SALT_LENGTH];
	char solution[MAX_PASSWORD_LENGTH];
} HashEntry;

typedef struct HashEntries {
	char* filepath;
	int entriesCount;
	int currentTotal;
	HashEntry* entries;
} HashEntries;


void readHashEntriesFromFile(const char *filepath, HashEntries *entries);

void reorganizeNotSolvedEntries(HashEntries *entries);

void printHashEntries(HashEntries *entries);

void printHashEntry(HashEntry entry);

void saveHashEntriesToFile(HashEntries *entries);

bool containsNewSolution(HashEntries *entries);

bool containsSolutionHashEntry(HashEntry *entry);

void hexToBytes(const char *hexString, unsigned char *bytes, unsigned int len);

#endif  // HASH_TYPES_H
