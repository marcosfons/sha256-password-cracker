#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "hash_entry.h"

void initHashEntry(HashEntry *entry, const char *user, const char *hashBytes,
                   const char *salt, const char *password) {
  strncpy(entry->username, user, USERNAME_LENGTH);
  hexToBytes(hashBytes, entry->hashBytes.bytes, HASH_BYTES_LENGTH);
  hexToBytes(salt, entry->salt, SALT_LENGTH);

	memset(entry->solution, 0, MAX_PASSWORD_LENGTH);
	if (password != NULL) {
		strcpy(entry->solution, password);
	}
}

void readHashEntriesFromFile(const char *filepath, HashEntries *entries) {
	FILE* file = fopen(filepath, "r");
	if (file == NULL) {
		printf("An error has occurred while reading file\n");
		perror("Error opening the file");
		return;
	}

	entries->filepath = (char*) malloc(strlen(filepath) + 1);
	strcpy(entries->filepath, filepath);

	entries->entriesCount = 0;
	entries->currentTotal = 0;
	entries->entries = (HashEntry*) malloc(1);

	char line[512]; // Lines will not be much longer than 512 chars
	while (fgets(line, sizeof(line), file) != NULL) {
		char* user = strtok(line, ":");
		char* salt = strtok(NULL, ":");
		char* hash = strtok(NULL, ":\n");
		char* password = strtok(NULL, "\n");
	
		if (hash != NULL && salt != NULL) {
			entries->entriesCount += 1;
			entries->currentTotal += 1;
			entries->entries = (HashEntry*) realloc(entries->entries, sizeof(HashEntry) * entries->entriesCount);
			if (entries->entries == NULL) {
					perror("Error reallocating memory while reading the file");
					fclose(file);
					return;
			}

			if (password != NULL) {
				entries->currentTotal -= 1;
			}

			initHashEntry(entries->entries + (entries->entriesCount - 1), user, hash, salt, password);
		}
	}

	fclose(file);
	return;

}

void printHashEntries(HashEntries *entries) {
	for (int i = 0; i < entries->entriesCount; i++) {
		printHashEntry(entries->entries[i]);
		printf("\n");
	}
}

void printHashEntry(HashEntry entry) {
	printf("User: %.5s  ", entry.username);
	printf("Hash: ");
	for (size_t i = 0; i < HASH_BYTES_LENGTH; i++) {
		printf("%02x", entry.hashBytes.bytes[i]);
	}
	printf("  Salt: ");
	for (size_t i = 0; i < SALT_LENGTH; i++) {
		printf("%02x", entry.salt[i]);
	}

	if (containsSolutionHashEntry(&entry)) {
		printf("  Pass: ");
		for (int i = 0; i < MAX_PASSWORD_LENGTH; i++) {
			printf("%c", entry.solution[i]);
		}
	}
}

void saveHashEntriesToFile(HashEntries *entries) {
	FILE* file = fopen(entries->filepath, "w");
	if (file == NULL) {
		perror("Error opening the file for write");
		return;
	}
	
	for (int i = 0; i < entries->entriesCount; i++) {
		HashEntry *entry = entries->entries + i;

		fprintf(file, "%.5s:", entry->username);

		for (int j = 0; j < SALT_LENGTH; j++) {
			fprintf(file, "%02x", entry->salt[j]);
		}
		fprintf(file, ":");
		for (int j = 0; j < HASH_BYTES_LENGTH; j++) {
			fprintf(file, "%02x", entry->hashBytes.bytes[j]);
		}
		
		if (containsSolutionHashEntry(entry)) {
			fprintf(file, ":%s", entry->solution);
		}

		fprintf(file, "\n");
	}

	fclose(file);
}

void reorganizeNotSolvedEntries(HashEntries *entries) {
	for (int i = 0; i < entries->currentTotal; i++) {
		if (containsSolutionHashEntry(entries->entries + i)) {
			int finalIndex = (entries->currentTotal) - 1;
			HashEntry entryCopy = entries->entries[i];
			entries->entries[i] = entries->entries[finalIndex];
			entries->entries[finalIndex] = entryCopy;

			entries->currentTotal = entries->currentTotal - 1;
			i -= 1;
		}
	}

	saveHashEntriesToFile(entries);
}

bool containsSolutionHashEntry(HashEntry* entry) {
	for (int i = 0; i < MAX_PASSWORD_LENGTH; i++) {
		if (entry->solution[i] != 0) {
			return true;
		}
	}
	
	return false;
}

bool containsNewSolution(HashEntries *entries) {
	for (int i = 0; i < entries->currentTotal; i++) {
		if (containsSolutionHashEntry(entries->entries + i)) {
			return true;
		}
	}

	return false;
}

void hexToBytes(const char* hexString, unsigned char* bytes, unsigned int length) {
  for (unsigned int i = 0; i < length; i += 1) {
		sscanf(&hexString[i * 2], "%02x", (unsigned int *) &bytes[i]);
  }
}
