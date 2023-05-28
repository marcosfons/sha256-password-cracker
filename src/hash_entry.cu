#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hash_entry.cuh"


void init_hash_entry(hash_entry* entry, const char hash_bytes[HASH_BYTES_LENGTH], const char salt[SALT_LENGTH]) {
	hex_to_bytes(hash_bytes, entry->hash_bytes);
	strncpy((char*) entry->salt, salt, SALT_LENGTH);

	for (int i = 0; i < MAX_PASSWORD_LENGTH; i++) {
		entry->solution[i] = 0;
	}
}

void read_entries_from_file(const char* filepath, hash_entry** entries, int* size) {
	FILE* file = fopen(filepath, "r");  // Replace "input.txt" with your file path
	if (file == NULL) {
		perror("Error opening the file");
		return;
	}

	char line[512]; // Lines will not be much longer than 512 chars
	while (fgets(line, sizeof(line), file) != NULL) {
		char* hash = strtok(line, ":");  // Split the line by colon
		char* salt = strtok(NULL, "\n");  // Get the remaining part of the line (salt)

		if (hash != NULL && salt != NULL) {
			hash_entry entry;
			init_hash_entry(&entry, hash, salt);

			*size += 1;
			*entries = (hash_entry*) realloc(*entries, sizeof(hash_entry) * (*size));
			if (*entries == NULL) {
					perror("Error reallocating memory while reading the file");
					fclose(file);
					return;
			}

			(*entries)[*size - 1] = entry;
		}
	}

	fclose(file);
	return;
}

void print_hash_entry(hash_entry entry) {
	printf("Hash: ");
	for (size_t i = 0; i < HASH_BYTES_LENGTH; i++) {
		printf("%02x", entry.hash_bytes[i]);
	}
	printf("  Salt: %.32s", entry.salt);
	for (int i = 0; i < MAX_PASSWORD_LENGTH; i++) {
		if (entry.solution[i]) {
			printf("  Pass: %s\n", entry.solution);
			return;
		}
	}
	printf("\n");
}

void hex_to_bytes(const char* hex_string, unsigned char bytes[HASH_BYTES_LENGTH]) {
  for (unsigned int i = 0; i < HASH_BYTES_LENGTH; i += 1) {
		sscanf(&hex_string[i * 2], "%02x", (unsigned int *) &bytes[i]);
  }
}

