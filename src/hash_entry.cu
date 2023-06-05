#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "hash_entry.cuh"


void init_hash_entry(hash_entry* entry, const char* hash_bytes, const char* salt) {
	hex_to_bytes(hash_bytes, entry->hash_bytes.hash_bytes);
	hex_to_bytes_with_len(salt, entry->salt, SALT_LENGTH);

	// for (int i = 0; i < MAX_PASSWORD_LENGTH; i++) {
	// 	entry->solution[i] = 0;
	// }
}

void read_entries_from_file(const char* filepath, hash_entry** entries, int* size) {
	FILE* file = fopen(filepath, "r");  // Replace "input.txt" with your file path
	if (file == NULL) {
		printf("An error has occurred while reading file\n");
		perror("Error opening the file");
		return;
	}

	char line[512]; // Lines will not be much longer than 512 chars
	while (fgets(line, sizeof(line), file) != NULL) {
		char* hash = strtok(line, ":");  // Split the line by colon
		char* salt = strtok(NULL, "\n");  // Get the remaining part of the line (salt)
	
		if (hash != NULL && salt != NULL) {
			*size += 1;
			*entries = (hash_entry*) realloc(*entries, sizeof(hash_entry) * (*size));
			if (*entries == NULL) {
					perror("Error reallocating memory while reading the file");
					fclose(file);
					return;
			}

			init_hash_entry((*entries) + (*size - 1), hash, salt);
		}
	}

	fclose(file);
	return;
}

void print_hash_entry(hash_entry entry) {
	printf("Hash: ");
	for (size_t i = 0; i < HASH_BYTES_LENGTH; i++) {
		printf("%02x", entry.hash_bytes.hash_bytes[i]);
	}
	printf("  Salt: ");
	for (size_t i = 0; i < SALT_LENGTH; i++) {
		printf("%02x", entry.salt[i]);
	}
}

void hex_to_bytes(const char* hex_string, unsigned char bytes[HASH_BYTES_LENGTH]) {
  for (unsigned int i = 0; i < HASH_BYTES_LENGTH; i += 1) {
		sscanf(&hex_string[i * 2], "%02x", (unsigned int *) &bytes[i]);
  }
}

void hex_to_bytes_with_len(const char* hex_string, unsigned char* bytes, unsigned int len) {
  for (unsigned int i = 0; i < len; i += 1) {
		sscanf(&hex_string[i * 2], "%02x", (unsigned int *) &bytes[i]);
		// sscanf(&hex_string[i * 2], "%02x", (unsigned int *) &bytes[i]);
  }
}
