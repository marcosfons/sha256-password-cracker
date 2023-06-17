#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "hash_entry.cuh"

void init_hash_entry(hash_entry *entry, const char *user,
                     const char *hash_bytes, const char *salt,
                     const char *password) {
  strncpy(entry->username, user, USERNAME_LENGTH);
  hex_to_bytes_with_len(hash_bytes, entry->hash_bytes.hash_bytes, HASH_BYTES_LENGTH);
  hex_to_bytes_with_len(salt, entry->salt, SALT_LENGTH);

	memset(entry->solution, 0, MAX_PASSWORD_LENGTH);
	if (password != NULL) {
		strcpy(entry->solution, password);
	}
}

void read_hash_entries_from_file(const char *filepath, hash_entries *entries) {
	FILE* file = fopen(filepath, "r");
	if (file == NULL) {
		printf("An error has occurred while reading file\n");
		perror("Error opening the file");
		return;
	}

	entries->filepath = (char*) malloc(strlen(filepath) + 1);
	strcpy(entries->filepath, filepath);

	entries->entries_count = 0;
	entries->current_total = 0;
	entries->entries = (hash_entry*) malloc(1);

	char line[512]; // Lines will not be much longer than 512 chars
	while (fgets(line, sizeof(line), file) != NULL) {
		char* user = strtok(line, ":");
		char* salt = strtok(NULL, ":");
		char* hash = strtok(NULL, ":\n");
		char* password = strtok(NULL, "\n");
	
		if (hash != NULL && salt != NULL) {
			entries->entries_count += 1;
			entries->current_total += 1;
			entries->entries = (hash_entry*) realloc(entries->entries, sizeof(hash_entry) * entries->entries_count);
			if (entries->entries == NULL) {
					perror("Error reallocating memory while reading the file");
					fclose(file);
					return;
			}

			if (password != NULL) {
				entries->current_total -= 1;
			}

			init_hash_entry(entries->entries + (entries->entries_count - 1), user, hash, salt, password);
		}
	}

	fclose(file);
	return;

}

void print_hash_entries(hash_entries *entries) {
	for (int i = 0; i < entries->entries_count; i++) {
		print_hash_entry(entries->entries[i]);
		printf("\n");
	}
}

void print_hash_entry(hash_entry entry) {
	printf("User: %.5s  ", entry.username);
	printf("Hash: ");
	for (size_t i = 0; i < HASH_BYTES_LENGTH; i++) {
		printf("%02x", entry.hash_bytes.hash_bytes[i]);
	}
	printf("  Salt: ");
	for (size_t i = 0; i < SALT_LENGTH; i++) {
		printf("%02x", entry.salt[i]);
	}

	if (contains_solution_hash_entry(&entry)) {
		printf("  Pass: ");
		for (int i = 0; i < MAX_PASSWORD_LENGTH; i++) {
			printf("%c", entry.solution[i]);
		}
	}
}

void save_hash_entries_to_file(hash_entries *entries) {
	FILE* file = fopen(entries->filepath, "w");
	if (file == NULL) {
		perror("Error opening the file for write");
		return;
	}
	
	for (int i = 0; i < entries->entries_count; i++) {
		hash_entry *entry = entries->entries + i;

		fprintf(file, "%.5s:", entry->username);

		for (int j = 0; j < SALT_LENGTH; j++) {
			fprintf(file, "%02x", entry->salt[j]);
		}
		fprintf(file, ":");
		for (int j = 0; j < HASH_BYTES_LENGTH; j++) {
			fprintf(file, "%02x", entry->hash_bytes.hash_bytes[j]);
		}
		
		if (contains_solution_hash_entry(entry)) {
			fprintf(file, ":%s", entry->solution);
		}

		fprintf(file, "\n");
	}

	fclose(file);
}

void reorganize_not_solved_entries(hash_entries *entries) {
	for (int i = 0; i < entries->current_total; i++) {
		if (contains_solution_hash_entry(entries->entries + i)) {
			int final_index = (entries->current_total) - 1;
			hash_entry entry_copy = entries->entries[i];
			entries->entries[i] = entries->entries[final_index];
			entries->entries[final_index] = entry_copy;

			entries->current_total = entries->current_total - 1;
			i -= 1;
		}
	}

	save_hash_entries_to_file(entries);
}

bool contains_solution_hash_entry(hash_entry* entry) {
	for (int i = 0; i < MAX_PASSWORD_LENGTH; i++) {
		if (entry->solution[i] != 0) {
			return true;
		}
	}
	
	return false;
}

bool contains_new_solution(hash_entries *entries) {
	for (int i = 0; i < entries->current_total; i++) {
		if (contains_solution_hash_entry(entries->entries + i)) {
			return true;
		}
	}

	return false;
}

void hex_to_bytes_with_len(const char* hex_string, unsigned char* bytes, unsigned int len) {
  for (unsigned int i = 0; i < len; i += 1) {
		sscanf(&hex_string[i * 2], "%02x", (unsigned int *) &bytes[i]);
		// sscanf(&hex_string[i * 2], "%02x", (unsigned int *) &bytes[i]);
  }
}
