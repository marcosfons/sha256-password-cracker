#ifndef H_HASH_TYPES_H
#define H_HASH_TYPES_H


#include <stdbool.h>


#define SALT_LENGTH 16

#define MAX_PASSWORD_LENGTH 16

#define MIN_PASSWORD_CHECK 7

#define MAX_INPUT_LENGTH (SALT_LENGTH + MAX_PASSWORD_LENGTH)

#define HASH_BYTES_LENGTH 32

#define HASH_STRING_LENGTH 256

#define USERNAME_LENGTH 5


union u_hash_bytes {
	unsigned char hash_bytes[HASH_BYTES_LENGTH];
	unsigned long long hash_number[4];
	unsigned int hash_number_long[8];
};

typedef struct hash_entry {
	char username[USERNAME_LENGTH];
	u_hash_bytes hash_bytes;
	unsigned char salt[SALT_LENGTH];
	char solution[MAX_PASSWORD_LENGTH];
} hash_entry;

typedef struct hash_entries {
	char* filepath;
	int entries_count;
	int current_total;
	hash_entry* entries;
} hash_entries;


void read_hash_entries_from_file(const char *filepath, hash_entries *entries);

void reorganize_not_solved_entries(hash_entries *entries);

void print_hash_entries(hash_entries *entries);

void print_hash_entry(hash_entry entry);

bool contains_new_solution(hash_entries *entries);

bool contains_solution_hash_entry(hash_entry *entry);

void hex_to_bytes_with_len(const char *hex_string, unsigned char *bytes,
                           unsigned int len);


#endif  // HASH_TYPES_H
