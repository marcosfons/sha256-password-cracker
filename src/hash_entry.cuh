#ifndef HASH_TYPES_H
#define HASH_TYPES_H


// Defined in the problem. The hash size is 32 and the maximum password characters is 16
#define SALT_LENGTH 32
// #define MAX_PASSWORD_LENGTH 16

#define MAX_PASSWORD_LENGTH 16

#define MIN_PASSWORD_CHECK 5

#define MAX_INPUT_LENGTH (SALT_LENGTH + MAX_PASSWORD_LENGTH)

#define HASH_BYTES_LENGTH 32

#define HASH_STRING_LENGTH 256


typedef struct hash_entry {
	unsigned char hash_bytes[HASH_BYTES_LENGTH];
	unsigned char salt[SALT_LENGTH];
	char solution[MAX_PASSWORD_LENGTH];
} hash_entry;


void read_entries_from_file(const char* filepath, hash_entry** entries, int* size);

void init_hash_entry(hash_entry* entry, const char hash_bytes[HASH_BYTES_LENGTH], const char salt[SALT_LENGTH]);

void print_hash_entry(hash_entry entry);

void hex_to_bytes(const char* hex_string, unsigned char bytes[HASH_BYTES_LENGTH]);



#endif  // HASH_TYPES_H
