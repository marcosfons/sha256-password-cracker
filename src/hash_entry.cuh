#ifndef H_HASH_TYPES_H
#define H_HASH_TYPES_H


#define SALT_LENGTH 16

#define MAX_PASSWORD_LENGTH 16

#define MIN_PASSWORD_CHECK 4

#define MAX_INPUT_LENGTH (SALT_LENGTH + MAX_PASSWORD_LENGTH)

#define HASH_BYTES_LENGTH 32

#define HASH_STRING_LENGTH 256


union u_hash_bytes {
	unsigned char hash_bytes[HASH_BYTES_LENGTH];
	unsigned long long hash_number[4];
	unsigned int hash_number_long[8];
	// unsigned int4 test[2];
};

typedef struct hash_entry {
	// unsigned char hash_bytes[HASH_BYTES_LENGTH];
	u_hash_bytes hash_bytes;
	unsigned char salt[SALT_LENGTH];
	// char solution[MAX_PASSWORD_LENGTH];
} hash_entry;


void read_entries_from_file(const char* filepath, hash_entry** entries, int* size);

void init_hash_entry(hash_entry* entry, const char hash_bytes[HASH_BYTES_LENGTH], const char salt[SALT_LENGTH]);

void print_hash_entry(hash_entry entry);

void hex_to_bytes(const char* hex_string, unsigned char bytes[HASH_BYTES_LENGTH]);
void hex_to_bytes_with_len(const char* hex_string, unsigned char *bytes, unsigned int len);



#endif  // HASH_TYPES_H
