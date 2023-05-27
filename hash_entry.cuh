#ifndef HASH_TYPES_H
#define HASH_TYPES_H


#define SALT_LENGTH 32
#define MAX_PASSWORD_LENGTH 16

#define MAX_INPUT_LENGTH (SALT_LENGTH + MAX_PASSWORD_LENGTH)

#define HASH_BYTES_LENGTH 32

#define HASH_STRING_LENGTH 256


typedef struct hash_entry {
	BYTE hash_bytes[HASH_BYTES_LENGTH];
	char salt[SALT_LENGTH];
	char solution[MAX_PASSWORD_LENGTH];
} hash_entry;


#endif
