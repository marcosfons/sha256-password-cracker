#ifndef WORDLIST_H
#define WORDLIST_H


#include "hash_entry.cuh"


typedef struct word {
	char* word;
	unsigned short length;
} word;

typedef struct wordlist {
	word* words;
	unsigned long words_count;
} wordlist;

typedef struct sequential_wordlist {
	char* words;
	unsigned int word_length;
	unsigned long long current;
	unsigned long long words_count;
	unsigned long long character_count;
} sequential_wordlist;



void create_sequential_wordlist(sequential_wordlist* wordlist, int length, const char* charset, size_t charset_length, size_t max_size);
bool generate_sequential_wordlist(sequential_wordlist* wordlist, size_t max_character, const char* charset, size_t charset_length);

void read_wordlist_from_file(const char* filepath, wordlist* wordlist);
void read_sequential_wordlist_from_file(const char* filepath, sequential_wordlist* wordlist);



#endif // WORDLIST_H
