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
	unsigned long words_count;
	unsigned long character_count;
} sequential_wordlist;

void read_wordlist_from_file(const char* filepath, wordlist* wordlist);
void read_sequential_wordlist_from_file(const char* filepath, sequential_wordlist* wordlist);



#endif // WORDLIST_H
