#ifndef WORDLIST_H
#define WORDLIST_H


#include "hash_entry.h"


typedef struct word {
	char* word;
	unsigned short length;
} word;

typedef struct Wordlist {
	word* words;
	unsigned long wordsCount;
} wordlist;

typedef struct SequentialWordlist {
	char* words;
	unsigned int wordLength;
	unsigned long long current;
	unsigned long long wordsCount;
	unsigned long long characterCount;
} sequential_wordlist;



void createSequentialWordlist(SequentialWordlist* wordlist, int length, const char* charset, size_t charset_length, size_t max_size);
bool generateSequentialWordlist(SequentialWordlist* wordlist, size_t max_character, const char* charset, size_t charset_length);

void readWordlistFromFile(const char* filepath, Wordlist* wordlist);
void readSequentialWordlistFromFile(const char* filepath, SequentialWordlist* wordlist);



#endif // WORDLIST_H
