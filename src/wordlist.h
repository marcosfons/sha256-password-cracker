#ifndef WORDLIST_H
#define WORDLIST_H


#include <stddef.h>
#include <stdio.h>

#include "hash_entry.h"


typedef struct SequentialWordlist {
	FILE* file;
	char* words;
	bool copied;
	bool finished;
	unsigned long long characterCount;
	unsigned long long maxChunkSize;
} sequential_wordlist;


void createSequentialWordlist(SequentialWordlist *wordlist, size_t maxSize);

void changeSequentialWordlistFile(SequentialWordlist *wordlist,
                                  const char *filepath);

bool readNextChunkFromSequentialWordlist(SequentialWordlist *wordlist,
                                         const char *charset);

#endif // WORDLIST_H
