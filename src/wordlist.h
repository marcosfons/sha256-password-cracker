#ifndef WORDLIST_H
#define WORDLIST_H


#include <stddef.h>
#include <stdio.h>

#include "hash_entry.h"


typedef struct SequentialWordlist {
	FILE* file;
	char* words;
	unsigned long long wordsCount;
	unsigned long long characterCount;
	unsigned long long maxChunkSize;
} sequential_wordlist;

// void createSequentialWordlist(SequentialWordlist *wordlist, int length,
//                               const char *charset, size_t charsetLength,
//                               size_t maxSize);
//
// bool generateSequentialWordlist(SequentialWordlist *wordlist,
//                                 size_t maxCharacter, const char *charset,
//                                 size_t charsetLength);

void createSequentialWordlistFromFile(SequentialWordlist *wordlist,
                                      const char *filepath, size_t maxSize);

bool readNextChunkFromSequentialWordlist(SequentialWordlist *wordlist,
                                         const char *charset);

void readSequentialWordlistFromFile(SequentialWordlist *wordlist,
                                    const char *charset);

#endif // WORDLIST_H
