#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#include "hash_entry.h"
#include "wordlist.h"


// void getInputFromNumber(unsigned long long current, char* input, int length, const char* charset, size_t charsetLength) {
// 	for (unsigned short i = 0; i < length; i++) {
// 		input[i] = charset[current % charsetLength];
// 		current /= charsetLength;
// 	}
// 	return;
// }
//
// void createSequentialWordlist(SequentialWordlist* wordlist, int length, const char* charset, size_t charsetLength, size_t maxCharacter) {
// 	wordlist->current = 0;
// 	wordlist->wordLength = length;
// 	wordlist->wordsCount = pow(charsetLength, length);
// 	wordlist->characterCount = wordlist->wordsCount * length;
//
// 	if (wordlist->characterCount > maxCharacter) {
// 		wordlist->characterCount = (maxCharacter / length) * length;
// 		wordlist->wordsCount = (maxCharacter / length);
// 	}
//
// 	wordlist->words = (char*) malloc(sizeof(char) * wordlist->characterCount);
//
// 	for (size_t i = wordlist->current; i < wordlist->wordsCount; i++) {
// 		getInputFromNumber(i, wordlist->words + (i * length), length, charset, charsetLength);
// 	}
// }

// bool generateSequentialWordlist(SequentialWordlist* wordlist, size_t maxCharacter, const char* charset, size_t charsetLength) {
// 	wordlist->current += wordlist->wordsCount;
// 	size_t max_words = pow(charsetLength, wordlist->wordLength);
//
// 	if (wordlist->current >= max_words) {
// 		return false;
// 	}
//
// 	wordlist->wordsCount = (max_words - wordlist->current);
// 	wordlist->characterCount = wordlist->wordsCount * wordlist->wordLength;
//
// 	if (wordlist->characterCount > maxCharacter) {
// 		wordlist->characterCount = (maxCharacter / wordlist->wordLength) * wordlist->wordLength;
// 		wordlist->wordsCount = (maxCharacter / wordlist->wordLength);
// 	}
//
// 	for (size_t i = 0; i < wordlist->wordsCount; i++) {
// 		getInputFromNumber(wordlist->current + i, wordlist->words + (i * wordlist->wordLength), wordlist->wordLength, charset, charsetLength);
// 	}
//
// 	return true;
// }

void createSequentialWordlist(SequentialWordlist *wordlist, size_t maxSize) {
	wordlist->file = NULL;
	wordlist->copied = false;
	wordlist->finished = false;
	wordlist->words = (char*) malloc(maxSize * sizeof(char));
	wordlist->characterCount = 0;
	wordlist->maxChunkSize = maxSize;
}

void changeSequentialWordlistFile(SequentialWordlist *wordlist, const char *filepath) {
	if (wordlist->file != NULL) {
		fclose(wordlist->file);
		wordlist->file = NULL;
	}

	wordlist->file = fopen(filepath, "r");
	if (wordlist->file == NULL) {
		perror("Error while opening sequential wordlist file");
		return;
	}

	wordlist->finished = false;
}

bool readNextChunkFromSequentialWordlist(SequentialWordlist* wordlist, const char* charset) {
	wordlist->copied = false;

	int charsetLength = strlen(charset);

	size_t ret = fread(wordlist->words, sizeof(unsigned char), wordlist->maxChunkSize - 1, wordlist->file);
	if (ret <= 1) {
		wordlist->finished = true;
		return false;
	}
	size_t j = 0;
	for (size_t i = 0; i < ret; i++) {
		// TODO(marcosfons): Remove this unnecessary if
		if (wordlist->words[i] == '\n') {
		} else if (strchr(charset, wordlist->words[i]) == NULL || wordlist->words[i] == '\0') {
			wordlist->words[j++] = charset[rand() % charsetLength];
		} else {
			wordlist->words[j++] = wordlist->words[i];
		}
	}

	for (size_t i = j; i < ret; i++) {
		wordlist->words[i] = charset[rand() % charsetLength];
	}
	wordlist->characterCount = ret;

	return true;
}

