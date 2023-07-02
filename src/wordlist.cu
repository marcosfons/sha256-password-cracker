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

void createSequentialWordlistFromFile(SequentialWordlist *wordlist, const char *filepath, size_t maxSize) {
	wordlist->file = fopen(filepath, "r");
	if (wordlist->file == NULL) {
		perror("Error while opening sequential wordlist file");
		return;
	}

	wordlist->words = (char*) malloc(maxSize * sizeof(char));
	wordlist->wordsCount = 0;
	wordlist->characterCount = 0;
	wordlist->maxChunkSize = maxSize;
}

bool readNextChunkFromSequentialWordlist(SequentialWordlist* wordlist, const char* charset) {
	int charsetLength = strlen(charset);

	wordlist->wordsCount = 0;
	wordlist->characterCount = 0;

	bool changed = false;

	char line[5000];

	// size_t ret = fread(wordlist->words, sizeof(unsigned char), wordlist->maxChunkSize, wordlist->file);
	//
	while (fgets(line, sizeof(line), wordlist->file) != NULL) {
		line[strcspn(line, "\n")] = '\0';
		unsigned int length = strlen(line);

		if (length <= 0) {
			continue;
		}

		for (int i = 0; i < length; i++) {
			if (strchr(charset, line[i]) == NULL) {
				line[i] = charset[rand() % charsetLength];
			}
		}

		if ((wordlist->characterCount + length) >= wordlist->maxChunkSize) {
			printf("Filled a chunk with %llu characters\n", wordlist->maxChunkSize);
			break;
		}

		wordlist->characterCount += length;
		wordlist->wordsCount += 1;

		strncpy(wordlist->words + wordlist->characterCount - length, line, length);
		changed = true;
	}

	return changed;
}

// void readSequentialWordlistFromFile(const char* filepath, SequentialWordlist* wordlist, const char* charset) {
// 	FILE* file = fopen(filepath, "r");
// 	if (file == NULL) {
// 		perror("Error while reading sequential wordlist file");
// 		return;
// 	}
//
// 	int charset_length = strlen(charset);
//
// 	wordlist->words = (char*) malloc(sizeof(char));
// 	wordlist->wordsCount = 0;
// 	wordlist->characterCount = 0;
//
// 	char line[1000];
// 	while (fgets(line, sizeof(line), file) != NULL) {
// 		if (line[0] == '\0') {
// 			continue;
// 		}
// 		unsigned int length = strlen(line) - 1;
//
// 		if (line[length] == '\n') {
// 			line[length] = '\0';
// 			length -= 1;
// 		}
//
// 		for (int i = 0; i < length + 1; i++) {
// 			if (strchr(charset, line[i]) == NULL) {
// 				line[i] = charset[rand() % charset_length];
// 			}
// 		}
//
// 		wordlist->characterCount += length;
// 		wordlist->wordsCount += 1;
//
// 		wordlist->words = (char*) realloc(wordlist->words, sizeof(char) * (wordlist->characterCount));
//
// 		strncpy(wordlist->words + wordlist->characterCount - length, line, length);
//
// 		if (wordlist->characterCount > 30000000) {
// 			printf("Passou do trem\n");
// 			break;
// 		}
// 	}
//
// 	wordlist->words = (char*) realloc(wordlist->words, sizeof(char) * (wordlist->characterCount + 1));
// 	wordlist->words[wordlist->characterCount] = '\0';
//
// 	fclose(file);
// 	return;
// }
