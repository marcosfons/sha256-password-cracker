#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#include "hash_entry.h"
#include "wordlist.h"


void getInputFromNumber(unsigned long long current, char* input, int length, const char* charset, size_t charset_length) {
	for (unsigned short i = 0; i < length; i++) {
		input[i] = charset[current % charset_length];
		current /= charset_length;
	}
	return;
}

void createSequentialWordlist(SequentialWordlist* wordlist, int length, const char* charset, size_t charsetLength, size_t maxCharacter) {
	wordlist->current = 0;
	wordlist->wordLength = length;
	wordlist->wordsCount = pow(charsetLength, length);
	wordlist->characterCount = wordlist->wordsCount * length;

	if (wordlist->characterCount > maxCharacter) {
		wordlist->characterCount = (maxCharacter / length) * length;
		wordlist->wordsCount = (maxCharacter / length);
	}

	wordlist->words = (char*) malloc(sizeof(char) * wordlist->characterCount);

	for (size_t i = wordlist->current; i < wordlist->wordsCount; i++) {
		getInputFromNumber(i, wordlist->words + (i * length), length, charset, charsetLength);
	}
}

bool generateSequentialWordlist(SequentialWordlist* wordlist, size_t maxCharacter, const char* charset, size_t charsetLength) {
	wordlist->current += wordlist->wordsCount;
	size_t max_words = pow(charsetLength, wordlist->wordLength);

	if (wordlist->current >= max_words) {
		return false;
	}

	wordlist->wordsCount = (max_words - wordlist->current);
	wordlist->characterCount = wordlist->wordsCount * wordlist->wordLength;

	if (wordlist->characterCount > maxCharacter) {
		wordlist->characterCount = (maxCharacter / wordlist->wordLength) * wordlist->wordLength;
		wordlist->wordsCount = (maxCharacter / wordlist->wordLength);
	}

	for (size_t i = 0; i < wordlist->wordsCount; i++) {
		getInputFromNumber(wordlist->current + i, wordlist->words + (i * wordlist->wordLength), wordlist->wordLength, charset, charsetLength);
	}

	return true;
}

void readWordlistFromFile(const char* filepath, Wordlist* wordlist) {
	FILE* file = fopen(filepath, "r");
	if (file == NULL) {
		perror("Error while reading wordlist file");
		return;
	}

	// Just to realloc to work. If does not do this get error realloc(): invalid pointer
	wordlist->words = (word*) malloc(sizeof(word));
	wordlist->wordsCount = 0;

	char line[100]; // Words will not be much longer than 512 chars
	while (fgets(line, sizeof(line), file) != NULL) {
		unsigned short length = strlen(line);

		if (length > 15) {
			wordlist->wordsCount += 1;
			wordlist->words = (word*) realloc(wordlist->words, sizeof(word) * (wordlist->wordsCount));

			wordlist->words[wordlist->wordsCount - 1].length = length;
			wordlist->words[wordlist->wordsCount - 1].word = (char*) malloc(length * sizeof(char));

			strcpy(wordlist->words[wordlist->wordsCount - 1].word, line);
		}

	}

	fclose(file);
	return;
}

void readSequentialWordlistFromFile(const char* filepath, SequentialWordlist* wordlist) {
	FILE* file = fopen(filepath, "r");
	if (file == NULL) {
		perror("Error while reading sequential wordlist file");
		return;
	}

	wordlist->words = (char*) malloc(sizeof(char));
	wordlist->wordsCount = 0;
	wordlist->characterCount = 0;

	char line[100]; // Words will not be much longer than 512 chars
	while (fgets(line, sizeof(line), file) != NULL) {
		unsigned short length = strlen(line) - 1;

		wordlist->characterCount += length;
		wordlist->wordsCount += 1;

		wordlist->words = (char*) realloc(wordlist->words, sizeof(char) * (wordlist->characterCount));

		strncpy(wordlist->words + wordlist->characterCount - length, line, length);
	}

	wordlist->words = (char*) realloc(wordlist->words, sizeof(char) * (wordlist->characterCount + 1));
	wordlist->words[wordlist->characterCount] = '\0';

	fclose(file);
	return;
}
