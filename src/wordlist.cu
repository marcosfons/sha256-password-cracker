#include <stdio.h>

#include "wordlist.cuh"

void read_wordlist_from_file(const char* filepath, wordlist* wordlist) {
	FILE* file = fopen(filepath, "r");
	if (file == NULL) {
		perror("Error while reading wordlist file");
		return;
	}

	// Just to realloc to work. If does not do this get error realloc(): invalid pointer
	wordlist->words = (word*) malloc(sizeof(word));
	wordlist->words_count = 0;

	char line[100]; // Words will not be much longer than 512 chars
	while (fgets(line, sizeof(line), file) != NULL) {
		unsigned short length = strlen(line);

		if (length > 15) {
			wordlist->words_count += 1;
			wordlist->words = (word*) realloc(wordlist->words, sizeof(word) * (wordlist->words_count));

			wordlist->words[wordlist->words_count - 1].length = length;
			wordlist->words[wordlist->words_count - 1].word = (char*) malloc(length * sizeof(char));

			strcpy(wordlist->words[wordlist->words_count - 1].word, line);
		}

	}

	fclose(file);
	return;
}

void read_sequential_wordlist_from_file(const char* filepath, sequential_wordlist* wordlist) {
	FILE* file = fopen(filepath, "r");
	if (file == NULL) {
		perror("Error while reading sequential wordlist file");
		return;
	}

	wordlist->words = (char*) malloc(sizeof(char));
	wordlist->words_count = 0;
	wordlist->character_count = 0;

	int i = 0; 

	char line[100]; // Words will not be much longer than 512 chars
	while (fgets(line, sizeof(line), file) != NULL) {
		unsigned short length = strlen(line) - 1;

		wordlist->character_count += length;
		wordlist->words_count += 1;

		wordlist->words = (char*) realloc(wordlist->words, sizeof(char) * (wordlist->character_count));

		strncpy(wordlist->words + wordlist->character_count - length, line, length);
	}

	wordlist->words = (char*) realloc(wordlist->words, sizeof(char) * (wordlist->character_count + 1));
	wordlist->words[wordlist->character_count] = '\0';

	fclose(file);
	return;
}
