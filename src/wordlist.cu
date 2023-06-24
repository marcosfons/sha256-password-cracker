#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#include "hash_entry.cuh"
#include "wordlist.cuh"

void get_input_from_number(unsigned long long current, char* input, int length, const char* charset, size_t charset_length) {
	for (unsigned short i = 0; i < length; i++) {
		input[i] = charset[current % charset_length];
		current /= charset_length;
	}
	return;
}

void create_sequential_wordlist(sequential_wordlist* wordlist, int length, const char* charset, size_t charset_length, size_t max_character) {
	wordlist->current = 0;
	wordlist->word_length = length;
	wordlist->words_count = pow(charset_length, length);
	wordlist->character_count = wordlist->words_count * length;

	if (wordlist->character_count > max_character) {
		wordlist->character_count = (max_character / length) * length;
		wordlist->words_count = (max_character / length);
	}

	wordlist->words = (char*) malloc(sizeof(char) * wordlist->character_count);

	for (size_t i = wordlist->current; i < wordlist->words_count; i++) {
		get_input_from_number(i, wordlist->words + (i * length), length, charset, charset_length);
	}
}

bool generate_sequential_wordlist(sequential_wordlist* wordlist, size_t max_character, const char* charset, size_t charset_length) {
	wordlist->current += wordlist->words_count;
	size_t max_words = pow(charset_length, wordlist->word_length);

	if (wordlist->current >= max_words) {
		return false;
	}

	wordlist->words_count = (max_words - wordlist->current);
	wordlist->character_count = wordlist->words_count * wordlist->word_length;

	if (wordlist->character_count > max_character) {
		wordlist->character_count = (max_character / wordlist->word_length) * wordlist->word_length;
		wordlist->words_count = (max_character / wordlist->word_length);
	}

	for (size_t i = 0; i < wordlist->words_count; i++) {
		get_input_from_number(wordlist->current + i, wordlist->words + (i * wordlist->word_length), wordlist->word_length, charset, charset_length);
	}

	return true;
}

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
