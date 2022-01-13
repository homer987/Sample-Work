// Implements a dictionary's functionality

#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <cs50.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>

#include "dictionary.h"

// Represents a node in a hash table
typedef struct node
{
    char word[LENGTH + 1];
    struct node *next;
}
node;

//  N to 26x26x26 = 17576
const unsigned int N = 17576;
int word_count = 0;

// Hash table
node *table[N];

// Returns true if word is in dictionary, else false
bool check(const char *word)
{
    // hash word to obtain hash value
    long bucket = hash(word);

    // access linked list at that hash value at that index in the hash table
    node *cursor = table[bucket];

    //traverse linked list, looking for the word using strcasecmp.
    // make a loop for cursur moving when it's !NUll.
    while (cursor != NULL)
    {
        // if word found, return true.
        if (strcasecmp( cursor -> word, word) == 0)
        {
            return true;
        }
        else // if word not found, keep moving the cursor
        {
            cursor = cursor -> next;
        }
    }
    return false;
}

// Hashes word to a number
unsigned int hash(const char *word)
{
    // hash function used is djb2
    unsigned long hash = 5381;
    int c;

    while ((c = toupper (*word++)))
    {
        hash = ((hash << 5) + hash) + c;
    }
    return (hash % N);
}

// Loads dictionary into memory, returning true if successful, else false
bool load(const char *dictionary)
{


    //open dictionary file
    // use fopen and check if return value is NUll.
    FILE *infile = fopen(dictionary, "r");
    if (infile == NULL)
    {
        return false;
    }

    if (infile != NULL)
    {
        // read strings from the file

        // making character array
        char word[LENGTH + 1];

        //use fscanf(file, "%s", word), where file is the result of calling fopen, and word is a character array.
        // loop - fscanf until the end of file.
        while (fscanf(infile, "%s", word) != EOF)
        {
            //create a new node, use malloc
            node *n = malloc(sizeof(node));
            if ( n == NULL) // if node is null, free and return false.
            {
                free(n);
                return false;
            }
            // otherwise
            //copy word into node using strcpy
            strcpy(n->word, word);

            //insert the node into the hash table
                //use hash function
                // function takes a string and returns an index.
            //hash the word.
            long bucket = hash(word);

            if (table[bucket] != NULL)
            {
                //making new pointer.
                n-> next = table[bucket];
                table[bucket] = n;
            }
            else if (table[bucket] == NULL) // if table[] is null, new pointer is the head.
            {
                table[bucket] = n;
            }
            word_count++; // adding word count for size function.
        }
        fclose(infile);
        return true;
    }
    return false;
}

// Returns number of words in dictionary if loaded, else 0 if not yet loaded
unsigned int size(void)
{
    // return word count from load function
    return word_count;
}

// Unloads dictionary from memory, returning true if successful, else false
bool unload(void)
{

    //creating a loop to move cursor while freeing temp.
    for (int i = 0; i < N; i++)
    {
        //creating cursor and temp cursor for freeing space while cursor keeps moving.
        node *cursor = table[i];
        node *tmp = table[i];

        while (cursor != NULL)
        {
            // moving cursor while freeing tmp.
            cursor = cursor -> next;
            free (tmp);
            tmp = cursor;
        }
    }
    return true;
}
