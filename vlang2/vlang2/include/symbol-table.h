#ifndef SYMBOL_TABLE_H_
#define SYMBOL_TABLE_H_
#ifdef __cplusplus
extern "C"{
#endif

#include <stdbool.h>

typedef enum variable_type
{
    VECTOR,
    SCALAR,
    UNDEFINED
}variable_type;

typedef struct var
{
    variable_type type;
    int size; // is only set when type is VECTOR
}var;

void add_scalar_symbol(const char*);
void add_vector_symbol(const char*,int);
var get_symbol(const char*);


#ifdef __cplusplus
}
#endif

#endif
