#include <unordered_map>
#include <string>
#include "symbol-table.h"

using namespace std;

std::unordered_map<string,var> variables;

void add_scalar_symbol(const char* name)
{
    var v;
    v.type = SCALAR;
    variables[name] = v;
}
void add_vector_symbol(const char* name,int size)
{
    var v;
    v.type = VECTOR;
    v.size = size;
    variables[name] = v;
}
var get_symbol(const char* name)
{
    if(variables.find(name) == variables.end())
    {
        var v;
        v.type = UNDEFINED;
        return v;
    }
    return variables[name];
}

