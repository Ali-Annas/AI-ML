#include <iostream>

using namespace std;

extern "C" int yyparse();

FILE* cppfile;

int main(int argc,const char* argv[])
{
    cppfile = fopen("output.cpp","wb");
    if(!cppfile)
        return 1;
    fputs("#include \"include/utils.h\"\nint main()\n",cppfile);
    yyparse();
    fclose(cppfile);
    return 0;
}
