%{
    #define YYSTYPE char*
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include "symbol-table.h"
    int yyerror(const char*);
    int yylex();
    extern FILE* cppfile;
    extern int line_num;
    char* merge(const char* str1,const char* str2)
    {
        size_t a = strlen(str1);
        size_t b = strlen(str2);
        char* result = (char*)malloc(sizeof(char)*(a+b+1));
        memcpy(result,str1,a);
        memcpy(result+a,str2,b);
        result[a+b] = 0;
        return result;
    }
    char buffer[1024];
    int indent = 0;
    void add_indent()
    {
        for(int i=1; i<= indent; i++)
            fputc('\t',cppfile);
    }
%}

%token COMMA
%token COLON
%token PRINT
%token LOOP
%token SCL
%token VEC
%token NUMBER
%token IDENTIFIER
%token ADD
%token SUB
%token MUL
%token DIV
%token AT
%token BEGIN_CURLY_BRACKET
%token END_CURLY_BRACKET
%token BEGIN_SQUARE_BRACKET
%token END_SQUARE_BRACKET
%token BEGIN_PARANTHESIS
%token END_PARANTHESIS
%token IF
%token EQUAL
%token SEMICOLON
%token STRING

%left ADD SUB
%left DIV MUL
%left AT COLON



%%
program: block
       ;
block_begin: BEGIN_CURLY_BRACKET {add_indent(); fputs("{\n",cppfile);indent++;};
block_end: END_CURLY_BRACKET {indent--; add_indent(); fputs("}\n",cppfile); };           
block: block_begin statement_list block_end;
statement_list: statement_list statement | statement;
statement: conditional_statement | loop_statement | print_statement | equal_statement | vec_statement | scl_statement;
if_begin: IF expr {add_indent(); fprintf(cppfile,"if(%s)\n",$2);};
conditional_statement: if_begin block;
loop_begin: LOOP expr {add_indent(); fprintf(cppfile,"for(int _ = 1; _ <= %s; _++)\n",$2);};
loop_statement: loop_begin block;
print_statement: PRINT STRING COLON argument_list SEMICOLON {add_indent(); fprintf(cppfile,"cout << %s << \": \" << %s << endl;\n",$2,$4); };

argument_list: argument_list COMMA expr {snprintf(buffer,1024,"%s << %s",$1,$3); free($1); free($3); $$ = strdup(buffer);}| expr;
             
equal_statement: IDENTIFIER EQUAL expr SEMICOLON {
    var v = get_symbol($1);
    if(v.type == UNDEFINED)
        yyerror("assignment to undefined variable");
    add_indent();
    if(v.type == VECTOR)
        fprintf(cppfile,"assign_vector(%s,%s);\n",$1,$3);
    else
        fprintf(cppfile,"%s = %s;\n",$1,$3);
    free($1);
    free($3);
} |
    IDENTIFIER COLON index EQUAL expr SEMICOLON {
    var v = get_symbol($1);
    if(v.type == UNDEFINED)
        yyerror("assignment to undefined varaible");
    if(v.type != VECTOR)
        yyerror("cannot at assign index of scalar type");
    add_indent();
    fprintf(cppfile,"%s[%s] = %s;\n",$1,$3,$5); 
    free($1);
    free($3);
    free($5);
};
vec_statement: VEC IDENTIFIER BEGIN_CURLY_BRACKET NUMBER END_CURLY_BRACKET SEMICOLON {
    var v = get_symbol($2);
    if(v.type != UNDEFINED)
        yyerror("redeclaration of variable");
    int i = atoi($4);
    add_vector_symbol($2,i);
    add_indent();
    fprintf(cppfile,"vector<int> %s(%d);\n",$2,i);
    free($2);
    free($4);
};
scl_statement: SCL IDENTIFIER SEMICOLON {
    var v = get_symbol($2);
    if(v.type != UNDEFINED)
        yyerror("redeclaration of variable");
    add_scalar_symbol($2);
    add_indent();
    fprintf(cppfile,"int %s;\n",$2);
    free($2);

};
expr: add_expr  | mul_expr | sub_expr | div_expr |index_expr | vector | prod_expr | NUMBER |
    BEGIN_PARANTHESIS expr END_PARANTHESIS {snprintf(buffer,1024,"(%s)",$2); free($2); $$ = strdup(buffer); }|
    IDENTIFIER {
        var v = get_symbol($1);
        if(v.type == UNDEFINED)
            yyerror("undefined identifier used in expression");
        $$ = $1;
    };

add_expr: expr ADD expr { snprintf(buffer,1024,"%s + %s",$1,$3); $$ = strdup(buffer); free($1); free($3); };
mul_expr: expr MUL expr { snprintf(buffer,1024,"%s * %s",$1,$3); $$ = strdup(buffer); free($1); free($3); };
div_expr: expr DIV expr { snprintf(buffer,1024,"%s / %s",$1,$3); $$ = strdup(buffer); free($1); free($3); };
sub_expr: expr SUB expr { snprintf(buffer,1024,"%s - %s",$1,$3); $$ = strdup(buffer); free($1); free($3); };

index_expr: expr COLON expr {
    snprintf(buffer,1024,"index_vector(%s,%s)",$1,$3);
    free($1);
    free($3);
    $$ = strdup(buffer);
};


index: NUMBER | IDENTIFIER {
        var v = get_symbol($1);
        if(v.type == UNDEFINED)
            yyerror("undefined identifier used as index");
        $$ = $1;
     };
prod_expr: vector AT vector {snprintf(buffer,1024,"vector_dotprod(%s,%s)",$1,$3); free($1); free($3); $$ = strdup(buffer); }|
           vector AT IDENTIFIER {snprintf(buffer,1024,"vector_dotprod(%s,%s)",$1,$3); free($1); free($3); $$ = strdup(buffer); }|
           IDENTIFIER AT vector {snprintf(buffer,1024,"vector_dotprod(%s,%s)",$1,$3); free($1); free($3); $$ = strdup(buffer); }|
           IDENTIFIER AT IDENTIFIER {snprintf(buffer,1024,"vector_dotprod(%s,%s)",$1,$3); free($1); free($3); $$ = strdup(buffer); };

vector: BEGIN_SQUARE_BRACKET num_list END_SQUARE_BRACKET {snprintf(buffer,1024,"vector<int>({%s})",$2); $$ = strdup(buffer);};
num_list: num_list COMMA NUMBER {snprintf(buffer,1024,"%s,%s",$1,$3); $$ = strdup(buffer); free($1); free($3); } | NUMBER ;

%%

int yyerror(const char* err)
{
    fprintf(stderr,"line %d: %s\n",line_num,err);
    // exit(1);  // is line ko comment kar dein
}
