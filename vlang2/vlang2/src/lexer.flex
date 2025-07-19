%{
    #define YYSTYPE char*
    #include "parser.tab.h"
    extern int yyparse();
    int line_num = 1;
%}

%option noyywrap

%%
if {return IF;}
loop {return LOOP;}
print {return PRINT;}
scl {return SCL;}
vec {return VEC;}
\"[^\n\"]*\" {yylval = strdup(yytext);return STRING;}
[a-zA-Z_]+[a-zA-Z_0-9]* {yylval = strdup(yytext);return IDENTIFIER;}
[0-9]+ {yylval = strdup(yytext); return NUMBER;}
"{" {return BEGIN_CURLY_BRACKET;}
"}" {return END_CURLY_BRACKET;}
"[" {return BEGIN_SQUARE_BRACKET;}
"]" {return END_SQUARE_BRACKET;}
"(" {return BEGIN_PARANTHESIS;}
")" {return END_PARANTHESIS;}
@   {return AT;}
"/" {return DIV;}
"*" {return MUL;}
"-" {return SUB;}
"+" {return ADD;}
"," {return COMMA;}
:   {return COLON;}
"=" {return EQUAL;}
\n {line_num++;}
" "|\t {}
; {return SEMICOLON;}

. {fprintf(stderr,"line %d: lexical error %s\n",line_num,yytext);exit(1);}
%%
