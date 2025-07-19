all:
	flex -o src/lex.yy.c src/lexer.flex
	bison -d -v src/parser.y -o src/parser.tab.c
	gcc -c src/lex.yy.c -o obj/lex.yy.o -Iinclude -g
	gcc -c src/parser.tab.c -o obj/parser.tab.o -Iinclude -g
	g++ -c src/main.cpp -o obj/main.o -Iinclude -g
	g++ -c src/symbol-table.cpp -o obj/symbol-table.o -Iinclude -g
	g++ obj/*.o -o vlang
