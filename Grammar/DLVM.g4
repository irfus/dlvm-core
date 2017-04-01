grammar DLVM;

// Grammar



literal
    :
    | scalar_literal
    | boolean_literal
    | tuple_literal
    | array_literal
    | tensor_literal
    ;

scalar_literal :

// Lexes

WS : [ \t\n\r]+ -> channel(HIDDEN) ;
ID : [a-zA-Z_]+ [a-zA-Z0-9_]* ;
LINE_COMMENT : '//' ~'\n'* '\n' -> channel(HIDDEN) ;
COMMENT : '/*' .*? '*/' -> channel(HIDDEN) ;
INT : [0-9]+ ;

FLOAT
    :   '-'? INT '.' INT EXP?   // 1.35, 1.35E-9, 0.3, -4.5
    |   '-'? INT EXP            // 1e10 -3e4
    |   '-'? INT                // -3, 45
    ;
fragment INT :   '0' | [1-9] [0-9]* ; // no leading zeros
fragment EXP :   [Ee] [+\-]? INT ;

STRING :  '"' (ESC | ~["\\])* '"' ;
fragment ESC :   '\\' (["\\/bfnrt] | UNICODE) ;
fragment UNICODE : 'u' HEX HEX HEX HEX ;
fragment HEX : [0-9a-fA-F] ;

DOT    	: '.' ;
LCURLY 	: '{' ;
LPAREN 	: '(' ;
LBRACK 	: '[' ;
RCURLY 	: '}' ;
RPAREN 	: ')' ;
RBRACK 	: ']' ;
COMMA  	: ',' ;
COLON  	: ':' ;
SEMI   	: ';' ;
LT 		: '<' ;
GT 		: '>' ;
UNDERSCORE : '_' ;
BANG 	: '!' ;
QUESTION: '?' ;
AT 		: '@' ;
AND 	: '&' ;
SUB 	: '-' ;
EQUAL 	: '=' ;
OR 		: '|' ;
DIV 	: '/' ;
ADD 	: '+' ;
MUL 	: '*' ;
MOD 	: '%' ;
CARET 	: '^' ;
TILDE 	: '~' ;
