//
//  Grammar.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

import Parsey

/// Sample
/// ```
/// #type float16
/// 
/// x: in[2x1]
/// 
/// W1: param[auto] = 0.0
/// h1: layer[4x1] = tanh(W1 x + b1)
/// 
/// recurrent t {
///     W2, W3, W4, W5: param[auto] = random(from: 0.0, to: 1.0)
///     b2, b3, b4, b5: param[auto] = 0.0
///     h2: layer[16x1] = tanh(W2 [h1.t, h5.(t-1)] + b2)
///     h3: layer[128x1] = tanh(W3 h2 + b3)
///     h4: layer[128x1] = relu(W4 h3 + b4)
///     h5: layer[16x1] = tanh(W5 h4 + b5)
/// }
/// 
/// W6: param[auto] = random(from: 0.5, to: 1.0)
/// b6: param[auto] = 0.0
/// h6: layer[16x1] = sigmoid(W6 h5 + b6)
/// 
/// W7: param[auto] = random(from: 0.0, to: 1.0)
/// b7: param[auto] = 0.0
/// o: out[16x1] = softmax(W7 h6 + b7)
/// ``````


extension Lexer {
    static let identifier = Lexer.regex("[a-zA-Z][a-zA-Z0-9]*")
    static let number = Lexer.signedDecimal ^^ { Int($0)! }
    static let dataType = Lexer.regex("(int|float)(8|16|32|64)")
    static let lineComment = Lexer.regex("//.*?\n")
    static let space = (Lexer.whitespace | Lexer.tab)+
}

public enum Grammar {
    static let typeRole =
        Lexer.token("in[") ^^ { _ in AST.TensorType.Role.input }
      | Lexer.token("out[") ^^ { _ in AST.TensorType.Role.output }
      | Lexer.token("hidden[") ^^ { _ in AST.TensorType.Role.hidden }
    static let type =
        typeRole ~~ Lexer.number.many(separatedBy: Lexer.character("x"))
            <~~ "]" ^^ { AST.TensorType(role: $0, shape: $1) }
    static let macro =
        Lexer.character("#") ~~> Lexer.space ~~> Lexer.dataType
}
