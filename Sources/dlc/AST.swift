//
//  AST.swift
//  DLVM
//
//  Created by Richard Wei on 12/25/16.
//
//

import struct Parsey.SourceRange
import enum DLVM.TypeBase
import protocol DLVM.DataType
import struct DLVM.ScalarType
import struct DLVM.TensorType
import enum DLVM.Immediate

protocol ASTNode {
    var sourceRange: SourceRange { get }
}

struct ModuleNode : ASTNode {
    let name: String
    let declarations: [DeclarationNode]
    let basicBlocks: [BasicBlockNode]
    let sourceRange: SourceRange
}

struct DeclarationNode : ASTNode {
    enum Role {
        case input, output, parameter
    }
    let name: String
    let sourceRange: SourceRange
}

struct BasicBlockNode : ASTNode {
    let name: String
    let isGradient: Bool
    let instructions: [InstructionNode]
    let sourceRange: SourceRange
}

enum InstructionNode : ASTNode {
    case tmul(OperandNode, OperandNode, SourceRange)
    
    var sourceRange: SourceRange {
        switch self {
        case let .tmul(_, _, sr):
            return sr
        }
    }
}

enum OperandNode : ASTNode {
    case immediate(ScalarType, Immediate, SourceRange)
    case globalValue(DataType, String, SourceRange)
    case temporary(DataType, String, SourceRange)

    var sourceRange: SourceRange {
        switch self {
        case let .immediate(_, _, sr),
             let .globalValue(_, _, sr),
             let .temporary(_, _, sr):
            return sr
        }
    }
}
