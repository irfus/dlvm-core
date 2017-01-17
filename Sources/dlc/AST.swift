//
//  AST.swift
//  DLVM
//
//  Created by Richard Wei on 12/25/16.
//
//

import struct Parsey.SourceRange

protocol ASTNode {
    var range: SourceRange { get }
}

struct ModuleNode : ASTNode {
    let items: [TopLevelItemNode]
    let range: SourceRange
}

enum TopLevelItemNode : ASTNode {
    case moduleName(String, SourceRange)
    case declaration(DeclarationNode, SourceRange)
    case basicBlock(BasicBlockNode, SourceRange)

    var range: SourceRange {
        switch self {
        case let .moduleName(_, sr), let .declaration(_, sr), let .basicBlock(_, sr):
            return sr
        }
    }
}

struct DeclarationNode : ASTNode {
    enum Role {
        case input, output, parameter
    }
    let role: Role
    let operand: OperandNode
    let initializer: Initializer?
    let range: SourceRange
}

struct ImmediateValueNode : ASTNode {
    let type: TypeNode
    let immediate: ImmediateNode

    var range: SourceRange {
        return type.range.lowerBound..<immediate.range.upperBound
    }
}

enum Initializer : ASTNode {
    case immediate(ImmediateValueNode, SourceRange)
    case random(ImmediateValueNode, ImmediateValueNode, SourceRange)
    case repeating(ImmediateValueNode, SourceRange)

    var range: SourceRange {
        switch self {
        case let .immediate(_, sr), let .random(_, _, sr), let .repeating(_, sr):
            return sr
        }
    }
}

struct BasicBlockNode : ASTNode {
    let name: String
    let isGradient: Bool
    let instructions: [InstructionDeclarationNode]
    let range: SourceRange
}

struct ShapeNode : ASTNode {
    let dimensions: [Int]
    let range: SourceRange
}

struct InstructionDeclarationNode : ASTNode {
    let name: String?
    let instruction: InstructionNode
    let range: SourceRange
}

enum InstructionNode : ASTNode {
    case neg(OperandNode, SourceRange)
    case tmul(OperandNode, OperandNode, SourceRange)
    case add(OperandNode, OperandNode, SourceRange)
    case sub(OperandNode, OperandNode, SourceRange)
    case mul(OperandNode, OperandNode, SourceRange)
    case div(OperandNode, OperandNode, SourceRange)
    case min(OperandNode, OperandNode, SourceRange)
    case max(OperandNode, OperandNode, SourceRange)
    case sigmoid(OperandNode, SourceRange)
    case tanh(OperandNode, SourceRange)
    case relu(OperandNode, SourceRange)
    case log(OperandNode, SourceRange)
    case softmax(OperandNode, SourceRange)
    case concat([OperandNode], Int?, SourceRange)
    case shapeCast(OperandNode, ShapeNode, SourceRange)
    case typeCast(OperandNode, TypeNode, SourceRange)
    case load(OperandNode, SourceRange)
    case store(OperandNode, OperandNode, SourceRange)

    var range: SourceRange {
        switch self {
        case let .neg(_, sr),
             let .tmul(_, _, sr),
             let .add(_, _, sr), let .sub(_, _, sr),
             let .mul(_, _, sr), let .div(_, _, sr),
             let .min(_, _, sr), let .max(_, _, sr),
             let .softmax(_, sr),
             let .sigmoid(_, sr), let .relu(_, sr), let .tanh(_, sr), let .log(_, sr),
             let .concat(_, _, sr),
             let .shapeCast(_, _, sr),
             let .typeCast(_, _, sr),
             let .load(_, sr),
             let .store(_, _, sr):
            return sr
        }
    }
}

enum ImmediateNode : ASTNode {
    case bool(Bool, SourceRange), int(Int, SourceRange), float(Double, SourceRange)

    var range: SourceRange {
        switch self {
        case .bool(_, let sr), .int(_, let sr), .float(_, let sr):
            return sr
        }
    }
}

struct OperandNode : ASTNode {
    let type: TypeNode
    let shape: ShapeNode?
    let variable: VariableNode

    var range: SourceRange {
        return type.range.lowerBound..<variable.range.upperBound
    }
}

enum VariableNode : ASTNode {
    case immediate(ImmediateNode, SourceRange)
    case global(String, SourceRange)
    case temporary(String, SourceRange)

    var range: SourceRange {
        switch self {
        case let .immediate(_, sr), let .global(_, sr), let .temporary(_, sr):
            return sr
        }
    }
}

enum TypeNode : ASTNode {
    case bool(Int, SourceRange), int(Int, SourceRange), float(Int, SourceRange)

    var range: SourceRange {
        switch self {
        case .bool(_, let sr), .int(_, let sr), .float(_, let sr):
            return sr
        }
    }
}

