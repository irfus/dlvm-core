//
//  AST.swift
//  DLVM
//
//  Created by Richard Wei on 12/25/16.
//
//

import struct Parsey.SourceRange
import enum DLVM.ElementwiseFunction
import enum DLVM.ComparisonPredicate
import enum DLVM.AggregationFunction
import enum DLVM.ArithmeticOperator
import enum DLVM.ReductionFunction
import enum DLVM.BinaryIntegrationFunction

protocol ASTNode {
    var range: SourceRange { get }
}

struct ModuleNode : ASTNode {
    let name: String
    let declarations: [DeclarationNode]
    let basicBlocks: [BasicBlockNode]
    let range: SourceRange
}

struct DeclarationNode : ASTNode {
    enum Role {
        case input, output, parameter
    }
    let role: Role
    let type: TypeNode
    let shape: ShapeNode
    let name: String
    let initializer: InitializerNode?
    let range: SourceRange
}

struct ImmediateValueNode : ASTNode {
    let type: TypeNode
    let immediate: ImmediateNode

    var range: SourceRange {
        return type.range.lowerBound..<immediate.range.upperBound
    }
}

enum InitializerNode : ASTNode {
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
    case matrixMultiply(OperandNode, OperandNode, SourceRange)
    case tensorMultiply(OperandNode, OperandNode, SourceRange)
    case arithmetic(ArithmeticOperator, OperandNode, OperandNode, SourceRange)
    case binaryReduction(BinaryIntegrationFunction, OperandNode, OperandNode, SourceRange)
    case reduce(ReductionFunction, OperandNode, SourceRange)
    case elementwise(ElementwiseFunction, OperandNode, SourceRange)
    case aggregate(AggregationFunction, OperandNode, SourceRange)
    case comparison(ComparisonPredicate, OperandNode, OperandNode, SourceRange)
    case concatenate([OperandNode], Int?, SourceRange)
    case shapeCast(OperandNode, ShapeNode, SourceRange)
    case typeCast(OperandNode, TypeNode, SourceRange)
    case load(OperandNode, SourceRange)
    case store(OperandNode, OperandNode, SourceRange)

    var range: SourceRange {
        switch self {
        case let .matrixMultiply(_, _, sr),
             let .tensorMultiply(_, _, sr),
             let .arithmetic(_, _, _, sr),
             let .binaryReduction(_, _, _, sr),
             let .reduce(_, _, sr),
             let .elementwise(_, _, sr),
             let .aggregate(_, _, sr),
             let .comparison(_, _, _, sr),
             let .concatenate(_, _, sr),
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

