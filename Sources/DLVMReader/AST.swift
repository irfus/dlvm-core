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
import enum DLVM.LogicOperator
import enum DLVM.ReductionFunction
import enum DLVM.BinaryIntegrationFunction

public protocol ASTNode {
    var range: SourceRange { get }
}

public struct ModuleNode : ASTNode {
    public let name: String
    public let declarations: [DeclarationNode]
    public let basicBlocks: [BasicBlockNode]
    public let range: SourceRange
}

public struct DeclarationNode : ASTNode {
    public enum Role {
        case input, output, parameter, constant
    }
    public let role: Role
    public let type: TypeNode
    public let shape: ShapeNode?
    public let name: String
    public let initializer: InitializerNode?
    public let range: SourceRange
}

public struct ImmediateValueNode : ASTNode {
    public let type: TypeNode
    public let immediate: ImmediateNode

    public var range: SourceRange {
        return type.range.lowerBound..<immediate.range.upperBound
    }
}

public enum InitializerNode : ASTNode {
    case immediate(ImmediateValueNode, SourceRange)
    case random(ImmediateValueNode, ImmediateValueNode, SourceRange)
    case repeating(ImmediateValueNode, SourceRange)

    public var range: SourceRange {
        switch self {
        case let .immediate(_, sr), let .random(_, _, sr), let .repeating(_, sr):
            return sr
        }
    }
}

public struct BasicBlockNode : ASTNode {
    public let name: String
    public let extensionTypeName: String?
    public let instructions: [InstructionDeclarationNode]
    public let range: SourceRange
}

public struct ShapeNode : ASTNode {
    public let dimensions: [Int]
    public let range: SourceRange
}

public struct InstructionDeclarationNode : ASTNode {
    public let name: String?
    public let instruction: InstructionNode
    public let range: SourceRange
}

public enum LoopConditionNode : ASTNode {
    case times(OperandNode, SourceRange)
    case untilEqual(OperandNode, OperandNode, SourceRange)

    public var range: SourceRange {
        switch self {
        case .times(_, let sr), .untilEqual(_, _, let sr):
            return sr
        }
    }
}

public enum InstructionNode : ASTNode {
    case matrixMultiply(OperandNode, OperandNode, SourceRange)
    case tensorMultiply(OperandNode, OperandNode, SourceRange)
    case arithmetic(ArithmeticOperator, OperandNode, OperandNode, SourceRange)
    case logic(LogicOperator, OperandNode, OperandNode, SourceRange)
    case binaryReduction(BinaryIntegrationFunction, OperandNode, OperandNode, SourceRange)
    case reduce(ReductionFunction, OperandNode, SourceRange)
    case elementwise(ElementwiseFunction, OperandNode, SourceRange)
    case aggregate(AggregationFunction, OperandNode, SourceRange)
    case comparison(ComparisonPredicate, OperandNode, OperandNode, SourceRange)
    case concatenate([OperandNode], Int?, SourceRange)
    case shapeCast(OperandNode, ShapeNode, SourceRange)
    case typeCast(OperandNode, TypeNode, SourceRange)
    case load(OperandNode, SourceRange)
    case export(OperandNode, OperandNode, SourceRange)
    case store(OperandNode, OperandNode, SourceRange)
    case loop(BasicBlockNode, LoopConditionNode, SourceRange)

    public var range: SourceRange {
        switch self {
        case let .matrixMultiply(_, _, sr),
             let .tensorMultiply(_, _, sr),
             let .arithmetic(_, _, _, sr),
             let .logic(_, _, _, sr),
             let .binaryReduction(_, _, _, sr),
             let .reduce(_, _, sr),
             let .elementwise(_, _, sr),
             let .aggregate(_, _, sr),
             let .comparison(_, _, _, sr),
             let .concatenate(_, _, sr),
             let .shapeCast(_, _, sr),
             let .typeCast(_, _, sr),
             let .load(_, sr),
             let .store(_, _, sr),
             let .export(_, _, sr),
             let .loop(_, _, sr):
            return sr
        }
    }
}

public enum ImmediateNode : ASTNode {
    case bool(Bool, SourceRange), int(Int, SourceRange), float(Double, SourceRange)

    public var range: SourceRange {
        switch self {
        case .bool(_, let sr), .int(_, let sr), .float(_, let sr):
            return sr
        }
    }
}

public struct OperandNode : ASTNode {
    public let type: TypeNode
    public let shape: ShapeNode?
    public let variable: VariableNode

    public var range: SourceRange {
        return type.range.lowerBound..<variable.range.upperBound
    }
}

public enum VariableNode : ASTNode {
    case immediate(ImmediateNode, SourceRange)
    case input(String, SourceRange)
    case constant(String, SourceRange)
    case output(String, SourceRange)
    case parameter(String, SourceRange)
    case temporary(String, SourceRange)

    public var range: SourceRange {
        switch self {
        case let .immediate(_, sr),
             let .input(_, sr),
             let .constant(_, sr),
             let .output(_, sr),
             let .parameter(_, sr),
             let .temporary(_, sr):
            return sr
        }
    }
}

public enum TypeNode : ASTNode {
    case bool(Int, SourceRange), int(Int, SourceRange), float(Int, SourceRange)

    public var range: SourceRange {
        switch self {
        case .bool(_, let sr), .int(_, let sr), .float(_, let sr):
            return sr
        }
    }
}

