//
//  Types.swift
//  DLVM
//
//  Created by Richard Wei on 1/21/17.
//
//

/// This file contains the somewhat formal type(shape) system of TEL

///                  d1 : Int
///                  d2 : Int
///                 ... : Int
///                  dn : Int
/// ---------------------------
///  d1 x d2 x ... x dn : Shape

///                     s1 : Shape
///                     s2 : Shape
///                    ... : Shape
///                     sn : Shape
///                      r : Shape
/// ---------------------------------
/// (s1, s2, ..., sn) -> r : Function

import struct DLVM.TensorShape

public enum TypeSystemError : Error {
    case inconcreteType(DimensionVariable, DimensionExpression)
    case argumentMismatch([[GenericDimensionElement]], [TensorShape])
    case genericArgumentNotDefined(GenericVariable)
}

public class DimensionVariable : Hashable {
    public var identifier: String?

    public init(identifier: String? = nil) {
        self.identifier = identifier
    }

    public static func == (lhs: DimensionVariable, rhs: DimensionVariable) -> Bool {
        return lhs === rhs
    }

    public var hashValue: Int {
        return ObjectIdentifier(self).hashValue
    }
}

public class GenericVariable : Hashable {
    public var identifier: String?

    public init(identifier: String? = nil) {
        self.identifier = identifier
    }

    public static func == (lhs: GenericVariable, rhs: GenericVariable) -> Bool {
        return lhs === rhs
    }

    public var hashValue: Int {
        return ObjectIdentifier(self).hashValue
    }
}

public enum ArithmeticTypeOperator {
    case add, subtract, multiply, divide

    public var lowered: (Int, Int) -> Int {
        switch self {
        case .add:      return (+)
        case .subtract: return (-)
        case .multiply: return (*)
        case .divide:   return (/)
        }
    }
}

public enum DimensionElement {
    case number(Int)
    case variable(DimensionVariable)
}

public typealias GenericShapeEnvironment = [GenericVariable : [DimensionElement]]
public typealias DimensionEnvironment = [DimensionVariable : Int]

public indirect enum GenericDimensionElement {
    case number(Int)
    case variable(DimensionVariable)
    case genericVariable(GenericVariable)

    public func evaluated(in env: GenericShapeEnvironment) throws -> [DimensionElement] {
        switch self {
        case let .number(i):
            return [.number(i)]
        case let .variable(v):
            return [.variable(v)]
        case let .genericVariable(v):
            guard let dims = env[v] else {
                throw TypeSystemError.genericArgumentNotDefined(v)
            }
            return dims
        }
    }
}

public struct GenericShape {
    public var elements: [GenericDimensionElement]
}

public indirect enum DimensionExpression {
    case dimension(Int)
    case variable(DimensionVariable)
    case arithmetic(ArithmeticTypeOperator, DimensionExpression, DimensionExpression)

    public func evaluated(in env: DimensionEnvironment) throws -> Int {
        switch self {
        case let .dimension(dim):
            return dim
        case let .variable(v):
            guard let val = env[v] else { throw TypeSystemError.inconcreteType(v, self) }
            return val
        case let .arithmetic(op, lhs, rhs):
            let lhsVal = try lhs.evaluated(in: env), rhsVal = try rhs.evaluated(in: env)
            return op.lowered(lhsVal, rhsVal)
        }
    }
}

public struct ShapeExpression {
    public var dimensions: [DimensionExpression]

    public func evaluated(in env: DimensionEnvironment) throws -> TensorShape {
        let dimensions = try self.dimensions.flatMap { try $0.evaluated(in: env) }
        return TensorShape(dimensions)
    }
}

public struct FunctionType {
    public var generics: [GenericVariable]
    public var arguments: [[GenericDimensionElement]]
    public var resultExpression: ShapeExpression
}

public extension FunctionType {
    /// Example: tmul :: <A, C> ([AxB], [BxC]) -> [AxC]
    public func result(forArguments actualArgs: [TensorShape]) throws -> TensorShape {
        guard actualArgs.count == arguments.count else {
            throw TypeSystemError.argumentMismatch(arguments, actualArgs)
        }
        fatalError("Unimplemented")
    }
}

public func withDimensionVariable<Result>(
    _ body: (DimensionVariable) throws -> Result) rethrows -> Result {
    return try body(DimensionVariable())
}
