//
//  Literal.swift
//  DLVM
//
//  Created by Richard Wei on 3/16/17.
//
//

/// Scalar literal
public enum ScalarLiteral {
    case int(IntegerLiteralType)
    case float(FloatLiteralType)
    case bool(BooleanLiteralType)

    public var typeBase: DataType.Base {
        switch self {
        case .bool: return .bool
        case .int: return .int
        case .float: return .float
        }
    }
}

extension ScalarLiteral : Equatable {
    public static func == (lhs: ScalarLiteral, rhs: ScalarLiteral) -> Bool {
        switch (lhs, rhs) {
        case let (.int(i1), .int(i2)): return i1 == i2
        case let (.float(f1), .float(f2)): return f1 == f2
        case let (.bool(b1), .bool(b2)): return b1 == b2
        default: return false
        }
    }
}

/// Tensor literal
/// - TODO: Need verification
public enum TensorLiteral {
    case item(ScalarLiteral)
    case repeating(ScalarLiteral)
    case elements([TensorLiteral])

    public var typeBase: DataType.Base {
        switch self {
        case let .item(lit):
            return lit.typeBase
        case let .repeating(lit):
            return lit.typeBase
        case let .elements(elements):
            return elements[0].typeBase
        }
    }
}

extension TensorLiteral : Equatable {
    public static func == (lhs: TensorLiteral, rhs: TensorLiteral) -> Bool {
        switch (lhs, rhs) {
        case let (.elements(ll1), .elements(ll2)):
            return ll1 == ll2
        case let (.repeating(l1), .repeating(l2)):
            return l1 == l2
        case let (.item(l1), .item(l2)):
            return l1 == l2
        default:
            return false
        }
    }
}

/// Scalar or tensor literal, literally
/// - Note: It has no type or shape, because a `Literal` is not a `Value`.
/// But `LiteralValue`, that uses `Literal`, is a value.
public enum Literal {
    case undefined
    case zero
    case scalar(ScalarLiteral)
    case tensor(TensorLiteral)
    case tuple([LiteralValue])
    case array([LiteralValue])
    case function(Function)
    case globalValue(GlobalValue)
}

extension Literal : Equatable {
    public static func == (lhs: Literal, rhs: Literal) -> Bool {
        switch (lhs, rhs) {
        case (.zero, .zero), (.undefined, .undefined): return true
        case let (.scalar(s1), .scalar(s2)): return s1 == s2
        case let (.tensor(t1), .tensor(t2)): return t1 == t2
        case let (.tuple(tt1), .tuple(tt2)): return tt1 == tt2
        case let (.array(tt1), .array(tt2)): return tt1 == tt2
        case let (.function(f1), .function(f2)): return f1 === f2
        case let (.globalValue(v1), .globalValue(v2)): return v1 === v2
        default: return false
        }
    }
}

/// Literal value. It wraps `Literal` into a value
public struct LiteralValue : Value {
    public var type: Type
    public var literal: Literal

    public init(type: Type, literal: Literal) {
        self.type = type
        self.literal = literal
    }

    public func makeUse() -> Use {
        return .literal(type, self)
    }
}

extension LiteralValue : Equatable {
    public static func == (lhs: LiteralValue, rhs: LiteralValue) -> Bool {
        return lhs.type == rhs.type
            && lhs.literal == rhs.literal
    }
}

public extension LiteralValue {
    init(shape: TensorShape, dataType: DataType, repeating number: Int) {
        let lit: ScalarLiteral
        switch dataType.base {
        case .int: lit = .int(number)
        case .float: lit = .float(Double(number))
        case .bool: lit = .bool(number == 0 ? false : true)
        }
        let type: Type = .tensor(shape, dataType)
        self.init(type: type,
                  literal: shape.isScalar ? .scalar(lit) : .tensor(.repeating(lit)))
    }
}
