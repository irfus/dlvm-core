//
//  Value.swift
//  DLVM
//
//  Created by Richard Wei on 1/10/17.
//
//

@_exported import DLVMTensor

/// Scope of value
public enum Scope {
    case global
    case local
    case none
}

/// Value base
public protocol Value {
    var type: Type { get }
}

public protocol SimpleValue : Value {
    var shape: TensorShape { get }
    var dataType: DataType { get }
}

public extension SimpleValue {
    var type: Type {
        return .tensor(shape, dataType)
    }
}

/// Scalar literal
public enum ScalarLiteral {
    case int(Int), float(Double), bool(Bool)

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
public enum TensorLiteral {
    case elements([ScalarLiteral])
    case random(from: ScalarLiteral, to: ScalarLiteral)
    case repeating(ScalarLiteral)

    public var typeBase: DataType.Base {
        switch self {
        case let .elements(elements):
            return elements[0].typeBase
        case let .random(from: lowerbound, to: _):
            return lowerbound.typeBase
        case let .repeating(value):
            return value.typeBase
        }
    }
}

extension TensorLiteral : Equatable {
    public static func == (lhs: TensorLiteral, rhs: TensorLiteral) -> Bool {
        switch (lhs, rhs) {
        case let (.elements(ll1), .elements(ll2)):
            return ll1 == ll2
        case let (.random(from: lo1, to: hi1), .random(from: lo2, to: hi2)):
            return lo1 == lo2 && hi1 == hi2
        case let (.repeating(l1), .repeating(l2)):
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
    case scalar(ScalarLiteral)
    case tensor(TensorLiteral)
}

extension Literal : Equatable {
    public static func == (lhs: Literal, rhs: Literal) -> Bool {
        switch (lhs, rhs) {
        case let (.scalar(s1), .scalar(s2)): return s1 == s2
        case let (.tensor(t1), .tensor(t2)): return t1 == t2
        default: return false
        }
    }
}

/// Literal value. It wraps `Literal` into a value
public struct LiteralValue : SimpleValue {
    public var shape: TensorShape
    public var dataType: DataType
    public var literal: Literal

    public init(shape: TensorShape, dataType: DataType, literal: Literal) {
        self.shape = shape
        self.dataType = dataType
        self.literal = literal
    }
}

extension LiteralValue : Equatable {
    public static func == (lhs: LiteralValue, rhs: LiteralValue) -> Bool {
        return lhs.shape == rhs.shape
            && lhs.dataType == rhs.dataType
            && lhs.literal == rhs.literal
    }
}

/// Anything that has a name
public protocol Named {
    var name: String { get }
}

/// Anything that may have a name
public protocol MaybeNamed {
    var name: String? { get }
}

public protocol Definition : class, MaybeNamed, Value {
}

// MARK: - Value helper factories
public extension SimpleValue {
    /// Returns a zero value of the same shape and the same data type
    func makeLiteral(_ integerLiteral: IntegerLiteralType) -> LiteralValue {
        let literal: Literal
        switch (dataType.base, shape) {
        case (.bool, []):
            literal = .scalar(.bool(integerLiteral != 0))
        case (.float, []):
            literal = .scalar(.float(Double(integerLiteral)))
        case (.int, []):
            literal = .scalar(.int(integerLiteral))
        case (.bool, _):
            literal = .tensor(.repeating(.bool(integerLiteral != 0)))
        case (.float, _):
            literal = .tensor(.repeating(.float(Double(integerLiteral))))
        case (.int, _):
            literal = .tensor(.repeating(.int(integerLiteral)))
        }
        return LiteralValue(shape: shape, dataType: dataType, literal: literal)
    }

    /// Returns a zero value of the same shape and the same data type
    func makeScalarLiteral(_ integerLiteral: IntegerLiteralType) -> LiteralValue {
        let literal: Literal
        switch dataType.base {
        case .bool:
            literal = .scalar(.bool(integerLiteral != 0))
        case .float:
            literal = .scalar(.float(Double(integerLiteral)))
        case .int:
            literal = .scalar(.int(integerLiteral))
        }
        return LiteralValue(shape: .scalar, dataType: dataType, literal: literal)
    }
}

/// User, anything that can use a value
public protocol User {
    var operands: [Use] { get }
}
