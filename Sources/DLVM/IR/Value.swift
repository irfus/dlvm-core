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
    static var scope: Scope { get }
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

/// Scalar or tensor literal, literally
/// - Note: It has no type or shape, because a `Literal` is not a `Value`.
/// But `LiteralValue`, that uses `Literal`, is a value.
public enum Literal {
    case scalar(ScalarLiteral)
    case tensor(TensorLiteral)
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

/// Literal value. It wraps `Literal` into a value
public struct LiteralValue : SimpleValue {
    public var shape: TensorShape
    public var dataType: DataType
    public var literal: Literal
    public static let scope: Scope = .none

    public init(shape: TensorShape, dataType: DataType, literal: Literal) {
        self.shape = shape
        self.dataType = dataType
        self.literal = literal
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

public protocol Definition : class, Named, Value {
    var name: String { get }
    var type: Type { get }
}

/// When a value has a name, it's a unique Def!
public class Def<ValueType : Value> : Named, Definition, Value, HashableByReference {
    public typealias UserType = Instruction
    public var name: String
    public var type: Type
    public var value: ValueType

    public static var scope: Scope {
        return ValueType.scope
    }
    
    public init(name: String, value: ValueType) {
        self.name = name
        self.type = value.type
        self.value = value
    }
}

/// A value is potentially recurrent if it's potentially recurrent
public protocol PotentiallyRecurrentValue : Value {
    var isRecurrent: Bool { get }
}

// MARK: - Recurrent value helper
public extension Def where ValueType : PotentiallyRecurrentValue {
    var isRecurrent: Bool {
        return value.isRecurrent
    }
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

// MARK: - Simple value accessor
public extension Def where ValueType : SimpleValue {
    var shape: TensorShape {
        return value.shape
    }

    var dataType: DataType {
        return value.dataType
    }

    func makeLiteral(_ integerLiteral: IntegerLiteralType) -> LiteralValue {
        return value.makeLiteral(integerLiteral)
    }

    func makeScalarLiteral(_ integerLiteral: IntegerLiteralType) -> LiteralValue {
        return value.makeScalarLiteral(integerLiteral)
    }
}

/// User, anything that can use a value
public protocol User {
    var operands: [Use] { get }
}
