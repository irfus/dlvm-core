//
//  Value.swift
//  DLVM
//
//  Created by Richard Wei on 1/10/17.
//
//

///
/// Base
///

public protocol Value : TextOutputStreamable {
    var type: DataType { get set }
    var shape: TensorShape { get set }
}

public extension Value {
    public var isMatrix: Bool {
        return shape.isMatrix
    }

    public var isVector: Bool {
        return shape.isVector
    }

    public var isScalar: Bool {
        return shape.isScalar
    }
}

public protocol Named {
    var name: String { get set }
}

public protocol NamedValue : class, Value, Named {
    var users: NamedObjectSet<Instruction> { get }
}

public protocol GlobalValue : NamedValue {
    weak var parent: Module? { get }
}

extension Value {
    public var isGlobal: Bool {
        return self is GlobalValue
    }
}

public final class Input : GlobalValue {
    public var name: String
    public var type: DataType
    public var shape: TensorShape
    public var isRecurrent: Bool = false
    public var users: NamedObjectSet<Instruction> = []
    public internal(set) weak var parent: Module?

    public init(name: String, type: DataType, shape: TensorShape) {
        self.name = name
        self.type = type
        self.shape = shape
    }
}

public final class Parameter : GlobalValue {
    public var name: String
    public var type: DataType
    public var shape: TensorShape
    public var initializer: Initializer
    public var users: NamedObjectSet<Instruction> = []
    public internal(set) weak var parent: Module?

    public init(name: String, type: DataType,
                shape: TensorShape, initializer: Initializer) {
        self.name = name
        self.type = type
        self.shape = shape
        self.initializer = initializer
    }
}

public final class Output : GlobalValue {
    public var name: String
    public var type: DataType
    public var shape: TensorShape
    public var isRecurrent: Bool = false
    public var users: NamedObjectSet<Instruction> = []
    public internal(set) weak var parent: Module?

    public init(name: String, type: DataType, shape: TensorShape) {
        self.name = name
        self.type = type
        self.shape = shape
    }
}

public protocol Initializer : TextOutputStreamable {
    var typeBase: DataType.Base { get }
}

public struct ImmediateValue : Value {
    public var type: DataType
    public var shape: TensorShape
    public var immediate: Immediate

    public init(type: DataType, shape: TensorShape = .scalar, immediate: Immediate) {
        self.type = type
        self.shape = shape
        self.immediate = immediate
    }
}

public enum Immediate : Initializer {
    case int(Int), float(Double), bool(Bool)

    public var typeBase: DataType.Base {
        switch self {
        case .bool: return .bool
        case .int: return .int
        case .float: return .float
        }
    }
}

public enum TensorInitializer : Initializer {
    case elements([ImmediateValue])
    case random(from: ImmediateValue, to: ImmediateValue)
    case repeating(ImmediateValue)

    public var typeBase: DataType.Base {
        switch self {
        case let .elements(elements):
            return elements[0].type.base
        case let .random(from: lowerbound, to: _):
            return lowerbound.type.base
        case let .repeating(value):
            return value.type.base
        }
    }
}
