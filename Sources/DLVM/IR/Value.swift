//
//  Value.swift
//  DLVM
//
//  Created by Richard Wei on 1/10/17.
//
//

///
/// ## Analysis information protocols
///

public protocol Usee : class {
    var users: NamedObjectSet<Instruction> { get }
}

internal protocol ManagedUsee : Usee {
    var users: NamedObjectSet<Instruction> { get set }
    func addUser(_ user: Instruction)
    func removeUser(_ user: Instruction)
    func removeAllUsers()
}

internal extension ManagedUsee {
    func addUser(_ user: Instruction) {
        users.insert(user)
    }

    func removeUser(_ user: Instruction) {
        users.remove(user)
    }

    func removeAllUsers() {
        users.removeAll()
    }
}

public typealias User = Instruction

///
/// Base
///

public protocol ValueRepresentation : TextOutputStreamable {
    var shape: TensorShape { get set }
    var type: DataType { get set }
}

public protocol Value : ValueRepresentation, TextOutputStreamable {
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

public protocol Global : Named, TextOutputStreamable {
    weak var module: Module? { get }
}

public typealias NamedValue = Value & Named & Usee

public protocol GlobalValue : NamedValue, Global {
}

public protocol GlobalPlaceholder : Named, Usee, Global, ValueRepresentation {
    weak var module: Module? { get }
    var type: DataType { get set }
    var shape: TensorShape { get set }
}

public extension Value {
    public var isGlobal: Bool {
        return self is GlobalValue
    }
}

public final class Input : GlobalPlaceholder, ManagedUsee {
    public var name: String
    public var type: DataType
    public var shape: TensorShape
    public var isRecurrent: Bool = false
    public internal(set) var users: NamedObjectSet<Instruction> = []
    public internal(set) weak var module: Module?

    public init(name: String, type: DataType, shape: TensorShape) {
        self.name = name
        self.type = type
        self.shape = shape
    }
}

public final class Constant : GlobalValue, ManagedUsee {
    public var name: String
    public var type: DataType
    public var shape: TensorShape
    public var defaultInitializer: Initializer
    public internal(set) var users: NamedObjectSet<Instruction> = []
    public internal(set) weak var module: Module?

    public init(name: String, type: DataType, shape: TensorShape,
                defaultInitializer: Initializer) {
        self.name = name
        self.type = type
        self.shape = shape
        self.defaultInitializer = defaultInitializer
    }
}

public final class Parameter : GlobalValue, ManagedUsee {
    public var name: String
    public var type: DataType
    public var shape: TensorShape
    public weak var gradientValue: NamedValue?
    public var initializer: Initializer
    public internal(set) var users: NamedObjectSet<Instruction> = []
    public internal(set) weak var module: Module?

    public init(name: String, type: DataType,
                shape: TensorShape, initializer: Initializer) {
        self.name = name
        self.type = type
        self.shape = shape
        self.initializer = initializer
    }
}

public final class Output : GlobalPlaceholder, ManagedUsee {
    public var name: String
    public var type: DataType
    public var shape: TensorShape
    public weak var errorValue: NamedValue?
    public var isRecurrent: Bool = false
    public internal(set) var users: NamedObjectSet<Instruction> = []
    public internal(set) weak var module: Module?

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
