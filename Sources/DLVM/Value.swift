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
}

public extension Value {
    public var isTensor: Bool {
        return type is TensorType
    }

    public var isScalar: Bool {
        return type is ScalarType
    }
}

public protocol NamedValue : Value {
    var name: String { get set }
}

public protocol GlobalValue : NamedValue, IRObject {
    typealias Parent = Module
}

public class Input : GlobalValue {
    public var name: String
    public var type: DataType
    public var isRecurrent: Bool = false
    public weak var parent: Module?

    public init(name: String, type: DataType) {
        self.name = name
        self.type = type
    }
}

public class Parameter : GlobalValue {
    public var name: String
    public var type: DataType
    public var initializer: Initializer
    public weak var parent: Module?

    public init(name: String, type: DataType, initializer: Initializer) {
        self.name = name
        self.type = type
        self.initializer = initializer
    }
}

public class Output : GlobalValue {
    public var name: String
    public var type: DataType
    public var isRecurrent: Bool = false
    public weak var parent: Module?

    public init(name: String, type: DataType) {
        self.name = name
        self.type = type
    }
}

public protocol Initializer : TextOutputStreamable {
    var typeBase: TypeBase { get }
}

public struct ImmediateValue : Value {
    public var type: DataType
    public var immediate: Immediate

    public init(type: DataType, immediate: Immediate) {
        self.type = type
        self.immediate = immediate
    }
}

public enum Immediate : Initializer {
    case int(Int), float(Double), bool(Bool)

    public var typeBase: TypeBase {
        switch self {
        case .bool: return .bool
        case .int: return .int
        case .float: return .float
        }
    }
}

public enum TensorInitializer : Initializer {
    case elements([Immediate])
    case random(from: Immediate, to: Immediate)
    case repeating(Immediate)

    public var typeBase: TypeBase {
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
