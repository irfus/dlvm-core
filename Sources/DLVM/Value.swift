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

public protocol Value : class {
    var name: String? { get set }
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

public enum Immediate {
    case int(Int), float(Float), bool(Bool)
}

public class Input : Value {
    public var name: String?
    public var type: DataType

    public init(type: DataType) {
        self.type = type
    }
}

public class Parameter : Value {
    public var name: String?
    public var type: DataType

    fileprivate init(type: DataType) {
        self.type = type
    }
}

///
/// Constant types
///

public class ScalarParameter : Parameter {
    public var immediate: Immediate

    public init(type: DataType, immediate: Immediate) {
        self.immediate = immediate
        super.init(type: type)
    }
}

public class TensorParameter : Parameter {
    public enum Initializer {
        case randomized(Immediate, Immediate)
        case repeated(Immediate)
        case elements([Immediate])
    }

    public var initializer: Initializer
    
    public init(type: DataType, initializer: Initializer) {
        self.initializer = initializer
        super.init(type: type)
    }
}
