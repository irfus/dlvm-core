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
    var shape: TensorShape? { get set }
}

public extension Value {

    public var isTensor: Bool {
        return shape != nil
    }

    public var isScalar: Bool {
        return shape == nil
    }

}

public enum Immediate {
    case int(Int), float(Float), bool(Bool)
}

public protocol Constant : Value {

}

///
/// Constant types
///

public class ConstantScalar : Constant {
    public var name: String? = nil
    public var type: DataType
    public var shape: TensorShape? = nil
    public var immediate: Immediate

    public init(type: DataType, immediate: Immediate) {
        self.type = type
        self.immediate = immediate
    }
}

public enum TensorInitializer {
    case randomized(Immediate, Immediate)
    case repeated(Immediate)
    case elements([Immediate])
}

public class ConstantTensor : Constant {
    public var name: String? = nil
    public var shape: TensorShape?
    public var type: DataType
    public var initializer: TensorInitializer


    public init(type: DataType, shape: TensorShape,
                initializer: TensorInitializer) {
        self.type = type
        self.shape = shape
        self.initializer = initializer
    }
}
