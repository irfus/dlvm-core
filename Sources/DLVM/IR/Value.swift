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

/// Anything that has a name
public protocol Named {
    var name: String { get }
}

/// Anything that may have a name
public protocol MaybeNamed {
    var name: String? { get }
}

public protocol Definition : class, Value {
}

/// User, anything that can use a value
public protocol User {
    var operands: [Use] { get }
}
