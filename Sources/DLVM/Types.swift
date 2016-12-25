//
//  Types.swift
//  DLVM
//
//  Created by Richard Wei on 12/20/16.
//
//

public enum DataType {
    case int8, int16, int32, int64
    case float8, float16, float32, float64

    public var isInt: Bool {
        switch self {
        case .int8, .int16, .int32, .int64:
            return true
        default:
            return false
        }
    }

    public var isFloat: Bool {
        return !isInt
    }
}

public enum ScalarType {
    case bool, int, float
}

public protocol TensorBase {
    static var dataType: DataType { get }
}
