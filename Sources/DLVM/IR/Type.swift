//
// Created by Richard Wei on 3/10/17.
//

import DLVMTensor

public indirect enum Type {
    case tensor(TensorShape, DataType)
    case array(Type, Int)
    case tuple([Type])
    case pointer(Type)
    case void
    case invalid
    case `struct`(StructType)
}

public extension Type {
    var isTensor: Bool {
        switch self {
        case .tensor: return true
        default: return false
        }
    }

    var isScalar: Bool {
        switch self {
        case .tensor([], _): return true
        default: return false
        }
    }

    var isVoid: Bool {
        switch self {
        case .void: return true
        default: return false
        }
    }
}

extension Type : Equatable {
    public static func ==(lhs: Type, rhs: Type) -> Bool {
        switch (lhs, rhs) {
        case let (.tensor(s1, t1), .tensor(s2, t2)):
            return s1 == s2 && t1 == t2
        case let (.tuple(ts1), .tuple(ts2)):
            return ts1 == ts2
        default:
            return false
        }
    }
}

// MARK: - Validation
public extension Type {
    public var isValid: Bool {
        switch self {
        case .invalid:
            return false
        case .tensor, .void:
            return true
        case let .array(subtype, _),
             let .pointer(subtype):
            return subtype.isValid
        case let .tuple(subtypes):
            return subtypes.reduce(true, { $0 && $1.isValid })
        case let .struct(st):
            return st.fields.reduce(true, { $0 && $1.isValid })
        }
    }
}
