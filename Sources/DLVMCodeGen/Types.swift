//
//  Types.swift
//  DLVM
//
//  Created by Richard Wei on 3/25/17.
//
//

import DLVM
import LLVM

public extension DataType {
    var llType: IRType {
        switch self {
        case .bool: return i1
        case let .int(w): return IntType(width: w)
        case .float(.half): return FloatType.half
        case .float(.single): return FloatType.float
        case .float(.double): return FloatType.double
        }
    }
}

// MARK: - Boolean lowering
extension Bool {
    var constant: Constant<Signed> {
        return IntType.int1.constant(self ? 1 : 0)
    }
}

// MARK: - Constant convertible
/// - Note: This is used to create target-specific enum constants
public protocol LLConstantConvertible : RawRepresentable {
    associatedtype RawValue : SignedInteger
    var constantType: IntType { get }
}

public extension LLConstantConvertible {
    var constant: IRValue {
        return constantType.constant(rawValue)
    }
}
