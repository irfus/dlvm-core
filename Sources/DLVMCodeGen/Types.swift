//
//  Types.swift
//  DLVM
//
//  Created by Richard Wei on 3/25/17.
//
//

import DLVM
import LLVM

public extension TypeAlias {
    var llType: IRType {
        switch self {
        case let .opaque(name):
            return StructType(name: name)
        case let .transparent(_, ty):
            return ty.llType
        }
    }
}

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

public extension Type {
    var llType: IRType {
        switch self {
        case let .tensor(shape, dt):
            return shape.reduce(dt.llType, { acc, dim in
                ArrayType(elementType: acc, count: dim)
            })
        case .void:
            return VoidType()
        case let .pointer(subt):
            return PointerType(pointee: subt.llType)
        case .invalid:
            return VoidType()
        case let .tuple(subtt):
            return StructType(elementTypes: subtt.map{$0.llType})
        case let .array(subt, n):
            return ArrayType(elementType: subt.llType, count: n)
        case let .function(args, ret):
            return FunctionType(argTypes: args.map{$0.llType}, returnType: ret.llType)
        case let .alias(alias):
            return alias.llType
        }
    }
}
