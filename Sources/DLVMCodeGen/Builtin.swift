//
//  Builtin.swift
//  DLVM
//
//  Created by Richard Wei on 4/4/17.
//
//

import LLVM
import DLVM

public class Builtin {
    public var functions: [AnyHashable : LLVM.Function] = [:]
    public var module: LLVM.Module
    public required init(module: LLVM.Module) {
        self.module = module
    }
}

public extension Builtin {
    enum Memory {
        case memcpy(to: IRValue, from: IRValue, count: IRValue, align: IRValue, isVolatile: IRValue)
        case malloc(size: IRValue)
        case free(IRValue)
    }
}

extension Builtin.Memory : LLFunctionPrototype {
    public var name: StaticString {
        switch self {
        case .memcpy: return "llvm.memcpy.p0i8.p0i8.i64"
        case .malloc: return "malloc"
        case .free: return "free"
        }
    }

    public var arguments: [IRValue] {
        switch self {
        case let .free(val):
            return [val]
        case let .malloc(size: val):
            return [val]
        case let .memcpy(to: v1, from: v2, count: v3, align: v4, isVolatile: v5):
            return [v1, v2, v3, v4, v5]
        }
    }
    
    public var type: FunctionType {
        switch self {
        case .free: return [i8*] => void
        case .malloc: return [i64*] => i8*
        case .memcpy: return [i8*, i8*, i64, i32, i1] => void
        }
    }
}
