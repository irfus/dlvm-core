//
//  Utilities.swift
//  DLVM
//
//  Created by Richard Wei on 3/22/17.
//
//

import LLVM

/// Extremely sweet IR sugar
var i1: IntType { return .int1 }
var i8: IntType { return .int8 }
var i16: IntType { return .int16 }
var i32: IntType { return .int32 }
var i64: IntType { return .int64 }
var f32: FloatType { return .float }
var f64: FloatType { return .double }
var void: VoidType { return VoidType() }

postfix operator *
postfix func *(type: IRType) -> IRType {
    return PointerType(pointee: type)
}

postfix operator **
postfix func **(type: IRType) -> IRType {
    return PointerType(pointee: type*)
}

postfix operator ***
postfix func ***(type: IRType) -> IRType {
    return PointerType(pointee: type**)
}

infix operator =>
func => (lhs: [IRType], rhs: IRType) -> FunctionType {
    return FunctionType(argTypes: lhs, returnType: rhs)
}

extension Bool {
    var value: Constant<Signed> {
        return IntType.int1.constant(self ? 1 : 0)
    }
}

public protocol IRValueConvertible : RawRepresentable {
    associatedtype RawValue : SignedInteger
    var constantType: IntType { get }
}

public extension IRValueConvertible {
    var constant: IRValue {
        return constantType.constant(rawValue)
    }
}

import cllvm

extension StructType {
    /// Opaque type initialization
    init(name: String) {
        let type = LLVMStructCreateNamed(LLVMGetGlobalContext(), name)!
        self.init(llvm: type)
    }
}
