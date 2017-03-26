//
//  Utilities.swift
//  DLVM
//
//  Created by Richard Wei on 3/22/17.
//
//

import LLVM

/// Primitive scalar types
var i1: IntType { return .int1 }
var i8: IntType { return .int8 }
var i16: IntType { return .int16 }
var i32: IntType { return .int32 }
var i64: IntType { return .int64 }
var f16: FloatType { return .half }
var f32: FloatType { return .float }
var f64: FloatType { return .double }
var void: VoidType { return VoidType() }

/// Pointer type sugar
postfix operator *
postfix operator **
postfix operator ***
postfix func *(type: IRType) -> IRType {
    return PointerType(pointee: type)
}
postfix func **(type: IRType) -> IRType {
    return PointerType(pointee: type*)
}
postfix func ***(type: IRType) -> IRType {
    return PointerType(pointee: type**)
}

/// Form a function type
infix operator =>
func => (lhs: [IRType], rhs: IRType) -> FunctionType {
    return FunctionType(argTypes: lhs, returnType: rhs)
}

import func cllvm.LLVMGetGlobalContext
import func cllvm.LLVMStructCreateNamed

// MARK: - Opaque type initializer
extension StructType {
    init(name: String) {
        let type = LLVMStructCreateNamed(LLVMGetGlobalContext(), name)!
        self.init(llvm: type)
    }
}

// MARK: - Impossible case crasher
/// - Note: This is called when LLGen encounters ill-formed DLVM IR.
/// All well-formedness should be checked by the module verifier, not by LLGen
@discardableResult
func DLImpossibleResult<T>() -> T {
    fatalError("Impossible case \(T.self). Something's wrong in CodeGen")
}

func DLImpossible() {
    let _: () = DLImpossibleResult()
}
