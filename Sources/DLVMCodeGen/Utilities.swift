//
//  Utilities.swift
//  DLVM
//
//  Copyright 2016-2017 Richard Wei.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
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

    static var empty: StructType {
        return StructType(elementTypes: [])
    }
}

// MARK: - Impossible case crasher
/// - Note: This is called when LLGen encounters ill-formed DLVM IR.
/// All well-formedness should be checked by the module verifier, not by LLGen
@discardableResult
func DLImpossibleResult<T>(function: String = #function,
                           file: StaticString = #file,
                           line: UInt = #line) -> T {
    fatalError("Impossible case \(T.self). Something's wrong before LLGen")
}

func DLImpossible(function: String = #function,
                  file: StaticString = #file,
                  line: UInt = #line) -> Never {
    let _: () = DLImpossibleResult(function: function, file: file, line: line)
    fatalError()
}

import Foundation

public func environmentVariable(named name: String) -> String? {
    return ProcessInfo.processInfo.environment[name]
}
