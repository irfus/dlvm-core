//
//  Types.swift
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

import DLVM
import LLVM_C

/// Primitive scalar types
var i1: LLVMTypeRef { return LLVMInt1Type() }
var i8: LLVMTypeRef { return LLVMInt8Type() }
var i16: LLVMTypeRef { return LLVMInt16Type() }
var i32: LLVMTypeRef { return LLVMInt32Type() }
var i64: LLVMTypeRef { return LLVMInt64Type() }
var f16: LLVMTypeRef { return LLVMHalfType() }
var f32: LLVMTypeRef { return LLVMFloatType() }
var f64: LLVMTypeRef { return LLVMDoubleType() }
var void: LLVMTypeRef { return LLVMVoidType() }

/// Pointer type sugar
postfix operator *
postfix operator **
postfix operator ***
postfix func *(type: LLVMTypeRef) -> LLVMTypeRef {
    return LLVMPointerType(type, LLAddressSpace.generic.rawValue)
}
postfix func **(type: LLVMTypeRef) -> LLVMTypeRef {
    return LLVMPointerType(type*, LLAddressSpace.generic.rawValue)
}
postfix func ***(type: LLVMTypeRef) -> LLVMTypeRef {
    return LLVMPointerType(type**, LLAddressSpace.generic.rawValue)
}

/// Array type sugar
func * (lhs: Int, rhs: LLVMTypeRef) -> LLVMTypeRef {
    return LLVMArrayType(rhs, UInt32(lhs))
}

extension LLVMBool {
    static var `true`: LLVMBool {
        return 1
    }
    static var `false`: LLVMBool {
        return 0
    }
}

prefix operator ^

extension String {
    /// Make opaque type
    static prefix func ^ (value: String) -> LLVMTypeRef {
        return LLVMStructCreateNamed(LLVMGetGlobalContext(), value)
    }
}

infix operator =>
prefix operator <^>

// MARK: - Opaque type initializer
extension Sequence where Element == LLVMTypeRef {

    /// Form a function type
    static func => (lhs: Self, rhs: LLVMTypeRef) -> LLVMTypeRef {
        var lhs: [LLVMTypeRef?] = lhs.map{$0}
        return LLVMFunctionType(rhs, &lhs, UInt32(lhs.count), .true)
    }

    /// Form an unpacked struct type
    static prefix func ^ (fields: Self) -> LLVMTypeRef {
        var fields: [LLVMTypeRef?] = fields.map{$0}
        return LLVMStructType(&fields, UInt32(fields.count), .false)
    }
    
    /// Form a packed struct type
    static prefix func <^> (fields: Self) -> LLVMTypeRef {
        var fields: [LLVMTypeRef?] = fields.map{$0}
        return LLVMStructType(&fields, UInt32(fields.count), .true)
    }
}


// MARK: - Constant convertible
protocol LLConstantConvertible {
    var llType: LLVMTypeRef { get }
    var constant: LLVMValueRef { get }
}

extension LLConstantConvertible {
    static prefix func %(value: Self) -> LLVMValueRef {
        return value.constant
    }
}

extension LLConstantConvertible where Self : FixedWidthInteger {
    var llType: LLVMTypeRef {
        return LLVMIntType(UInt32(bitWidth))
    }

    var constant: LLVMValueRef {
        return LLVMConstInt(llType, UInt64(bitPattern: Int64(self)),
                            Self.isSigned ? .true : .false)
    }
}

extension LLConstantConvertible where Self : RawRepresentable, Self.RawValue : LLConstantConvertible {
    var llType: LLVMTypeRef {
        return rawValue.llType
    }

    var constant: LLVMValueRef {
        return %rawValue
    }
}

extension Int : LLConstantConvertible {
    var llType: LLVMTypeRef {
        return i64
    }

    var constant: LLVMValueRef {
        return LLVMConstInt(llType, UInt64(bitPattern: Int64(self)), .true)
    }
}

extension Int16 : LLConstantConvertible {}
extension Int32 : LLConstantConvertible {}
extension Int64 : LLConstantConvertible {}
extension UInt16 : LLConstantConvertible {}
extension UInt32 : LLConstantConvertible {}
extension UInt64 : LLConstantConvertible {}

extension Bool : LLConstantConvertible {
    var llType: LLVMTypeRef {
        return i1
    }

    var constant: LLVMValueRef {
        return LLVMConstInt(i1, self ? 1 : 0, .false)
    }
}

extension DataType {
    var llType: LLVMTypeRef {
        switch self {
        case .bool: return i1
        case .int(let w): return LLVMIntType(UInt32(w))
        case .float(.half): return LLVMHalfType()
        case .float(.single): return LLVMFloatType()
        case .float(.double): return LLVMDoubleType()
        }
    }
}
