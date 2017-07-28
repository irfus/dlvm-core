//
//  LLVMTypeHelper.swift
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

/// This file contains syntactic sugar for creating LLVM types for converting 
/// Swift values to LLVM constants

import DLVM
import LLVM_C

// MARK: - Operator declarations

/// For creating constants from Swift types
prefix operator %
/// For creating packed struct constants from Swift collections
prefix operator <%>
/// For creating constants of a specified type from Swift numeric values
infix operator ~
/// For creating LLVM types
prefix operator ^
/// For forming function types
infix operator =>
/// For creating packed struct types
prefix operator <^>
/// For creating pointers
postfix operator *
/// For creating second-order pointers
postfix operator **
/// For creating third-order pointers
postfix operator ***

// MARK: - LLVM primitive types

var i1: LLVMTypeRef { return LLVMInt1Type() }
var i8: LLVMTypeRef { return LLVMInt8Type() }
var i16: LLVMTypeRef { return LLVMInt16Type() }
var i32: LLVMTypeRef { return LLVMInt32Type() }
var i64: LLVMTypeRef { return LLVMInt64Type() }
var f16: LLVMTypeRef { return LLVMHalfType() }
var f32: LLVMTypeRef { return LLVMFloatType() }
var f64: LLVMTypeRef { return LLVMDoubleType() }
var void: LLVMTypeRef { return LLVMVoidType() }

// MARK: - Syntactic sugar for creating LLVM types

/// Create a first-order pointer of a type
postfix func *(type: LLVMTypeRef) -> LLVMTypeRef {
    return LLVMPointerType(type, LLAddressSpace.generic.rawValue)
}

/// Create a second-order pointer of a type
postfix func **(type: LLVMTypeRef) -> LLVMTypeRef {
    return LLVMPointerType(type*, LLAddressSpace.generic.rawValue)
}

/// Create a third-order pointer of a type
postfix func ***(type: LLVMTypeRef) -> LLVMTypeRef {
    return LLVMPointerType(type**, LLAddressSpace.generic.rawValue)
}

/// Create an array type
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

extension String {
    /// Make opaque type from string
    static prefix func ^ (value: String) -> LLVMTypeRef {
        return LLVMStructCreateNamed(LLVMGetGlobalContext(), value)
    }
}

extension Sequence where Iterator.Element == LLVMTypeRef {

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

// MARK: - Swift value - LLVM constant conversion

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

extension Float32 : LLConstantConvertible {
    var llType: LLVMTypeRef {
        return f32
    }

    var constant: LLVMValueRef {
        return LLVMConstReal(llType, Double(self))
    }
}

extension Float64 : LLConstantConvertible {
    var llType: LLVMTypeRef {
        return f64
    }

    var constant: LLVMValueRef {
        return LLVMConstReal(llType, Double(self))
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

// MARK: - DLVM DataType - LLVM type conversion
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

extension FixedWidthInteger where Self : SignedInteger {
    func llValue(ofType type: LLVMTypeRef) -> LLVMValueRef {
        return LLVMConstInt(LLVMIntType(UInt32(bitWidth)), UInt64(self),
                            self >= 0 ? .true : .false)
    }
}

extension Double {
    func llValue(ofType type: LLVMTypeRef) -> LLVMValueRef {
        return LLVMConstReal(type, self)
    }
}

// MARK: - Aggregate value factories

extension Sequence where Iterator.Element == LLVMValueRef {
    /// Create an array constant
    func array(ofType elementType: LLVMTypeRef) -> LLVMValueRef {
        var elements: [LLVMValueRef?] = Array(self)
        return LLVMConstArray(elementType, &elements, UInt32(elements.count))
    }

    /// Create a struct constant
    static prefix func % (values: Self) -> LLVMValueRef {
        var values: [LLVMValueRef?] = values.map{$0}
        return LLVMConstStruct(&values, UInt32(values.count), .false)
    }
    
    /// Create a packed struct constant
    static prefix func <%> (values: Self) -> LLVMValueRef {
        var values: [LLVMValueRef?] = values.map{$0}
        return LLVMConstStruct(&values, UInt32(values.count), .true)
    }
}
