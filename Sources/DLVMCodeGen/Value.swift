//
//  Value.swift
//  DLVM
//
//  Created by Richard Wei on 5/21/17.
//
//

import LLVM_C

prefix operator %
prefix operator <%>
infix operator ~

extension Sequence where Iterator.Element == LLVMValueRef {
    func array(ofType elementType: LLVMTypeRef) -> LLVMValueRef {
        var elements: [LLVMValueRef?] = Array(self)
        return LLVMConstArray(elementType, &elements, UInt32(elements.count))
    }
    
    static prefix func % (values: Self) -> LLVMValueRef {
        var values: [LLVMValueRef?] = values.map{$0}
        return LLVMConstStruct(&values, UInt32(values.count), .false)
    }
    
    static prefix func <%> (values: Self) -> LLVMValueRef {
        var values: [LLVMValueRef?] = values.map{$0}
        return LLVMConstStruct(&values, UInt32(values.count), .true)
    }
}

#if swift(>=4.0)
extension FixedWidthInteger where Self : LLConstantConvertible {
    static func ~ (value: Self, type: LLVMTypeRef) -> LLVMValueRef {
        return LLVMConstInt(type, UInt64(value), Self.isSigned ? .true : .false)
    }
}
#else
extension SignedInteger where Self : LLConstantConvertible {
    static func ~ (value: Self, type: LLVMTypeRef) -> LLVMValueRef {
        return LLVMConstInt(type, UInt64(bitPattern: Int64(value.toIntMax())), .true)
    }
}
extension UnsignedInteger where Self : LLConstantConvertible {
    static func ~ (value: Self, type: LLVMTypeRef) -> LLVMValueRef {
        return LLVMConstInt(type, UInt64(value.toUIntMax()), .false)
    }
}
#endif
