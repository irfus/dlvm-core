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

extension Sequence where Element == LLVMValueRef {
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

extension FixedWidthInteger where Self : LLConstantConvertible {
    static func ~ (value: Self, type: LLVMTypeRef) -> LLVMValueRef {
        return LLVMConstInt(type, UInt64(value), Self.isSigned ? .true : .false)
    }
}
