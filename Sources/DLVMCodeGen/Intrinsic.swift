//
//  Intrinsic.swift
//  DLVM
//
//  Created by Richard Wei on 3/21/17.
//
//

import LLVM_C

public protocol IRIntrinsic : Hashable {
    var type: LLVMTypeRef { get }
}

public protocol Target {
    associatedtype Intrinsic : IRIntrinsic
    subscript(intrinsic: Intrinsic) -> LLVMValueRef { mutating get }
}

extension StaticString : Equatable {
    public static func == (lhs: StaticString, rhs: StaticString) -> Bool {
        return lhs.utf8Start == rhs.utf8Start
    }
}
