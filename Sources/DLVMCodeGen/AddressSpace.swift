//
//  AddressSpace.swift
//  DLVM
//
//  Created by Richard Wei on 3/25/17.
//
//

import LLVM

/// Address space
public enum LLAddressSpace: Int {
    case generic = 0
    case global = 1
    case `internal` = 2
    case shared = 3
    case constant = 4
    case local = 5
}

// MARK: - Initializer with `LLAddressSpace`
public extension PointerType {
    public init(pointee: IRType, addressSpace: LLAddressSpace) {
        self.init(pointee: pointee, addressSpace: addressSpace.hashValue)
    }
}
