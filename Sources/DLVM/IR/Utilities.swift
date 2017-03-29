//
//  Utilities.swift
//  DLVM
//
//  Created by Richard Wei on 3/29/17.
//
//

import Foundation

// MARK: - Impossible case crasher
@discardableResult
func DLImpossibleResult<T>() -> T {
    fatalError("Impossible case \(T.self). Something's wrong in Core DLVM")
}

func DLImpossible() {
    let _: () = DLImpossibleResult()
}

@discardableResult
func DLUnimplemented<T>() -> T {
    fatalError("Unimplemented result \(T.self)")
}
