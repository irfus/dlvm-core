//
//  Utilities.swift
//  DLVM
//
//  Created by Richard Wei on 3/29/17.
//
//

import Foundation

func DLUnimplemented(_ function: String = #function, file: StaticString = #file, line: UInt = #line) -> Never {
    fatalError("\(function) is not fully implemented. \(file):\(line)")
}
