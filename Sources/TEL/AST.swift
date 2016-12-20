//
//  AST.swift
//  DLVM
//
//  Created by Richard Wei on 12/20/16.
//
//

import enum DLVM.DataType
import struct DLVM.TensorShape

public enum AST {

    enum Macro {
        case type(DataType)
    }

    struct TensorType {
        enum Role {
            case input, output, hidden
        }
        var role: Role
        var shape: [Int]
    }
    
}
