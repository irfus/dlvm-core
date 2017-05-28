//
//  AST.swift
//  DLVM
//
//  Created by Richard Wei on 5/25/17.
//
//

import CoreTensor
import DLVM
import struct Parsey.SourceRange

protocol ASTNode {
    var range: SourceRange { get }
}

