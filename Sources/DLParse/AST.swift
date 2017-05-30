//
//  AST.swift
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

import CoreTensor
import DLVM
import struct Parsey.SourceRange

protocol ASTNode {
    var range: SourceRange { get }
}

public indirect enum LiteralNode {
    case undefined
    case null
    case zero
    case scalar(Literal.Scalar)
    case tensor([UseNode])
    case tuple([UseNode])
    case array([UseNode])
    case `struct`([(String, UseNode)])
}

public struct UseNode : ASTNode {
    public enum Kind {
        case argument(String)
        case temporary(String)
        case global(String)
        case literal(LiteralNode)
        case function(String)
        case constant(InstructionKind)
    }
    public let type: TypeNode
    public let kind: Kind
    public let range: SourceRange
}

public indirect enum TypeNode {
    case tensor(TensorShape, DataType, SourceRange)
    case array(Int, TypeNode, SourceRange)
    case tuple([TypeNode], SourceRange)
    case pointer(TypeNode, SourceRange)
    case box(TypeNode, SourceRange)
    case function([TypeNode], TypeNode, SourceRange)
    case nominal(String, SourceRange)
    case void(SourceRange)
    case invalid(SourceRange)
}

extension TypeNode : ASTNode {
    var range: SourceRange {
        switch self {
        case .tensor(_, _, let sr),
             .array(_, _, let sr),
             .tuple(_, let sr),
             .pointer(_, let sr),
             .box(_, let sr),
             .function(_, _, let sr),
             .nominal(_, let sr),
             .void(let sr),
             .invalid(let sr):
            return sr
        }
    }
}

public struct InstructionNode : ASTNode {
    public let opcode: InstructionKind.Opcode
    public let range: SourceRange
}

