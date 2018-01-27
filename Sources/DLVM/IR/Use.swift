//
//  Use.swift
//  DLVM
//
//  Copyright 2016-2018 The DLVM Team.
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

import CoreOp

public indirect enum Use : Equatable {
    case argument(Argument)
    case instruction(Instruction)
    case variable(Variable)
    case literal(Type, Literal)
    case function(Function)
}

public extension Use {
    var type: Type {
        get {
            switch self {
            case .argument(let x):
                return x.type
            case .instruction(let x):
                return x.type
            case .variable(let x):
                return x.type
            case .function(let x):
                return x.type
            case .literal(let t, _):
                return t
            }
        }
        set(newType) {
            switch self {
            case let .argument(x):
                self = .argument(x)
            case let .instruction(x):
                self = .instruction(x)
            case let .variable(x):
                self = .variable(x)
            case let .literal(_, x):
                self = .literal(newType, x)
            case let .function(x):
                self = .function(x)
            }
        }
    }

    var tensorType: TensorType? {
        return type.tensorType
    }

    var value: Value {
        switch self {
        case let .argument(val): return val
        case let .variable(val): return val
        case let .instruction(val): return val
        case let .function(val): return val
        case let .literal(ty, lit): return LiteralValue(type: ty, literal: lit)
        }
    }

    var definition: Definition? {
        return value as? Definition
    }

    var name: String? {
        switch self {
        case let .variable(def): return def.name
        case let .instruction(def): return def.name
        case let .argument(def): return def.name
        case let .function(def): return def.name
        case .literal: return nil
        }
    }

    var instruction: Instruction? {
        guard case let .instruction(inst) = self else {
            return nil
        }
        return inst
    }
}

infix operator ~

public func ~ (lhs: Literal, rhs: Type) -> Use {
    return .literal(rhs, lhs)
}
