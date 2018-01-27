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

public indirect enum Use : Equatable {
    case argument(Type, Argument)
    case instruction(Type, Instruction)
    case variable(Type, Variable)
    case literal(Type, Literal)
    case function(Type, Function)
}

public extension Use {
    var type: Type {
        get {
            switch self {
            case .argument(let t, _),
                 .instruction(let t, _),
                 .variable(let t, _),
                 .literal(let t, _),
                 .function(let t, _):
                return t
            }
        }
        set(newType) {
            switch self {
            case let .argument(_, x):
                self = .argument(newType, x)
            case let .instruction(_, x):
                self = .instruction(newType, x)
            case let .variable(_, x):
                self = .variable(newType, x)
            case let .literal(_, x):
                self = .literal(newType, x)
            case let .function(_, x):
                self = .function(newType, x)
            }
        }
    }

    var tensorType: TensorType? {
        return type.tensorType
    }

    var value: Value {
        switch self {
        case let .argument(_, val): return val
        case let .variable(_, val): return val
        case let .instruction(_, val): return val
        case let .function(_, val): return val
        case let .literal(ty, lit): return LiteralValue(type: ty, literal: lit)
        }
    }

    var definition: Definition? {
        return value as? Definition
    }

    var name: String? {
        switch self {
        case let .variable(_, def): return def.name
        case let .instruction(_, def): return def.name
        case let .argument(_, def): return def.name
        case let .function(_, def): return def.name
        case .literal: return nil
        }
    }

    var instruction: Instruction? {
        guard case let .instruction(_, inst) = self else {
            return nil
        }
        return inst
    }
}

infix operator ~

public func ~ (lhs: Literal, rhs: Type) -> Use {
    return .literal(rhs, lhs)
}
