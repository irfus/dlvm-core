//
//  LiteralBroadcastingPromotion.swift
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

open class LiteralBroadcastingPromotion : TransformPass {
    public typealias Body = BasicBlock

    open class func run(on body: BasicBlock) -> Bool {
        var changed = false
        for inst in body {
            /// Instruction must be broadcastable
            /// (currently only includes elementwise arithmetic ops)
            guard inst.kind.isElementwiseArithmetic else {
                continue
            }
            for var operand in inst.operands {
                /// Operand must have tensor type
                guard case let .tensor(_, dt) = operand.type else {
                    continue
                }
                /// Operand must be a literal or literal instruction
                switch operand {
                case .literal(_, .scalar(_)):
                    changed = true
                    operand.type = .scalar(dt)
                    inst.substitute(operand, for: operand)
                case .instruction(_, let i):
                    guard case let .literal(lit, ty) = i.kind,
                        case let .tensor(_, dt) = ty else {
                            break
                    }
                    changed = true
                    inst.substitute(.literal(.scalar(dt), lit), for: operand)
                default:
                    break
                }
            }
        }
        return changed
    }
}
