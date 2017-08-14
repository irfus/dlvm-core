//
//  LiteralBroadcastingPromotion.swift
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

open class LiteralBroadcastingPromotion : TransformPass {
    public typealias Body = BasicBlock

    open class func run(on body: BasicBlock) -> Bool {
        var changed = false
        for inst in body {
            /// `zipWith` is the only instruction kind supporting broadcasting
            guard case .numericBinary(let op, var lhs, var rhs) = inst.kind else {
                continue
            }
            /// Must have tensor type
            guard case let (.tensor(s1, dt1),
                            .tensor(s2, dt2)) = (lhs.type.canonical,
                                                 rhs.type.canonical) else
                { continue }
            /// If s1 != s2, it's either the case that this IR is malformed or 
            /// broadcasting is already used
            guard s1 == s2, !s1.isScalar else { continue }
            /// Broadcast the scalar side
            switch (lhs, rhs) {
            case (.literal(_, .scalar(_)), _):
                changed = true
                lhs.type = .scalar(dt1)
                inst.kind = .numericBinary(op, lhs, rhs)
            case (_, .literal(_, .scalar(_))):
                changed = true
                rhs.type = .scalar(dt2)
                inst.kind = .numericBinary(op, lhs, rhs)
            default:
                break
            }
        }
        return changed
    }
}
