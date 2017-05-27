//
//  AlgebraSimplification.swift
//  DLVM
//
//  Created by Richard Wei on 5/22/17.
//
//

import Foundation

/// Algebra Simplification simplifies the following expressions
/// - x^0 => 1
/// - x^1 => x
/// - x^2 => x * x
/// - (e^x - e^(-x)) / 2 => sinh(x)
/// - (e^x + e^(-x)) / 2 => cosh(x)
/// - (e^x - e^(-x)) / (e^x + e^(-x)) => tanh(x)
/// - sin(0) => 0
/// - cos(0) => 1
/// - tanh(0) => 0
open class AlgebraSimplification : TransformPass {
    public typealias Body = BasicBlock

    public typealias Substitution = (old: Use, new: Use)

    open class func run(on body: BasicBlock) throws -> Bool {
        var changed = false
        var newlyChanged: Bool
        var substWorkList: [Substitution] = []
        /// Repeat until no more changes
        repeat {
            let users = try body.parent.analysis(from: UserAnalysis.self)
            newlyChanged = false
            for inst in body {
                /// Do use-substitutions in the work list for this instruction
                for (old: old, new: new) in substWorkList {
                    inst.substitute(old, for: new)
                    /// - TODO: Handle broadcasting
                }
                /// Match patterns
                switch inst.kind {
                /// x^k
                case let .binary(.associative(.arithmetic(.power)), x, .literal(_,  lit), _):
                    switch lit {
                    case 0:
                        substWorkList.append((old: %inst, new: .scalar(.int(1)) ~ x.type))
                    case 1:
                        substWorkList.append((old: %inst, new: x))
                    case 2:
                        /// Replace with `multiply`
                        let mult = Instruction(kind: .matrixMultiply(x, x), parent: body)
                        body.insert(mult, before: inst)
                        substWorkList.append((old: %inst, new: %mult))
                    default:
                        continue
                    }
                /// sin(0)
                case let .unary(.sin, .literal(ty, 0)):
                    substWorkList.append((old: %inst, new: .scalar(.int(0)) ~ ty))
                /// cos(0)
                case let .unary(.cos, .literal(ty, 0)):
                    substWorkList.append((old: %inst, new: .scalar(.int(1)) ~ ty))
                /// tan(0)
                case let .unary(.tan, .literal(ty, 0)):
                    substWorkList.append((old: %inst, new: .scalar(.int(0)) ~ ty))
                /// (e^x - e^(-x)) / 2 => sinh(x)
                /// (e^x + e^(-x)) / 2 => cosh(x)
                case let .binary(.associative(.arithmetic(.divide)),
                                 .instruction(_, lhs), .literal(_, 2), _) where users[lhs].isEmpty:
                    switch lhs.kind {
                    case let .binary(.associative(.arithmetic(lhsOp)),
                                     .instruction(_, llhs), .instruction(_, lrhs), _)
                        where (lhsOp == .add || lhsOp == .subtract) &&  users[llhs].isEmpty && users[lrhs].isEmpty:
                        switch (llhs.kind, lrhs.kind) {
                        case (.unary(.exp, let x),
                              .unary(.exp, .instruction(_, let lrrhs)))
                            where users[lrrhs].isEmpty:
                            switch lrrhs.kind {
                            case .binary(.associative(.arithmetic(.subtract)), .literal(_, 0), x, _),
                                 .unary(.negate, x):
                                /// Insert sinh or cosh
                                let newOp: UnaryOp = lhsOp == .add ? .cos : .sin
                                let simplified = Instruction(kind: .unary(newOp, x), parent: body)
                                body.insert(simplified, before: inst)
                                /// Remove intermediate nodes
                                [lhs, llhs, lrhs, lrrhs].forEach { $0.removeFromParent() }
                            default: continue
                            }
                        default: continue
                        }
                    default: continue
                    }
                /// - TODO: More patterns
                default:
                    continue
                }
                inst.removeFromParent()
                newlyChanged = true
            }
            changed = changed || newlyChanged
        } while newlyChanged
        return changed
    }
}
