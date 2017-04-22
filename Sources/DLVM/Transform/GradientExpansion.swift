//
//  GradientExpander.swift
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

/// WIP: Richard

/// Replace every `gradient` instruction to a `call` to a function that
/// produces the gradient
public class GradientExpansion: TransformPass<Module> {
    public override class func run(on module: Module) throws -> Bool {
        var changed = false
        /// Run analysis before the transformation
        /// Only to check if this pass has been previously run
        let globalGradInfo = try module.analysis(from: GlobalGradientAnalysis.self)

        var expanded: [Function : Function] = [:]

        for function in module {
            for instruction in function.instructions {
                if case let .gradient(.function(funcToDiff), from: diffIndex, wrt: varIndices) = instruction.kind,
                    funcToDiff.isDifferentiable {
                    if let _ = globalGradInfo.gradient(of: function, from: diffIndex, wrt: varIndices) {
                        continue
                    }
                }
            }
        }

            /// If function is not differentiable, do nothing
//            guard function.isDifferentiable else { continue }
//            /// If gradient function exists, do nothing
//            if let _ = globalGradInfo.gradient(of: function) { continue }
//            /// Clone function
////            let grad = function.makeClone(named: func)
//            /// Expand this function
//            try expand(function)
//            changed = true
        return changed
    }
}

/// - Note: A new approach of adjoint code generation will be used.
/// Function will first be cloned and then adjoint code gets generated in
/// a real adjoint fasion

// MARK: - Function cloning
/// - Note: Big, ugly, not-so-safe, imperative code written in 4 minutes
public extension Function {
    public func makeClone(named name: String) -> Function {
        precondition(!parent.containsElement(named: name),
                     "Module already contains a function with the same name")

        let newFunc = Function(name: name,
                               arguments: arguments.map{($0.name, $0.type)},
                               result: result,
                               attributes: attributes,
                               parent: parent)

        /// Mappings from old IR units to new IR units
        var newArgs: [Argument : Argument] = [:]
        var newBlocks: [BasicBlock : BasicBlock] = [:]
        var newInsts: [Instruction : Instruction] = [:]

        func newUse(from old: Use) -> Use {
            switch old {
            /// If recursion, change function to the new function
            case .function(self): return .function(newFunc)
            case .function, .global, .literal: return old
            case let .argument(t, arg): return .argument(t, newArgs[arg]!)
            case let .instruction(t, inst): return .instruction(t, newInsts[inst]!)
            }
        }

        /// Clone basic blocks
        for oldBB in self {
            /// Entry block is a special case which always exists in a function
            let newBB = oldBB.isEntry ? newFunc.entry : {
                let newBB = BasicBlock(name: oldBB.name,
                                       arguments: oldBB.arguments.map{($0.name, $0.type)},
                                       parent: newFunc)
                newFunc.append(newBB)
                return newBB
            }()

            /// Insert argument mappings
            for (oldArg, newArg) in zip(oldBB.arguments, newBB.arguments) {
                newArgs[oldArg] = newArg
            }
            newBlocks[oldBB] = newBB
        }

        /// Clone instructions
        for oldBB in self {
            let newBB = newBlocks[oldBB]!
            /// Clone instructions
            for oldInst in oldBB {
                let newInst = Instruction(name: oldInst.name, kind: oldInst.kind, parent: newBB)
                /// - Note: Slow but clean for now
                for oldUse in newInst.operands {
                    newInst.substitute(oldUse, for: newUse(from: oldUse))
                }
                /// If branching, switch old BBs to new BBs
                switch newInst.kind {
                case let .branch(dest, args):
                    newInst.kind = .branch(newBlocks[dest]!, args)
                case let .conditional(cond, thenBB, thenArgs, elseBB, elseArgs):
                    newInst.kind = .conditional(cond, newBlocks[thenBB]!, thenArgs,
                                                newBlocks[elseBB]!, elseArgs)
                default: break
                }
            }
        }

        return newFunc
    }
}

fileprivate extension Type {
    func makeLiteralValue(_ number: IntegerLiteralType) -> LiteralValue {
        switch canonical {
        case let .tensor(shape, type):
            return LiteralValue(shape: shape, dataType: type, repeating: number)
        case let .tuple(subtypes):
            let sublits = subtypes.map{$0.makeLiteralValue(number).makeUse()}
            return LiteralValue(type: self, literal: .tuple(sublits))
        case let .array(subtype, i):
            let sublit = subtype.makeLiteralValue(number).makeUse()
            let sublits = Array(repeating: sublit, count: i)
            return LiteralValue(type: self, literal: .array(sublits))
        case _:
            fatalError()
        }
    }
}

/// Integer literal builder based on the unsafe (but convenient)
/// assumption that the value is a tensor
fileprivate extension Value {
    func makeTensor(repeating repeatedValue: IntegerLiteralType) -> Use {
        let canType = type.canonical
        guard case let .tensor(shape, dtype) = canType else {
            preconditionFailure("\(self), a.k.a. \(canType), is not tensor")
        }
        let val = LiteralValue(shape: shape, dataType: dtype, repeating: repeatedValue)
        return val.makeUse()
    }

    func makeScalar(_ value: IntegerLiteralType) -> Use {
        let canType = type.canonical
        guard case let .tensor(_, dtype) = canType else {
            preconditionFailure("\(self), a.k.a. \(canType), is not tensor")
        }
        let val = LiteralValue(shape: .scalar, dataType: dtype, repeating: value)
        return val.makeUse()
    }
}

/// Integer literal builder based on the unsafe (but convenient)
/// assumption that the value is a tensor
fileprivate extension Use {
    func makeTensor(repeating repeatedValue: IntegerLiteralType) -> Use {
        return value.makeTensor(repeating: repeatedValue)
    }

    func makeScalar(_ value: IntegerLiteralType) -> Use {
        return self.value.makeScalar(value)
    }
}

fileprivate class ADContext {
    var blocks: [BasicBlock : BasicBlock] = [:]
    var adjoints: [AnyHashable : Use] = [:]
    var clones: [AnyHashable : Use] = [:]

    unowned let forward: Function
    unowned let gradient: Function
    lazy var builder: IRBuilder = IRBuilder(function: self.gradient)

    init(forward: Function, gradient: Function) {
        self.forward = forward
        self.gradient = gradient
    }
}

fileprivate extension GradientExpansion {

    static func expand(_ function: Function) throws {
        let builder = IRBuilder(module: function.parent)
        /// Seed on return instructions
        let exits = try function.premise().exits
        for (block, returnInst) in exits {
            let retVal = returnInst.operands[0]
            let seed = retVal.type.makeLiteralValue(1).makeUse()
//            context.insertAdjoint(seed, for: retVal)

            /// Differentiate
            for inst in block.reversed().dropFirst() {
//                differentiate(inst, using: builder, in: context)
            }

//            let tupleLit = LiteralValue(type: .tuple(function.arguments.map{$0.type}),
//                                        literal: .tuple(gradients))
//            builder.return(tupleLit.makeUse())
        }

    }

    static func differentiate(_ instruction: Instruction,
                              using bd: IRBuilder,
                              in context: ADContext) {
        guard let adjoint = context.adjoints[instruction] else {
            fatalError("Adjoint seed not found in AD")
        }
        var grad: [(operand: Use, derivative: Use)]
        switch instruction.kind {
        /* Basic arithmetic */
        case let .binary(.associative(.arithmetic(.add)), lhs, rhs):
            grad = [
                (lhs, adjoint), /// ∂f/∂x = D
                (rhs, adjoint), /// ∂f/∂y = D
            ]
        case let .binary(.associative(.arithmetic(.subtract)), lhs, rhs):
            grad = [
                (lhs, adjoint),                                     /// ∂f/∂x = D
                (rhs, bd.subtract(adjoint.makeScalar(0), adjoint)), /// ∂f/∂y = -D
            ]
        case let .binary(.associative(.arithmetic(.multiply)), lhs, rhs):
            grad = [
                (lhs, rhs), /// ∂f/∂x = y
                (rhs, lhs), /// ∂f/∂y = x
            ]
        case let .binary(.associative(.arithmetic(.divide)), lhs, rhs):
            let lhsClone = lhs
            let rhsClone = rhs
            grad = [
                (lhs, bd.divide(adjoint, by: rhsClone)),  /// ∂f/∂x = D/y
                (rhs, bd.subtract(lhsClone.makeScalar(0), /// ∂f/∂y = -x/y^2
                                  bd.divide(lhsClone, by: bd.multiply(rhsClone, by: rhsClone))))
            ]

        /* Matrix multiplication */
        case let .matrixMultiply(lhs, rhs):
            let lhsClone = lhs
            let rhsClone = rhs
            grad = [
                /// ∂f/∂x = D • y^T
                (lhs, bd.matrixMultiply(bd.transpose(lhsClone), adjoint)),
                /// ∂f/∂y = x^T • D
                (rhs, bd.matrixMultiply(adjoint, bd.transpose(rhsClone))),
            ]


        /* Unary elementwise transformations */
        case let .unary(.elementwise(.exp), x):
            let cloned = instruction.makeUse()
            grad = [
                (x, cloned)
            ]

        case let .unary(.elementwise(.tanh), x):
            let cloned = instruction.makeUse()
            grad = [
                (x, bd.subtract(cloned, bd.subtract(x.makeScalar(1),
                                                    bd.multiply(cloned, by: cloned))))
            ]

        case let .extract(from: x, at: _):
            grad = [
                (x, x.type.makeLiteralValue(1).makeUse())
            ]

        default:
            /// - TODO: Implement all cases!
            fatalError("Unimplemented \(instruction)")
        }
    }

}
