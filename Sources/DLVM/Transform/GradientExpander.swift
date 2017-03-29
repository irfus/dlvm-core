//
//  GradientExpander.swift
//  DLVM
//
//  Created by Richard Wei on 2/21/17.
//
//

/// Replace every `gradient` instruction to a `call` to a function that
/// produces the gradient
public class GradientExpander: TransformPass<Module> {
    public override class func run(on module: Module) throws -> Bool {
        var changed = false

        for function in module {
            /// NOTE: For testing purposes, we are differentiating every diff'able function
            /// instead of expanding `gradient` instructions. We'll move to that later when
            /// AD is working

            /// - TODO: Run differentiability verification pass to make sure
            /// the function is differentiable

            /// If function is not differentiable, do nothing
            guard function.isDifferentiable else { continue }
            /// If gradient function exists, do nothing
            let globalGradInfo = try function.parent.analysis(from: GlobalGradientAnalysis.self)
            if let _ = globalGradInfo.gradient(of: function) { continue }
            /// Expand this function
            try expand(function)
            changed = true
        }

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
            case .function(let t, self): return .function(t, newFunc)
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
        case .void, .pointer, .function:
            fatalError("Cannot make numeric literal for type \(self)")
        case .invalid:
            fatalError("Invalid type. You kiddin?")
        case .alias(_):
            fatalError("Opaque type is certainly not differentiable")
        case let .tensor(shape, type):
            return LiteralValue(shape: shape, dataType: type, repeating: number)
        case let .tuple(subtypes):
            let sublits = subtypes.map{$0.makeLiteralValue(number).makeUse()}
            return LiteralValue(type: self, literal: .tuple(sublits))
        case let .array(subtype, i):
            let sublit = subtype.makeLiteralValue(number).makeUse()
            let sublits = Array(repeating: sublit, count: i)
            return LiteralValue(type: self, literal: .array(sublits))
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

class ADContext {
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

extension ADContext {
    func adjoint(for original: Use) -> Use? {
        switch original {
        case let .argument(_, def): return adjoints[def]
        case let .instruction(_, def): return adjoints[def]
        default: return nil
        }
    }
    
    func insertAdjoint(_ derivative: Use, for original: Use) {
        switch original {
        case let .argument(_, def): adjoints[def] = derivative
        case let .instruction(_, def): adjoints[def] = derivative
        default: return
        }
    }

    /// Copy forward definition from original definition, if applicable
    /// There are four cases
    /// 1. literal, global, or function: return itself
    /// 3. instruction: recursively clone
    /// 4. argument: replace it with argument in the gradient function
    /// - Note: Nodes that are already cloned are stored in the context.
    /// Nodes must be cloned to their parent's correspodning basic block
    func clone(_ use: Use) -> Use {
        switch use {
        case .literal, .global, .function: return use
        case let .instruction(_, inst):
            /// Return cached if present
            if let cached = clones[inst] {
                return cached
            }
            
            /// Make sure the clone is happening in the right basic block
            let forwardBB = inst.parent
            guard let gradBB = blocks[forwardBB] else {
                preconditionFailure("Cannot find \(forwardBB.name)'s corresponding gradient block")
            }
            builder.move(to: gradBB)
            
            /// Recursively clone operands
            let clonedOperands = inst.operands.map(clone)
            /// Clone current instruction
            let newInstUse = builder.buildInstruction(inst.kind)
            /// Unwrap and get the instruction
            guard case let .instruction(_, newInst) = newInstUse else { fatalError() }
            /// Replace old operands with cloned ones
            /// - Note: Currently we use a not-so-efficient way to replace
            /// all uses to avoid matching all kinds of instructions all
            /// over again.
            for (new, old) in zip (clonedOperands, inst.operands) {
                newInst.substitute(new, for: old)
            }
            /// Cache the clone (this is extremely important)
            clones[inst] = newInstUse
            return newInstUse
            
        case let .argument(_, arg):
            guard let cached = clones[arg] else {
                fatalError("Cannot find argument %\(arg.name)'s corresponding argument in the gradient function. This should not happen")
            }
            return cached
        }
    }
}

fileprivate extension GradientExpander {

    static func expand(_ function: Function) throws {
        let builder = IRBuilder(module: function.parent)
        /// Build gradient function
        let grad = builder.buildFunction(named: function.name + "_gradient",
                                         arguments: function.arguments.map { ($0.name, $0.type) },
                                         result: .tuple(function.arguments.map { ($0.type) }),
                                         attributes: [ .differentiable, .differentiating(function) ])

        /// Create AD context
        let context = ADContext(forward: function, gradient: grad)

        /// Create a phantom (temporarily) basic block in the gradient
        /// function for each basic block in the original function
        for bb in function {
            let newBB = builder.buildBasicBlock(named: bb.name,
                                                arguments: bb.arguments.map{($0.name, $0.type)},
                                                in: grad)
            context.blocks[bb] = newBB
            /// Insert newly created arguments as clones
            for (old, new) in zip(bb.arguments, newBB.arguments) {
                context.clones[old] = new.makeUse()
            }
        }

        /// Seed on return instructions
        let exits = try function.premise().exits
        for (block, returnInst) in exits {
            let retVal = returnInst.operands[0]
            let seed = retVal.type.makeLiteralValue(1).makeUse()
            context.insertAdjoint(seed, for: retVal)

            /// Get corresponding gradient BB
            let gradBlock = context.blocks[block]!
            builder.move(to: gradBlock)

            /// Differentiate
            for inst in block.reversed().dropFirst() {
                differentiate(inst, using: builder, in: context)
            }

            /// Collect gradient values. If no gradient exists, return 0
            let gradients: [Use] = function.arguments.map {
                context.adjoints[$0] ?? $0.type.makeZero().makeUse()
            }

            let tupleLit = LiteralValue(type: .tuple(function.arguments.map{$0.type}),
                                        literal: .tuple(gradients))
            builder.return(tupleLit.makeUse())
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
                (lhs, context.clone(rhs)), /// ∂f/∂x = y
                (rhs, context.clone(lhs)), /// ∂f/∂y = x
            ]
        case let .binary(.associative(.arithmetic(.divide)), lhs, rhs):
            let lhsClone = context.clone(lhs)
            let rhsClone = context.clone(rhs)
            grad = [
                (lhs, bd.divide(adjoint, by: rhsClone)),  /// ∂f/∂x = D/y
                (rhs, bd.subtract(lhsClone.makeScalar(0), /// ∂f/∂y = -x/y^2
                                  bd.divide(lhsClone, by: bd.multiply(rhsClone, by: rhsClone))))
            ]

        /* Matrix multiplication */
        case let .matrixMultiply(lhs, rhs):
            let lhsClone = context.clone(lhs)
            let rhsClone = context.clone(rhs)
            grad = [
                /// ∂f/∂x = D • y^T
                (lhs, bd.matrixMultiply(bd.transpose(lhsClone), adjoint)),
                /// ∂f/∂y = x^T • D
                (rhs, bd.matrixMultiply(adjoint, bd.transpose(rhsClone))),
            ]


        /* Unary elementwise transformations */
        case let .unary(.elementwise(.exp), x):
            let cloned = context.clone(instruction.makeUse())
            grad = [
                (x, cloned)
            ]

        case let .unary(.elementwise(.tanh), x):
            let cloned = context.clone(instruction.makeUse())
            grad = [
                (x, bd.subtract(cloned, bd.subtract(x.makeScalar(1),
                                                    bd.multiply(cloned, by: cloned))))
            ]

        case let .extract(from: x, at: _):
            grad = [
                (x, x.type.makeLiteralValue(1).makeUse())
            ]

        case let .call(fun, args):
            let fgrads = bd.gradient(fun, args)
            grad = []
            for (i, arg) in args.enumerated() {
                grad.append((operand: arg,
                             derivative: bd.buildInstruction(.extract(from: fgrads, at: [i]))))
            }

        default:
            /// - TODO: Implement all cases!
            fatalError("Unimplemented \(instruction)")
        }

        /// Accumulate to adjacent
        for (operand, derivative) in grad where operand.type.isTensor {
            if let adj = context.adjoint(for: operand) {
                let acc = bd.add(adj, derivative)
                context.insertAdjoint(acc, for: operand)
            }
            context.insertAdjoint(derivative, for: operand)
        }
    }

}
