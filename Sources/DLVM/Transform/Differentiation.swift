//
//  Differentiation.swift
//  DLVM
//
//  Copyright 2016-2017 The DLVM Team.
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
/// - TODO: Refactor bigly

/// Replace every `gradient` instruction to a `call` to a function that
/// produces the gradient
open class Differentiation: TransformPass {
    public typealias Body = Module
    
    open class func run(on module: Module) -> Bool {
        var changed = false

        var expanded: [Function : Function] = [:]

        for (i, function) in module.enumerated() {
            if case let .gradient(funcToDiff,
                                   from: diffIndex,
                                   wrt: varIndices,
                                   keeping: outputIndices,
                                   seedable: isSeedable)? = function.declarationKind {
                /// Clone func to diff and set argument types/return type
                let newFunc = funcToDiff.makeClone(named: function.name)
                newFunc.returnType = function.returnType
                if isSeedable {
                    newFunc.argumentTypes = function.argumentTypes
                    var seedArgName = "seed", count = 0
                    while newFunc[0].arguments.contains(where: { $0.name == seedArgName }) {
                        seedArgName = "seed\(count)"
                        count += 1
                    }
                    newFunc[0].arguments.append(Argument(name: seedArgName,
                                                         type: funcToDiff.returnType,
                                                         parent: newFunc[0]))
                }

                let context = ADContext(forward: funcToDiff, gradient: function)
                expand(newFunc, in: context,
                       from: diffIndex, wrt: (varIndices ?? Array(0..<funcToDiff.argumentTypes.count)),
                       keeping: outputIndices, seedable: isSeedable)
                print("new func", newFunc)

                newFunc.parent = module
                module.insert(newFunc, at: i)
                function.removeFromParent()
                changed = true

                for bb in newFunc {
                    assert(bb.existsInParent)
                    for inst in bb {
                        assert(inst.existsInParent)
                    }
                }
            }
        }
        
        module.stage = .optimizable

        return changed
    }
}

/// - Note: A new approach of adjoint code generation will be used.
/// Function will first be cloned and then adjoint code gets generated in
/// a real adjoint fasion

fileprivate extension Type {
    func makeLiteral(_ number: IntegerLiteralType) -> Use {
        switch canonical {
        case let .tensor(shape, type):
            return .tensor(shape, type, repeating: number)
        case let .tuple(elementTypes):
            let sublits = elementTypes.map{$0.makeLiteral(number)}
            return .literal(self, .tuple(sublits))
        case let .array(i, elementType):
            let sublit = elementType.makeLiteral(number)
            let sublits = Array(repeating: sublit, count: i)
            return .literal(self, .array(sublits))
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
        return .tensor(shape, dtype, repeating: repeatedValue)
    }

    func makeScalar(_ value: IntegerLiteralType) -> Use {
        let canType = type.canonical
        guard case let .tensor(_, dtype) = canType else {
            preconditionFailure("\(self), a.k.a. \(canType), is not tensor")
        }
        return .tensor(.scalar, dtype, repeating: value)
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

    func getAdjoint(for key: Use) -> Use? {
        guard let definition = key.definition else {
            fatalError("\(key) has no definition")
        }
        return adjoints[ObjectIdentifier(definition)]
    }

    func getAdjoint(for key: Instruction) -> Use? {
        return adjoints[ObjectIdentifier(key as Definition)]
    }

    func insertAdjoint(_ value: Use, for key: Use) {
        guard let definition = key.definition else {
            fatalError("\(key) has no definition")
        }
        adjoints[ObjectIdentifier(definition)] = value
    }

    func insertAdjoint(_ value: Use, for key: Instruction) {
        adjoints[ObjectIdentifier(key as Definition)] = value
    }
}

fileprivate extension Differentiation {

    static func expand(_ function: Function, in context: ADContext,
                       from diffIndex: Int?, wrt varIndices: [Int],
                       keeping outputIndices: [Int], seedable isSeedable: Bool) {
        let builder = IRBuilder(module: function.parent)

        /// Initialize adjoints
        for bb in function {
            for inst in bb {
                context.insertAdjoint(%inst.makeLiteral(0), for: inst)
                for operand in inst.operands {
                    context.insertAdjoint(%operand.makeLiteral(1), for: operand)
                }
            }
        }

        /// Seed on return instructions
        let exits = function.premise.exits
        for (block, returnInst) in exits {
            let retVal: Use = returnInst.operands[diffIndex ?? 0]
            let seed: Use
            if !isSeedable {
                seed = %retVal.makeLiteral(1)
            } else {
                guard let seedArg = function[0].arguments.last else {
                    fatalError("No seed argument")
                }
                seed = %seedArg
            }
            context.insertAdjoint(seed, for: retVal)

            /// Differentiate
            for inst in block.reversed().dropFirst() {
                differentiate(inst, using: builder, in: context)
            }

            /// Remove old return
            returnInst.removeFromParent()

            /// Build new return
            var newReturn: [Use] = []
            newReturn += varIndices.map { context.getAdjoint(for: %function[0].arguments[$0])! }
            newReturn += outputIndices.map { returnInst.operands[$0] }

            let tupleLit = builder.buildInstruction(.literal(.tuple(newReturn), function.returnType))
            builder.return(%tupleLit)
        }
    }

    static func differentiate(_ instruction: Instruction,
                              using bd: IRBuilder,
                              in context: ADContext) {
        /// Move builder
        bd.move(to: instruction.parent)

        /// Get adjoint for instruction
        guard let adjoint = context.getAdjoint(for: instruction) else {
            fatalError("Adjoint not found in AD")
        }

        /// Get adjoints for operands
        var grad: [(operand: Use, derivative: Use)]
        switch instruction.kind {
        /* Basic arithmetic */
        case let .numericBinary(.add, lhs, rhs):
            grad = [
                (lhs, adjoint), /// ∂f/∂x = D
                (rhs, adjoint), /// ∂f/∂y = D
            ]
        case let .numericBinary(.subtract, lhs, rhs):
            grad = [
                (lhs, adjoint),                                     /// ∂f/∂x = D
                (rhs, %bd.subtract(adjoint.makeScalar(0), adjoint)), /// ∂f/∂y = -D
            ]
        case let .numericBinary(.multiply, lhs, rhs):
            grad = [
                (lhs, rhs), /// ∂f/∂x = y
                (rhs, lhs), /// ∂f/∂y = x
            ]
        case let .numericBinary(.divide, lhs, rhs):
            let lhsClone = lhs
            let rhsClone = rhs
            grad = [
                (lhs, %bd.divide(adjoint, rhsClone)),  /// ∂f/∂x = D/y
                (rhs, %bd.subtract(lhsClone.makeScalar(0), /// ∂f/∂y = -x/y^2
                                   %bd.divide(lhsClone, %bd.multiply(rhsClone, rhsClone))))
            ]

        /* Dot */
        case let .dot(lhs, rhs):
            let lhsClone = lhs
            let rhsClone = rhs
            grad = [
                /// ∂f/∂x = D • y^T
                (lhs, %bd.dot(adjoint, %bd.transpose(rhsClone))),
                /// ∂f/∂y = x^T • D
                (rhs, %bd.dot(%bd.transpose(lhsClone), adjoint)),
            ]

        /* Unary elementwise transformations */
        case let .numericUnary(.exp, x):
            let cloned = instruction.makeUse()
            grad = [
                (x, cloned)
            ]

        case let .numericUnary(.tanh, x):
            let cloned = instruction.makeUse()
            grad = [
                (x, %bd.subtract(cloned, %bd.subtract(x.makeScalar(1),
                                                      %bd.multiply(cloned, cloned))))
            ]

        case let .extract(from: x, at: _):
            grad = [
                (x, x.type.makeLiteral(1))
            ]


        default:
            /// - TODO: Implement all cases!
            fatalError("Unimplemented \(instruction)")
        }

        /// Update adjoints
        for (operand, derivative) in grad {
            /// Update instruction adjoint
            let newAdjoint: Use
            switch (adjoint.type, derivative.type) {
            case (.tensor(.scalar, _), .tensor(.scalar, _)): newAdjoint = %bd.multiply(adjoint, derivative)
            // case (.tensor(_, _), .tensor(_, _)): newAdjoint = %bd.dot(adjoint, derivative)
            case (.tensor(_, _), .tensor(_, _)): newAdjoint = %bd.multiply(adjoint, derivative)
            default: fatalError("Adjoint is not a tensor")
            }
            context.insertAdjoint(newAdjoint, for: instruction)

            /// Update operand adjoint
            guard let operandAdjoint = context.getAdjoint(for: operand) else {
                fatalError("Adjoint not found in AD")
            }
            context.insertAdjoint(%bd.add(operandAdjoint, newAdjoint), for: operand)
        }
    }
}
