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
                /// Clone function to diff
                let newFunc = funcToDiff.makeClone(named: function.name)
                /// Set return type and argument types
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

                /// Expand new function
                let context = ADContext(forward: funcToDiff, gradient: function)
                expand(newFunc, in: context,
                       from: diffIndex, wrt: (varIndices ?? Array(0..<funcToDiff.argumentTypes.count)),
                       keeping: outputIndices, seedable: isSeedable)

                /// Insert new function
                module.insert(newFunc, at: i)
                function.removeFromParent()
                changed = true

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

    func hasAdjoint(for key: Use) -> Bool {
        return getAdjoint(for: key) != nil
    }

    func hasAdjoint(for key: Instruction) -> Bool {
        return getAdjoint(for: key) != nil
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

            /// Differentiate
            for inst in block.reversed().dropFirst() {
                differentiate(inst, using: builder, in: context,
                              returnValue: retVal, returnSeed: seed)
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
                              in context: ADContext,
                              returnValue: Use, returnSeed: Use) {
        /// Move builder
        bd.move(to: instruction.parent)

        /// Get adjoint for instruction
        var adjoint = context.getAdjoint(for: instruction) ?? %instruction.makeLiteral(0)
        if let returnInst = returnValue.instruction, instruction == returnInst {
            adjoint = returnSeed
        }

        /// Get adjoints for operands
        var grad: [(operand: Use, derivative: Use)]
        switch instruction.kind {
        /* Basic arithmetic */
        case let .numericBinary(.add, lhs, rhs):
            grad = [
                /// ∂f/∂x = D
                (lhs, adjoint),
                /// ∂f/∂y = D
                (rhs, adjoint),
            ]
        case let .numericBinary(.subtract, lhs, rhs):
            grad = [
                /// ∂f/∂x = D
                (lhs, adjoint),
                /// ∂f/∂y = -D
                (rhs, %bd.numeric(.negate, adjoint)),
            ]
        case let .numericBinary(.multiply, lhs, rhs):
            grad = [
                /// ∂f/∂x = y
                (lhs, rhs),
                /// ∂f/∂y = x
                (rhs, lhs),
            ]
        case let .numericBinary(.divide, lhs, rhs):
            let lhsClone = lhs
            let rhsClone = rhs
            grad = [
                /// ∂f/∂x = D/y
                (lhs, %bd.divide(adjoint, rhsClone)),
                /// ∂f/∂y = -x/y^2
                (rhs, %bd.numeric(.negate,
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

        /* Transpose */
        case let .transpose(x):
            grad = [
                /// ∂f/∂x = D^T
                (x, %bd.transpose(adjoint))
            ]

        /* Unary elementwise transformations */
        case let .numericUnary(.log, x):
            grad = [
                /// ∂f/∂x = D / x
                (x, %bd.divide(adjoint, x))
            ]

        case let .numericUnary(.cos, x):
            grad = [
                /// ∂f/∂x = -D * sin(x)
                (x, %bd.multiply(%bd.numeric(.negate, adjoint), %bd.numeric(.sin, x)))
            ]

        case let .numericUnary(.sin, x):
            grad = [
                /// ∂f/∂x = D * cos(x)
                (x, %bd.multiply(adjoint, %bd.numeric(.cos, x)))
            ]

        case let .numericUnary(.tan, x):
            let cosx = %bd.numeric(.cos, x)
            grad = [
                /// ∂f/∂x = D / (cos(x) * cos(x))
                (x, %bd.divide(adjoint, %bd.multiply(cosx, cosx)))
            ]

        case let .numericUnary(.cosh, x):
            grad = [
                /// ∂f/∂x = D * sinh(x)
                (x, %bd.multiply(adjoint, %bd.numeric(.sinh, x)))
            ]

        case let .numericUnary(.sinh, x):
            grad = [
                /// ∂f/∂x = D * cosh(x)
                (x, %bd.multiply(adjoint, %bd.numeric(.cosh, x)))
            ]

        case let .numericUnary(.tanh, x):
            let cloned = instruction.makeUse()
            grad = [
                /// ∂f/∂x = D * (1 - (f * f))
                (x, %bd.multiply(adjoint, %bd.subtract(x.makeScalar(1), %bd.multiply(cloned, cloned))))
            ]

        case let .numericUnary(.acos, x):
            let xClone = x
            grad = [
                /// ∂f/∂x = -D / sqrt(1 - (x * x))
                (x, %bd.divide(%bd.numeric(.negate, adjoint),
                               %bd.numeric(.sqrt, %bd.subtract(x.makeScalar(1), %bd.multiply(xClone, xClone)))))
            ]

        case let .numericUnary(.asin, x):
            let xClone = x
            grad = [
                /// ∂f/∂x = D / sqrt(1 - (x * x))
                (x, %bd.divide(adjoint,
                               %bd.numeric(.sqrt, %bd.subtract(x.makeScalar(1), %bd.multiply(xClone, xClone)))))
            ]

        case let .numericUnary(.atan, x):
            let xClone = x
            grad = [
                /// ∂f/∂x = D / (1 + (x * x))
                (x, %bd.divide(adjoint, %bd.add(x.makeScalar(1), %bd.multiply(xClone, xClone))))
            ]

        case let .numericUnary(.exp, x):
            let cloned = instruction.makeUse()
            grad = [
                /// ∂f/∂x = f * D
                (x, %bd.multiply(cloned, adjoint))
            ]

        case let .numericUnary(.sqrt, x):
            let cloned = instruction.makeUse()
            grad = [
                /// ∂f/∂x = D / (2 * f)
                (x, %bd.divide(adjoint, %bd.multiply(instruction.makeScalar(2), cloned)))
            ]

        /* Element extraction */
        case let .extract(from: x, at: _):
            grad = [
                (x, x.makeScalar(1))
            ]

        default:
            /// - TODO: Implement all cases!
            fatalError("Unimplemented \(instruction)")
        }

        /// Update operand adjoints
        for (operand, derivative) in grad {
            if let operandAdjoint = context.getAdjoint(for: operand) {
                context.insertAdjoint(%bd.add(operandAdjoint, derivative), for: operand)
            } else {
                context.insertAdjoint(derivative, for: operand)
            }
        }
    }
}
