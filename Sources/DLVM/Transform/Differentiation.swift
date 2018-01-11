//
//  Differentiation.swift
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

open class Differentiation: TransformPass {
    public typealias Body = Module
    private typealias GradientMapping =
        [Function : [(config: GradientConfiguration, adjoint: Function)]]

    open class func run(on module: Module) -> Bool {
        var changed = false
        var workList: [Function] = Array(module)
        var adjoints: GradientMapping = [:]
        while !workList.isEmpty {
            let adjoint = workList.removeFirst()
            guard case let .gradient(config)? = adjoint.declarationKind
                else { continue }
            /// Copy contents of primal function to adjoint function
            config.primal.copyContents(to: adjoint)
            /// Add seed argument if necessary
            if config.isSeedable {
                let seedArgName = makeFreshName("seed", in: adjoint[0].arguments)
                let seedArg = Argument(name: seedArgName,
                                       type: config.primal.returnType,
                                       parent: adjoint[0])
                adjoint[0].arguments.append(seedArg)
            }
            /// Expand adjoint function
            let context = ADContext(primal: config.primal, adjoint: adjoint)
            expand(adjoint, in: context, from: config.sourceIndex,
                   wrt: (config.argumentIndices ?? Array(config.primal.argumentTypes.indices)),
                   keeping: config.keptIndices, seedable: config.isSeedable,
                   workList: &workList, adjoints: &adjoints)
            /// Remove gradient declaration
            adjoint.declarationKind = nil
            /// Add primal and adjoint functions to mapping
            let newAdjoints = (adjoints[config.primal] ?? []) + [(config, adjoint)]
            adjoints[config.primal] = newAdjoints
            changed = true
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

fileprivate extension Module {
    func replaceAllUses(of oldUse: Use, with newUse: Use) {
        for fn in self {
            fn.replaceAllUses(of: oldUse, with: newUse)
        }
    }
}

fileprivate func makeFreshName<S : Sequence>(_ name: String, in names: S) -> String
    where S.Element : Named
{
    var result = name
    var count = 0
    while names.contains(where: { $0.name == result }) {
        result = "name\(count)"
        count += 1
    }
    return result
}

fileprivate class ADContext {
    var blocks: [BasicBlock : BasicBlock] = [:]
    var adjoints: [AnyHashable : Use] = [:]
    var clones: [AnyHashable : Use] = [:]

    unowned let primal: Function
    unowned let adjoint: Function
    lazy var builder: IRBuilder = IRBuilder(function: self.adjoint)

    init(primal: Function, adjoint: Function) {
        self.primal = primal
        self.adjoint = adjoint
    }

    func hasAdjoint(for key: Use) -> Bool {
        return adjoint(for: key) != nil
    }

    func hasAdjoint(for key: Instruction) -> Bool {
        return adjoint(for: key) != nil
    }

    func adjoint(for key: Use) -> Use? {
        guard let definition = key.definition else {
            fatalError("\(key) has no definition")
        }
        return adjoints[ObjectIdentifier(definition)]
    }

    func adjoint(for key: Instruction) -> Use? {
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
    private static func expand(_ function: Function, in context: ADContext,
                               from sourceIndex: Int?, wrt argIndices: [Int],
                               keeping keptIndices: [Int],
                               seedable isSeedable: Bool,
                               workList: inout [Function],
                               adjoints: inout GradientMapping) {
        let builder = IRBuilder(module: function.parent)
        /// Seed on return instructions
        let exits = function.premise.exits
        for (block, returnInst) in exits {
            let retVal: Use = returnInst.operands[sourceIndex ?? 0]
            let seed: Use
            if isSeedable {
                guard let seedArg = function[0].arguments.last else {
                    fatalError("No seed argument")
                }
                seed = %seedArg
            } else {
                builder.move(to: block, index: 0)
                seed = retVal.makeLiteral(1, using: builder).makeUse()
                builder.move(to: block)
            }
            /// Mark instructions to differentiate
            var instsToDiff: Set<Instruction> = Set(returnInst.predecessors)
            func markInstruction(_ inst: Instruction) {
                instsToDiff.insert(inst)
                inst.predecessors.forEach(markInstruction)
            }
            instsToDiff.forEach(markInstruction)
            /// Iterate through instructions in reverse order and differentiate
            for inst in function.instructions.lazy.reversed()
                where instsToDiff.contains(inst) {
                    differentiate(inst, using: builder, in: context,
                                  returnValue: retVal, returnSeed: seed,
                                  workList: &workList, gradients: &adjoints)
            }
            /// Remove old return
            returnInst.removeFromParent()
            /// Build new return
            var newReturn: [Use] = []
            newReturn += argIndices.map { context.adjoint(for: %function[0].arguments[$0])! }
            newReturn += keptIndices.map { returnInst.operands[$0] }
            let tupleLit = builder.buildInstruction(.literal(.tuple(newReturn), function.returnType))
            builder.return(%tupleLit)
        }
    }

    private static func differentiate(_ instruction: Instruction,
                                      using bd: IRBuilder,
                                      in context: ADContext,
                                      returnValue: Use, returnSeed: Use,
                                      workList: inout [Function],
                                      gradients: inout GradientMapping) {
        /// Move builder
        bd.move(to: instruction.parent)
        /// Get adjoint for instruction
        var instAdjoint: Use
        if let oldAdjoint = context.adjoint(for: instruction) {
            instAdjoint = oldAdjoint
        } else if let returnInst = returnValue.instruction, instruction == returnInst {
            instAdjoint = returnSeed
        } else {
            instAdjoint = instruction.makeLiteral(0, using: bd).makeUse()
        }
        /// Get adjoints for operands
        var grad: [(operand: Use, derivative: Use)]
        switch instruction.kind {
        /* Literal value */
        case let .literal(lit, _):
            grad = lit.operands.map {
                ($0, $0.makeLiteral(0, using: bd).makeUse())
            }

        /* Basic arithmetic */
        case let .numericBinary(.add, lhs, rhs):
            grad = [
                /// ∂f/∂x = D
                (lhs, instAdjoint),
                /// ∂f/∂y = D
                (rhs, instAdjoint)
            ]
        case let .numericBinary(.subtract, lhs, rhs):
            grad = [
                /// ∂f/∂x = D
                (lhs, instAdjoint),
                /// ∂f/∂y = -D
                (rhs, %bd.numeric(.negate, instAdjoint))
            ]
        case let .numericBinary(.multiply, lhs, rhs):
            grad = [
                /// ∂f/∂x = y
                (lhs, rhs),
                /// ∂f/∂y = x
                (rhs, lhs)
            ]
        case let .numericBinary(.divide, lhs, rhs):
            grad = [
                /// ∂f/∂x = D/y
                (lhs, %bd.divide(instAdjoint, rhs)),
                /// ∂f/∂y = -x/y^2
                (rhs, %bd.numeric(.negate, %bd.divide(lhs, %bd.multiply(rhs, rhs))))
            ]

        /* Dot */
        case let .dot(lhs, rhs):
            grad = [
                /// ∂f/∂x = D • y^T
                (lhs, %bd.dot(instAdjoint, %bd.transpose(rhs))),
                /// ∂f/∂y = x^T • D
                (rhs, %bd.dot(%bd.transpose(lhs), instAdjoint))
            ]

        /* Transpose */
        case let .transpose(x):
            grad = [
                /// ∂f/∂x = D^T
                (x, %bd.transpose(instAdjoint))
            ]

        /* Unary elementwise transformations */
        case let .numericUnary(.log, x):
            grad = [
                /// ∂f/∂x = D / x
                (x, %bd.divide(instAdjoint, x))
            ]

        case let .numericUnary(.cos, x):
            grad = [
                /// ∂f/∂x = -D * sin(x)
                (x, %bd.multiply(%bd.numeric(.negate, instAdjoint), %bd.numeric(.sin, x)))
            ]

        case let .numericUnary(.sin, x):
            grad = [
                /// ∂f/∂x = D * cos(x)
                (x, %bd.multiply(instAdjoint, %bd.numeric(.cos, x)))
            ]

        case let .numericUnary(.tan, x):
            let cosx = %bd.numeric(.cos, x)
            grad = [
                /// ∂f/∂x = D / (cos(x) * cos(x))
                (x, %bd.divide(instAdjoint, %bd.multiply(cosx, cosx)))
            ]

        case let .numericUnary(.cosh, x):
            grad = [
                /// ∂f/∂x = D * sinh(x)
                (x, %bd.multiply(instAdjoint, %bd.numeric(.sinh, x)))
            ]

        case let .numericUnary(.sinh, x):
            grad = [
                /// ∂f/∂x = D * cosh(x)
                (x, %bd.multiply(instAdjoint, %bd.numeric(.cosh, x)))
            ]

        case let .numericUnary(.tanh, x):
            let cloned = instruction.makeUse()
            grad = [
                /// ∂f/∂x = D * (1 - (f * f))
                (x, %bd.multiply(instAdjoint, %bd.subtract(x.makeScalar(1), %bd.multiply(cloned, cloned))))
            ]

        case let .numericUnary(.acos, x):
            grad = [
                /// ∂f/∂x = -D / sqrt(1 - (x * x))
                (x, %bd.divide(%bd.numeric(.negate, instAdjoint),
                               %bd.numeric(.sqrt, %bd.subtract(x.makeScalar(1), %bd.multiply(x, x)))))
            ]

        case let .numericUnary(.asin, x):
            grad = [
                /// ∂f/∂x = D / sqrt(1 - (x * x))
                (x, %bd.divide(instAdjoint,
                               %bd.numeric(.sqrt, %bd.subtract(x.makeScalar(1), %bd.multiply(x, x)))))
            ]

        case let .numericUnary(.atan, x):
            grad = [
                /// ∂f/∂x = D / (1 + (x * x))
                (x, %bd.divide(instAdjoint, %bd.add(x.makeScalar(1), %bd.multiply(x, x))))
            ]

        case let .numericUnary(.exp, x):
            let cloned = instruction.makeUse()
            grad = [
                /// ∂f/∂x = f * D
                (x, %bd.multiply(cloned, instAdjoint))
            ]

        case let .numericUnary(.sqrt, x):
            let cloned = instruction.makeUse()
            grad = [
                /// ∂f/∂x = D / (2 * f)
                (x, %bd.divide(instAdjoint, %bd.multiply(instruction.makeScalar(2), cloned)))
            ]

        /* Element extraction */
        case let .extract(from: x, at: _):
            grad = [
                (x, x.makeScalar(1))
            ]

        /* Function application */
        case let .apply(.function(_, fn), operands):
            let adjoint: Function
            let config = GradientConfiguration(
                primal: fn, sourceIndex: nil, argumentIndices: nil,
                keptIndices: [], isSeedable: true
            )
            if let funcAdjoints = gradients[fn],
                let gradIndex = funcAdjoints.index(where: { $0.config == config }) {
                adjoint = funcAdjoints[gradIndex].adjoint
            } else {
                guard let gradientType = fn.gradientType(from: nil,
                                                         wrt: nil,
                                                         keeping: [],
                                                         seedable: true),
                    case let .function(argumentTypes, returnType) = gradientType else {
                        fatalError("Function \(fn.name) is not differentiable")
                }
                let module = fn.parent
                let adjointName = makeFreshName("\(fn.name)_grad", in: module)
                adjoint = Function(name: adjointName,
                                   argumentTypes: argumentTypes,
                                   returnType: returnType,
                                   declarationKind: .gradient(config),
                                   parent: module)
                module.insert(adjoint, after: context.adjoint)
                workList.append(adjoint)
            }
            let operandAdjoint = bd.apply(%adjoint, operands + [instAdjoint])
            grad = operands.enumerated().map { (i, operand) in
                return (operand, %bd.extract(from: %operandAdjoint, at: [.index(i)]))
            }

        default:
            /// - TODO: Implement all cases!
            fatalError("Unimplemented \(instruction)")
        }

        /// Update operand adjoints
        for (operand, newAdjoint) in grad {
            if let operandAdjoint = context.adjoint(for: operand) {
                context.insertAdjoint(%bd.add(operandAdjoint, newAdjoint), for: operand)
            } else {
                context.insertAdjoint(newAdjoint, for: operand)
            }
        }
    }
}
