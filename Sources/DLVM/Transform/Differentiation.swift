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
    private typealias AdjointMapping =
        [Function : [(config: AdjointConfiguration, adjoint: Function)]]

    open class func run(on module: Module) -> Bool {
        var changed = false
        var workList: [Function] = Array(module)
        var adjoints: AdjointMapping = [:]
        while !workList.isEmpty {
            let adjoint = workList.removeFirst()
            guard case let .adjoint(config)? = adjoint.declarationKind
                else { continue }
            /// Remove adjoint declaration
            adjoint.declarationKind = nil
            /// Copy contents of primal function to adjoint function
            config.primal.copyContents(to: adjoint)
            /// Add seed argument if necessary
            if config.isSeedable {
                let seedArgName = adjoint.makeFreshName("seed")
                let seedArg = Argument(name: seedArgName,
                                       type: config.seedType,
                                       parent: adjoint[0])
                adjoint[0].arguments.append(seedArg)
            }
            /// Expand adjoint function
            let context = ADContext(primal: config.primal, adjoint: adjoint)
            expand(adjoint, in: context, from: config.sourceIndex,
                   wrt: (config.argumentIndices ?? Array(config.primal.argumentTypes.indices)),
                   keeping: config.keptIndices, seedable: config.isSeedable,
                   workList: &workList, adjoints: &adjoints)
            /// Add primal and adjoint functions to mapping
            let newAdjoints = (adjoints[config.primal] ?? []) + [(config, adjoint)]
            adjoints[config.primal] = newAdjoints
            changed = true
        }
        module.stage = .optimizable
        return changed
    }
}

fileprivate extension AdjointConfiguration {
    var seedType: Type {
        if case let .tuple(elements) = primal.returnType {
            guard let sourceIndex = sourceIndex,
                elements.indices.contains(sourceIndex) else {
                fatalError("Invalid source index for primal returning tuple")
            }
            return elements[sourceIndex]
        } else {
            return primal.returnType
        }
    }
}

fileprivate extension Module {
    func replaceAllUses(of oldUse: Use, with newUse: Use) {
        for fn in self {
            fn.replaceAllUses(of: oldUse, with: newUse)
        }
    }
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
                               adjoints: inout AdjointMapping) {
        let builder = IRBuilder(module: function.parent)
        /// Canonicalize loops
        let cfg = function.analysis(from: ControlFlowGraphAnalysis.self)
        var loopInfo = function.analysis(from: LoopAnalysis.self)
        let loops = Set(loopInfo.innerMostLoops.values)
        for loop in loops {
            /// Create a unique loop preheader
            if loop.preheader == nil {
                /// Gather original predecessors of header
                let preds = cfg.predecessors(of: loop.header)
                    .filter { !loop.contains($0) }
                /// Create preheader and connect it with header
                let preheader = BasicBlock(
                    name: function.makeFreshName("preheader"),
                    arguments: loop.header.arguments.map{
                        (function.makeFreshName($0.name), $0.type)
                    },
                    parent: function)
                function.insert(preheader, before: loop.header)
                builder.move(to: preheader)
                builder.branch(loop.header, preheader.arguments.map(%))
                /// Change all original predecessors to branch to preheader
                preds.forEach { pred in
                    guard let terminator = pred.terminator else { return }
                    terminator.substituteBranches(to: loop.header, with: preheader)
                }
                /// Make the preheader a part of the parent loop if it exists
                if let parent = loop.parent {
                    loopInfo.innerMostLoops[preheader] = parent
                    parent.blocks.insert(preheader, before: loop.header)
                }
            }
        }
        /// Seed on return instructions
        let exits = function.premise.exits
        for (block, returnInst) in exits {
            /// Get return value
            let retVal: Use
            if let sourceIndex = sourceIndex {
                guard case let .instruction(ty, inst) = returnInst.operands[0],
                    case .literal(let lit, ty) = inst.kind,
                    case let .tuple(elements) = lit,
                    elements.indices.contains(sourceIndex) else {
                    fatalError("""
                        Invalid return instruction \(returnInst) for source \
                        index \(sourceIndex)
                        """)
                }
                retVal = elements[sourceIndex]
            } else {
                retVal = returnInst.operands[0]
            }
            /// Get seed value and insert into context
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
            context.insertAdjoint(seed, for: retVal)
            /// Mark instructions to differentiate
            var instsToDiff: Set<Instruction>
            if sourceIndex != nil {
                switch retVal {
                case let .instruction(_, inst):
                    instsToDiff = Set(inst.predecessors).union([inst])
                default:
                    instsToDiff = []
                }
            } else {
                instsToDiff = Set(returnInst.predecessors)
            }
            func markInstruction(_ inst: Instruction) {
                instsToDiff.insert(inst)
                inst.predecessors.forEach(markInstruction)
            }
            instsToDiff.forEach(markInstruction)
            /// Iterate through instructions in reverse order and differentiate
            for inst in function.instructions.reversed()
                where instsToDiff.contains(inst) {
                differentiate(inst, using: builder, in: context,
                              returnValue: retVal, workList: &workList,
                              adjoints: &adjoints)
            }
            /// Remove old return
            returnInst.removeFromParent()
            /// Build new return
            var newReturn: [Use] = []
            newReturn += argIndices.map { i in
                guard let argAdjoint = context.adjoint(for: %function[0].arguments[i]) else {
                    fatalError("""
                        Adjoint not found for argument \(function[0].arguments[i]) \
                        in function \(function.name)
                        """)
                }
                return argAdjoint
            }
            newReturn += keptIndices.map { returnInst.operands[$0] }
            let tupleLit = builder.buildInstruction(.literal(.tuple(newReturn), function.returnType))
            builder.return(%tupleLit)
        }
    }

    private static func differentiate(_ inst: Instruction,
                                      using bd: IRBuilder,
                                      in context: ADContext,
                                      returnValue: Use,
                                      workList: inout [Function],
                                      adjoints: inout AdjointMapping) {
        /// Move builder
        bd.move(to: inst.parent)
        /// Get adjoint for instruction
        var instAdjoint: Use
        if let oldAdjoint = context.adjoint(for: inst) {
            instAdjoint = oldAdjoint
        } else {
            instAdjoint = returnValue.makeLiteral(0, using: bd).makeUse()
        }
        /// Get adjoints for operands
        var operandAdjoints: [(operand: Use, adjoint: Use)]
        switch inst.kind {
        /* Function application */
        case let .apply(.function(_, fn), operands):
            let adjoint: Function
            let config = AdjointConfiguration(
                primal: fn, sourceIndex: nil, argumentIndices: nil,
                keptIndices: [], isSeedable: true
            )
            if let funcAdjoints = adjoints[fn],
                let gradIndex = funcAdjoints.index(where: { $0.config == config }) {
                adjoint = funcAdjoints[gradIndex].adjoint
            } else {
                guard let adjointType = fn.adjointType(from: nil,
                                                         wrt: nil,
                                                         keeping: [],
                                                         seedable: true),
                    case let .function(argumentTypes, returnType) = adjointType else {
                        fatalError("Function @\(fn.name) is not differentiable")
                }
                let module = fn.parent
                let adjointName = module.makeFreshFunctionName("\(fn.name)_grad")
                adjoint = Function(name: adjointName,
                                   argumentTypes: argumentTypes,
                                   returnType: returnType,
                                   declarationKind: .adjoint(config),
                                   parent: module)
                module.insert(adjoint, after: context.adjoint)
                workList.append(adjoint)
            }
            let operandAdjoint = bd.apply(%adjoint, operands + [instAdjoint])
            operandAdjoints = operands.enumerated().map { (i, operand) in
                return (operand, %bd.extract(from: %operandAdjoint, at: [.index(i)]))
            }

        /* Default case, use defined adjoints */
        default:
            guard let tmp = inst.kind.operandAdjoints(
                using: bd, primal: %inst, seed: instAdjoint) else {
                fatalError("Unimplemented \(inst)")
            }
            operandAdjoints = tmp
        }

        /// Update operand adjoints
        for (operand, newAdjoint) in operandAdjoints {
            /// Adjoints for immediate literals are never needed, do not store
            if case .literal = operand { continue }
            if let operandAdjoint = context.adjoint(for: operand) {
                context.insertAdjoint(%bd.add(operandAdjoint, newAdjoint), for: operand)
            } else {
                context.insertAdjoint(newAdjoint, for: operand)
            }
        }
    }
}
