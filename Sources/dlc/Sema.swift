//
//  Sema.swift
//  DLVM
//
//  Created by Richard Wei on 1/16/17.
//
//

import DLVM

enum SemanticError : Error {
    case redeclaredTemporary(InstructionDeclarationNode)
    case redeclaredGlobal(DeclarationNode)
    case redeclaredBasicBlock(BasicBlockNode)
    case extraneousInitializer(DeclarationNode, InitializerNode)
    case missingInitializer(DeclarationNode)
    case undeclaredVariable(VariableNode)
    case cannotFindMainBlock(BasicBlockNode)
    case typeMismatch(OperandNode, DataType)
    case shapeMismatch(OperandNode, TensorShape)
    case notGlobal(OperandNode)
    case extraneousName(InstructionDeclarationNode)
    case missingName(InstructionDeclarationNode)
    case initializerTypeMismatch(InitializerNode, TypeNode)
    case unsupportedExtensionType(BasicBlockNode)
    case extensionInNonextension(BasicBlockNode)
    case nonextensionInExtension(BasicBlockNode)
    case extensionTypeMismatchWithParent(BasicBlockNode)
    case redeclaredExtension(BasicBlockNode)
}

extension ModuleNode {
    func makeModule() throws -> Module {
        let module = Module(name: name)

        for decl in declarations {
            let value = try decl.makeDeclaration(in: module)
            module.add(value)
        }

        for bbNode in basicBlocks {
            let bb = try bbNode.makeBasicBlock(in: nil, module: module)
            if !bb.isExtension {
                module.append(bb)
            }
        }
        return module
    }
}

extension BasicBlockNode {
    func makeBasicBlock(in env: BasicBlock?, module: Module) throws -> BasicBlock {
        let bb: BasicBlock
        let extType = try extensionTypeName.map { (extTypeName) throws -> BasicBlock.ExtensionType in
            guard let extType = BasicBlock.ExtensionType.lexicon[extTypeName] else {
                throw SemanticError.unsupportedExtensionType(self)
            }
            return extType
        }
        switch (env, extType) {
        /// Non-extension cannot live in an extension
        case let (env?, nil) where env.isExtension:
            throw SemanticError.nonextensionInExtension(self)

        /// Extension cannot live in a non-extension
        case let (env?, _?) where !env.isExtension:
            throw SemanticError.extensionInNonextension(self)

        /// Extension must have the same type as parent's
        case let (env?, extType?) where extType != env.extensionType:
            throw SemanticError.extensionTypeMismatchWithParent(self)

        /// Non-extension cannot be duplicate
        case (nil, nil) where module.containsBasicBlock(named: name):
            throw SemanticError.redeclaredBasicBlock(self)

        /// Nested non-extension's name must not be duplicate
        case let (env?, nil) where module.containsBasicBlock(named: name)
                                || env.hasDescendant(named: name):
            throw SemanticError.redeclaredBasicBlock(self)

        /// Global extension
        case let (nil, extType?):
            /// Search for basic block
            guard let mainBB = module.basicBlock(named: name)
                ?? module.basicBlocks.flatMap({$0.descendant(named: name)}).first else {
                throw SemanticError.cannotFindMainBlock(self)
            }
            guard !mainBB.hasExtension(ofType: extType) else {
                throw SemanticError.redeclaredExtension(self)
            }
            bb = mainBB.makeExtension(ofType: extType)

        /// Nested extension
        case let (env?, extType?):
            guard let mainBB = env.mainBlock?.descendant(named: name) else {
                throw SemanticError.cannotFindMainBlock(self)
            }
            guard !mainBB.hasExtension(ofType: extType) else {
                throw SemanticError.redeclaredExtension(self)
            }
            bb = mainBB.makeExtension(ofType: extType)

        /// Global non-extension
        case (nil, nil):
            bb = BasicBlock(name: name)

        /// Nested non-extension
        case let (env?, nil):
            bb = env.makeChild(named: name)
        }
        
        for instNode in instructions {
            let inst = try instNode.makeInstruction(in: bb, module: module)
            bb.append(inst)
        }
        return bb
    }
}

extension DeclarationNode {
    func makeDeclaration(in env: Module) throws -> GlobalValue {
        guard !env.containsGlobalValue(named: name) else {
            throw SemanticError.redeclaredGlobal(self)
        }
        switch role {
        /// Error cases
        case .input where initializer != nil,
             .output where initializer != nil:
            throw SemanticError.extraneousInitializer(self, initializer!)

        case .parameter:
            guard let initializer = initializer else {
                throw SemanticError.missingInitializer(self)
            }
            let declType = type.makeType()
            switch initializer {
            case let .immediate(immInit, _):
                let immType = immInit.type.makeType()
                guard immType == declType else {
                    throw SemanticError.initializerTypeMismatch(initializer, type)
                }
            case let .random(lo, hi, _):
                let loType = lo.type.makeType(), hiType = hi.type.makeType()
                guard loType == declType, hiType == declType else {
                    throw SemanticError.initializerTypeMismatch(initializer, type)
                }

            case let .repeating(immInit, _):
                let immType = immInit.type.makeType()
                guard immType == declType else {
                    throw SemanticError.initializerTypeMismatch(initializer, type)
                }
            }
            return Parameter(name: name, type: declType,
                             shape: shape?.makeShape() ?? [],
                             initializer: initializer.makeInitializer())

        case .input:
            return Input(name: name, type: type.makeType(), shape: shape?.makeShape() ?? [])

        case .output:
            return Output(name: name, type: type.makeType(), shape: shape?.makeShape() ?? [])
        }
    }
}

extension TypeNode {
    func makeType() -> DataType {
        switch self {
        case .bool: return .bool
        case let .int(size, _): return .int(size)
        case let .float(size, _): return .float(size)
        }
    }
}

extension ShapeNode {
    func makeShape() -> TensorShape {
        return TensorShape(dimensions)
    }
}

extension ImmediateNode {
    func makeImmediate() -> Immediate {
        switch self {
        case let .bool(b, _): return .bool(b)
        case let .int(i, _):  return .int(i)
        case let .float(f, _): return .float(f)
        }
    }
}

extension ImmediateValueNode {
    func makeImmediateValue() -> ImmediateValue {
        return ImmediateValue(type: type.makeType(), immediate: immediate.makeImmediate())
    }
}

extension InitializerNode {
    func makeInitializer() -> Initializer {
        switch self {
        case let .immediate(immVal, _):
            return immVal.immediate.makeImmediate()
        case let .random(lo, hi, _):
            return TensorInitializer.random(from: lo.makeImmediateValue(),
                                            to: hi.makeImmediateValue())
        case let .repeating(immVal, _):
            return TensorInitializer.repeating(immVal.makeImmediateValue())
        }
    }
}

extension OperandNode {
    func makeValue(in env: BasicBlock, module: Module) throws -> Value {
        let type = self.type.makeType()
        let shape = self.shape?.makeShape() ?? .scalar
        switch variable {
        case let .global(name, _):
            guard let global = module.globalValue(named: name) else {
                throw SemanticError.undeclaredVariable(variable)
            }
            guard type == global.type else {
                throw SemanticError.typeMismatch(self, type)
            }
            guard shape.canBroadcast(to: global.shape)
               || global.shape.canBroadcast(to: shape) else {
                throw SemanticError.shapeMismatch(self, shape)
            }
            return global

        case let .immediate(imm, _):
            let immidiate = imm.makeImmediate()
            guard immidiate.typeBase == type.base else {
                let expectedType = DataType(base: immidiate.typeBase, size: type.size)
                throw SemanticError.typeMismatch(self, expectedType)
            }
            return ImmediateValue(type: type, shape: shape, immediate: immidiate)

        case let .temporary(name, _):
            guard let temporary = env.contextualInstruction(named: name) else {
                throw SemanticError.undeclaredVariable(variable)
            }
            guard type == temporary.type else {
                throw SemanticError.typeMismatch(self, type)
            }
            guard shape.canBroadcast(to: temporary.shape)
               || temporary.shape.canBroadcast(to: shape) else {
                throw SemanticError.shapeMismatch(self, shape)
            }
            return temporary
        }
    }
}

extension LoopConditionNode {
    func makeLoopCondition(in env: BasicBlock, module: Module) throws -> LoopInstruction.Condition {
        switch self {
        case let .times(op, _):
            let val = try op.makeValue(in: env, module: module)
            return .times(val)
        case let .untilEqual(lhs, rhs, _):
            let lVal = try lhs.makeValue(in: env, module: module)
            let rVal = try rhs.makeValue(in: env, module: module)
            return .untilEqual(lVal, rVal)
        }
    }
}

extension InstructionDeclarationNode {
    func makeInstruction(in env: BasicBlock, module: Module) throws -> Instruction {
        /// Named instruction
        if let name = name {
            guard !env.containsInstruction(named: name) else {
                throw SemanticError.redeclaredTemporary(self)
            }
            switch instruction {
            case let .aggregate(fun, op, _):
                return AggregationInstruction(name: name,
                                              function: fun,
                                              operand: try op.makeValue(in: env, module: module))

            case let .arithmetic(fun, lhs, rhs, _):
                return ArithmeticInstruction(name: name,
                                             function: fun,
                                             firstOperand: try lhs.makeValue(in: env, module: module),
                                             secondOperand: try rhs.makeValue(in: env, module: module))

            case let .logic(fun, lhs, rhs, _):
                return LogicInstruction(name: name,
                                        function: fun,
                                        firstOperand: try lhs.makeValue(in: env, module: module),
                                        secondOperand: try rhs.makeValue(in: env, module: module))

            case let .binaryReduction(fun, lhs, rhs, _):
                return BinaryReductionInstruction(name: name,
                                                  function: fun,
                                                  firstOperand: try lhs.makeValue(in: env, module: module),
                                                  secondOperand: try rhs.makeValue(in: env, module: module))

            case let .comparison(fun, lhs, rhs, _):
                return ComparisonInstruction(name: name,
                                             function: fun,
                                             firstOperand: try lhs.makeValue(in: env, module: module),
                                             secondOperand: try rhs.makeValue(in: env, module: module))

            case let .concatenate(ops, axis, _):
                let vals = try ops.map { [unowned env] in try $0.makeValue(in: env, module: module) }
                return ConcatenationInstruction(name: name, operands: vals, axis: axis ?? 0)

            case let .elementwise(fun, op, _):
                let val = try op.makeValue(in: env, module: module)
                return ElementwiseInstruction(name: name, function: fun, operand: val)

            case let .load(op, _):
                guard let val = try op.makeValue(in: env, module: module) as? GlobalValue else {
                    throw SemanticError.notGlobal(op)
                }
                return LoadInstruction(name: name, source: val)

            case let .matrixMultiply(lhs, rhs, _):
                return MatrixMultiplicationInstruction(name: name,
                                                       firstOperand: try lhs.makeValue(in: env, module: module),
                                                       secondOperand: try rhs.makeValue(in: env, module: module))

            case let .reduce(fun, op, _):
                return ReductionInstruction(name: name,
                                            function: fun,
                                            operand: try op.makeValue(in: env, module: module))

            case let .shapeCast(op, shape, _):
                return ShapeCastInstruction(name: name,
                                            operand: try op.makeValue(in: env, module: module),
                                            targetShape: shape.makeShape())

            case let .tensorMultiply(lhs, rhs, _):
                return TensorMultiplicationInstruction(name: name,
                                                       firstOperand: try lhs.makeValue(in: env, module: module),
                                                       secondOperand: try rhs.makeValue(in: env, module: module))

            case let .typeCast(op, ty, _):
                return TypeCastInstruction(name: name,
                                           operand: try op.makeValue(in: env, module: module),
                                           targetType: ty.makeType())

            default:
                throw SemanticError.extraneousName(self)
            }
        }
        /// Unnamed instruction
        else {
            switch instruction {
            case let .store(src, dest, _):
                let srcVal = try src.makeValue(in: env, module: module)
                guard let destVal = try dest.makeValue(in: env, module: module) as? GlobalValue else {
                    throw SemanticError.notGlobal(dest)
                }
                return StoreInstruction(source: srcVal, destination: destVal)
            case let .loop(bb, cond, _):
                let condVal = try cond.makeLoopCondition(in: env, module: module)
                let bbVal = try bb.makeBasicBlock(in: env, module: module)
                return LoopInstruction(condition: condVal, body: bbVal)
            default:
                throw SemanticError.missingName(self)
            }
        }
    }
}
