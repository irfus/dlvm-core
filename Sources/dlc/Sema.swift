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
    case typeMismatch(OperandNode, DataType)
    case shapeMismatch(OperandNode, TensorShape)
    case storeDestinationNotGlobal(OperandNode)
    case loadSourceNotGlobal(OperandNode)
    case extraneousName(InstructionDeclarationNode)
    case missingName(InstructionDeclarationNode)
    case initializerTypeMismatch(InitializerNode, TypeNode)
}

extension ModuleNode {
    func makeModule() throws -> Module {
        let module = Module(name: name)
        let env = VerificationEnvironment()

        for decl in declarations {
            let value = try decl.makeDeclaration(in: env)
            env.insertGlobal(value)
            module.add(value)
        }

        for bbNode in basicBlocks {
            let bb = try bbNode.makeBasicBlock(in: env)
            module.append(bb)
        }
        return module
    }
}

extension BasicBlockNode {
    func makeBasicBlock(in env: VerificationEnvironment) throws -> BasicBlock {
        guard !env.containsBasicBlock(named: name) else {
            throw SemanticError.redeclaredBasicBlock(self)
        }
        let bb = BasicBlock(name: name)
        for instNode in instructions {
            let inst = try instNode.makeInstruction(in: env)
            if let temp = inst as? NamedValue {
                env.insertTemporary(temp)
            }
            bb.append(inst)
        }
        return bb
    }
}

extension DeclarationNode {
    func makeDeclaration(in env: VerificationEnvironment) throws -> GlobalValue {
        guard !env.containsGlobal(named: name) else {
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
                             shape: shape.makeShape(),
                             initializer: initializer.makeInitializer())

        case .input:
            return Input(name: name, type: type.makeType(), shape: shape.makeShape())

        case .output:
            return Output(name: name, type: type.makeType(), shape: shape.makeShape())

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
            return immVal.immediate.makeImmediate()
        }
    }
}

extension OperandNode {
    func makeValue(in env: VerificationEnvironment) throws -> Value {
        let type = self.type.makeType()
        let shape = self.shape?.makeShape() ?? .scalar
        switch variable {
        case let .global(name, _):
            guard let global = env.global(named: name) else {
                throw SemanticError.undeclaredVariable(variable)
            }
            guard type == global.type else {
                throw SemanticError.typeMismatch(self, type)
            }
            guard shape == global.shape else {
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
            guard let temporary = env.temporary(named: name) else {
                throw SemanticError.undeclaredVariable(variable)
            }
            guard type == temporary.type else {
                throw SemanticError.typeMismatch(self, type)
            }
            guard shape == temporary.shape else {
                throw SemanticError.shapeMismatch(self, shape)
            }
            return temporary
        }
    }
}

extension InstructionDeclarationNode {
    func makeInstruction(in env: VerificationEnvironment) throws -> Instruction {
        /// Named instruction
        if let name = name {
            guard !env.containsTemporary(named: name) else {
                throw SemanticError.redeclaredTemporary(self)
            }
            switch instruction {
            case let .aggregate(fun, op, _):
                return AggregationInstruction(name: name,
                                              function: fun,
                                              operand: try op.makeValue(in: env))

            case let .arithmetic(fun, lhs, rhs, _):
                return ArithmeticInstruction(name: name,
                                             function: fun,
                                             firstOperand: try lhs.makeValue(in: env),
                                             secondOperand: try rhs.makeValue(in: env))

            case let .binaryReduction(fun, lhs, rhs, _):
                return BinaryReductionInstruction(name: name,
                                                  function: fun,
                                                  firstOperand: try lhs.makeValue(in: env),
                                                  secondOperand: try rhs.makeValue(in: env))

            case let .comparison(fun, lhs, rhs, _):
                return ComparisonInstruction(name: name,
                                             function: fun,
                                             firstOperand: try lhs.makeValue(in: env),
                                             secondOperand: try rhs.makeValue(in: env))

            case let .concatenate(ops, axis, _):
                let vals = try ops.map { [unowned env] in try $0.makeValue(in: env) }
                return ConcatenationInstruction(name: name, operands: vals, axis: axis ?? 0)

            case let .elementwise(fun, op, _):
                let val = try op.makeValue(in: env)
                return ElementwiseInstruction(name: name, function: fun, operand: val)

            case let .load(op, _):
                guard let val = try op.makeValue(in: env) as? GlobalValue else {
                    throw SemanticError.loadSourceNotGlobal(op)
                }
                return LoadInstruction(name: name, source: val)

            case let .matrixMultiply(lhs, rhs, _):
                return MatrixMultiplicationInstruction(name: name,
                                                       firstOperand: try lhs.makeValue(in: env),
                                                       secondOperand: try rhs.makeValue(in: env))

            case let .reduce(fun, op, _):
                return ReductionInstruction(name: name,
                                            function: fun,
                                            operand: try op.makeValue(in: env))

            case let .shapeCast(op, shape, _):
                return ShapeCastInstruction(name: name,
                                            operand: try op.makeValue(in: env),
                                            targetShape: shape.makeShape())

            case let .tensorMultiply(lhs, rhs, _):
                return TensorMultiplicationInstruction(name: name,
                                                       firstOperand: try lhs.makeValue(in: env),
                                                       secondOperand: try rhs.makeValue(in: env))

            case let .typeCast(op, ty, _):
                return TypeCastInstruction(name: name,
                                           operand: try op.makeValue(in: env),
                                           targetType: ty.makeType())

            default:
                throw SemanticError.extraneousName(self)
            }
        }
        /// Unnamed instruction
        else {
            switch instruction {
            case let .store(src, dest, _):
                let srcVal = try src.makeValue(in: env)
                guard let destVal = try dest.makeValue(in: env) as? GlobalValue else {
                    throw SemanticError.storeDestinationNotGlobal(dest)
                }
                return StoreInstruction(source: srcVal, destination: destVal)
            default:
                throw SemanticError.missingName(self)
            }
        }
    }
}
