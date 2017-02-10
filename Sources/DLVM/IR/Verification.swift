//
//  Verification.swift
//  DLVM
//
//  Created by Richard Wei on 12/25/16.
//
//

public enum VerificationError : Error {
    case globalMissingParent(Global)
    case globalRedeclared(Global)
    case basicBlockRedeclared(BasicBlock)
    case extensionTypeMismatch(BasicBlock, parent: BasicBlock)
    case redeclaredInstruction(DefiningInstruction)
    case blockMissingModule(BasicBlock)
    case blockModuleMismatch(BasicBlock, Module)
    case blockParentMismatch(BasicBlock, parent: BasicBlock)
    case instructionMissingParent(Instruction)
    case instructionParentMismatch(Instruction, parent: BasicBlock)
    case shapeMismatch(ValueRepresentation, ValueRepresentation, Instruction)
    case typeMismatch(ValueRepresentation, ValueRepresentation, Instruction)
    case unexpectedShape(DefiningInstruction, TensorShape)
    case unexpectedType(DefiningInstruction, DataType)
    case logicOperandNotLogic(DefiningInstruction)
    case cannotShapeCast(ShapeCastInstruction)
    case cannotTensorMultiply(TensorMultiplicationInstruction)
    case cannotMatrixMultiply(MatrixMultiplicationInstruction)
    case cannotConcatenate(Value, ConcatenationInstruction)
    case operandsEmpty(Instruction)
}

public protocol SelfVerifiable {
    func verify() throws
}

extension Module : SelfVerifiable {

    open func verify() throws {
        /// Update
        updateAnalysisInformation()
        /// Check for redeclared global values
        var globalValueNames: Set<String> = []
        func checkForRedeclaration<T: Global, S: Sequence>(in values: S) throws
            where S.Iterator.Element == T {
            for value in values {
                _ = try value.parent()
                guard !globalValueNames.contains(value.name) else {
                    throw VerificationError.globalRedeclared(value)
                }
            }
        }
        try checkForRedeclaration(in: inputs)
        try checkForRedeclaration(in: parameters)
        try checkForRedeclaration(in: outputs)

        /// Check basic blocks
        for bb in basicBlocks {
            /// Check module reference
            guard let bbModule = bb.module else {
                throw VerificationError.blockMissingModule(bb)
            }
            guard self === bbModule else {
                throw VerificationError.blockModuleMismatch(bb, self)
            }
            try bb.verify()
        }
    }
    
}

extension BasicBlock : SelfVerifiable {

    open func verify() throws {
        /// Check instructions
        var instNames: Set<String> = []
        for inst in instructions {
            let parent = try inst.parent()
            guard parent === self else {
                throw VerificationError.instructionParentMismatch(inst, parent: self)
            }
            if let defInst = inst as? DefiningInstruction {
                guard !instNames.contains(defInst.name) else {
                    throw VerificationError.redeclaredInstruction(defInst)
                }
                instNames.insert(defInst.name)
            }
            try inst.verify()
        }
    }

}

fileprivate extension Global {
    func parent() throws -> Module {
        guard let module = module else {
            throw VerificationError.globalMissingParent(self)
        }
        return module
    }
}

fileprivate extension Instruction {
    func parent() throws -> BasicBlock {
        guard let bb = parent else {
            throw VerificationError.instructionMissingParent(self)
        }
        return bb
    }

    func broadcastedShape(_ lhs: ValueRepresentation, _ rhs: ValueRepresentation) throws -> TensorShape {
        guard let shape = lhs.shape.broadcasted(to: rhs.shape)
                       ?? rhs.shape.broadcasted(to: lhs.shape) else {
            throw VerificationError.shapeMismatch(lhs, rhs, self)
        }
        return shape
    }

    func homomorphicShape(_ lhs: ValueRepresentation, _ rhs: ValueRepresentation) throws -> TensorShape {
        guard lhs.shape == rhs.shape else {
            throw VerificationError.shapeMismatch(lhs, rhs, self)
        }
        return lhs.shape
    }

    func homogeneousType(_ lhs: ValueRepresentation, _ rhs: ValueRepresentation) throws -> DataType {
        guard lhs.type == rhs.type else {
            throw VerificationError.typeMismatch(lhs, rhs, self)
        }
        return lhs.type
    }

    func verifyHomomorphic(_ lhs: ValueRepresentation, _ rhs: ValueRepresentation) throws {
        _ = try homogeneousType(lhs, rhs)
        _ = try broadcastedShape(lhs, rhs)
    }

    func verifyHomomorphicBroadcasted(_ lhs: ValueRepresentation, _ rhs: ValueRepresentation) throws {
        _ = try homogeneousType(lhs, rhs)
        _ = try broadcastedShape(lhs, rhs)
    }
}

// MARK: - Verification helpers
fileprivate extension DefiningInstruction {
    func verifyDeclaration(type: DataType, shape: TensorShape) throws {
        guard self.type == type else {
            throw VerificationError.unexpectedType(self, type)
        }
        guard self.shape == shape else {
            throw VerificationError.unexpectedShape(self, shape)
        }
    }

    func verifyHomomorphicBroadcasted(_ lhs: ValueRepresentation, _ rhs: ValueRepresentation) throws {
        let type = try homogeneousType(lhs, rhs)
        let shape = try broadcastedShape(lhs, rhs)
        try verifyDeclaration(type: type, shape: shape)
    }

    func verifyHomomorphic(_ value: ValueRepresentation) throws {
        try verifyDeclaration(type: value.type, shape: value.shape)
    }
}

public extension ReductionInstruction {
    public func verify() throws {
        try verifyDeclaration(type: operand.type,
                              shape: axis.flatMap(operand.shape.droppingDimension) ?? .scalar)
    }
}

public extension HomomorphicBinaryInstruction {
    public func verify() throws {
        try verifyHomomorphicBroadcasted(firstOperand, secondOperand)
        if function is LogicInstruction, type != .bool {
            throw VerificationError.logicOperandNotLogic(self)
        }
    }
}

public extension HomomorphicUnaryInstruction {
    public func verify() throws {
        try verifyHomomorphic(operand)
    }
}

public extension ComparisonInstruction {
    public func verify() throws {
        _ = try homogeneousType(firstOperand, secondOperand)
        let shape = try broadcastedShape(firstOperand, secondOperand)
        try verifyDeclaration(type: .bool, shape: shape)
    }
}

public extension LoadInstruction {
    public func verify() throws {
        try verifyDeclaration(type: source.type, shape: source.shape)
    }
}

public extension StoreInstruction {
    public func verify() throws {
        try verifyHomomorphic(source, destination)
    }
}

public extension ExportInstruction {
    public func verify() throws {
        try verifyHomomorphic(source, destination)
    }
}

public extension TypeCastInstruction {
    public func verify() throws {
        /// TODO: Check for data type conformance
        try verifyDeclaration(type: target, shape: operand.shape)
    }
}

public extension ShapeCastInstruction {
    public func verify() throws {
        guard operand.shape.canBroadcast(to: target)
           || operand.shape.contiguousSize == target.contiguousSize else {
            throw VerificationError.cannotShapeCast(self)
        }
        try verifyDeclaration(type: operand.type, shape: target)
    }
}

public extension TensorMultiplicationInstruction {
    public func verify() throws {
        let type = try homogeneousType(firstOperand, secondOperand)
        guard let product = firstOperand.shape.multiplied(with: secondOperand.shape)
                         ?? secondOperand.shape.multiplied(with: firstOperand.shape) else {
            throw VerificationError.cannotTensorMultiply(self)
        }
        try verifyDeclaration(type: type, shape: product)
    }
}

public extension MatrixMultiplicationInstruction {
    public func verify() throws {
        let type = try homogeneousType(firstOperand, secondOperand)
        guard let product = firstOperand.shape.matrixMultiplied(with: secondOperand.shape)
                         ?? secondOperand.shape.matrixMultiplied(with: firstOperand.shape) else {
            throw VerificationError.cannotMatrixMultiply(self)
        }
        try verifyDeclaration(type: type, shape: product)
    }
}

public extension ConcatenationInstruction {
    public func verify() throws {
        guard let firstOp = operands.first else {
            throw VerificationError.operandsEmpty(self)
        }
        /// TODO: Check repeated operands
        let (type, shape) = try operands.dropFirst()
                                        .reduce((firstOp.type, firstOp.shape),
                                        { acc, next in
            let type = try homogeneousType(operands[0], next)
            guard let shape = acc.1.concatenating(with: next.shape, alongDimension: axis) else {
                throw VerificationError.cannotConcatenate(next, self)
            }
            return (type, shape)
        })
        try verifyDeclaration(type: type, shape: shape)
    }
}

public extension PhiInstruction {
    public func verify() throws {
        guard let firstOp = operands.first else {
            throw VerificationError.operandsEmpty(self)
        }
        /// TODO: Check repeated operands
        let (type, shape) = try operands.dropFirst()
                                        .reduce((firstOp.type, firstOp.shape),
                                        { acc, next in
            let type = try homogeneousType(firstOp, next)
            let shape = try homomorphicShape(firstOp, next)
            return (type, shape)
        })
        try verifyDeclaration(type: type, shape: shape)
    }
}

public extension BranchInstruction {
    public func verify() throws {
    }
}
