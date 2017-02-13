//
//  Verification.swift
//  DLVM
//
//  Created by Richard Wei on 12/25/16.
//
//

public enum VerificationError : Error {
    case basicBlockRedeclared(BasicBlock)
    case extensionTypeMismatch(BasicBlock, parent: BasicBlock)
    case redeclaredInstruction(Instruction)
    case blockMissingModule(BasicBlock)
    case blockModuleMismatch(BasicBlock, Module)
    case shapeMismatch(Use, Use, Instruction)
    case typeMismatch(Use, Use, Instruction)
    case unexpectedShape(Use, TensorShape)
    case unexpectedType(Use, DataType)
    case cannotShapeCast(Instruction)
    case cannotMatrixMultiply(Instruction)
    case cannotConcatenate(Instruction)
    case operandsEmpty(Instruction)
}

public protocol SelfVerifiable {
    func verify() throws
}

extension Module : SelfVerifiable {

    open func verify() throws {
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
            if let name = inst.name {
                guard !instNames.contains(name) else {
                    throw VerificationError.redeclaredInstruction(inst)
                }
                instNames.insert(name)
            }
            // try inst.verify()
        }
    }

}

fileprivate extension Instruction {

    func broadcastedShape(_ lhs: Use, _ rhs: Use) throws -> TensorShape {
        guard let shape = lhs.shape.broadcasted(to: rhs.shape)
                       ?? rhs.shape.broadcasted(to: lhs.shape) else {
            throw VerificationError.shapeMismatch(lhs, rhs, self)
        }
        return shape
    }

    func homomorphicShape(_ lhs: Use, _ rhs: Use) throws -> TensorShape {
        guard lhs.shape == rhs.shape else {
            throw VerificationError.shapeMismatch(lhs, rhs, self)
        }
        return lhs.shape
    }

    func homogeneousType(_ lhs: Use, _ rhs: Use) throws -> DataType {
        guard lhs.type == rhs.type else {
            throw VerificationError.typeMismatch(lhs, rhs, self)
        }
        return lhs.type
    }

    func verifyHomomorphic(_ lhs: Use, _ rhs: Use) throws {
        _ = try homogeneousType(lhs, rhs)
        _ = try broadcastedShape(lhs, rhs)
    }

    func verifyHomomorphicBroadcasted(_ lhs: Use, _ rhs: Use) throws {
        _ = try homogeneousType(lhs, rhs)
        _ = try broadcastedShape(lhs, rhs)
    }
}

/*

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

 */
