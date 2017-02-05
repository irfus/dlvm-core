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
    case instructionMissingParent(Instruction)
    case instructionParentMismatch(Instruction)
    case shapeMismatch(Value, Value, Instruction)
    case typeMismatch(Value, Value, Instruction)
    case unexpectedShape(DefiningInstruction, TensorShape)
    case unexpectedType(DefiningInstruction, DataType)
    case logicOperandNotLogic(DefiningInstruction)
    case loadSourceNotGlobal(LoadInstruction)
    case storeDestinationNotGlobal(StoreInstruction)
    case operandCannotBeGlobal(NamedValue)
}

public protocol SelfVerifiable {
    func verify() throws
}

extension Module : SelfVerifiable {

    open func verify() throws {
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
        
        /// Check naming of child blocks
        /// Check extension type
        var bbNames: Set<String> = []
        for subBlock in descendants {
            guard subBlock.extensionType == self.extensionType else {
                throw VerificationError.extensionTypeMismatch(subBlock, parent: self)
            }
            guard !bbNames.contains(subBlock.name) else {
                throw VerificationError.basicBlockRedeclared(subBlock)
            }
            bbNames.insert(subBlock.name)
        }

        /// Check instructions
        var instNames: Set<String> = []
        for inst in instructions {
            _ = try inst.parent()
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
        guard let module = parent else {
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

    func broadcastedShape(_ lhs: Value, _ rhs: Value) throws -> TensorShape {
        guard let shape = lhs.shape.broadcasted(to: rhs.shape)
                       ?? rhs.shape.broadcasted(to: lhs.shape) else {
            throw VerificationError.shapeMismatch(lhs, rhs, self)
        }
        return shape
    }

    func homogeneousType(_ lhs: Value, _ rhs: Value) throws -> DataType {
        guard lhs.type == rhs.type else {
            throw VerificationError.typeMismatch(lhs, rhs, self)
        }
        return lhs.type
    }
}

// MARK: - Verification helpers
fileprivate extension DefiningInstruction {
    func verify(type: DataType, shape: TensorShape) throws {
        guard self.type == type else {
            throw VerificationError.unexpectedType(self, type)
        }
        guard self.shape == shape else {
            throw VerificationError.unexpectedShape(self, shape)
        }
    }

    func verifyHomomorphic(_ lhs: Value, _ rhs: Value) throws {
        let type = try homogeneousType(lhs, rhs)
        let shape = try broadcastedShape(lhs, rhs)
        try verify(type: type, shape: shape)
    }

    func verifyHomomorphic(_ value: Value) throws {
        try verify(type: value.type, shape: value.shape)
    }
}

public extension Instruction {
    public func verify() throws {
    }
}

public extension ReductionInstruction {
    public func verify() throws {
        try verify(type: operand.type, shape: .scalar)
    }
}

public extension HomomorphicBinaryInstruction {
    public func verify() throws {
        try verifyHomomorphic(firstOperand, secondOperand)
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
        try verify(type: .bool, shape: shape)
    }
}

public extension LoadInstruction {
    public func verify() throws {
        try verify(type: source.type, shape: source.shape)
    }
}

public extension StoreInstruction {
    public func verify() throws {
        /// TODO
        guard destination.isGlobal else {
            throw VerificationError.storeDestinationNotGlobal(self)
        }
    }
}
