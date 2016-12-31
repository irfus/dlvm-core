//
//  IRBuilder.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

public class IRBuilder {
    public let module: Module
    public var currentBlock: BasicBlock?

    fileprivate var globalNameId: Int = 0
    fileprivate var nameIdTable: [String : Int] = [:]

    public init() {
        module = Module()
    }
}

extension IRBuilder {

    func makeName() -> String {
        return disambiguatedName(for: "v\(globalNameId)")
    }

    func disambiguatedName(for name: String) -> String {
        if let id = nameIdTable[name] {
            nameIdTable[name] = id + 1
            return name + ".\(id)"
        }
        nameIdTable[name] = 1
        return name
    }

    @discardableResult
    func build(_ instructionKind: Instruction.Kind, named name: String? = nil) -> VariableOperand {
        precondition(currentBlock != nil, "No current basic block")
        let instruction = Instruction(kind: instructionKind)
        currentBlock!.append(instruction)
        return instruction.makeVariable(named: name ?? makeName())
    }
    
}

// MARK: - Main builder API
public extension IRBuilder {

    @discardableResult
    public func declareTensor(named name: String, dataType: DataType,
                              shape: TensorShape) -> TensorVariable {
        let tensor = TensorVariable(name: name, dataType: dataType, shape: shape)
        module.addDeclaration(tensor)
        return tensor
    }

    @discardableResult
    public func declareScalar(named name: String,
                              type: ScalarType) -> ScalarVariable {
        let scalar = ScalarVariable(name: name, type: type)
        module.addDeclaration(scalar)
        return scalar
    }

    @discardableResult
    public func makeBasicBlock(named name: String) -> BasicBlock {
        let block = BasicBlock(name: disambiguatedName(for: name))
        currentBlock = block
        module.append(block)
        return block
    }

    @discardableResult
    public func makeScalar(_ operand: ScalarOperand, name: String? = nil) -> ScalarVariable {
        return build(.scalar(operand), named: name) as! ScalarVariable
    }

    @discardableResult
    public func makeTensor(dataType: DataType, shape: TensorShape,
                           repeating operand: ScalarOperand, name: String? = nil) -> TensorVariable {
        return build(.tensor(dataType, shape, operand), named: name) as! TensorVariable
    }

    @discardableResult
    public func makeRandom(dataType: DataType, shape: TensorShape,
                           lowerBound: ScalarOperand, upperBound: ScalarOperand,
                           name: String? = nil) -> TensorVariable {
        return build(.random(dataType, shape, lowerBound, upperBound), named: name) as! TensorVariable
    }

    /// Addition of the same type
    @discardableResult
    public func makeBinaryOperation<T:VariableOperand>(
        _ `operator`: Instruction.BinaryOperator,
        _ lhs: T, _ rhs: T, name: String? = nil) -> T {
        return build(.binaryOp(`operator`, lhs, rhs), named: name) as! T
    }

    /// Addition of any operand with tensor
    @discardableResult
    public func makeBinaryOperation(
        _ `operator`: Instruction.BinaryOperator
        , _ lhs: Operand, _ rhs: TensorVariable, name: String? = nil) -> TensorVariable {
        return build(.binaryOp(`operator`, lhs, rhs), named: name) as! TensorVariable
    }

    /// Addition of any operand with tensor
    @discardableResult
    public func makeBinaryOperation(
        _ `operator`: Instruction.BinaryOperator
        , _ lhs: TensorVariable, _ rhs: Operand, name: String? = nil) -> TensorVariable {
        return build(.binaryOp(`operator`, lhs, rhs), named: name) as! TensorVariable
    }

    @discardableResult
    public func makeComparison(_ `operator`: Instruction.ComparisonOperator,
                               _ lhs: Operand, _ rhs: Operand,
                               name: String? = nil) -> ScalarVariable {
        return build(.compare(`operator`, lhs, rhs), named: name) as! ScalarVariable // bool
    }

    @discardableResult
    public func makeDotProduct(_ lhs: TensorVariable, _ rhs: TensorVariable,
                               name: String? = nil) -> TensorVariable {
        return build(.dotProduct(lhs, rhs), named: name) as! TensorVariable
    }

    @discardableResult
    public func makeProduct(_ lhs: TensorVariable, _ rhs: TensorVariable,
                            name: String? = nil) -> TensorVariable {
        return build(.product(lhs, rhs), named: name) as! TensorVariable
    }

    @discardableResult
    public func makeActivation(_ function: Instruction.ActivationFunction,
                               _ argument: TensorVariable, name: String? = nil) -> TensorVariable {
        return build(.activation(function, argument), named: name) as! TensorVariable
    }

    @discardableResult
    public func makeTransformation(
        _ function: Instruction.TransformationFunction, _ argument: TensorVariable,
        name: String? = nil) -> TensorVariable {
        return build(.transformation(function, argument), named: name) as! TensorVariable
    }

    @discardableResult
    public func makeConcatenation(
        _ arguments: [TensorVariable], name: String? = nil) -> TensorVariable {
        return build(.concat(arguments), named: name) as! TensorVariable
    }

    @discardableResult
    public func makePhi<T:VariableOperand>(_ variables: T..., name: String? = nil) -> T {
        return build(.phi(variables), named: name) as! T
    }

    public func makeBranch(condition: VariableOperand,
                           thenBlock: BasicBlock, elseBlock: BasicBlock) {
        build(.condBranch(condition, then: thenBlock, else: elseBlock))
    }

    public func makeBranch(_ basicBlock: BasicBlock) {
        build(.uncondBranch(basicBlock))
    }

    public func makeBranch(condition: VariableOperand,
                           thenBlock: String, elseBlock: String) {
        guard let thenBB = module.basicBlock(named: thenBlock),
            let elseBB = module.basicBlock(named: elseBlock) else {
            preconditionFailure("Basic block not present")
        }
        build(.condBranch(condition, then: thenBB, else: elseBB))
    }

    public func makeBranch(_ basicBlockName: String) {
        guard let block = module.basicBlock(named: basicBlockName) else {
            preconditionFailure("Basic block not present")
        }
        build(.uncondBranch(block))
    }

}
