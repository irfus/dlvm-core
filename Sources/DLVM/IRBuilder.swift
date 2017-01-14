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
    
    public init(moduleName: String) {
        module = Module(name: moduleName)
    }
}

extension IRBuilder {
    
    func makeName() -> String {
        return disambiguatedName(for: "v\(globalNameId)")
    }
    
    fileprivate func disambiguatedName(for name: String) -> String {
        if let id = nameIdTable[name] {
            nameIdTable[name] = id + 1
            return name + ".\(id)"
        }
        nameIdTable[name] = 1
        return name
    }

}

// MARK: - Main builder API
public extension IRBuilder {

    func build<Inst: Instruction>(_ instruction: Inst, name: String?) -> Inst {
        if let inst = instruction as? DefiningInstruction {
            inst.name = name ?? makeName()
        }
        guard let block = currentBlock else {
            preconditionFailure("Current block doesn't exist")
        }
        block.append(instruction)
        return instruction
    }

    @discardableResult
    public func declareInput(type: DataType, name: String) -> Input {
        let input = Input(type: type)
        input.name = name
        module.add(input)
        return input
    }

    @discardableResult
    public func declareParameter(type: DataType,
                                 initializer: Initializer,
                                 name: String) -> Parameter {
        let parameter = Parameter(type: type, initializer: initializer)
        parameter.name = name
        module.add(parameter)
        return parameter
    }
    
    @discardableResult
    public func makeBasicBlock(named name: String) -> BasicBlock {
        let block = BasicBlock(name: disambiguatedName(for: name))
        currentBlock = block
        module.append(block)
        return block
    }

    /// Addition of the same type
    @discardableResult
    public func makeArithmeticOperation(_ `operator`: ArithmeticOperator,
                                        _ lhs: Value, _ rhs: Value,
                                        name: String? = nil) -> DefiningInstruction {
        let inst = ArithmeticInstruction(
            operator: `operator`, leftOperand: lhs, rightOperand: rhs)
        return build(inst, name: name)
    }
    
    @discardableResult
    public func makeComparison(_ `operator`: ComparisonPredicate,
                               _ lhs: Value, _ rhs: Value,
                               name: String? = nil) -> DefiningInstruction {
        let inst = ComparisonInstruction(
            predicate: `operator`, leftOperand: lhs, rightOperand: rhs)
        return build(inst, name: name)
    }
    
    @discardableResult
    public func makeTensorProduct(_ lhs: Value, _ rhs: Value,
                                  name: String? = nil) -> DefiningInstruction {
        let inst = TensorProductInstruction(leftOperand: lhs, rightOperand: rhs)
        return build(inst, name: name)
    }
    
    @discardableResult
    public func makeElementwiseCall(_ function: ElementwiseFunction,
                                    _ operand: Value, name: String? = nil) -> DefiningInstruction {
        let inst = ElementwiseCallInstruction(function: function, operand: operand)
        return build(inst, name: name)
    }
    
    @discardableResult
    public func makeAggregateCall(_ function: AggregateFunction,
                                  _ operand: Value, name: String? = nil) -> DefiningInstruction {
        let inst = AggregateCallInstruction(function: function, operand: operand)
        return build(inst, name: name)
    }
    
    @discardableResult
    public func makeConcatenation(
        _ operands: [Value], axis: Int, name: String? = nil) -> DefiningInstruction {
        let inst = ConcatenationInstruction(operands: operands, axis: axis)
        return build(inst, name: name)
    }

    @discardableResult
    public func makeStore<T : GlobalValue>(source: DefiningInstruction, destination: T) -> Instruction {
        let inst = StoreInstruction(source: source, destination: destination)
        return build(inst, name: nil)
    }

}
