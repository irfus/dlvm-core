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

    func build<Inst: Instruction>(_ instruction: Inst) -> Inst {
        guard let block = currentBlock else {
            preconditionFailure("Current block doesn't exist")
        }
        block.append(instruction)
        return instruction
    }

    @discardableResult
    public func declare<T : GlobalValue>(_ globalValue: T) -> T {
        module.add(globalValue)
        return globalValue
    }

    @discardableResult
    public func declareInput(name: String, type: DataType) -> Input {
        let input = Input(name: name, type: type)
        module.add(input)
        return input
    }

    @discardableResult
    public func declareOutput(name: String, type: DataType) -> Output {
        let output = Output(name: name, type: type)
        module.add(output)
        return output
    }

    @discardableResult
    public func declareParameter(name: String, type: DataType,
                                 initializer: Initializer) -> Parameter {
        let parameter = Parameter(name: name, type: type, initializer: initializer)
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

    @discardableResult
    public func makeArithmeticOperation(_ `operator`: ArithmeticOperator,
                                        _ lhs: Value, _ rhs: Value,
                                        name: String? = nil) -> DefiningInstruction {
        let inst = ArithmeticInstruction(name: name ?? makeName(),
                                         operator: `operator`,
                                         leftOperand: lhs, rightOperand: rhs)
        return build(inst)
    }

    @discardableResult
    public func makeNegation(_ operand: Value, name: String? = nil) -> Value {
        let inst = NegationInstruction(name: name ?? makeName(), operand: operand)
        return build(inst)
    }

    @discardableResult
    public func makeComparison(_ `operator`: ComparisonPredicate,
                               _ lhs: Value, _ rhs: Value,
                               name: String? = nil) -> DefiningInstruction {
        let inst = ComparisonInstruction(name: name ?? makeName(),
                                         predicate: `operator`,
                                         leftOperand: lhs, rightOperand: rhs)
        return build(inst)
    }
    
    @discardableResult
    public func makeTensorProduct(_ lhs: Value, _ rhs: Value,
                                  name: String? = nil) -> DefiningInstruction {
        let inst = TensorProductInstruction(name: name ?? makeName(),
                                            leftOperand: lhs, rightOperand: rhs)
        return build(inst)
    }

    @discardableResult
    public func makeElementwiseCall(_ function: ElementwiseFunction,
                                    _ operand: Value, name: String? = nil) -> DefiningInstruction {
        let inst = ElementwiseCallInstruction(name: name ?? makeName(),
                                              function: function, operand: operand)
        return build(inst)
    }
    
    @discardableResult
    public func makeAggregateCall(_ function: AggregateFunction,
                                  _ operand: Value, name: String? = nil) -> DefiningInstruction {
        let inst = AggregateCallInstruction(name: name ?? makeName(),
                                            function: function, operand: operand)
        return build(inst)
    }
    
    @discardableResult
    public func makeConcatenation(
        _ operands: [Value], axis: Int, name: String? = nil) -> DefiningInstruction {
        let inst = ConcatenationInstruction(name: name ?? makeName(),
                                            operands: operands, axis: axis)
        return build(inst)
    }

    @discardableResult
    public func makeStore<T : GlobalValue>(source: DefiningInstruction, destination: T) -> Instruction {
        let inst = StoreInstruction(source: source, destination: destination)
        return build(inst)
    }

}
