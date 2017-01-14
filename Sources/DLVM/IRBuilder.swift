//
//  IRBuilder.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

open class IRBuilder {
    open let module: Module
    open var currentBlock: BasicBlock?
    
    fileprivate var globalNameId: Int = 0
    fileprivate var nameIdTable: [String : Int] = [:]

    public init(moduleName: String) {
        module = Module(name: moduleName)
    }
}

extension IRBuilder {
    
    func makeName() -> String {
        defer { globalNameId += 1 }
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

}

// MARK: - Main builder API
extension IRBuilder {

    private func build<Inst : Instruction>(_ instruction: Inst) -> Inst {
        guard let block = currentBlock else {
            preconditionFailure("Current block doesn't exist")
        }
        block.append(instruction)
        return instruction
    }

    @discardableResult
    open func declare<T : GlobalValue>(_ globalValue: T) -> T {
        module.add(globalValue)
        return globalValue
    }

    @discardableResult
    open func declareInput(name: String, type: DataType) -> Input {
        let input = Input(name: name, type: type)
        module.add(input)
        return input
    }

    @discardableResult
    open func declareOutput(name: String, type: DataType) -> Output {
        let output = Output(name: name, type: type)
        module.add(output)
        return output
    }

    @discardableResult
    open func declareParameter(name: String, type: DataType,
                                 initializer: Initializer) -> Parameter {
        let parameter = Parameter(name: name, type: type, initializer: initializer)
        parameter.name = name
        module.add(parameter)
        return parameter
    }
    
    @discardableResult
    open func makeBasicBlock(named name: String) -> BasicBlock {
        let block = BasicBlock(name: disambiguatedName(for: name))
        currentBlock = block
        module.append(block)
        return block
    }

    @discardableResult
    open func makeArithmeticOperation(_ `operator`: ArithmeticOperator,
                                        _ lhs: Value, _ rhs: Value,
                                        name: String? = nil) -> DefiningInstruction {
        let inst = ArithmeticInstruction(name: name ?? makeName(),
                                         operator: `operator`,
                                         leftOperand: lhs, rightOperand: rhs)
        return build(inst)
    }

    @discardableResult
    open func makeNegation(_ operand: Value, name: String? = nil) -> DefiningInstruction {
        let inst = NegationInstruction(name: name ?? makeName(), operand: operand)
        return build(inst)
    }

    @discardableResult
    open func makeComparison(_ `operator`: ComparisonPredicate,
                               _ lhs: Value, _ rhs: Value,
                               name: String? = nil) -> DefiningInstruction {
        let inst = ComparisonInstruction(name: name ?? makeName(),
                                         predicate: `operator`,
                                         leftOperand: lhs, rightOperand: rhs)
        return build(inst)
    }
    
    @discardableResult
    open func makeTensorProduct(_ lhs: Value, _ rhs: Value,
                                  name: String? = nil) -> DefiningInstruction {
        let inst = TensorProductInstruction(name: name ?? makeName(),
                                            leftOperand: lhs, rightOperand: rhs)
        return build(inst)
    }

    @discardableResult
    open func makeElementwiseCall(_ function: ElementwiseFunction,
                                    _ operand: Value, name: String? = nil) -> DefiningInstruction {
        let inst = ElementwiseCallInstruction(name: name ?? makeName(),
                                              function: function, operand: operand)
        return build(inst)
    }
    
    @discardableResult
    open func makeAggregateCall(_ function: AggregateFunction,
                                  _ operand: Value, name: String? = nil) -> DefiningInstruction {
        let inst = AggregateCallInstruction(name: name ?? makeName(),
                                            function: function, operand: operand)
        return build(inst)
    }
    
    @discardableResult
    open func makeConcatenation(
        _ operands: [Value], axis: Int, name: String? = nil) -> DefiningInstruction {
        let inst = ConcatenationInstruction(name: name ?? makeName(),
                                            operands: operands, axis: axis)
        return build(inst)
    }

    @discardableResult
    open func makeShapeCast(_ operand: Value, targetShape: TensorShape,
                            name: String? = nil) -> DefiningInstruction {
        let inst = ShapeCastInstruction(name: name ?? makeName(),
                                        operand: operand, targetShape: targetShape)
        return build(inst)
    }

    @discardableResult
    open func makeTypeCast(_ operand: Value, targetBase: TypeBase, targetSize: Int,
                           name: String? = nil) -> DefiningInstruction {
        let inst = TypeCastInstruction(name: name ?? makeName(),
                                       operand: operand,
                                       targetBase: targetBase, targetSize: targetSize)
        return build(inst)
    }

    @discardableResult
    open func makeStore<T : GlobalValue>(source: Value, destination: T) -> Instruction {
        let inst = StoreInstruction(source: source, destination: destination)
        return build(inst)
    }

}
