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
                                      name: String? = nil) -> ArithmeticInstruction {
        let inst = ArithmeticInstruction(name: name ?? makeName(),
                                         function: `operator`,
                                         firstOperand: lhs, secondOperand: rhs)
        return build(inst)
    }
    
    @discardableResult
    open func makeComparison(_ `operator`: ComparisonPredicate,
                             _ lhs: Value, _ rhs: Value,
                             name: String? = nil) -> ComparisonInstruction {
        let inst = ComparisonInstruction(name: name ?? makeName(),
                                         predicate: `operator`,
                                         firstOperand: lhs, secondOperand: rhs)
        return build(inst)
    }
    
    @discardableResult
    open func makeTensorMultiplication(_ lhs: Value, _ rhs: Value,
                                       name: String? = nil) -> TensorMultiplicationInstruction {
        let inst = TensorMultiplicationInstruction(name: name ?? makeName(),
                                                   firstOperand: lhs, secondOperand: rhs)
        return build(inst)
    }
    
    @discardableResult
    open func makeMatrixMultiplication(_ lhs: Value, _ rhs: Value,
                                       name: String? = nil) -> MatrixMultiplicationInstruction {
        let inst = MatrixMultiplicationInstruction(name: name ?? makeName(),
                                                   firstOperand: lhs, secondOperand: rhs)
        return build(inst)
    }
    
    @discardableResult
    open func makeElementwiseTransformation(_ function: ElementwiseFunction,
                                            _ operand: Value,
                                            name: String? = nil) -> ElementwiseTransformationInstruction {
        let inst = ElementwiseTransformationInstruction(name: name ?? makeName(),
                                                        function: function, operand: operand)
        return build(inst)
    }
    
    @discardableResult
    open func makeAggregateTransformation(_ function: AggregateFunction,
                                          _ operand: Value,
                                          name: String? = nil) -> AggregateTransformationInstruction {
        let inst = AggregateTransformationInstruction(name: name ?? makeName(),
                                                      function: function, operand: operand)
        return build(inst)
    }

    @discardableResult
    open func makeScan(_ function: ScanFunction, _ operand: Value,
                       name: String? = nil) -> ScanInstruction {
        let inst = ScanInstruction(name: name ?? makeName(), function: function, operand: operand)
        return build(inst)
    }

    @discardableResult
    open func makeReduction(_ function: ReductionFunction, _ operand: Value,
                            name: String? = nil) -> ReductionInstruction {
        let inst = ReductionInstruction(name: name ?? makeName(), function: function, operand: operand)
        return build(inst)
    }

    @discardableResult
    open func makeBinaryReduction(_ function: BinaryReductionFunction,
                                  _ firstOperand: Value, _ secondOperand: Value,
                                  name: String? = nil) -> BinaryReductionInstruction {
        let inst = BinaryReductionInstruction(name: name ?? makeName(), function: function,
                                              firstOperand: firstOperand, secondOperand: secondOperand)
        return build(inst)
    }
    
    @discardableResult
    open func makeConcatenation(_ operands: [Value], axis: Int,
                                name: String? = nil) -> ConcatenationInstruction {
        let inst = ConcatenationInstruction(name: name ?? makeName(),
                                            operands: operands, axis: axis)
        return build(inst)
    }
    
    @discardableResult
    open func makeShapeCast(_ operand: Value, targetShape: TensorShape,
                            name: String? = nil) -> ShapeCastInstruction {
        let inst = ShapeCastInstruction(name: name ?? makeName(),
                                        operand: operand, targetShape: targetShape)
        return build(inst)
    }
    
    @discardableResult
    open func makeTypeCast(_ operand: Value, targetBase: TypeBase, targetSize: Int,
                           name: String? = nil) -> TypeCastInstruction {
        let inst = TypeCastInstruction(name: name ?? makeName(),
                                       operand: operand,
                                       targetBase: targetBase, targetSize: targetSize)
        return build(inst)
    }
    
    @discardableResult
    open func makeLoad(_ source: Input, name: String? = nil) -> LoadInstruction {
        let inst = LoadInstruction(name: name ?? makeName(), source: source)
        return build(inst)
    }
    
    @discardableResult
    open func makeStore(_ source: Value, to destination: GlobalValue) -> StoreInstruction {
        let inst = StoreInstruction(source: source, destination: destination)
        return build(inst)
    }
    
}
