//
//  DLGen.swift
//  DLVM
//
//  Created by Richard Wei on 12/23/16.
//
//  This file contains DLVM IR generator from TEL
//

import DLVM

public extension Program {
    
    public func makeModule() -> Module {
        let cgen = CodeGenerator(program: self)
        let module = cgen.makeModule()
        return module
    }
    
}

struct CodeGenEnvironment {
    var variables: [String : DLVM.VariableOperand] = [:]
    
    subscript(key: String) -> DLVM.VariableOperand? {
        get {
            return variables[key]
        }
        set {
            variables[key] = newValue
        }
    }
}

class CodeGenerator {
    
    let program: Program
    lazy var builder: IRBuilder = IRBuilder(moduleName: self.program.moduleName)
    var env = CodeGenEnvironment()
    
    init(program: Program) {
        self.program = program
    }
    
    func makeModule() -> Module {
        /// Declare input
        for input in program.inputs {
            let variable = builder.declareTensor(named: input.name,
                                                 dataType: program.dataType,
                                                 shape: input.shape)
            env[variable.name] = variable
        }
        /// Define globals
        for param in program.parameters {
            let def: TensorDefinition
            switch param.initializer {
            case let .constant(.int(i)):
                def = ImmediateTensorDefinition(dataType: program.dataType,
                                                shape: param.shape,
                                                value: .int(i))

            case let .constant(.float(f)):
                def = ImmediateTensorDefinition(dataType: program.dataType,
                                                shape: param.shape,
                                                value: .float(f))
                
            case let .random(.int(i1), .int(i2)):
                def = RandomizingTensorDefinition(
                    dataType: program.dataType, shape: param.shape,
                    lowerBound: .int(i1), upperBound: .int(i2)
                )

            case let .random(.float(f1), .float(f2)):
                def = RandomizingTensorDefinition(
                    dataType: program.dataType, shape: param.shape,
                    lowerBound: .float(f1), upperBound: .float(f2)
                )

            default:
                preconditionFailure("This should not have passed Sema")
            }

            let variable = builder.declareTensor(def, name: param.name)
            env.variables[variable.name] = variable
        }

        /// Entry block
        let initBB = builder.makeBasicBlock(named: "init")
        builder.module.entryBlock = initBB

        /// TODO
        return builder.module
    }

}
