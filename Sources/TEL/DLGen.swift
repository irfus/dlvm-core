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
    var variables: [String : DLVM.Variable] = [:]
    
    subscript(key: String) -> DLVM.Variable? {
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
    let builder = IRBuilder()
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
        /// Initialize parameters
        let initBB = builder.makeBasicBlock(named: "init")
        builder.module.entryBlock = initBB
        for param in program.parameters {
            let variable: TensorVariable
            switch param.initializer {
            case let .constant(.int(i)):
                variable = builder.makeTensor(dataType: program.dataType,
                                              shape: param.shape,
                                              repeating: Immediate.int(i),
                                              name: param.name)
            case let .constant(.float(f)):
                variable = builder.makeTensor(dataType: program.dataType,
                                              shape: param.shape,
                                              repeating: Immediate.float(f),
                                              name: param.name)
            case let .random(.int(i1), .int(i2)):
                variable = builder.makeRandom(dataType: program.dataType,
                                              shape: param.shape,
                                              lowerBound: Immediate.int(i1),
                                              upperBound: Immediate.int(i2),
                                              name: param.name)
            case let .random(.float(f1), .float(f2)):
                variable = builder.makeRandom(dataType: program.dataType,
                                              shape: param.shape,
                                              lowerBound: Immediate.float(f1),
                                              upperBound: Immediate.float(f2),
                                              name: param.name)
            default:
                preconditionFailure("This should not have passed Sema")
            }
            env.variables[variable.name] = variable
        }
        /// TODO
        return builder.module
    }
    
}
