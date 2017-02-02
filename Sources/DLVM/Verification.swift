//
//  Verification.swift
//  DLVM
//
//  Created by Richard Wei on 12/25/16.
//
//

public enum VerificationError : Error {
    case globalValueRedeclared(Value)
    case typeMismatch(Value, Value)
}

extension Module {

    open func verify() throws {
        /// Check for redeclared global values
        var globalValueNames: Set<String> = []
        func checkForRedeclaration<T: GlobalValue, S: Sequence>(in values: S) throws
            where S.Iterator.Element == T {
            for value in values {
                guard !globalValueNames.contains(value.name) else {
                    throw VerificationError.globalValueRedeclared(value)
                }
            }
        }
        try checkForRedeclaration(in: inputs)
        try checkForRedeclaration(in: parameters)
        try checkForRedeclaration(in: outputs)

        /// Check basic blocks
        for bb in basicBlocks {
            try bb.verify(in: self)
        }
    }
    
}

extension BasicBlock {

    open func verify(in module: Module) throws {
        
    }
    
}
