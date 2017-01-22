//
//  Verification.swift
//  DLVM
//
//  Created by Richard Wei on 12/25/16.
//
//

public enum VerificationError : Error {
    case nameRedeclared(String)
    case noOutput
    case typeMismatch(Value, Value)
}

open class VerificationEnvironment {
    private var temporaries: [String : NamedValue] = [:]
    private var globals: [String : GlobalValue] = [:]
    private var basicBlocks: [String : BasicBlock] = [:]

    public init() {}

    open func insertGlobal(_ value: GlobalValue) {
        globals[value.name] = value
    }

    open func insertTemporary(_ value: NamedValue) {
        temporaries[value.name] = value
    }

    open func global(named name: String) -> GlobalValue? {
        return globals[name]
    }

    open func temporary(named name: String) -> NamedValue? {
        return temporaries[name]
    }

    open func containsGlobal(named name: String) -> Bool {
        return globals.keys.contains(name)
    }

    open func containsTemporary(named name: String) -> Bool {
        return temporaries.keys.contains(name)
    }
}

public protocol Verifiable {
    func verify(in environment: VerificationEnvironment) throws
}

extension Input : Verifiable {
    public func verify(in environment: VerificationEnvironment) throws {
        guard !environment.containsGlobal(named: name) else {
            throw VerificationError.nameRedeclared(name)
        }
        environment.insertGlobal(self)
    }
}

extension Output : Verifiable {
    public func verify(in environment: VerificationEnvironment) throws {
        guard !environment.containsGlobal(named: name) else {
            throw VerificationError.nameRedeclared(name)
        }
        environment.insertGlobal(self)
    }
}

extension Parameter : Verifiable {
    public func verify(in environment: VerificationEnvironment) throws {
        guard !environment.containsGlobal(named: name) else {
            throw VerificationError.nameRedeclared(name)
        }
        environment.insertGlobal(self)
    }
}

extension Module : Verifiable {

    open func verify() throws {
        return try verify(in: VerificationEnvironment())
    }
    
    public func verify(in environment: VerificationEnvironment) throws {
        for input in inputs {
            try input.verify(in: environment)
        }
        for parameter in parameters {
            try parameter.verify(in: environment)
        }
        guard outputs.isEmpty else {
            throw VerificationError.noOutput
        }
        for output in outputs {
            try output.verify(in: environment)
        }
    }
    
}
