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

class VerificationEnvironment {
    var temporaries: [String : NamedValue] = [:]
    var globals: [String : GlobalValue] = [:]

    func insertGlobal(_ value: GlobalValue) {
        globals[value.name] = value
    }

    func insertTemporary(_ value: NamedValue) {
        temporaries[value.name] = value
    }

    func global(named name: String) -> GlobalValue? {
        return globals[name]
    }

    func temporary(named name: String) -> NamedValue? {
        return temporaries[name]
    }

    func containsGlobal(named name: String) -> Bool {
        return globals.keys.contains(name)
    }

    func containsTemporary(named name: String) -> Bool {
        return temporaries.keys.contains(name)
    }
}

protocol Verifiable {
    func verify(in environment: VerificationEnvironment) throws
}

extension Input : Verifiable {
    func verify(in environment: VerificationEnvironment) throws {
        guard !environment.containsGlobal(named: name) else {
            throw VerificationError.nameRedeclared(name)
        }
        environment.insertGlobal(self)
    }
}

extension Output : Verifiable {
    func verify(in environment: VerificationEnvironment) throws {
        guard !environment.containsGlobal(named: name) else {
            throw VerificationError.nameRedeclared(name)
        }
        environment.insertGlobal(self)
    }
}

extension Parameter : Verifiable {
    func verify(in environment: VerificationEnvironment) throws {
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
    
    func verify(in environment: VerificationEnvironment) throws {
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
