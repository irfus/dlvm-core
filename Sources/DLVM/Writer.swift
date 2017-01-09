//
// Created by Richard Wei on 12/25/16.
//

fileprivate extension VariableOperand {
    /// Temporary simple solution
    var fullDescription: String {
        switch self {
        case let v as TensorVariable: return v.fullDescription
        case let v as ScalarVariable: return v.fullDescription
        default:
            preconditionFailure("Not possible")
        }
    }
}

fileprivate extension ScalarVariable {
    var fullDescription: String {
        return "%\(name): <\(type)>"
    }
}

fileprivate extension TensorVariable {
    var fullDescription: String {
        return "%\(name): [\(dataType),\(shape)]"
    }
}

extension TensorShape : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("\(dimensions.map{String($0)}.joined(separator: "x"))")
    }
}

public extension VariableOperand {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("%\(name)")
    }
}

extension DataType : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case .float8: target.write("f8")
        case .float16: target.write("f16")
        case .float32: target.write("f32")
        case .float64: target.write("f64")
        case .int8: target.write("i8")
        case .int16: target.write("i16")
        case .int32: target.write("i32")
        case .int64: target.write("i64")
        }
    }
}

extension ScalarType : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case .bool: target.write("bool")
        case .int: target.write("int")
        case .float: target.write("float")
        }
    }
}

extension ImmediateOperand : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case let .bool(b):
            target.write("bool \(b)")
        case let .int(i):
            target.write("int \(i)")
        case let .float(f):
            target.write("float \(f)")
        }
    }
}

extension RandomizingTensorDefinition : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("random ")
        lowerBound.write(to: &target)
        target.write(", ")
        upperBound.write(to: &target)
    }
}

extension ImmediateTensorDefinition : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("repeat ")
        value.write(to: &target)
    }
}

extension ImmediateScalarDefinition : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        value.write(to: &target)
    }
}

extension Instruction.ActivationFunction : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case .relu: target.write("relu")
        case .tanh: target.write("tanh")
        case .sigmoid: target.write("sigmoid")
        }
    }
}

extension Instruction.TransformationFunction : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case .log: target.write("log")
        case .softmax: target.write("softmax")
        }
    }
}

extension Instruction.ArithmeticOperator : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case .add: target.write("add")
        case .sub: target.write("sub")
        case .mul: target.write("mul")
        case .div: target.write("div")
        case .min: target.write("min")
        case .max: target.write("max")
        }
    }
}

extension Instruction.ComparisonOperator : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case .eq: target.write("eq")
        case .neq: target.write("neq")
        case .geq: target.write("geq")
        case .leq: target.write("leq")
        case .gt: target.write("gt")
        case .lt: target.write("lt")
        }
    }
}

extension Instruction : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        if let variable = self.variable {
            target.write("\(variable.fullDescription) = ")
        }
        switch kind {
        // TODO
        case let .negate(op):
            target.write("neg \(op)")
            
        case let .arithOp(op, lhs, rhs):
            target.write("arith.\(op) \(lhs), \(rhs)")
            
        case let .compare(op, lhs, rhs):
            target.write("cmp.\(op) \(lhs), \(rhs)")
            
        case let .output(v):
            target.write("out \(v)")
            
        case let .activation(f, v):
            target.write("activ.\(f) \(v)")
            
        case let .transformation(f, v):
            target.write("trans.\(f) \(v)")
            
        case let .shapeCast(s, v):
            target.write("shapecast [\(s)], \(v)")
            
        case let .product(lhs, rhs):
            target.write("prod \(lhs), \(rhs)")
            
        case let .dotProduct(lhs, rhs):
            target.write("dotp \(lhs), \(rhs)")
            
        case let .uncondBranch(bb):
            target.write("br @\(bb.name)")
            
        case let .phi(vars):
            target.write("phi \(vars.map{"\($0)"}.joined(separator: ", "))")
            
        case let .condBranch(op, then: thenBB, else: elseBB):
            target.write("condbr \(op), @\(thenBB.name), @\(elseBB.name)")
            
        case let .concat(vars, dimension: dim):
            target.write("concat.\(dim) \(vars.map{"\($0)"}.joined(separator: ", "))")
        }
    }
}

extension BasicBlock : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        name.write(to: &target)
        target.write(":\n")
        for inst in elements {
            target.write("\t")
            inst.write(to: &target)
            target.write("\n")
        }
    }
}

extension Module : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("module \(name)\n\n")
        for variable in externalVariables {
            target.write("extern \(variable.fullDescription)\n")
        }
        target.write("\n")
        for variable in definedVariables {
            target.write("define \(variable.fullDescription) = \(variable.definition!)\n")
        }
        target.write("\n")
        for bb in basicBlocks {
            bb.write(to: &target)
            target.write("\n")
        }
    }
}
