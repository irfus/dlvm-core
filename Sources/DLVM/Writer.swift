//
// Created by Richard Wei on 12/25/16.
//

extension NamedValue {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("\(type) %\(name)")
    }
}

extension GlobalValue {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("\(type) @\(name)")
    }
}

extension ImmediateValue : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("\(type) \(immediate)")
    }
}

extension Immediate : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case let .bool(b): target.write(b.description)
        case let .int(i): target.write(i.description)
        case let .float(f): target.write(f.description)
        }
    }
}

extension TensorInitializer : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case let .elements(elements):
            target.write("elements [")
            for elem in elements {
                elem.write(to: &target)
                target.write(", ")
            }
            target.write("]")
        case let .repeating(immediate):
            target.write("repeating \(immediate)")
        case let .random(from: lowerBound, to: upperBound):
            target.write("random from \(lowerBound) to \(upperBound)")
        }
    }
}

extension TensorShape : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("[\(dimensions.map{String($0)}.joined(separator: "x"))]")
    }
}

extension TypeBase : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case .float: target.write("f")
        case .int: target.write("i")
        case .bool: target.write("b")
        }
    }
}

extension ScalarType : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        base.write(to: &target)
        size.description.write(to: &target)
    }
}

extension TensorType : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        base.write(to: &target)
        size.description.write(to: &target)
        target.write(" ")
        shape.write(to: &target)
    }
}

extension ElementwiseFunction : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case .relu: target.write("relu")
        case .tanh: target.write("tanh")
        case .sigmoid: target.write("sigmoid")
        case .log: target.write("log")
        }
    }
}

extension AggregateFunction : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case .softmax: target.write("softmax")
        }
    }
}

extension ArithmeticOperator : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case .add: target.write("add")
        case .subtract: target.write("sub")
        case .multiply: target.write("mul")
        case .divide: target.write("div")
        case .min: target.write("min")
        case .max: target.write("max")
        }
    }
}

extension ComparisonPredicate : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case .equalTo: target.write("eq")
        case .notEqualTo: target.write("neq")
        case .greaterThanOrEqualTo: target.write("geq")
        case .lessThanOrEqualTo: target.write("leq")
        case .greaterThan: target.write("gt")
        case .lessThan: target.write("lt")
        }
    }
}

extension BasicBlock : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        name.write(to: &target)
        target.write(":\n")
        for inst in elements {
            target.write("\t")
            if let defInst = inst as? DefiningInstruction {
                target.write("%\(defInst.name) = ")
            }
            switch inst {
            case let inst as NegationInstruction:
                target.write("neg \(inst.operand)")
            case let inst as TensorProductInstruction:
                target.write("tmul \(inst.leftOperand), \(inst.rightOperand)")
            case let inst as ArithmeticInstruction:
                target.write("\(inst.operator) \(inst.leftOperand), \(inst.rightOperand)")
            case let inst as ComparisonInstruction:
                target.write("cmp \(inst.predicate) \(inst.leftOperand), \(inst.rightOperand)")
            case let inst as ElementwiseCallInstruction:
                target.write("\(inst.function) \(inst.operand)")
            case let inst as AggregateCallInstruction:
                target.write("\(inst.function) \(inst.operand)")
            case let inst as StoreInstruction:
                target.write("store \(inst.source) to \(inst.destination)")
            case let inst as ConcatenationInstruction:
                target.write("concat ")
                inst.operands.map{"\($0)"}.joined(separator: ", ").write(to: &target)
                target.write(" along \(inst.axis)")
            case let inst as ShapeCastInstruction:
                target.write("shapecast \(inst.operand) to \(inst.targetShape)")
            case let inst as TypeCastInstruction:
                target.write("typecast \(inst.operand) to \(inst.targetBase)\(inst.targetSize)")
            default:
                preconditionFailure("Unsupported instruction class \(type(of: inst))")
                break
            }
            target.write("\n")
        }
    }
}

extension Module : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("module \(name)\n\n")
        for input in inputs {
            target.write("declare input ")
            input.write(to: &target)
            target.write("\n")
        }
        target.write("\n")
        for parameter in parameters {
            target.write("declare parameter ")
            parameter.write(to: &target)
            target.write(" = ")
            parameter.initializer.write(to: &target)
            target.write("\n")
        }
        target.write("\n")
        for output in outputs {
            target.write("declare output ")
            output.write(to: &target)
            target.write("\n")
        }
        target.write("\n")
        for bb in basicBlocks {
            bb.write(to: &target)
            target.write("\n")
        }
    }
}
