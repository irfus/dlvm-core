//
// Created by Richard Wei on 12/25/16.
//

extension NamedValue {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        if shape.isScalar {
            target.write("\(type) %\(name)")
        } else {
            target.write("\(type) \(shape) %\(name)")
        }
    }
}

extension GlobalValue {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        if shape.isScalar {
            target.write("\(type) @\(name)")
        } else {
            target.write("\(type) \(shape) @\(name)")
        }
    }
}

extension ImmediateValue : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        if shape.isScalar {
            target.write("\(type) \(immediate)")
        } else {
            target.write("\(type) \(shape) \(immediate)")
        }
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
        if !isScalar {
            target.write("[\(dimensions.map{String($0)}.joined(separator: "x"))]")
        }
    }
}

extension DataType.Base : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case .float: target.write("f")
        case .int: target.write("i")
        case .bool: target.write("b")
        }
    }
}

extension DataType : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        base.write(to: &target)
        size.description.write(to: &target)
    }
}

extension AggregationFunction: TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case .softmax: target.write("softmax")
        case .logSoftmax: target.write("logSoftmax")
        case let .scan(fun): target.write("scan \(fun)")
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
        case .truncateDivide: target.write("truncDiv")
        case .floorDivide: target.write("floorDiv")
        case .mod: target.write("mod")
        case .power: target.write("pow")
        case .mean: target.write("mean")
        }
    }
}

extension ReductionFunction : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case let .logical(fun): target.write("\(fun)")
        case let .arithmetic(fun): target.write("\(fun)")
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

extension BinaryIntegrationFunction: TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case .crossEntropy: target.write("crossEnt")
        }
    }
}

extension BasicBlock : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        name.write(to: &target)
        target.write(":\n")
        for inst in elements {
            target.write("    ")
            if let defInst = inst as? DefiningInstruction {
                target.write("%\(defInst.name) = ")
            }
            switch inst {
            case let inst as TensorMultiplicationInstruction:
                target.write("tmul \(inst.firstOperand), \(inst.secondOperand)")
            case let inst as MatrixMultiplicationInstruction:
                target.write("mmul \(inst.firstOperand), \(inst.secondOperand)")
            case let inst as ArithmeticInstruction:
                target.write("\(inst.function) \(inst.firstOperand), \(inst.secondOperand)")
            case let inst as ComparisonInstruction:
                target.write("cmp \(inst.function) \(inst.firstOperand), \(inst.secondOperand)")
            case let inst as ElementwiseInstruction:
                target.write("\(inst.function) \(inst.operand)")
            case let inst as AggregationInstruction:
                target.write("\(inst.function) \(inst.operand)")
            case let inst as StoreInstruction:
                target.write("store \(inst.source) to \(inst.destination)")
            case let inst as LoadInstruction:
                target.write("load \(inst.source)")
            case let inst as ConcatenationInstruction:
                target.write("concat ")
                inst.operands.map{"\($0)"}.joined(separator: ", ").write(to: &target)
                target.write(" along \(inst.axis)")
            case let inst as ReductionInstruction:
                target.write("reduce \(inst.function) \(inst.operand)")
            case let inst as BinaryReductionInstruction:
                target.write("\(inst.function) \(inst.firstOperand), \(inst.secondOperand)")
            case let inst as ReductionInstruction:
                target.write("\(inst.function) \(inst.operand)")
            case let inst as ShapeCastInstruction:
                target.write("shapecast \(inst.operand) to \(inst.targetShape)")
            case let inst as TypeCastInstruction:
                target.write("typecast \(inst.operand) to \(inst.targetType)")
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
