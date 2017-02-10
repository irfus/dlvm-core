//
// Created by Richard Wei on 12/25/16.
//

extension DefiningInstruction where Self : Named {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        type.write(to: &target)
        target.write(shape.isScalar ? " %" : " \(shape) %")
        name.write(to: &target)
    }
}

extension Variable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        type.write(to: &target)
        target.write(shape.isScalar ? " $" : " \(shape) $")
        name.write(to: &target)
    }
}

extension Input {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        type.write(to: &target)
        target.write(shape.isScalar ? " @" : " \(shape) @")
        name.write(to: &target)
    }
}

extension Output {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        type.write(to: &target)
        target.write(shape.isScalar ? " &" : " \(shape) &")
        name.write(to: &target)
    }
}

extension Constant {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        type.write(to: &target)
        target.write(shape.isScalar ? " '" : " \(shape) '")
        name.write(to: &target)
    }
}

extension ImmediateValue : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        type.write(to: &target)
        target.write(shape.isScalar ? " " : " \(shape) ")
        immediate.write(to: &target)
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
            target.write("[\(map{String($0)}.joined(separator: "x"))]")
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
        case .argmax: target.write("argmax")
        case .argmin: target.write("argmin")
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
        case .modulo: target.write("mod")
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

extension BasicBlock : TextOutputStreamable {

    private func makeIndentation() -> String {
        return "    "
    }

    public func write<Target : TextOutputStream>(to target: inout Target) {
        if let extensionType = extensionType {
            target.write("extension !\(name) for \(extensionType)")
        } else {
            target.write("!\(name)")
        }
        /// Begin block
        target.write(" {\n")
        for inst in elements {
            /// Write indentation
            makeIndentation().write(to: &target)
            target.write("    ")
            inst.writeDefinition(to: &target)
            target.write("\n")
        }
        makeIndentation().write(to: &target)
        /// End block
        target.write("}")
    }
}

extension Instruction {
    public func writeDefinition<Target : TextOutputStream>(to target: inout Target) {
        if let defInst = self as? DefiningInstruction {
            target.write("%\(defInst.name) = ")
        }
        switch self {
        case let inst as TensorMultiplicationInstruction:
            target.write("tmul \(inst.firstOperand), \(inst.secondOperand)")
        case let inst as MatrixMultiplicationInstruction:
            target.write("mmul \(inst.firstOperand), \(inst.secondOperand)")
        case let inst as ArithmeticInstruction:
            target.write("\(inst.function) \(inst.firstOperand), \(inst.secondOperand)")
        case let inst as LogicInstruction:
            target.write("\(inst.function) \(inst.firstOperand), \(inst.secondOperand)")
        case let inst as ComparisonInstruction:
            target.write("cmp \(inst.function) \(inst.firstOperand), \(inst.secondOperand)")
        case let inst as ElementwiseInstruction:
            target.write("\(inst.function) \(inst.operand)")
        case let inst as AggregationInstruction:
            target.write("\(inst.function) \(inst.operand)")
        case let inst as StoreInstruction:
            target.write("store \(inst.source) to \(inst.destination)")
        case let inst as ExportInstruction:
            target.write("export \(inst.source) to \(inst.destination)")
        case let inst as LoadInstruction:
            target.write("load \(inst.source)")
        case let inst as ConcatenationInstruction:
            target.write("concat ")
            inst.operands.map{"\($0)"}.joined(separator: ", ").write(to: &target)
            target.write(" along \(inst.axis)")
        case let inst as ReductionInstruction:
            target.write("reduce \(inst.function) \(inst.operand)")
        case let inst as ShapeCastInstruction:
            target.write("shapecast \(inst.operand) to \(inst.target)")
        case let inst as TypeCastInstruction:
            target.write("typecast \(inst.operand) to \(inst.target)")
        case let inst as PhiInstruction:
            target.write("phi \(inst.incomingValues.map{"(\($0) \($1))"}.joined(separator: ", "))")
        case let inst as BranchInstruction:
            target.write("br \(inst.destination)")
        default:
            preconditionFailure("Unsupported instruction class \(type(of: self))")
            break
        }
    }
}

extension Module : TextOutputStreamable {

    private func writeOperandNotationDescription<Target : TextOutputStream>(to target: inout Target) {
        target.write("// Operand notations:\n")
        target.write("//   @ ---- input\n")
        target.write("//   ' ---- constant\n")
        target.write("//   $ ---- parameter\n")
        target.write("//   & ---- output\n")
        target.write("//   % ---- temporary\n")
    }
    
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("module \(name)\n\n")
        /// Need add an empty line between different kinds of globals
        /// We check for non-emptiness, which looks crappy but works :)
        if !inputs.isEmpty {
            for input in inputs {
                target.write("declare input \(input.type) \(input.shape) \(input.name)\n")
            }
            target.write("\n")
        }
        if !constants.isEmpty {
            for constant in constants {
                target.write("declare constant \(constant.type) \(constant.shape) \(constant.name) = \(constant.defaultInitializer)\n")
            }
            target.write("\n")
        }
        if !parameters.isEmpty {
            for parameter in parameters {
                target.write("declare parameter \(parameter.type) \(parameter.shape) \(parameter.name) = \(parameter.initializer)\n")
            }
            target.write("\n")
        }
        if !outputs.isEmpty {
            for output in outputs {
                target.write("declare output \(output.type) \(output.shape) \(output.name)\n")
            }
            target.write("\n")
        }
        writeOperandNotationDescription(to: &target)
        target.write("\n")
        for bb in basicBlocks {
            bb.write(to: &target)
            /// Write out all the extensions
            for ext in bb.extensions.values {
                target.write("\n\n")
                ext.write(to: &target)
            }
            target.write("\n\n")
        }
    }
}
