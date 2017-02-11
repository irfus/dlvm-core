//
// Created by Richard Wei on 12/25/16.
//

extension GlobalValue : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("declare \(kind) \(shape) \(type) @\(name) = \(initializer)")
    }
}

extension Placeholder : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("declare placeholder \(shape) \(type) @\(name)")
    }
}

extension LiteralValue : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        type.write(to: &target)
        target.write(shape.isScalar ? " " : " \(shape) ")
        literal.write(to: &target)
    }
}

extension ScalarLiteral : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case let .bool(b): target.write(b.description)
        case let .int(i): target.write(i.description)
        case let .float(f): target.write(f.description)
        }
    }
}

extension TensorLiteral : TextOutputStreamable {
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

extension Literal : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case .scalar(let lit): lit.write(to: &target)
        case .tensor(let lit): lit.write(to: &target)
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

extension IntegrationFunction: TextOutputStreamable {
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

extension Control : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case let .branch(bb):
            target.write("br \(bb)")
        case let .condBranch(op, thenBB, elseBB):
            target.write("condbr \(op), \(thenBB), \(elseBB)")
        case let .export(op, out):
            target.write("export \(op) to \(out)")
        case let .store(op, v):
            target.write("store \(op) to \(v)")
        }
    }
}

extension Operation : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case let .arithmetic(f, op1, op2):
            target.write("\(f) \(op1), \(op2)")
        case let .compare(f, op1, op2):
            target.write("\(f) \(op1), \(op2)")
        case let .logical(f, op1, op2):
            target.write("\(f) \(op1), \(op2)")
        case let .matrixMultiply(op1, op2):
            target.write("matmul \(op1), \(op2)")
        case let .integrate(f, op):
            target.write("\(f) \(op)")
        case let .transform(f, op):
            target.write("\(f) \(op)")
        case let .reduce(f, op, axis: axis):
            target.write("\(f) \(op) along \(axis)")
        case let .phi(ops):
            target.write("phi \(ops.map{"[\($0) \($1)]"}.joined(separator: ", "))")
        case let .concat(ops, axis: axis):
            target.write("concat \(ops.map{"\($0)"}.joined(separator: ", ")) along \(axis)")
        case let .typeCast(op, t):
            target.write("typecast \(op) to \(t)")
        case let .shapeCast(op, s):
            target.write("shapecast \(op) to \(s)")
        }
    }
}


/*

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
            inst.write(to: &target)
            target.write("\n")
        }
        makeIndentation().write(to: &target)
        /// End block
        target.write("}")
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

*/
