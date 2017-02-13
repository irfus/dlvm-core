//
// Created by Richard Wei on 12/25/16.
//

public extension Value {
    static var referencePrefix: String {
        switch Self.scope {
            case .global: return "@"
            case .local: return "%"
            case .none: return ""
        }
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

extension BooleanFunction : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case .and: target.write("and")
        case .or: target.write("or")
        case .xor: target.write("xor")
        }
    }
}

extension AssociativeFunction : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case let .arithmetic(fun): fun.write(to: &target)
        case let .boolean(fun): fun.write(to: &target)
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

extension BinaryFunction : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case let .associative(fun): fun.write(to: &target)
        case let .comparison(fun): fun.write(to: &target)
        }
    }
}

extension UnaryFunction : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case let .elementwise(fun): target.write("\(fun)")
        case let .integration(fun): target.write("\(fun)")
        case let .scan(fun): fun.write(to: &target)
        }
    }
}

extension Control : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case let .br(bb):
            target.write("br \(bb)")
        case .ret:
            target.write("ret")
        case let .condBr(op, thenBB, elseBB):
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
        case let .condLoad(p, bb):
            target.write("condload \(p), %\(bb.name)")
        case let .binary(f, op1, op2):
            target.write("\(f) \(op1), \(op2)")
        case let .matrixMultiply(op1, op2):
            target.write("matmul \(op1), \(op2)")
        case let .unary(f, op):
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

extension Use : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("\(shape) \(type) ")
        switch kind {
        case let .global(def):
            target.write("@\(def.name)")
        case let .local(def):
            target.write("%\(def.name)")
        case let .literal(lit):
            lit.literal.write(to: &target)
        }
    }
}

extension BasicBlock : TextOutputStreamable {
    private func makeIndentation() -> String {
        return "    "
    }

    public func write<Target : TextOutputStream>(to target: inout Target) {
        /// Begin block
        target.write("\(name):\n")
        for inst in elements {
            /// Write indentation
            makeIndentation().write(to: &target)
            target.write("    ")
            inst.write(to: &target)
            target.write("\n")
        }
        makeIndentation().write(to: &target)
    }
}

extension Instruction : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case let .control(control):
            control.write(to: &target)
        case let .operation(def):
            target.write("%\(def.name) = \(def.value)")
        }
    }
}

extension Global : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case let .placeholder(def):
            target.write("declare placeholder \(def.type) \(def.shape) @\(def.name)\n")
        case let .value(def):
            target.write("declare parameter \(def.type) \(def.shape) @\(def.name) = \(def.value.initializer)\n")
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
        if !globals.isEmpty {
            for global in globals {
                global.write(to: &target)
            }
            target.write("\n")
        }
        writeOperandNotationDescription(to: &target)
        target.write("\n")
        for bb in basicBlocks {
            bb.write(to: &target)
            target.write("\n\n")
        }
    }
}
