//
// Created by Richard Wei on 12/25/16.
//

import DLVMTensor

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
        target.write("[\(map{String($0)}.joined(separator: "x"))]")
    }
}

extension TensorIndex : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("(")
        map{"\($0)"}.joined(separator: ", ").write(to: &target)
        target.write(")")
    }
}

extension Type : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case let .simple(shape, dType):
            if !shape.isScalar {
                target.write("\(shape) ")
            }
            target.write("\(dType)")
            
        case let .tuple(components):
            target.write("(")
            target.write(components.map{"\($0)"}.joined(separator: ", "))
            target.write(")")
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

extension ArithmeticOp: TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case .add: target.write("add")
        case .subtract: target.write("sub")
        case .multiply: target.write("mul")
        case .divide: target.write("div")
        case .min: target.write("min")
        case .max: target.write("max")
        case .truncateDivide: target.write("truncdiv")
        case .floorDivide: target.write("floordiv")
        case .modulo: target.write("mod")
        case .power: target.write("pow")
        case .mean: target.write("mean")
        }
    }
}

extension BooleanOp: TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case .and: target.write("and")
        case .or: target.write("or")
        case .xor: target.write("xor")
        }
    }
}

extension AssociativeOp: TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case let .arithmetic(fun): fun.write(to: &target)
        case let .boolean(fun): fun.write(to: &target)
        }
    }
}

extension ComparisonOp: TextOutputStreamable {
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

extension BinaryOp: TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case let .associative(fun): fun.write(to: &target)
        case let .comparison(fun): fun.write(to: &target)
        }
    }
}

extension UnaryOp: TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case let .elementwise(fun): target.write("\(fun)")
        case let .integration(fun): target.write("\(fun)")
        }
    }
}

extension Control : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case let .br(bb, args):
            target.write("br %\(bb.name)(\(args.map{"\($0)"}.joined(separator: ", ")))")
        case let .condBr(op, thenBB, elseBB):
            target.write("condbr \(op), %\(thenBB.name), %\(elseBB.name)")
        case let .yield(op, v):
            target.write("yield \(op) to \(v)")
        case let .store(op, v):
            target.write("store \(op) to \(v)")
        case let .ret(op):
            target.write("ret")
            if let op = op { target.write(" \(op)") }
        case .cont:
            target.write("cont")
        case let .pull(ph, thenBB, elseBB):
            target.write("pull \(ph), %\(thenBB), %\(elseBB)")
        }
    }
}

extension Operation : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case let .binary(f, op1, op2):
            target.write("\(f) \(op1), \(op2)")
        case let .matMul(op1, op2):
            target.write("matmul \(op1), \(op2)")
        case let .unary(f, op):
            target.write("\(f) \(op)")
        case let .reduce(f, op, axis: axis):
            target.write("reduce \(f) \(op)")
            if let axis = axis {
                target.write(" along \(axis)")
            }
        case let .scan(f, op, axis: axis):
            target.write("scan \(f) \(op)")
            if let axis = axis {
                target.write(" along \(axis)")
            }
        case let .concat(shape, type, ops, axis: axis):
            target.write("concat \(ops.map{"\($0)"}.joined(separator: ", ")) to ")
            if !shape.isScalar {
                target.write("\(shape) ")
            }
            target.write("\(type) along \(axis)")
        case let .transpose(op):
            target.write("transpose \(op)")
        case let .typeCast(op, t):
            target.write("typecast \(op) to \(t)")
        case let .shapeCast(op, s):
            target.write("shapecast \(op) to \(s)")
        case let .get(def):
            target.write("get \(def)")
        case let .call(shape, type, fun, args):
            target.write("call ")
            if shape.isScalar {
                target.write("\(shape) ")
            }
            target.write("\(type) @\(fun.name)(\(args.map{"\($0)"}.joined(separator: ", ")))")
        case let .diff(fun, use, wrt: idx):
            target.write("diff ")
            if shape.isScalar {
                target.write("\(shape) ")
            }
            target.write("\(type) @\(fun.name) wrt #\(idx) from \(use)")
        case let .subtensor(use, idx):
            target.write("subtensor \(idx) of \(use)")
        case let .element(use, i):
            target.write("element \(i) of \(use)")
        case let .intrinsic(shape, type, intrin, args):
            target.write("intrin ")
            if shape.isScalar {
                target.write("\(shape) ")
            }
            target.write("\(type) @\(intrin.name)(\(args.map{"\($0)"}.joined(separator: ", ")))")
        }
    }
}

extension Instruction : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch kind {
        case let .control(control):
            control.write(to: &target)
        case let .operation(def):
            target.write("%\(def.name) = \(def.value)")
        }
    }
}

extension Global : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("declare ")
        if isRecurrent {
            target.write("recurrent ")
        }
        switch self {
        case let .placeholder(def):
            target.write("placeholder \(def)")
        case let .value(def):
            target.write("\(def.value.kind) ")
            def.write(to: &target)
            target.write(" = \(def.value.initializer)")
        case let .output(def):
            target.write("output \(def)")
        }
        target.write("\n")
    }
}

extension Use : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        if !shape.isScalar {
            target.write("\(shape) ")
        }
        target.write("\(type) ")
        switch kind {
        case let .global(ref):      target.write("@\(ref.name)")
        case let .local(ref):       target.write("%\(ref.name)")
        case let .argument(ref):    target.write("%\(ref.name)")
        case let .literal(lit):     lit.literal.write(to: &target)
        }
    }
}

extension Def : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        if !shape.isScalar {
            target.write("\(shape) ")
        }
        target.write("\(type) ")
        switch ValueType.scope {
        case .global: target.write("@")
        case .local: target.write("%")
        default: break
        }
        name.write(to: &target)
    }
}

extension BasicBlock : TextOutputStreamable {
    private func makeIndentation() -> String {
        return "    "
    }

    public func write<Target : TextOutputStream>(to target: inout Target) {
        /// Begin block
        target.write("\(name)(\(arguments.map{"\($0)"}.joined(separator: ", "))):\n")
        for inst in elements {
            /// Write indentation
            makeIndentation().write(to: &target)
            inst.write(to: &target)
            target.write("\n")
        }
    }
}

extension Function : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        if isDifferentiable {
            target.write("differentiable ")
        }
        target.write("function ")
        target.write("@\(name)(")
        arguments.map{"\($0)"}.joined(separator: ", ").write(to: &target)
        target.write(") ")
        if let result = result {
            target.write("-> \(result.shape) \(result.type) ")
        }
        target.write("{\n")
        for bb in self {
            bb.write(to: &target)
        }
        target.write("}")
    }
}

extension Module : TextOutputStreamable {

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
        target.write("\n")
        for fun in self {
            fun.write(to: &target)
            target.write("\n\n")
        }
    }
}
