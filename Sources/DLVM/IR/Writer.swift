//
// Created by Richard Wei on 12/25/16.
//

import DLVMTensor

extension LiteralValue : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        type.write(to: &target)
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
        target.write("\(map{String($0)}.joined(separator: "x"))")
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
    public func write<Target>(to target: inout Target) where Target : TextOutputStream {
        switch self {
        case .invalid:
            target.write("<<error>>")
        case let .struct(ty):
            ty.name.write(to: &target)
        case let .tensor([], t):
            t.write(to: &target)
        case let .tensor(s, t):
            target.write("[\(s):\(t)]")
        case let .tuple(subtypes):
            target.write("(\(subtypes.map{"\($0)"}.joined(separator: ", ")))")
        case .void:
            target.write("void")
        case let .array(subtype, n):
            target.write("<\(n) x \(subtype)>")
        case let .pointer(subtype):
            target.write("\(subtype)*")
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

extension AssociativeOp: TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case let .arithmetic(fun): String(describing: fun).write(to: &target)
        case let .boolean(fun): String(describing: fun).write(to: &target)
        }
    }
}

extension BinaryOp: TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case let .associative(fun): fun.write(to: &target)
        case let .comparison(fun): String(describing: fun).write(to: &target)
        }
    }
}

extension UnaryOp: TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case let .elementwise(fun): String(describing: fun).write(to: &target)
        case let .integration(fun): String(describing: fun).write(to: &target)
        }
    }
}

extension InstructionKind : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case let .branch(bb, args):
            target.write("branch %\(bb.name)(\(args.map{"\($0)"}.joined(separator: ", ")))")
        case let .conditional(op, thenBB, elseBB):
            target.write("conditional \(op), %\(thenBB.name), %\(elseBB.name)")
        case let .`return`(op):
            target.write("return")
            if let op = op { target.write(" \(op)") }
        case let .binary(f, op1, op2):
            target.write("\(f) \(op1), \(op2)")
        case let .matrixMultiply(op1, op2):
            target.write("matrixMultiply \(op1), \(op2)")
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
        case let .concatenate(ops, axis: axis):
            target.write("concatenate \(ops.map{"\($0)"}.joined(separator: ", ")) along \(axis)")
        case let .transpose(op):
            target.write("transpose \(op)")
        case let .dataTypeCast(op, t):
            target.write("dataTypeCast \(op) to \(t)")
        case let .shapeCast(op, s):
            target.write("shapeCast \(op) to \(s)")
        case let .call(fun, args):
            target.write("call \(fun.result) @\(fun.name)(\(args.map{"\($0)"}.joined(separator: ", ")))")
        case let .gradient(fun, args):
            target.write("gradient \(fun.result) @\(fun.name)(\(args.map{"\($0)"}.joined(separator: ", ")))")
        case let .subtensor(use, idx):
            target.write("subtensor \(idx) of \(use)")
        case let .tupleElement(use, i):
            target.write("element \(i) of \(use)")
        case let .tuple(uses):
            target.write("tuple \(uses.map{"\($0)"}.joined(separator: ", "))")
        case let .allocate(t, n):
            target.write("allocate \(t), \(n)")
        case let .store(v, p):
            target.write("store \(v) to \(p)")
        case let .load(v):
            target.write("load \(v)")
        case let .elementPointer(v, i):
            target.write("elementPointer \(v), \(i)")
        case let .bitCast(v, t):
            target.write("bitCast \(v) to \(t)")
        }
    }
}

extension Instruction : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        if let name = name {
            target.write("%\(name) = ")
        }
        kind.write(to: &target)
    }
}

extension GlobalValue : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("define \(kind) \(type) ")
        if let name = name {
            target.write("@\(name)")
        } else {
            target.write("_")
        }
        target.write(" = \(initializer)\n")
    }
}

extension Use : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("\(type) ")
        switch self {
        case let .global(_, ref):      target.write("\(ref.name.flatMap{"@\($0)"} ?? "@_")")
        case let .instruction(_, ref): target.write(ref.name.flatMap{"%\($0)"} ?? "%_")
        case let .argument(_, ref):    target.write(ref.name.flatMap{"%\($0)"} ?? "%_")
        case let .literal(_, lit):     lit.literal.write(to: &target)
        }
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

extension Argument : TextOutputStreamable {
    public func write<Target>(to target: inout Target) where Target : TextOutputStream {
        target.write("\(type) ")
        target.write(name.flatMap{"%\($0)"} ?? "%_")
    }
}

extension Function : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        if isDifferentiable {
            target.write("differentiable ")
        }
        target.write("func ")
        target.write("@\(name)(")
        arguments.map{"\($0)"}.joined(separator: ", ").write(to: &target)
        target.write(") ")
        if !result.isVoid {
            target.write("-> \(result) ")
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
