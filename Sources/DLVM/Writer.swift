//
// Created by Richard Wei on 12/25/16.
//

extension Input : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("input \(type) %\(name)")
    }
}

extension Output : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("output \(type) %\(name)")
    }
}

extension Parameter : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("parameter \(type) %\(name) = \(initializer)")
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

extension ScalarType : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch base {
        case .float: target.write("f")
        case .int: target.write("i")
        case .bool: target.write("b")
        }
        size.description.write(to: &target)
    }
}

extension TensorType : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch base {
        case .float: target.write("f")
        case .int: target.write("i")
        case .bool: target.write("b")
        }
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
            inst.write(to: &target)
            target.write("\n")
        }
    }
}

extension Module : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("module \(name)\n\n")
        target.write("\n")
        for bb in basicBlocks {
            bb.write(to: &target)
            target.write("\n")
        }
    }
}
