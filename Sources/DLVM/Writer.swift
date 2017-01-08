//
// Created by Richard Wei on 12/25/16.
//

extension UnavailableVariable : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("ε")
    }
}

extension ScalarVariable : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        type.write(to: &target)
        target.write(" @")
        name.write(to: &target)
    }
}

extension TensorShape : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("[\(dimensions.map{String($0)}.joined(separator: "x"))]")
    }
}

extension TensorVariable : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        dataType.write(to: &target)
        target.write(" ")
        shape.write(to: &target)
        target.write(" @")
        name.write(to: &target)
    }
}

extension DataType : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write(rawValue)
    }
}

extension ScalarType : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write(rawValue)
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

public protocol SelfDescribing : CustomStringConvertible {}

public extension SelfDescribing {
    public var description: String {
        return String(describing: self)
    }
}

public extension TextOutputStreamable where Self : SelfDescribing {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        description.write(to: &target)
    }
}

extension RandomizingTensorDefinition : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        dataType.write(to: &target)
        target.write(" ")
        shape.write(to: &target)
        target.write(" ")
        target.write("random ")
        lowerBound.write(to: &target)
        target.write(", ")
        upperBound.write(to: &target)
    }
}

extension ImmediateTensorDefinition : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        dataType.write(to: &target)
        target.write(" ")
        shape.write(to: &target)
        target.write(" ")
        target.write("repeating ")
        value.write(to: &target)
    }
}

extension ImmediateScalarDefinition : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        value.write(to: &target)
    }
}

extension Instruction.ActivationFunction : SelfDescribing, TextOutputStreamable {}
extension Instruction.TransformationFunction : SelfDescribing, TextOutputStreamable {}
extension Instruction.BinaryOperator : SelfDescribing, TextOutputStreamable {}
extension Instruction.ComparisonOperator : SelfDescribing, TextOutputStreamable {}

extension Instruction : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch kind {
        // TODO
        case let .negate(op):
            target.write("neg ")
            op.write(to: &target)

        case let .binaryOp(op, lhs, rhs):
            op.write(to: &target)
            target.write(" ")
            lhs.write(to: &target)
            target.write(", ")
            rhs.write(to: &target)

        case let .compare(op, lhs, rhs):
            target.write("cmp ")
            op.write(to: &target)
            target.write(" ")
            lhs.write(to: &target)
            target.write(", ")
            rhs.write(to: &target)

        case let .output(v):
            target.write("output ")
            v.write(to: &target)

        case let .activation(f as TextOutputStreamable, v),
             let .transformation(f as TextOutputStreamable, v):
            f.write(to: &target)
            target.write(" ")
            v.write(to: &target)
        default: break
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
        target.write("module " + name + "\n\n")
        for decl in declarations {
            if let def = decl.definition {
                target.write("define @" + decl.name + " = ")
                def.write(to: &target)
            }
            else {
                target.write("declare ")
                decl.write(to: &target)
            }
            target.write("\n")
        }
    }
}
