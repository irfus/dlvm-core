//
//  Writer.swift
//  DLVM
//
//  Copyright 2016-2018 The DLVM Team.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

import CoreTensor
import CoreOp

extension LiteralValue : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("\(literal) : \(type)")
    }
}

extension Literal.Scalar : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case let .bool(b): target.write(b.description)
        case let .int(i): target.write(i.description)
        case let .float(f): target.write(f.description)
        }
    }
}

extension Literal : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case let .scalar(lit):
            lit.write(to: &target)
        case let .tensor(vals):
            target.write("<\(vals.joinedDescription)>")
        case let .tuple(vals):
            target.write("(\(vals.joinedDescription))")
        case let .array(vals):
            target.write("[\(vals.joinedDescription)]")
        case let .struct(fields):
            target.write("{\(fields.map{"#\($0.0) = \($0.1)"}.joined(separator: ", "))}")
        case let .enumCase(name, associatedTypes):
            target.write("?\(name)(\(associatedTypes.joinedDescription))")
        case .zero:
            target.write("zero")
        case .undefined:
            target.write("undefined")
        case .null:
            target.write("null")
        }
    }
}

extension TensorShape : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write(isScalar ? "scalar" : "\(map{String($0)}.joined(separator: " x "))")
    }
}

extension TensorIndex : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("(")
        joinedDescription.write(to: &target)
        target.write(")")
    }
}

extension StructType : TextOutputStreamable {
    public func write<Target>(to target: inout Target) where Target : TextOutputStream {
        target.write("struct $\(name) {\n")
        let fieldsDescription = fields
            .map { (name, type) in "    #\(name): \(type)" }
            .description(joinedBy: ",\n")
        target.write(fieldsDescription)
        target.write("\n}")
    }
}

extension EnumType : TextOutputStreamable {
    public func write<Target>(to target: inout Target) where Target : TextOutputStream {
        target.write("enum $\(name) {\n")
        let casesDescription = cases
            .map { (name, assocTypes) in "    ?\(name)(\(assocTypes.joinedDescription))" }
            .description(joinedBy: ",\n")
        target.write(casesDescription)
        target.write("\n}")
    }
}

extension Type : TextOutputStreamable {
    public func write<Target>(to target: inout Target) where Target : TextOutputStream {
        switch self {
        case .invalid:
            target.write("<<error>>")
        case let .tensor([], t):
            t.write(to: &target)
        case let .tensor(s, t):
            target.write("<\(s) x \(t)>")
        case let .tuple(elementTypes):
            target.write("(\(elementTypes.joinedDescription))")
        case let .array(n, elementType):
            target.write("[\(n) x \(elementType)]")
        case let .pointer(elementType):
            target.write("*\(elementType)")
        case let .box(elementType):
            target.write("box{\(elementType)}")
        case let .function(args, ret):
            target.write("(\(args.joinedDescription)) -> \(ret)")
        case let .alias(a):
            target.write("$")
            a.name.write(to: &target)
        case let .struct(structTy):
            target.write("$")
            structTy.name.write(to: &target)
        case let .enum(enumTy):
            target.write("$")
            enumTy.name.write(to: &target)
        case .stack:
            target.write("stack")
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
        switch self {
        case .bool: target.write("bool")
        case let .int(w): target.write("i\(w)")
        case let .float(w): target.write("f\(w.rawValue)")
        }
    }
}

extension ReductionCombinator : TextOutputStreamable {
    public func write<Target>(to target: inout Target) where Target : TextOutputStream {
        switch self {
        case let .function(f): f.write(to: &target)
        case let .numeric(op): String(describing: op).write(to: &target)
        case let .boolean(op): String(describing: op).write(to: &target)
        }
    }
}

extension InstructionKind : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        switch self {
        case let .branch(bb, args):
            target.write("branch '\(bb.name)(\(args.joinedDescription))")
        case let .conditional(op, thenBB, thenArgs, elseBB, elseArgs):
            target.write("conditional \(op) then '\(thenBB.name)(\(thenArgs.joinedDescription)) else '\(elseBB.name)(\(elseArgs.joinedDescription))")
        case let .return(op):
            target.write("return")
            if let op = op {
                target.write(" \(op)")
            }
        case let .literal(lit, ty):
            target.write("literal \(lit): \(ty)")
        case let .numericUnary(f, op):
            target.write("\(f) \(op)")
        case let .numericBinary(f, op1, op2):
            target.write("\(f) \(op1), \(op2)")
        case let .booleanBinary(f, op1, op2):
            target.write("\(f) \(op1), \(op2)")
        case let .not(op):
            target.write("not \(op)")
        case let .compare(f, op1, op2):
            target.write("\(f) \(op1), \(op2)")
        case let .dot(op1, op2):
            target.write("dot \(op1), \(op2)")
        case let .reduce(comb, op, initial, dims):
            target.write("reduce \(op) by \(comb) init \(initial) along \(dims.joinedDescription)")
        case let .reduceWindow(comb, op, initial, dims, strides, padding):
            target.write("reduceWindow \(op) by \(comb) init \(initial) ")
            target.write("dims \(dims.joinedDescription) strides \(strides.joinedDescription) ")
            target.write("padding \(padding)")
        case let .scan(f, op, dims):
            target.write("scan \(op) by \(f) along \(dims.joinedDescription)")
        case let .concatenate(ops, axis: axis):
            target.write("concatenate \(ops.joinedDescription) along \(axis)")
        case let .transpose(op):
            target.write("transpose \(op)")
        case let .reverse(op, dims: dims):
            target.write("reverse \(op) along \(dims.joinedDescription)")
        case let .slice(v, at: range):
            target.write("slice \(v) from \(range.lowerBound) upto \(range.upperBound)")
        case let .convolve(v, kernel: k, strides: s, padding: p,
                           leftDilation: ld, rightDilation: rd, groups: g):
            target.write("convolve \(v) kernel \(k)")
            if let s = s {
                target.write(" strides \(s.joinedDescription)")
            }
            if let p = p {
                target.write(" padding \(p.map({ ($0.low, $0.high) }).joinedDescription)")
            }
            if let ld = ld {
                target.write(" leftDilation \(ld.joinedDescription)")
            }
            if let rd = rd {
                target.write(" rightDilation \(rd.joinedDescription)")
            }
            if let g = g {
                target.write(" groups \(g)")
            }
        case let .dataTypeCast(op, t):
            target.write("dataTypeCast \(op) to \(t)")
        case let .padShape(op, at: index):
            target.write("padShape \(op) at \(index)")
        case let .squeezeShape(op, at: index):
            target.write("squeezeShape \(op) at \(index)")
        case let .shapeCast(op, s):
            target.write("shapeCast \(op) to \(s)")
        case let .apply(f, args):
            target.write("apply \(f.identifier)(\(args.joinedDescription)): \(f.type)")
        case .createStack:
            target.write("createStack")
        case let .destroyStack(stack):
            target.write("destroyStack \(stack)")
        case let .push(v, to: stack):
            target.write("push \(v) to \(stack)")
        case let .pop(t, from: stack):
            target.write("pop \(t) from \(stack)")
        case let .extract(use, indices):
            target.write("extract \(indices.joinedDescription) from \(use)")
        case let .insert(src, to: dest, at: indices):
            target.write("insert \(src) to \(dest) at \(indices.joinedDescription)")
        case let .allocateStack(t, n):
            target.write("allocateStack \(t) count \(n)")
        case let .store(v, p):
            target.write("store \(v) to \(p)")
        case let .load(v):
            target.write("load \(v)")
        case let .elementPointer(v, ii):
            target.write("elementPointer \(v) at \(ii.joinedDescription)")
        case let .bitCast(v, t):
            target.write("bitCast \(v) to \(t)")
        case let .allocateHeap(t, count: c):
            target.write("allocateHeap \(t) count \(c)")
        case let .allocateBox(t):
            target.write("allocateBox \(t)")
        case let .deallocate(v):
            target.write("deallocate \(v)")
        case let .projectBox(v):
            target.write("projectBox \(v)")
        case let .retain(v):
            target.write("retain \(v)")
        case let .release(v):
            target.write("release \(v)")
        case let .copy(from: src, to: dest, count: count):
            target.write("copy from \(src) to \(dest) count \(count)")
        case let .random(shape, from: lo, upTo: hi):
            target.write("random \(shape) from \(lo) upto \(hi)")
        case let .select(left, right, by: flags):
            target.write("select \(left), \(right) by \(flags)")
        case .trap:
            target.write("trap")
        }
    }
}

extension Instruction : TextOutputStreamable {
    public var printedName: String? {
        return name ?? (type.isVoid ? nil : "\(parent.indexInParent).\(indexInParent)")
    }

    public func write<Target : TextOutputStream>(to target: inout Target) {
        if let name = printedName {
            target.write("%\(name) = ")
        }
        kind.write(to: &target)
    }
}

extension Variable: TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("var @\(name): \(type)")
    }
}

extension TypeAlias : TextOutputStreamable {
    public func write<Target>(to target: inout Target) where Target : TextOutputStream {
        target.write("type $\(name) = ")
        if let type = type {
            type.write(to: &target)
        } else {
            target.write("opaque")
        }
    }
}

extension ElementKey : TextOutputStreamable {
    public func write<Target>(to target: inout Target) where Target : TextOutputStream {
        switch self {
        case let .index(i): target.write("\(i)")
        case let .name(n): target.write("#" + n)
        case let .value(v): target.write("\(v)")
        }
    }
}

extension Use : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("\(identifier): \(type)")
    }

    public var identifier: String {
        switch self {
        case let .variable(_, ref):
            return "@\(ref.name)"
        case let .instruction(_, ref):
            return ref.printedName.flatMap{"%\($0)"} ?? "%_"
        case let .argument(_, ref):
            return "%\(ref.name)"
        case let .literal(_, lit):
            return lit.description
        case let .function(_, ref):
            return "@\(ref.name)"
        }
    }
}

extension BasicBlock : TextOutputStreamable {
    private func makeIndentation() -> String {
        return "    "
    }

    public func write<Target : TextOutputStream>(to target: inout Target) {
        /// Begin block
        target.write("'\(name)(\(arguments.map{"\($0)"}.joined(separator: ", "))):\n")
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
        target.write("%\(name): \(type)")
    }
}

extension Function.Attribute : TextOutputStreamable {
    public func write<Target>(to target: inout Target) where Target : TextOutputStream {
        target.write("!")
        switch self {
        case .inline: target.write("inline")
        }
    }
}

extension Function : TextOutputStreamable {
    public func write<Target : TextOutputStream>(to target: inout Target) {
        for attr in attributes {
            attr.write(to: &target)
            target.write("\n")
        }
        switch declarationKind {
        case .external?:
            target.write("[extern]\n")
        case let .gradient(config)?:
            target.write("[gradient @\(config.primal.name)")
            config.sourceIndex.ifAny {
                target.write(" from \($0)")
            }
            config.argumentIndices.ifAny {
                target.write(" wrt \($0.joinedDescription)")
            }
            if !config.keptIndices.isEmpty {
                target.write(" keeping \(config.keptIndices.joinedDescription)")
            }
            if config.isSeedable {
                target.write(" seedable")
            }
            target.write("]\n")
        default:
            break
        }
        target.write("func ")
        target.write("@\(name): \(type)")
        if isDefinition {
            target.write(" {\n")
            for bb in self {
                bb.write(to: &target)
            }
            target.write("}")
        }
    }
}

extension String {
    var literal: String {
        var out = ""
        for char in self {
            switch char {
            case "\"", "\\":
                out.append("\\")
                out.append(char)
            case "\n":
                out.append("\\n")
            case "\t":
                out.append("\\t")
            case "\r":
                out.append("\\r")
            default:
                out.append(char)
            }
        }
        return out
    }
}

extension Module : TextOutputStreamable {
    func write<C, T>(_ elements: C, to target: inout T)
        where C : Collection, T : TextOutputStream,
              C.Element : TextOutputStreamable
    {
        for element in elements {
            target.write("\n")
            element.write(to: &target)
            target.write("\n")
        }
    }

    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("module \"\(name.literal)\"\n")
        target.write("stage \(stage)\n")
        write(enums, to: &target)
        write(structs, to: &target)
        write(typeAliases, to: &target)
        write(variables, to: &target)
        write(elements, to: &target)
    }
}
