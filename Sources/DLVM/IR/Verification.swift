//
//  Verification.swift
//  DLVM
//
//  Created by Richard Wei on 12/25/16.
//
//

import DLVMTensor

public enum VerificationError<Node : SelfVerifiable> : Error {
    case redeclared(Node)
    case noParent(Node)
    case blockFunctionMismatch(BasicBlock, Node)
    case missingTerminator(Node)
    case unbroadcastableMismatch(Use, Use, Node)
    case typeMismatch(Use, Use, Node)
    case unexpectedShape(Use, TensorShape, Node)
    case unexpectedDataType(Use, DataType, Node)
    case unexpectedType(Use, Type, Node)
    case notTensor(Use, Node)
    case notTuple(Use, Node)
    case cannotShapeCast(Use, TensorShape, Node)
    case cannotTypeCast(Use, DataType, Node)
    case cannotMatrixMultiply(Use, Use, Node)
    case noOperands(Node)
    case concatenationShapeMismatch([Use], Int, Node)
    case useShapeMismatch(Node)
    case useTypeMismatch(Node)
    case noTopSection(Node)
    case noDimensions(Use, Node)
    case noSpecifiedDimension(Use, Int, Node)
    case definitionNotInBasicBlock(Use, BasicBlock, Node)
    case functionArgumentMismatch([Use], Type, Node)
    case notAFunctionCall(Use, Function, Node)
    case functionDiffArgumentMismatch(Use, Argument, Function, Node)
    case invalidTensorIndex(Use, TensorIndex, Node)
    case invalidIndex(Use, Int, Node)
    case multipleExits([BasicBlock], Node)
    case noEntry(Node)
    case noExit(Node)
    case noForwardPass(Node)
    case noReturn(Node)
    case useBeforeDef(user: Instruction, usee: Value, Node)
    case basicBlockArgumentMismatch([Use], BasicBlock, Node)
    case unexpectedBasicBlockType(BasicBlock, Node)
    case axisOutOfBounds(Int, Use, Node)
    case functionEntryArgumentMismatch(BasicBlock, Node)
    case returnTypeMismatch(Instruction, Node)
    case invalidType(Node)
    case namedVoidValue(Node)
    case notPointer(Use, Node)
    case invalidIndices(Use, [Int], Node)
    case invalidOffsets(Use, [Use], Node)
    case gradientTypeMismatch(Function.Attribute, Type, Node)
    case invalidLiteral(Type, Literal, Node)
    case missingIndices(Use, Node)
    case notDifferentiable(Function, Node)
}

public protocol SelfVerifiable {
    func performVerification() throws
}

extension Module : SelfVerifiable {
    public func performVerification() throws {
        for global in globalValues {
            try global.performVerification()
        }
        for fun in self {
            try fun.performVerification()
        }
    }
}

extension GlobalValue : SelfVerifiable {
    public func performVerification() throws {
        try initializer.performVerification()
    }
}

extension LiteralValue : SelfVerifiable {

    private func verifyUse(_ use: Use, _ subtype: Type) throws {
        try use.performVerification()
        guard use.type == subtype else {
            throw VerificationError.unexpectedType(use, subtype, self)
        }
    }
    
    public func performVerification() throws {
        switch (type.canonical, literal) {

        /* Simple literals */
            
        /// Anything can be undefined
        case (_, .undefined): break
        /// Any tensor can be zero initialized
        case (.tensor, .zero): break

        /// Scalar tensor with scalar literal
        case (.tensor([], let dt), .scalar(let lit)):
            guard lit.typeBase == dt.base else {
                throw VerificationError.invalidLiteral(type, literal, self)
            }

        /* Aggregate literals */

        /// Tensor literal
        case let (.tensor(shape, dt), .tensor(elements)):
            let subtype: Type = .tensor(shape.dropFirst(), dt)
            for use in elements {
                try verifyUse(use, subtype)
            }

        /// Tuple literal
        case let (.tuple(subtypes), .tuple(elements)) where subtypes.count == elements.count:
            for (subtype, use) in zip(subtypes, elements) {
                try verifyUse(use, subtype)
            }

        /// Array literal
        case let (.array(subtype, n), .array(elements)) where n == elements.count:
            for use in elements {
                try verifyUse(use, subtype)
            }
            
        default:
            throw VerificationError.invalidLiteral(type, literal, self)
        }
    }
}

extension Function: SelfVerifiable {
    public func performVerification() throws {
        let domTree = try analysis(from: DominanceAnalysis.self)

        /// Check entry block arguments
        guard entry.arguments.elementsEqual(arguments) else {
            throw VerificationError.functionEntryArgumentMismatch(entry, self)
        }

        var bbNames: Set<String> = []
        /// Verify basic blocks
        for bb in self {
            /// Check for redeclaration
            guard !bbNames.contains(bb.name) else {
                throw VerificationError.redeclared(bb)
            }
            bbNames.insert(bb.name)
            /// Verify bb
            try bb.performVerification()

            /// Check return type
            let bbPremise = try bb.premise()
            if case let .return(retVal) = bbPremise.terminator.kind {
                switch retVal {
                case let use? where use.type != result:
                    throw VerificationError.returnTypeMismatch(bbPremise.terminator, self)
                case nil where !result.isVoid:
                    throw VerificationError.returnTypeMismatch(bbPremise.terminator, self)
                default:
                    break
                }
            }

            /// Check dominance
            for user in bb {
                for use in user.operands {
                    guard domTree.properlyDominates(use, user) else {
                        throw VerificationError.useBeforeDef(user: user, usee: use.value, bb)
                    }
                }
            }
        }

        /// Verify attributes
        for attr in attributes {
            switch attr {
            /// Gradient function
            case let .differentiating(antigrad):
                /// Check for type mismatch
                let antigradArgTypes = antigrad.arguments.map{$0.type}
                let expectedType: Type = .function(antigradArgTypes, .tuple(antigradArgTypes))
                guard type == expectedType else {
                    throw VerificationError.gradientTypeMismatch(attr, expectedType, self)
                }
            default:
                break
            }
        }

        /// - TODO: Check if function marked !gradient contains non-differentiable operations
    }
}

extension BasicBlock : SelfVerifiable {
    public func performVerification() throws {
        /// Check for terminator
        guard hasTerminator else {
            throw VerificationError<BasicBlock>.missingTerminator(self)
        }
        /// Check for name duplication
        var names: Set<String> = []
        /// Check arguments
        for arg in arguments {
            guard !names.contains(arg.name) else {
                throw VerificationError.redeclared(arg)
            }
            names.insert(name)
            try arg.performVerification()
        }
        /// Check instructions
        for inst in self {
            if let name = inst.name {
                guard !names.contains(name) else {
                    throw VerificationError.redeclared(inst)
                }
                names.insert(name)
            }
            try inst.performVerification()
        }
    }
}

extension Argument : SelfVerifiable {
    public func performVerification() throws {
    }
}

extension Instruction : SelfVerifiable {
    public func performVerification() throws {
        /// Use type must match usee type
        for use in operands {
            try use.performVerification()
        }

        /// Visit kind
        try kind.performVerification()
        
        /// Check type
        switch type {
        case .void where name != nil:
            /// If void, it cannot have a name
            throw VerificationError.namedVoidValue(self)
        case .invalid:
            /// Cannot be invalid
            throw VerificationError.invalidType(self)
        default:
            break
        }
    }
}

extension InstructionKind : SelfVerifiable {
    public func performVerification() throws {
        switch self {
        case let .conditional(use, _, _):
            guard case let .tensor(s, t) = use.type.unaliased, s.isScalar, t.isBool else {
                throw VerificationError.unexpectedType(use, .tensor(.scalar, .bool), self)
            }

        case let .branch(bb, args):
            for (formal, arg) in zip(bb.arguments, args) where formal.type != arg.type {
                throw VerificationError.basicBlockArgumentMismatch(args, bb, self)
            }

        case .return: break /// Verified at Function

        case let .binary(_, lhs, rhs):
            guard case let .tensor(s1, t1) = lhs.type.unaliased,
                  case let .tensor(s2, t2) = rhs.type.unaliased,
                  let _ = s1 <> s2, t1 == t2
                else { throw VerificationError.unbroadcastableMismatch(lhs, rhs, self) }

        case let .matrixMultiply(lhs, rhs):
            guard case let .tensor(s1, t1) = lhs.type.unaliased,
                  case let .tensor(s2, t2) = rhs.type.unaliased,
                  s1.canMatrixMultiply(with: s2), t1 == t2
                else { throw VerificationError.cannotMatrixMultiply(lhs, rhs, self) }

        case let .concatenate(vv, axis: axis):
            guard let first = vv.first,
                  case let .tensor(s1, t1) = first.type.unaliased
                else { throw VerificationError.noOperands(self) }
            /// Check simple, data type equality, and concatenability
            var accShape: TensorShape = s1
            for v in vv.dropFirst() {
                guard case let .tensor(shape, type) = v.type.unaliased, type == t1,
                    let newShape = accShape.concatenating(with: shape,
                                                          alongDimension: axis)
                    else { throw VerificationError.concatenationShapeMismatch(vv, axis, self) }
                accShape = newShape
            }

        case let .reduce(_, v1, axis), let .scan(_, v1, axis: axis):
            guard case let .tensor(s1, _) = v1.type.unaliased else {
                throw VerificationError.notTensor(v1, self)
            }
            if let axis = axis, !s1.indices.contains(axis) {
                throw VerificationError.axisOutOfBounds(axis, v1, self)
            }

        case let .shapeCast(v1, target):
            guard case let .tensor(s1, _) = v1.type.unaliased else {
                throw VerificationError.notTensor(v1, self)
            }
            guard s1.contiguousSize == target.contiguousSize else {
                throw VerificationError.cannotShapeCast(v1, target, self)
            }

        case let .dataTypeCast(v1, target):
            guard case let .tensor(_, t1) = v1.type.unaliased else {
                throw VerificationError.notTensor(v1, self)
            }
            guard t1.canCast(to: target) else {
                throw VerificationError.cannotTypeCast(v1, target, self)
            }

        case let .call(fun, vv), let .gradient(fun, vv):
            let actual = vv.map{$0.type}
            switch fun.type.unaliased {
            case let .function(args, _), let .pointer(.function(args, _)):
                guard args == actual else { fallthrough }
            default:
                throw VerificationError.functionArgumentMismatch(vv, fun.type.unaliased, self)
            }

        case let .extract(v1, indices):
            guard !indices.isEmpty else {
                throw VerificationError.missingIndices(v1, self)
            }
            guard let subtype = v1.type.subtype(at: indices) else {
                throw VerificationError.invalidIndices(v1, indices, self)
            }

        case let .insert(src, to: dest, at: indices):
            guard !indices.isEmpty else {
                throw VerificationError.missingIndices(dest, self)
            }
            guard let subtype = dest.type.subtype(at: indices) else {
                throw VerificationError.invalidIndices(dest, indices, self)
            }
            guard subtype == src.type else {
                throw VerificationError.typeMismatch(src, dest, self)
            }

        case let .unary(_, v1), let .transpose(v1):
            guard case .tensor = v1.type.unaliased else {
                throw VerificationError.notTensor(v1, self)
            }

        case let .load(v1):
            guard case .pointer = v1.type.unaliased else {
                throw VerificationError.notPointer(v1, self)
            }

        case let .store(v1, to: v2):
            guard case let .pointer(subtype) = v2.type.unaliased else {
                throw VerificationError.notPointer(v2, self)
            }
            guard v1.type == subtype else {
                throw VerificationError.typeMismatch(v1, v2, self)
            }

        case let .elementPointer(v, ii):
            func gepCheck<C: Collection>(type: Type, indices: C) throws where C.Iterator.Element == Use {
                if indices.isEmpty { return }
                if case let .pointer(t) = type.unaliased {
                    try gepCheck(type: t, indices: ii.dropFirst())
                } else {
                    throw VerificationError.invalidOffsets(v, ii, self)
                }
            }
            try gepCheck(type: v.type, indices: ii)

        case .allocate, .bitCast: break
        }
    }
}

extension Use : SelfVerifiable {
    public func performVerification() throws {
        try value.performVerification()
        /// Type must be valid
        guard type.isValid else {
            throw VerificationError.invalidType(self)
        }
        /// Value type must match use type
        guard value.type == type else {
            throw VerificationError.useTypeMismatch(self)
        }
    }
}

// MARK: - Lazy verification
public extension IRUnit {
    public func verify() throws {
        _ = try analysis(from: Verifier.self)
    }
}

/// Verifier pass
public class Verifier<Body : IRUnit> : AnalysisPass<Body, Void> {
    public override class func run(on body: Body) throws {
        try body.performVerification()
    }
}
