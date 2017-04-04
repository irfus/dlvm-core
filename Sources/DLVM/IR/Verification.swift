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
    case cannotCastDataType(Use, DataType, Node)
    case cannotMatrixMultiply(Use, Use, Node)
    case noOperands(Node)
    case concatenationShapeMismatch([Use], Int, Node)
    case useShapeMismatch(Node)
    case useTypeMismatch(Node)
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
    case notDifferentiable(Node)
    case unexpectedMemoryLocation(Use, Node)
    case invalidCopyOperands(Use, Use, Node)
    case notComputeFunction(Function, Node)
    case notMutable(Use, Node)
    case computeGraphMismatch(Function, Node)
    case notBox(Use, Node)
    case notHeapObject(Use, Node)
    case cannotLoadFromCompute(Use, Node)
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
    private func verifyDifferentiability() throws {
        guard isDifferentiable else { return }
        /// All arguments have to be tensors or aggregate types of tensors
        /// No explicit pointer semantics are allowed
        /// - TODO: Check arguments
        for inst in instructions where inst.kind.accessesMemory {
            throw VerificationError.notDifferentiable(self)
        }
    }

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

        /// Verify differentiability
        try verifyDifferentiability()

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
        try kind.performVerification(in: self)
        
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

extension InstructionKind {
    public func performVerification(in instruction: Instruction) throws {
        let function = instruction.parent.parent
        switch self {
        case let .conditional(use, thenBB, thenArgs, elseBB, elseArgs):
            guard case let .tensor(s, t) = use.type.unaliased, s.isScalar, t.isBool else {
                throw VerificationError.unexpectedType(use, .tensor(.scalar, .bool), instruction)
            }
            guard thenBB.arguments.count == thenArgs.count,
                  zip(thenBB.arguments, thenArgs).forAll({$0.type == $1.type}) else {
                throw VerificationError.basicBlockArgumentMismatch(thenArgs, thenBB, instruction)
            }
            guard elseBB.arguments.count == elseArgs.count,
                  zip(elseBB.arguments, elseArgs).forAll({$0.type == $1.type}) else {
                throw VerificationError.basicBlockArgumentMismatch(elseArgs, elseBB, instruction)
            }

        case let .branch(bb, args):
            guard bb.arguments.count == args.count,
                  zip(bb.arguments, args).forAll({$0.type == $1.type}) else {
                throw VerificationError.basicBlockArgumentMismatch(args, bb, instruction)
            }

        case .return: break /// Verified at Function

        case let .unary(_, v1), let .transpose(v1):
            switch v1.type.unaliased {
            /// Non-compute scalar
            case .tensor([], _): break
            /// Compute tensor
            case .pointer(.tensor(_, _), .compute, .mutable): break
            default:
                throw VerificationError.notTensor(v1, instruction)
            }

        case let .binary(_, lhs, rhs):
            switch (lhs.type.unaliased, rhs.type.unaliased) {
            /// Non-compute scalar
            case let (.tensor([], t1), .tensor([], t2)) where t1 == t2:
                break
            /// Compute tensor
            case let (.pointer(.tensor(s1, t1), .compute, .mutable),
                      .pointer(.tensor(s2, t2), .compute, .mutable))
                where s1.canMutuallyBroadcast(with: s2) && t1 == t2:
                guard function.isCompute else {
                    throw VerificationError.notComputeFunction(function, instruction)
                }
            default:
                throw VerificationError.unbroadcastableMismatch(lhs, rhs, instruction)
            }

        /// Compute-only
        case let .matrixMultiply(lhs, rhs):
            guard function.isCompute else {
                throw VerificationError.notComputeFunction(function, instruction)
            }
            guard case let .pointer(.tensor(s1, t1), .compute, .mutable) = lhs.type.unaliased,
                  case let .pointer(.tensor(s2, t2), .compute, .mutable) = rhs.type.unaliased,
                  s1.canMatrixMultiply(with: s2), t1 == t2
                else { throw VerificationError.cannotMatrixMultiply(lhs, rhs, instruction) }

        /// Compute-only
        case let .concatenate(vv, axis: axis):
            guard function.isCompute else {
                throw VerificationError.notComputeFunction(function, instruction)
            }
            guard let first = vv.first,
                  case let .pointer(.tensor(s1, t1), .compute, .mutable) = first.type.unaliased
                else { throw VerificationError.noOperands(instruction) }
            /// Check simple, data type equality, and concatenability
            var accShape: TensorShape = s1
            for v in vv.dropFirst() {
                guard case let .pointer(.tensor(shape, type), .compute, .mutable) = v.type.unaliased,
                      type == t1,
                      let newShape = accShape.concatenating(with: shape, alongDimension: axis)
                    else { throw VerificationError.concatenationShapeMismatch(vv, axis, instruction) }
                accShape = newShape
            }

        /// Compute-only
        case let .reduce(_, v1, axis), let .scan(_, v1, axis: axis):
            guard function.isCompute else {
                throw VerificationError.notComputeFunction(function, instruction)
            }
            guard case let .pointer(.tensor(s1, _), .compute, .mutable) = v1.type.unaliased else {
                throw VerificationError.notTensor(v1, instruction)
            }
            if let axis = axis, !s1.indices.contains(axis) {
                throw VerificationError.axisOutOfBounds(axis, v1, instruction)
            }

        case let .shapeCast(v1, target):
            switch v1.type.unaliased {
            case let .tensor(s1, _) where target.contiguousSize == s1.contiguousSize: break
            case let .pointer(.tensor(s1, _), .compute, .mutable) where target.contiguousSize == s1.contiguousSize:
                guard function.isCompute else {
                    throw VerificationError.notComputeFunction(function, instruction)
                }
            default:
                throw VerificationError.notTensor(v1, instruction)
            }

        case let .dataTypeCast(v1, target):
            switch v1.type.unaliased {
            case let .tensor(_, t1),
                 let .pointer(.tensor(_, t1), .compute, .mutable):
                guard t1.canCast(to: target) else {
                    throw VerificationError.cannotCastDataType(v1, target, instruction)
                }
            default:
                throw VerificationError.notTensor(v1, instruction)
            }

        case let .apply(fun, vv), let .applyGradient(fun, vv):
            let actual = vv.map{$0.type}
            switch fun.type.unaliased {
            case let .function(args, _), let .pointer(.function(args, _), .host, _):
                guard actual.count == args.count
                   && zip(actual, args).forAll({$0.conforms(to: $1)}) else {
                    throw VerificationError.functionArgumentMismatch(vv, fun.type.unaliased, instruction)
                }
            case .pointer(_, .compute, _):
                throw VerificationError.unexpectedMemoryLocation(fun, instruction)
            default:
                throw VerificationError.invalidType(fun)
            }

        case let .compute(fun, vv, in: graph), let .computeGradient(fun, vv, in: graph):
            switch (fun, graph.type) {
            case let (.function(_, fd1), .computeGraph(fd2)) where fd1 == fd2:
                let actual = vv.map{$0.type}
                guard fd1.acceptsArguments(actual) else {
                    throw VerificationError.functionArgumentMismatch(vv, fun.type.unaliased, instruction)
                }
            default:
                throw VerificationError.invalidType(fun)
            }

        case let .projectBox(v):
            guard case .box = v.type.unaliased else {
                throw VerificationError.notBox(v, instruction)
            }

        case let .allocateHeap(_, _, count: v):
            guard case .tensor([], .int(64)) = v.type.unaliased else {
                throw VerificationError.unexpectedType(v, .tensor([], .int(64)), instruction)
            }

        case let .extract(v1, indices):
            guard !indices.isEmpty else {
                throw VerificationError.missingIndices(v1, instruction)
            }
            guard let _ = v1.type.subtype(at: indices) else {
                throw VerificationError.invalidIndices(v1, indices, instruction)
            }

        case let .insert(src, to: dest, at: indices):
            guard !indices.isEmpty else {
                throw VerificationError.missingIndices(dest, instruction)
            }
            guard let subtype = dest.type.subtype(at: indices) else {
                throw VerificationError.invalidIndices(dest, indices, instruction)
            }
            guard subtype == src.type else {
                throw VerificationError.typeMismatch(src, dest, instruction)
            }

        case let .load(v1):
            guard case .pointer(_, let loc, _) = v1.type.unaliased else {
                throw VerificationError.notPointer(v1, instruction)
            }
            guard loc == .host else {
                throw VerificationError.cannotLoadFromCompute(v1, instruction)
            }

        case let .store(v1, to: v2):
            guard case let .pointer(subtype, .host, mut) = v2.type.unaliased else {
                throw VerificationError.notPointer(v2, instruction)
            }
            guard mut == .mutable else {
                throw VerificationError.notMutable(v2, instruction)
            }
            guard v1.type == subtype else {
                throw VerificationError.typeMismatch(v1, v2, instruction)
            }

        case let .elementPointer(v, ii):
            guard type.isValid else {
                throw VerificationError.invalidOffsets(v, ii, instruction)
            }

        case let .copy(from: src, to: dest, count: count):
            guard case .tensor([], .int(64)) = count.type.unaliased else {
                throw VerificationError.unexpectedType(count, .scalar(.int(64)), instruction)
            }
            switch (src.type, dest.type) {
            case let (.pointer(t1, _, _), .pointer(t2, _, .mutable)),
                 let (.box(t1, _), .box(t2, _)):
                guard t1 == t2 else { fallthrough }
            default:
                throw VerificationError.invalidCopyOperands(src, dest, instruction)
            }

        case .bitCast(_, _):
            // TODO
            break

        case .deallocate(let v):
            switch v.type.unaliased {
            case .pointer, .box: break
            case _: throw VerificationError.notHeapObject(v, instruction)
            }

        case let .retain(v), let .release(v):
            guard case .box = v.type else {
                throw VerificationError.notBox(v, instruction)
            }

        case .allocateStack, .trap, .allocateBox: break
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
