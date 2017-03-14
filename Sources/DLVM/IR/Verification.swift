//
//  Verification.swift
//  DLVM
//
//  Created by Richard Wei on 12/25/16.
//
//

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
    case functionArgumentCountMismatch(Function, Node)
    case functionArgumentMismatch(Use, Argument, Function, Node)
    case notAFunctionCall(Use, Function, Node)
    case functionDiffArgumentMismatch(Use, Argument, Function, Node)
    case invalidTensorIndex(Use, TensorIndex, Node)
    case invalidIndex(Use, Int, Node)
    case multipleExits([BasicBlock], Node)
    case noEntry(Node)
    case noExit(Node)
    case noForwardPass(Node)
    case noReturn(Node)
    case unreachable(Instruction, from: Instruction, Node)
    case basicBlockArgumentMismatch([Use], BasicBlock, Node)
    case unexpectedBasicBlockType(BasicBlock, Node)
    case axisOutOfBounds(Int, Use, Node)
    case functionEntryArgumentMismatch(BasicBlock, Node)
    case returnTypeMismatch(Instruction, Node)
}

public protocol SelfVerifiable {
    func verify() throws
}

extension Module : SelfVerifiable {
    open func verify() throws {
        for fun in self {
            try fun.verify()
        }
    }
}

extension Function: SelfVerifiable {
    public func verify() throws {
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
            try bb.verify()

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
                    if case let .local(usee) = use.kind {
                        guard usee.properlyDominates(user, in: domTree) else {
                            throw VerificationError.unreachable(user, from: usee, self)
                        }
                    }
                }
            }
        }
    }
}

extension BasicBlock : SelfVerifiable {
    open func verify() throws {
        /// Check instructions
        var instNames: Set<String> = []
        guard hasTerminator else {
            throw VerificationError<BasicBlock>.missingTerminator(self)
        }
        for inst in self {
            if let name = inst.name {
                guard !instNames.contains(name) else {
                    throw VerificationError.redeclared(inst)
                }
                instNames.insert(name)
            }
            try inst.verify()
        }
    }
}

extension Instruction : SelfVerifiable {
    public func verify() throws {
        for operand in operands {
            try operand.verify()
        }
        try kind.verify()
    }
}

extension InstructionKind : SelfVerifiable {
    public func verify() throws {
        switch self {

        case let .conditional(use, _, _):
            guard case let .tensor(s, t) = use.type, s.isScalar, t.isBool else {
                throw VerificationError.unexpectedType(use, .tensor(.scalar, .bool), self)
            }

        case let .store(use, to: global):
            guard use.type == global.type else {
                throw VerificationError.unexpectedType(use, global.type, self)
            }

        case let .branch(bb, args):
            for (formal, arg) in zip(bb.arguments, args) where formal.type != arg.type {
                throw VerificationError.basicBlockArgumentMismatch(args, bb, self)
            }

        case .return: break /// Verified at Function

        case let .binary(_, lhs, rhs):
            guard case let .tensor(s1, t1) = lhs.type,
                  case let .tensor(s2, t2) = rhs.type,
                  let _ = s1 <> s2, t1 == t2
                else { throw VerificationError.unbroadcastableMismatch(lhs, rhs, self) }

        case let .matrixMultiply(lhs, rhs):
            guard case let .tensor(s1, t1) = lhs.type,
                  case let .tensor(s2, t2) = rhs.type,
                  s1.canMatrixMultiply(with: s2), t1 == t2
                else { throw VerificationError.cannotMatrixMultiply(lhs, rhs, self) }

        case let .concatenate(vv, axis: axis):
            guard let first = vv.first,
                  case let .tensor(s1, t1) = first.type
                else { throw VerificationError.noOperands(self) }
            /// Check simple, data type equality, and concatenability
            var accShape: TensorShape = s1
            for v in vv.dropFirst() {
                guard case let .tensor(shape, type) = v.type, type == t1,
                    let newShape = accShape.concatenating(with: shape,
                                                          alongDimension: axis)
                    else { throw VerificationError.concatenationShapeMismatch(vv, axis, self) }
                accShape = newShape
            }

        case let .reduce(_, v1, axis), let .scan(_, v1, axis: axis):
            guard case let .tensor(s1, _) = v1.type else {
                throw VerificationError.notTensor(v1, self)
            }
            if let axis = axis, !s1.indices.contains(axis) {
                throw VerificationError.axisOutOfBounds(axis, v1, self)
            }

        case let .shapeCast(v1, target):
            guard case let .tensor(s1, _) = v1.type else {
                throw VerificationError.notTensor(v1, self)
            }
            guard s1.contiguousSize == target.contiguousSize else {
                throw VerificationError.cannotShapeCast(v1, target, self)
            }

        case let .dataTypeCast(v1, target):
            guard case let .tensor(_, t1) = v1.type else {
                throw VerificationError.notTensor(v1, self)
            }
            guard t1.canCast(to: target) else {
                throw VerificationError.cannotTypeCast(v1, target, self)
            }

        case let .call(fun, vv):
            guard vv.count == fun.arguments.count else {
                throw VerificationError.functionArgumentCountMismatch(fun, self)
            }
            for (actual, formal) in zip(vv, fun.arguments) {
                guard actual.type == formal.type else {
                    throw VerificationError.functionArgumentMismatch(actual, formal, fun, self)
                }
            }

        case let .gradient(fun, vv):
            guard vv.count == fun.arguments.count else {
                throw VerificationError.functionArgumentCountMismatch(fun, self)
            }
            for (actual, formal) in zip(vv, fun.arguments) {
                guard actual.type == formal.type else {
                    throw VerificationError.functionArgumentMismatch(actual, formal, fun, self)
                }
            }

        case let .subtensor(v1, idx):
            guard case let .tensor(s1, _) = v1.type else {
                throw VerificationError.notTensor(v1, self)
            }
            guard let _ = s1[idx] else {
                throw VerificationError.invalidTensorIndex(v1, idx, self)
            }

        case let .tupleElement(v1, i):
            guard case let .tuple(subtypes) = v1.type else {
                throw VerificationError.notTuple(v1, self)
            }
            guard subtypes.indices.contains(i) else {
                throw VerificationError.invalidIndex(v1, i, self)
            }

        case let .unary(_, v1), let .transpose(v1):
            guard case .tensor = v1.type else {
                throw VerificationError.notTensor(v1, self)
            }

        case .tuple: break
        }
    }
}

extension Use : SelfVerifiable {
    public func verify() throws {
        guard value.type == self.type else {
            throw VerificationError<Use>.useTypeMismatch(self)
        }
    }
}

public class Verifier<Body : IRUnit & SelfVerifiable> : AnalysisPass<Body, Void> {
    public override class func run(on body: Body) throws {
        try body.verify()
    }
}
