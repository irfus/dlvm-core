//
//  Verification.swift
//  DLVM
//
//  Created by Richard Wei on 12/25/16.
//
//

public enum VerificationError<Node : SelfVerifiable> : Error {
    case basicBlockRedeclared(BasicBlock, Node)
    case redeclaredInstruction(Instruction, Node)
    case noParent(Node)
    case blockFunctionMismatch(BasicBlock, Node)
    case missingTerminator(Node)
    case unbroadcastableMismatch(Use, Use, Node)
    case typeMismatch(Use, Use, Node)
    case unexpectedShape(Use, TensorShape, Node)
    case unexpectedType(Use, DataType, Node)
    case cannotShapeCast(Use, TensorShape, Node)
    case cannotTypeCast(Use, DataType, Node)
    case cannotMatrixMultiply(Use, Use, Node)
    case cannotConcatenate(Use, Node)
    case concatenationShapeMismatch([Use], TensorShape, Node)
    case useShapeMismatch(Node)
    case useTypeMismatch(Node)
    case noOperands(Node)
    case noTopSection(Node)
    case noDimensions(Use, Node)
    case noSpecifiedDimension(Use, Int, Node)
    case definitionNotInBasicBlock(Use, BasicBlock, Node)
    case placeholderError(Def<Placeholder>, shouldBeRecurrent: Bool, Node)
    case functionResultMismatch(TensorShape, DataType, Function, Node)
    case functionArgumentCountMismatch(Function, Node)
    case functionArgumentMismatch(Use, Def<Argument>, Function, Node)
    case notAFunctionCall(Use, Function, Node)
    case functionDiffArgumentMismatch(Use, Def<Argument>, Function, Node)
    case invalidTensorIndex(Use, TensorIndex, Node)
    case invalidIndex(Use, Int, Node)
    case intrinsicArgError(Intrinsic, [Use], Node)
    case intrinsicResultMismatch(TensorShape, DataType, Intrinsic, Node)
    case multipleExits([BasicBlock], Node)
    case noEntry(Node)
    case noExit(Node)
    case noForwardPass(Node)
    case noReturn(Node)
    case postdominanceUnreachable(BasicBlock, Node)
    case dominanceUnreachable(BasicBlock, Node)
    case basicBlockArgumentMismatch([Use], BasicBlock, Node)
    case unexpectedBasicBlockType(BasicBlock, Node)
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
        /// TODO: Check for reachability in dominator tree and postdominator tree
    }
}

extension Section : SelfVerifiable {
    public func verify() throws {
        
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
                    throw VerificationError.redeclaredInstruction(inst, self)
                }
                instNames.insert(name)
            }
            try inst.verify()
        }
    }
}

extension Instruction : SelfVerifiable {
    public func verify() throws {
        switch kind {
        case let .control(ctrl):
            try ctrl.verify()
        case let .operation(def):
            try def.verify()
        }
    }
}

public extension Def where ValueType : SelfVerifiable {
    public func verify() throws {
        try value.verify()
    }
}

extension Control : SelfVerifiable {
    public func verify() throws {
        for operand in operands {
            try operand.verify()
        }
    }

    private func verifySemantics() throws {
        switch self {

        case let .condBr(use, _, _):
            /// Check bool use
            guard use.type == .bool else {
                throw VerificationError.unexpectedType(use, .bool, self)
            }
            guard use.shape == .scalar else {
                throw VerificationError.unexpectedShape(use, .scalar, self)
            }

        case let .pull(ph, thenBB, elseBB):
            /// `thenBB` must have one argument of the same type of `use`
            guard thenBB.arguments.count == 1,
                ph.type == thenBB.arguments[0].type,
                ph.shape == thenBB.arguments[0].shape else {
                throw VerificationError.unexpectedBasicBlockType(thenBB, self)
            }
            /// `elseBB` cannot take arguments
            guard elseBB.arguments.isEmpty else {
                throw VerificationError.unexpectedBasicBlockType(elseBB, self)
            }

        case let .yield(use, to: global as Definition),
             let .store(use, to: global as Definition):
            guard use.type == global.type else {
                throw VerificationError.unexpectedType(use, global.type, self)
            }
            guard use.shape == global.shape else {
                throw VerificationError.unexpectedShape(use, global.shape, self)
            }

        case let .br(bb, args):
            for (formal, arg) in zip(bb.arguments, args) {
                guard formal.shape == arg.shape, formal.type == arg.type else {
                    throw VerificationError.basicBlockArgumentMismatch(args, bb, self)
                }
            }
            
        case .ret: break
        case .cont: break

        }
    }
}

extension Operation : SelfVerifiable {
    public func verify() throws {
        for operand in operands {
            try operand.verify()
        }
        try verifySemantics()
    }

    private func verifyHomogeneousType(_ lhs: Use, _ rhs: Use) throws {
        guard lhs.type == rhs.type else {
            throw VerificationError.typeMismatch(lhs, rhs, self)
        }
    }

    private func verifySemantics() throws {
        switch self {
        case let .binary(_, lhs, rhs):
            try verifyHomogeneousType(lhs, rhs)
            guard lhs.shape.canMutuallyBroadcast(with: rhs.shape) else {
                throw VerificationError.unbroadcastableMismatch(lhs, rhs, self)
            }

        case let .matMul(lhs, rhs):
            try verifyHomogeneousType(lhs, rhs)
            guard lhs.shape.canMatrixMultiply(with: rhs.shape) else {
                throw VerificationError.cannotMatrixMultiply(lhs, rhs, self)
            }

        case let .concat(shape, type, uses, axis: axis):
            guard var finalShape = uses.first?.shape else {
                throw VerificationError.noOperands(self)
            }
            for use in uses.dropFirst() {
                guard use.type == type else {
                    throw VerificationError.unexpectedType(use, type, self)
                }
                guard let concatShape = finalShape.concatenating(with: use.shape, alongDimension: axis)
                    else { throw VerificationError.cannotConcatenate(use, self) }
                finalShape = concatShape
            }
            guard finalShape == shape else {
                throw VerificationError.concatenationShapeMismatch(uses, shape, self)
            }

        case let .reduce(_, use, axis: nil),
             let .scan(_, use, axis: nil):
            guard !use.shape.isScalar else {
                throw VerificationError.noDimensions(use, self)
            }

        case let .reduce(_, use, axis: axis?),
             let .scan(_, use, axis: axis?):
            guard use.shape.indices.contains(axis) else {
                throw VerificationError.noSpecifiedDimension(use, axis, self)
            }
            break

        case let .shapeCast(use, target):
            guard use.shape.contiguousSize == target.contiguousSize else {
                throw VerificationError.cannotShapeCast(use, shape, self)
            }

        case let .typeCast(use, type):
            guard use.type.size <= type.size else {
                throw VerificationError.cannotTypeCast(use, type, self)
            }

        case let .get(ph):
            guard !ph.isRecurrent else {
                throw VerificationError.placeholderError(ph, shouldBeRecurrent: false, self)
            }

        case .unary: break
        case .transpose: break

        case let .call(shape, type, fun, ops):
            guard shape == fun.result?.shape else {
                throw VerificationError.functionResultMismatch(shape, type, fun, self)
            }
            guard ops.count == fun.arguments.count else {
                throw VerificationError.functionArgumentCountMismatch(fun, self)
            }
            for (actual, formal) in zip(ops, fun.arguments) {
                guard actual.shape == formal.shape, actual.type == formal.type else {
                    throw VerificationError.functionArgumentMismatch(actual, formal, fun, self)
                }
            }

        case let .diff(fun, call, wrt: arg):
            guard fun.arguments.contains(arg) else {
                throw VerificationError.functionDiffArgumentMismatch(call, arg, fun, self)
            }
            guard case .local(let def) = call.kind,
                  case .call(_, _, fun, _) = def.value else {
                throw VerificationError.notAFunctionCall(call, fun, self)
            }

        case let .subtensor(use, idx):
            guard let _ = use.shape[idx] else {
                throw VerificationError.invalidTensorIndex(use, idx, self)
            }

        case let .element(use, i):
            guard let first = use.shape.first, i < first else {
                throw VerificationError.invalidIndex(use, i, self)
            }

        case let .intrinsic(shape, type, intrin, uses):
            guard let (rShape, rType) = intrin.result(forArguments: uses) else {
                throw VerificationError.intrinsicArgError(intrin, uses, self)
            }
            guard rShape == shape, rType == type else {
                throw VerificationError.intrinsicResultMismatch(shape, type, intrin, self)
            }
        }
    }
}

extension Use : SelfVerifiable {
    public func verify() throws {
        guard value.shape == self.shape else {
            throw VerificationError<Use>.useShapeMismatch(self)
        }
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
