//
//  Verification.swift
//  DLVM
//
//  Copyright 2016-2017 Richard Wei.
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

public enum VerificationError<Node : SelfVerifiable> : Error {
    case illegalModuleName(String, Node)
    case duplicateStructField(String, Node)
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
    case gradientArgumentMismatch(Function, Int, [Int], Node)
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
    case invalidIndices(Use, [ElementKey], Node)
    case invalidOffsets(Use, [ElementKey], Node)
    case gradientTypeMismatch(Function.Attribute, Type, Node)
    case invalidLiteral(Type, Literal, Node)
    case missingIndices(Use, Node)
    case notDifferentiable(Node)
    case unexpectedMemoryType(Use, Node)
    case invalidCopyOperands(Use, Use, Node)
    case notComputeFunction(Function, Node)
    case computeGraphMismatch(Function, Node)
    case notBox(Use, Node)
    case notHeapObject(Use, Node)
    case cannotLoadFromCompute(Use, Node)
    case notFunction(Use, Node)
    case invalidGradientArguments(Use, Node)
    case notConstantExpression(Node)
    case structFieldNameMismatch(StructType, Use, Node)
}

public protocol SelfVerifiable {
    func performVerification() throws
}

import Foundation
private let moduleNamePattern = try! NSRegularExpression(pattern: "[a-zA-Z_]", options: [])

extension Module : SelfVerifiable {
    private func verify<T: SelfVerifiable & Named>(_ declaration: T, namespace: inout Set<String>) throws {
        guard !namespace.contains(declaration.name) else {
            throw VerificationError.redeclared(declaration)
        }
        try declaration.performVerification()
        namespace.insert(declaration.name)
    }
    
    public func performVerification() throws {
        guard let _ = moduleNamePattern.firstMatch(in: name, options: [ .anchored ],
                                                   range: NSRange(0..<name.characters.count)) else {
            throw VerificationError.illegalModuleName(name, self)
        }
        /// Verify types and values
        var typeNameSet: Set<String> = []
        try typeAliases.forEach { try self.verify($0, namespace: &typeNameSet) }
        try structs.forEach { try self.verify($0, namespace: &typeNameSet) }
        var valueNameSet: Set<String> = []
        try elements.forEach { try self.verify($0, namespace: &valueNameSet) }
        try globalValues.forEach { try self.verify($0, namespace: &valueNameSet) }
    }
}

extension GlobalValue : SelfVerifiable {
    public func performVerification() throws {
        try initializer.performVerification()
    }
}

extension TypeAlias : SelfVerifiable {
    public func performVerification() throws {
        guard let type = type else { return }
        guard type.canonical.isValid else {
            throw VerificationError.invalidType(self)
        }
    }
}

extension StructType : SelfVerifiable {
    public func performVerification() throws {
        var set: Set<String> = []
        /// Verify struct fields' uniqueness and validity
        for (name, ty) in fields {
            guard !set.contains(name) else {
                throw VerificationError.duplicateStructField(name, self)
            }
            guard ty.isValid else {
                throw VerificationError.invalidType(self)
            }
            set.insert(name)
        }
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
        case (.tensor([], let dt), .scalar(let lit)) where lit.typeBase != dt.base:
            throw VerificationError.invalidLiteral(type, literal, self)

        /// High-dimensional compute tensor with scalar literal
        case(.box(.tensor(_, let dt), .compute), .scalar(let lit)) where lit.typeBase != dt.base:
            throw VerificationError.invalidLiteral(type, literal, self)

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
            
        case let (.struct(structTy), .struct(fields)) where structTy.fields.count == fields.count:
            for ((name: fmlName, type: fmlType), (name, val)) in zip(structTy.fields, fields) {
                guard fmlName == name else {
                    throw VerificationError.structFieldNameMismatch(structTy, val, self)
                }
                try verifyUse(val, fmlType)
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
            case let .differentiating(antigrad, from: diffIndex, wrt: varIndices, keepingOutputs: outputIndices):
                /// Check for type mismatch
                guard let expectedType = antigrad.gradientType(fromOutput: diffIndex,
                                                               withRespectTo: varIndices,
                                                               keepingOutputs: outputIndices) else {
                    throw VerificationError.gradientArgumentMismatch(antigrad, diffIndex, varIndices, self)
                }
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
    /// Verifies constant expression
    public func performVerification() throws {
        switch self {
        case _ where accessesMemory, .load, .store, .apply, .compute:
            throw VerificationError.notConstantExpression(self)
        default: return
        }
    }

    /// Verifies instruction
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
            case .box(.tensor(_, _), .compute): break
            default:
                throw VerificationError.notTensor(v1, instruction)
            }


        case let .binary(_, lhs, rhs, nil):
            switch (lhs.type.unaliased, rhs.type.unaliased) {
            /// Non-compute scalar
            case let (.tensor([], t1), .tensor([], t2)) where t1 == t2:
                break
            /// Compute tensor
            case let (.box(.tensor(s1, t1), .compute),
                      .box(.tensor(s2, t2), .compute))
                where s1 == s2 && t1 == t2:
                guard function.isCompute else {
                    throw VerificationError.notComputeFunction(function, instruction)
                }
            default:
                throw VerificationError.unbroadcastableMismatch(lhs, rhs, instruction)
            }

        case let .binary(_, lhs, rhs, bc?):
            switch (lhs.type.unaliased, rhs.type.unaliased) {
            /// Compute tensor
            case let (.box(.tensor(s1, t1), .compute),
                      .box(.tensor(s2, t2), .compute))
                where mutuallyBroadcast(s1, s2, at: bc) != nil && t1 == t2:
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
            guard case let .box(.tensor(s1, t1), .compute) = lhs.type.unaliased,
                  case let .box(.tensor(s2, t2), .compute) = rhs.type.unaliased,
                  s1.isMatrixMultiplicable(by: s2), t1 == t2
                else { throw VerificationError.cannotMatrixMultiply(lhs, rhs, instruction) }

        /// Compute-only
        case let .concatenate(vv, axis: axis):
            guard function.isCompute else {
                throw VerificationError.notComputeFunction(function, instruction)
            }
            guard let first = vv.first,
                  case let .box(.tensor(s1, t1), .compute) = first.type.unaliased
                else { throw VerificationError.noOperands(instruction) }
            /// Check simple, data type equality, and concatenability
            var accShape: TensorShape = s1
            for v in vv.dropFirst() {
                guard case let .box(.tensor(shape, type), .compute) = v.type.unaliased,
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
            guard case let .box(.tensor(s1, _), .compute) = v1.type.unaliased else {
                throw VerificationError.notTensor(v1, instruction)
            }
            if let axis = axis, !s1.indices.contains(axis) {
                throw VerificationError.axisOutOfBounds(axis, v1, instruction)
            }

        case let .shapeCast(v1, target):
            switch v1.type.unaliased {
            case let .tensor(s1, _) where target.contiguousSize == s1.contiguousSize: break
            case let .box(.tensor(s1, _), .compute) where target.contiguousSize == s1.contiguousSize:
                guard function.isCompute else {
                    throw VerificationError.notComputeFunction(function, instruction)
                }
            default:
                throw VerificationError.notTensor(v1, instruction)
            }

        case let .dataTypeCast(v1, target):
            switch v1.type.unaliased {
            case let .tensor(_, t1),
                 let .box(.tensor(_, t1), .compute):
                guard t1.canCast(to: target) else {
                    throw VerificationError.cannotCastDataType(v1, target, instruction)
                }
            default:
                throw VerificationError.notTensor(v1, instruction)
            }

        case let .apply(fun, vv):
            let actual = vv.map{$0.type}
            switch fun.type.unaliased {
            case let .function(args, _), let .pointer(.function(args, _)):
                guard actual.count == args.count
                   && zip(actual, args).forAll({$0.conforms(to: $1)}) else {
                    throw VerificationError.functionArgumentMismatch(vv, fun.type.unaliased, instruction)
                }
            default:
                throw VerificationError.invalidType(fun)
            }

        case let .compute(fun, vv, in: graph):
            switch (fun, graph.type) {
            case let (.function(fd1), .computeBuffer(fd2)) where fd1 == fd2:
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

        case let .allocateHeap(_, count: v):
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
            guard case .pointer(_) = v1.type.unaliased else {
                throw VerificationError.notPointer(v1, instruction)
            }

        case let .store(v1, to: v2):
            guard case let .pointer(subtype) = v2.type.unaliased else {
                throw VerificationError.notPointer(v2, instruction)
            }
            guard v1.type == subtype else {
                throw VerificationError.typeMismatch(v1, v2, instruction)
            }

        case let .elementPointer(v, ii):
            guard case let .pointer(t) = v.type.unaliased else {
                throw VerificationError.notPointer(v, instruction)
            }
            guard let _ = t.subtype(at: ii) else {
                throw VerificationError.invalidOffsets(v, ii, instruction)
            }

        case let .copy(from: src, to: dest, count: count):
            guard case .tensor([], .int(64)) = count.type.unaliased else {
                throw VerificationError.unexpectedType(count, .scalar(.int(64)), instruction)
            }
            switch (src.type, dest.type) {
            case let (.pointer(t1), .pointer(t2)) where t1 == t2:
                break
            case let (.box(t1, _), .box(t2, _)) where t1 == t2:
                /// Count must be literal 1
                guard case .literal(let litVal) = count,
                      case .scalar(.int(1)) = litVal.literal
                    else { fallthrough }
                
            default:
                throw VerificationError.invalidCopyOperands(src, dest, instruction)
            }

        case .bitCast(_, _):
            // TODO
            break

        case .deallocate(let v):
            switch v.type.unaliased {
            case .pointer, .box, .computeBuffer: break
            case _: throw VerificationError.notHeapObject(v, instruction)
            }

        case let .retain(v), let .release(v):
            guard case .box = v.type else {
                throw VerificationError.notBox(v, instruction)
            }

        case let .allocateCompute(v):
            guard case let .function(fref) = v else {
                throw VerificationError.notFunction(v, instruction)
            }
            guard fref.isCompute else {
                throw VerificationError.notComputeFunction(fref, instruction)
            }

        case let .gradient(v, from: diff, wrt: vars, keeping: outputIndices):
            guard case let .function(fref) = v else {
                throw VerificationError.notFunction(v, instruction)
            }
            guard fref.isCompute else {
                throw VerificationError.notComputeFunction(fref, instruction)
            }
            guard let _ = fref.gradientType(fromOutput: diff, withRespectTo: vars,
                                            keepingOutputs: outputIndices) else {
                throw VerificationError.invalidGradientArguments(v, instruction)
            }

        case let .requestMemory(v):
            guard case .box(_, .compute) = v.type.unaliased else {
                throw VerificationError.unexpectedMemoryType(v, instruction)
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
        func verify(_ lhs: Type, _ rhs: Type) throws {
            guard lhs == rhs else {
                throw VerificationError.useTypeMismatch(self)
            }
        }
        switch self {
        case let .argument(ty, def):
            try verify(ty, def.type)
        case let .instruction(ty, def):
            try verify(ty, def.type)
        case let .global(ty, gv):
            try verify(ty, gv.type.pointer)
        case .constant, .literal, .function:
            break
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
