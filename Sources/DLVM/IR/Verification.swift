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

public enum VerificationError<Node : Verifiable> : Error {
    case illegalName(String, Node)
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
    case cannotDot(Use, Use, Node)
    case noOperands(Node)
    case concatenationShapeMismatch([Use], Int, Node)
    case useShapeMismatch(Node)
    case useTypeMismatch(Node)
    case noDimensions(Use, Node)
    case invalidSlicingRange(CountableClosedRange<Int>, Node)
    case noSpecifiedDimension(Use, Int, Node)
    case definitionNotInBasicBlock(Use, BasicBlock, Node)
    case functionArgumentMismatch([Use], Type, Node)
    case notAFunctionCall(Use, Function, Node)
    case gradientArgumentMismatch(Function, Int?, [Int], Node)
    case invalidTensorIndex(Use, TensorIndex, Node)
    case invalidIndex(Use, Int, Node)
    case multipleExits([BasicBlock], Node)
    case noEntry(Node)
    case noExit(Node)
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
    case gradientTypeMismatch(Function.DeclarationKind, Type, Node)
    case invalidLiteral(Type, Literal, Node)
    case missingIndices(Use, Node)
    case notDifferentiable(Node)
    case unexpectedMemoryType(Use, Node)
    case invalidCopyOperands(Use, Use, Node)
    case notBox(Use, Node)
    case notHeapObject(Use, Node)
    case notFunction(Use, Node)
    case invalidGradientArguments(Use, Node)
    case notConstantExpression(Node)
    case structFieldNameMismatch(StructType, Use, Node)
    case invalidReductionDimensions([Int], Use, Node)
    case dataTypeNotNumeric(Use, Node)
    case invalidAllocationSize(Node)
    case declarationCannotHaveBody(Node)
    case nestedLiteralNotInLiteralInstruction(Literal, Node)
}

public protocol Verifiable {
    func performVerification() throws
}

import struct Foundation.NSRange
import class Foundation.NSRegularExpression
private let identifierPattern = try! NSRegularExpression(pattern: "[a-zA-Z_][a-zA-Z0-9_.]*",
                                                         options: [ .dotMatchesLineSeparators ])

private func verifyIdentifier<Unit : Verifiable>(_ id: String, in unit: Unit) throws {
    guard let _ = identifierPattern.firstMatch(in: id, options: [ .anchored ],
                                               range: NSRange(0..<id.count)) else {
        throw VerificationError.illegalName(id, unit)
    }
}

extension Module : Verifiable {
    private func verify<T: Verifiable & Named>(_ declaration: T, namespace: inout Set<String>) throws {
        guard !namespace.contains(declaration.name) else {
            throw VerificationError.redeclared(declaration)
        }
        try declaration.performVerification()
        namespace.insert(declaration.name)
    }
    
    public func performVerification() throws {
        try verifyIdentifier(name, in: self)
        /// Verify types and values
        var typeNameSet: Set<String> = []
        try typeAliases.forEach { try self.verify($0, namespace: &typeNameSet) }
        try structs.forEach { try self.verify($0, namespace: &typeNameSet) }
        var valueNameSet: Set<String> = []
        try elements.forEach { try self.verify($0, namespace: &valueNameSet) }
        try variables.forEach { try self.verify($0, namespace: &valueNameSet) }
    }
}

extension Variable: Verifiable {
    public func performVerification() throws {
    }
}

extension TypeAlias : Verifiable {
    public func performVerification() throws {
        guard let type = type else { return }
        guard type.canonical.isValid else {
            throw VerificationError.invalidType(self)
        }
    }
}

extension StructType : Verifiable {
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

extension LiteralValue : Verifiable {

    private func verifyUse(_ use: Use, _ elementType: Type) throws {
        try use.performVerification()
        guard use.type == elementType else {
            throw VerificationError.unexpectedType(use, elementType, self)
        }
    }
    
    public func performVerification() throws {
        switch (type.canonical, literal) {

        /* Simple literals */
            
        /// Anything can be undefined
        case (_, .undefined): break
        /// Any tensor can be zero initialized
        case (.tensor, .zero): break

        /// Any tensor with scalar literal
        case (.tensor(_, let dt), .scalar(let lit)) where dt.isExpressible(as: lit):
            break

        /* Aggregate literals */

        /// Tensor literal
        case let (.tensor(shape, dt), .tensor(elements)) where elements.count == shape.first:
            let elementType: Type = .tensor(shape.dropFirst(), dt)
            for use in elements {
                try verifyUse(use, elementType)
            }

        /// Tuple literal
        case let (.tuple(elementTypes), .tuple(elements)) where elementTypes.count == elements.count:
            for (elementType, use) in zip(elementTypes, elements) {
                try verifyUse(use, elementType)
            }

        /// Array literal
        case let (.array(n, elementType), .array(elements)) where n == elements.count:
            for use in elements {
                try verifyUse(use, elementType)
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

extension Function : Verifiable {
    private func verifyDifferentiability(from: Int, wrt: [Int]) throws {
        /// - TODO: Check along the differentiation floow
        /// let dfg = try analysis(from: DataFlowGraphAnalysis.self)
        /// All arguments have to be tensors or aggregate types of tensors
        /// No explicit pointer semantics are allowed
        /// - TODO: Check arguments
        for inst in instructions where inst.kind.accessesMemory {
            throw VerificationError.notDifferentiable(self)
        }
    }

    public func performVerification() throws {
        try verifyIdentifier(name, in: self)

        /// Verify declaration
        if let declarationKind = declarationKind {
            /// Declarations cannot have body
            guard isEmpty else {
                throw VerificationError.declarationCannotHaveBody(self)
            }
            switch declarationKind {
            /// Verify gradient function's type signature
            case let .gradient(antigrad, from: diffIndex, wrt: varIndices,
                               keeping: outputIndices, seedable: isSeedable):
                /// Check for type mismatch
                guard let expectedType = antigrad.gradientType(fromOutput: diffIndex,
                                                               withRespectTo: varIndices,
                                                               keepingOutputs: outputIndices,
                                                               isSeedable: isSeedable) else {
                    throw VerificationError.gradientArgumentMismatch(antigrad, diffIndex, varIndices, self)
                }
                guard type == expectedType else {
                    throw VerificationError.gradientTypeMismatch(declarationKind, expectedType, self)
                }
            case .external:
                break
            }
            /// Skip all CFG/DFG verifications because it's a declaration!
            return
        }

        let domTree = analysis(from: DominanceAnalysis.self)

        var bbNames: Set<String> = []
        /// Verify basic blocks
        for bb in self {
            /// Check for redeclaration/redefinition
            guard !bbNames.contains(bb.name) else {
                throw VerificationError.redeclared(bb)
            }
            /// Check entry block arguments
            guard !bb.isEntry || bb.arguments.map({$0.type}).elementsEqual(argumentTypes) else {
                throw VerificationError.functionEntryArgumentMismatch(bb, self)
            }
            bbNames.insert(bb.name)
            /// Verify bb
            try bb.performVerification()

            /// Check return type
            let bbPremise = try bb.verifyPremise()
            if case let .return(retVal) = bbPremise.terminator.kind {
                switch retVal {
                case let use? where use.type != returnType:
                    throw VerificationError.returnTypeMismatch(bbPremise.terminator, self)
                case nil where !returnType.isVoid:
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

    }
}

extension BasicBlock : Verifiable {
    public func performVerification() throws {
        try verifyIdentifier(name, in: self)
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

extension Argument : Verifiable {
    public func performVerification() throws {
        try verifyIdentifier(name, in: self)
    }
}

extension Instruction : Verifiable {
    public func performVerification() throws {
        if let name = name {
            try verifyIdentifier(name, in: self)
        }
        /// Use type must match usee type
        for use in operands {
            try use.performVerification()
            /// Special case: nested literals can only be in a `literal`
            /// instruction
            if opcode != .literal, case .literal(_, let lit) = use {
                guard lit.isScalar else {
                    throw VerificationError
                        .nestedLiteralNotInLiteralInstruction(lit, self)
                }
            }
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
    /// Verifies instruction
    public func performVerification(in instruction: Instruction) throws {
        switch self {
        case let .conditional(use, thenBB, thenArgs, elseBB, elseArgs):
            guard case let .tensor(s, t) = use.type.unaliased, s.isScalar, t.isBool else {
                throw VerificationError.unexpectedType(use, .tensor(.scalar, .bool), instruction)
            }
            guard thenBB.arguments.count == thenArgs.count,
                  zip(thenBB.arguments, thenArgs).forAll({$0.1.type == $0.1.type}) else {
                throw VerificationError.basicBlockArgumentMismatch(thenArgs, thenBB, instruction)
            }
            guard elseBB.arguments.count == elseArgs.count,
                  zip(elseBB.arguments, elseArgs).forAll({$0.0.type == $0.1.type}) else {
                throw VerificationError.basicBlockArgumentMismatch(elseArgs, elseBB, instruction)
            }

        case let .branch(bb, args):
            guard bb.arguments.count == args.count,
                  zip(bb.arguments, args).forAll({$0.0.type == $0.1.type}) else {
                throw VerificationError.basicBlockArgumentMismatch(args, bb, instruction)
            }

        case .return: break /// Verified at Function

        case let .literal(lit, ty):
            try LiteralValue(type: ty, literal: lit).performVerification()

        case let .map(_, v1), let .transpose(v1):
            guard case .tensor(_, _) = v1.type.unaliased else {
                throw VerificationError.notTensor(v1, instruction)
            }

        case let .slice(v, at: range):
            guard case let .tensor(shape, _) = v.type.unaliased else {
                throw VerificationError.notTensor(v, instruction)
            }
            guard let dim0 = shape.first else {
                throw VerificationError.noDimensions(v, instruction)
            }
            guard range.contains(dim0) else {
                throw VerificationError.invalidSlicingRange(range, instruction)
            }

        case let .zipWith(_, lhs, rhs):
            guard case let .tensor(s1, t1) = lhs.type.unaliased,
                  case let .tensor(s2, t2) = rhs.type.unaliased,
                  s1.isCompatible(with: s2), t1 == t2 else {
                throw VerificationError.unbroadcastableMismatch(lhs, rhs, instruction)
            }

        case let .dot(lhs, rhs):
            guard case let .tensor(s1, t1) = lhs.type.unaliased,
                  case let .tensor(s2, t2) = rhs.type.unaliased,
                  s1.isMatrixMultiplicable(by: s2), t1 == t2 else {
                throw VerificationError.cannotDot(lhs, rhs, instruction)
            }

        case let .concatenate(vv, axis: axis):
            guard let first = vv.first,
                  case let .tensor(s1, t1) = first.type.unaliased else {
                throw VerificationError.noOperands(instruction)
            }
            var accShape: TensorShape = s1
            for v in vv.dropFirst() {
                guard case let .tensor(shape, type) = v.type.unaliased, type == t1,
                      let newShape = accShape.concatenating(with: shape, alongDimension: axis) else {
                    throw VerificationError.concatenationShapeMismatch(vv, axis, instruction)
                }
                accShape = newShape
            }

        case let .scan(.op(op), v1, dims):
            let shape: TensorShape
            if op.isBoolean {
                guard case let .tensor(s1, .bool) = v1.type.unaliased else {
                    throw VerificationError.unexpectedDataType(v1, .bool, instruction)
                }
                shape = s1
            } else {
                guard case let .tensor(s1, t1) = v1.type.unaliased, t1.isNumeric else {
                    throw VerificationError.dataTypeNotNumeric(v1, instruction)
                }
                shape = s1
            }
            guard dims.count <= shape.rank, dims.forAll({$0 < shape.rank}), !dims.containsDuplicate else {
                throw VerificationError.invalidReductionDimensions(dims, v1, instruction)
            }
            
        case let .reduce(.op(op), v1, initial, dims):
            let shape: TensorShape
            let dtype: DataType
            if op.isBoolean {
                guard case let .tensor(s1, .bool) = v1.type.unaliased else {
                    throw VerificationError.unexpectedDataType(v1, .bool, instruction)
                }
                shape = s1
                dtype = .bool
            } else {
                guard case let .tensor(s1, t1) = v1.type.unaliased, t1.isNumeric else {
                    throw VerificationError.dataTypeNotNumeric(v1, instruction)
                }
                shape = s1
                dtype = t1
            }
            guard dims.count <= shape.rank, dims.forAll({$0 < shape.rank}), !dims.containsDuplicate else {
                throw VerificationError.invalidReductionDimensions(dims, v1, instruction)
            }
            /// Initial must be a scalar
            guard case .tensor([], dtype) = initial.type.canonical else {
                throw VerificationError.unexpectedShape(initial, .scalar, instruction)
            }

        case let .scan(.function(f), v1, dims):
            guard case let .tensor(s1, t1) = v1.type.unaliased else {
                throw VerificationError.notTensor(v1, instruction)
            }
            let expectedFuncType: Type = .function([.tensor([], t1)], .tensor([], t1))
            guard expectedFuncType == f.type.unaliased else {
                throw VerificationError.unexpectedType(f, expectedFuncType, instruction)
            }
            guard dims.count <= s1.rank, dims.forAll({$0 < s1.rank}), !dims.containsDuplicate else {
                throw VerificationError.invalidReductionDimensions(dims, v1, instruction)
            }
            
        case let .reduce(.function(f), v1, initial, dims):
            guard case let .tensor(s1, t1) = v1.type.unaliased else {
                throw VerificationError.notTensor(v1, instruction)
            }
            let expectedFuncType: Type = .function([.tensor([], t1)], .tensor([], t1))
            guard expectedFuncType == f.type.unaliased else {
                throw VerificationError.unexpectedType(f, expectedFuncType, instruction)
            }
            guard dims.count <= s1.rank, dims.forAll({$0 < s1.rank}), !dims.containsDuplicate else {
                throw VerificationError.invalidReductionDimensions(dims, v1, instruction)
            }
            /// Initial must be a scalar
            guard case .tensor([], t1) = initial.type.canonical else {
                throw VerificationError.unexpectedShape(initial, .scalar, instruction)
            }

        case let .shapeCast(v1, target):
            guard case let .tensor(s1, _) = v1.type.unaliased,
                  target.contiguousSize == s1.contiguousSize else {
                throw VerificationError.notTensor(v1, instruction)
            }

        case let .dataTypeCast(v1, target):
            guard case let .tensor(_, t1) = v1.type.unaliased else {
                throw VerificationError.notTensor(v1, instruction)
            }
            guard t1.canCast(to: target) else {
                throw VerificationError.cannotCastDataType(v1, target, instruction)
            }

        case let .apply(fun, vv):
            let actual = vv.map{$0.type}
            switch fun.type.unaliased {
            case let .function(args, _),
                 let .pointer(.function(args, _)):
                guard actual.count == args.count && zip(actual, args).forAll({$0.0.conforms(to: $0.1)}) else {
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
            guard let _ = v1.type.elementType(at: indices) else {
                throw VerificationError.invalidIndices(v1, indices, instruction)
            }

        case let .insert(src, to: dest, at: indices):
            guard !indices.isEmpty else {
                throw VerificationError.missingIndices(dest, instruction)
            }
            guard let elementType = dest.type.elementType(at: indices) else {
                throw VerificationError.invalidIndices(dest, indices, instruction)
            }
            guard elementType == src.type else {
                throw VerificationError.typeMismatch(src, dest, instruction)
            }

        case let .load(v1):
            guard case .pointer(_) = v1.type.unaliased else {
                throw VerificationError.notPointer(v1, instruction)
            }

        case let .store(v1, to: v2):
            guard case let .pointer(elementType) = v2.type.unaliased else {
                throw VerificationError.notPointer(v2, instruction)
            }
            guard v1.type == elementType else {
                throw VerificationError.typeMismatch(v1, v2, instruction)
            }

        case let .elementPointer(v, ii):
            guard case let .pointer(t) = v.type.unaliased else {
                throw VerificationError.notPointer(v, instruction)
            }
            guard let _ = t.elementType(at: ii) else {
                throw VerificationError.invalidOffsets(v, ii, instruction)
            }

        case let .copy(from: src, to: dest, count: count):
            guard case .tensor([], .int(64)) = count.type.unaliased else {
                throw VerificationError.unexpectedType(count, .scalar(.int(64)), instruction)
            }
            switch (src.type, dest.type) {
            case let (.pointer(t1), .pointer(t2)) where t1 == t2:
                break
            case let (.box(t1), .box(t2)) where t1 == t2:
                guard case .literal(_, .scalar(.int(1))) = count else { fallthrough }
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

        case let .allocateStack(_, n):
            guard n > 0 else {
                throw VerificationError.invalidAllocationSize(instruction)
            }
            
        case .trap, .allocateBox: break
        }
    }
}

extension Use : Verifiable {
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
        case let .variable(ty, gv):
            try verify(ty, gv.type.pointer)
        case .literal, .function:
            break
        }
    }
}

/// Verifier pass
public enum Verifier<Unit : IRCollection> : VerificationPass {
    public typealias Body = Unit
    public typealias Result = Void
    
    public static func run(on body: Body) throws {
        try body.performVerification()
    }
}

/// Cached verification
public extension IRCollection {
    func verify() throws {
        try runVerification(Verifier<Self>.self)
    }
}
