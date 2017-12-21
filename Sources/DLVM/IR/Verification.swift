//
//  Verification.swift
//  DLVM
//
//  Copyright 2016-2017 The DLVM Team.
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

public enum VerificationError<Node : Verifiable> : Error {
    case axisOutOfBounds(Int, Use, Node)
    case basicBlockArgumentMismatch([Use], BasicBlock, Node)
    case blockFunctionMismatch(BasicBlock, Node)
    case convolveInvalidShape(Use, Node)
    case convolveInputChannelMismatch(Use, Use, Node)
    case convolveInvalidDilation([Int], Node)
    case convolveInvalidDilationRank([Int], Int, Node)
    case convolveTypeMismatch(Use, Use, Node)
    case windowDimensionsMismatch([Int], Use, Node)
    case cannotCastDataType(Use, DataType, Node)
    case cannotDot(Use, Use, Node)
    case cannotShapeCast(Use, TensorShape, Node)
    case concatenationShapeMismatch([Use], Int, Node)
    case dataTypeMismatch(Use, Use, Node)
    case dataTypeNotBoolean(Use, Node)
    case dataTypeNotNumeric(Use, Node)
    case declarationCannotHaveBody(Node)
    case definitionNotInBasicBlock(Use, BasicBlock, Node)
    case duplicateStructField(String, Node)
    case functionArgumentMismatch([Use], Type, Node)
    case functionEntryArgumentMismatch(BasicBlock, Node)
    case gradientArgumentMismatch(Function, Int?, [Int]?, Node)
    case gradientTypeMismatch(Function.DeclarationKind, Type, Node)
    case illegalName(String, Node)
    case invalidAllocationSize(Node)
    case invalidCopyOperands(Use, Use, Node)
    case invalidGradientArguments(Use, Node)
    case invalidIndex(Use, Int, Node)
    case invalidIndices(Use, [ElementKey], Node)
    case invalidLiteral(Type, Literal, Node)
    case invalidOffsets(Use, [ElementKey], Node)
    case invalidReductionDimensions([Int], Use, Node)
    case invalidSlicingRange(CountableClosedRange<Int>, Node)
    case invalidTensorIndex(Use, TensorIndex, Node)
    case invalidType(Node)
    case missingIndices(Use, Node)
    case missingTerminator(Node)
    case multipleExits([BasicBlock], Node)
    case namedVoidValue(Node)
    case nestedLiteralNotInLiteralInstruction(Literal, Node)
    case noDimensions(Use, Node)
    case noEntry(Node)
    case noExit(Node)
    case noOperands(Node)
    case noParent(Node)
    case noReturn(Node)
    case noSpecifiedDimension(Use, Int, Node)
    case notAFunctionCall(Use, Function, Node)
    case notBox(Use, Node)
    case notConstantExpression(Node)
    case notDifferentiable(Node)
    case notFunction(Use, Node)
    case notHeapObject(Use, Node)
    case notPointer(Use, Node)
    case notStack(Use, Node)
    case notTensor(Use, Node)
    case notTuple(Use, Node)
    case randomBoundNotScalar(Use, Node)
    case redeclared(Node)
    case returnTypeMismatch(Instruction, Node)
    case structFieldNameMismatch(StructType, Use, Node)
    case typeMismatch(Use, Use, Node)
    case unbroadcastableMismatch([Use], Node)
    case unexpectedBasicBlockType(BasicBlock, Node)
    case unexpectedDataType(Use, DataType, Node)
    case unexpectedMemoryType(Use, Node)
    case unexpectedShape(Use, TensorShape, Node)
    case unexpectedType(Use, Type, Node)
    case useBeforeDef(user: Instruction, usee: Value, Node)
    case useShapeMismatch(Node)
    case useTypeMismatch(Node)
    case windowInvalidStrides([Int], Node)
    case windowInvalidStridesRank([Int], Int, Node)
    case windowInvalidPadding([(low: Int, high: Int)], Node)
    case windowInvalidPaddingRank([(low: Int, high: Int)], Int, Node)
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
    private func verifyDifferentiability(from diffIndex: Int?,
                                         wrt varIndices: [Int]) throws {
        /// - TODO: Check along the differentiation flow
        /// let dfg = try analysis(from: DataFlowGraphAnalysis.self)
        /// All arguments have to be tensors or aggregate types of tensors

        /// No explicit pointer semantics are allowed
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

/*
fileprivate class StackContext {
    var stacks: [ObjectIdentifier: [Type]] = [:]

    func insert(_ stack: Use, instruction: Instruction) throws {
        guard .stack == stack.type, let definition = stack.definition else {
            throw VerificationError.notStack(stack, instruction)
        }
        stacks[ObjectIdentifier(definition)] = []
    }

    func push(_ value: Use, to stack: Use, instruction: Instruction) throws {
        guard .stack == stack.type, let definition = stack.definition else {
            throw VerificationError.notStack(stack, instruction)
        }
        stacks[ObjectIdentifier(definition)].append(value.type)
    }

    func pop(_ type: Type, to stack: Use, instruction: Instruction) throws {
        guard let definition = stack.definition else {
            throw VerificationError.notStack(stack, instruction)
        }
        stacks[ObjectIdentifier(definition)].dropLast()
    }
}
 */

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

        case let .numericUnary(_, v1), let .transpose(v1):
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

        case let .numericBinary(_, lhs, rhs):
            guard case let .tensor(s1, dt1) = lhs.type.unaliased else {
                throw VerificationError.notTensor(lhs, instruction)
            }
            guard case let .tensor(s2, dt2) = rhs.type.unaliased else {
                throw VerificationError.notTensor(rhs, instruction)
            }
            guard dt1 == dt2 else {
                throw VerificationError.dataTypeMismatch(lhs, rhs, instruction)
            }
            guard dt1.isNumeric else {
                throw VerificationError.dataTypeNotNumeric(lhs, instruction)
            }
            guard dt2.isNumeric else {
                throw VerificationError.dataTypeNotNumeric(rhs, instruction)
            }
            guard s1.isCompatible(with: s2) else {
                throw VerificationError.unbroadcastableMismatch([lhs, rhs], instruction)
            }
            
        case let .booleanBinary(_, lhs, rhs):
            guard case let .tensor(s1, dt1) = lhs.type.unaliased else {
                throw VerificationError.notTensor(lhs, instruction)
            }
            guard case let .tensor(s2, dt2) = rhs.type.unaliased else {
                throw VerificationError.notTensor(rhs, instruction)
            }
            guard dt1 == dt2 else {
                throw VerificationError.dataTypeMismatch(lhs, rhs, instruction)
            }
            guard dt1.isBool else {
                throw VerificationError.dataTypeNotBoolean(lhs, instruction)
            }
            guard dt2.isBool else {
                throw VerificationError.dataTypeNotBoolean(rhs, instruction)
            }
            guard s1.isCompatible(with: s2) else {
                throw VerificationError.unbroadcastableMismatch([lhs, rhs], instruction)
            }
            
        case let .compare(_, lhs, rhs):
            guard case let .tensor(s1, dt1) = lhs.type.unaliased else {
                throw VerificationError.notTensor(lhs, instruction)
            }
            guard case let .tensor(s2, dt2) = rhs.type.unaliased else {
                throw VerificationError.notTensor(rhs, instruction)
            }
            guard dt1 == dt2 else {
                throw VerificationError.dataTypeMismatch(lhs, rhs, instruction)
            }
            guard s1.isCompatible(with: s2) else {
                throw VerificationError.unbroadcastableMismatch([lhs, rhs], instruction)
            }
            
        case let .not(v):
            guard case let .tensor(_, dt1) = v.type.unaliased else {
                throw VerificationError.notTensor(v, instruction)
            }
            guard dt1.isBool else {
                throw VerificationError.dataTypeNotNumeric(v, instruction)
            }

        case let .dot(lhs, rhs):
            guard case let .tensor(s1, t1) = lhs.type.unaliased,
                case let .tensor(s2, t2) = rhs.type.unaliased,
                /// Either matrix multiplication or vector dot
                s1.isMatrixMultiplicable(by: s2) || (s1.isVector && s1 == s2),
                t1 == t2
                else { throw VerificationError.cannotDot(lhs, rhs, instruction) }

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

        case let .scan(.numeric(_), v1, dims):
            guard case let .tensor(s1, t1) = v1.type.unaliased, t1.isNumeric else {
                throw VerificationError.dataTypeNotNumeric(v1, instruction)
            }
            let shape = s1
            guard dims.count <= shape.rank, dims.forAll({$0 < shape.rank}), !dims.containsDuplicate else {
                throw VerificationError.invalidReductionDimensions(dims, v1, instruction)
            }
            
        case let .scan(.boolean(_), v1, dims):
            guard case let .tensor(s1, .bool) = v1.type.unaliased else {
                throw VerificationError.unexpectedDataType(v1, .bool, instruction)
            }
            let shape = s1
            guard dims.count <= shape.rank, dims.forAll({$0 < shape.rank}), !dims.containsDuplicate else {
                throw VerificationError.invalidReductionDimensions(dims, v1, instruction)
            }
            
        case let .reduce(.numeric(_), v1, initial, dims):
            guard case let .tensor(s1, t1) = v1.type.unaliased, t1.isNumeric else {
                throw VerificationError.dataTypeNotNumeric(v1, instruction)
            }
            let shape = s1
            let dtype = t1
            guard dims.count <= shape.rank, dims.forAll({$0 < shape.rank}), !dims.containsDuplicate else {
                throw VerificationError.invalidReductionDimensions(dims, v1, instruction)
            }
            /// Initial must be a scalar
            guard case .tensor([], dtype) = initial.type.canonical else {
                throw VerificationError.unexpectedShape(initial, .scalar, instruction)
            }
            
        case let .reduce(.boolean(_), v1, initial, dims):
            guard case let .tensor(s1, .bool) = v1.type.unaliased else {
                throw VerificationError.unexpectedDataType(v1, .bool, instruction)
            }
            let shape = s1
            let dtype: DataType = .bool
            guard dims.count <= shape.rank, dims.forAll({$0 < shape.rank}), !dims.containsDuplicate else {
                throw VerificationError.invalidReductionDimensions(dims, v1, instruction)
            }
            /// Initial must be a scalar
            guard case .tensor([], dtype) = initial.type.canonical else {
                throw VerificationError.unexpectedShape(initial, .scalar, instruction)
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

        case let .reduceWindow(.numeric(_), v1, initial: initial, dimensions: dims,
                               strides: strides, padding: _):
            guard case let .tensor(s1, t1) = v1.type.unaliased, t1.isNumeric else {
                throw VerificationError.dataTypeNotNumeric(v1, instruction)
            }
            /// Window must have same rank as operand, window dims must be positive
            guard dims.count <= s1.rank, dims.forAll({ $0 > 0 }) else {
                throw VerificationError.invalidReductionDimensions(dims, v1, instruction)
            }
            /// Strides must be greater than one
            guard strides.forAll({ $0 >= 1 }) else {
                throw VerificationError.windowInvalidStrides(strides, instruction)
            }
            /// Initial must be a scalar
            guard case .tensor([], t1) = initial.type.canonical else {
                throw VerificationError.unexpectedShape(initial, .scalar, instruction)
            }

        case let .reduceWindow(.boolean(_), v1, initial: initial, dimensions: dims,
                               strides: strides, padding: _):
            guard case let .tensor(s1, .bool) = v1.type.unaliased else {
                throw VerificationError.unexpectedDataType(v1, .bool, instruction)
            }
            /// Window must have same rank as operand, window dims must be positive
            guard dims.count <= s1.rank, dims.forAll({ $0 > 0 }) else {
                throw VerificationError.invalidReductionDimensions(dims, v1, instruction)
            }
            /// Strides must be greater than one
            guard strides.forAll({ $0 >= 1 }) else {
                throw VerificationError.windowInvalidStrides(strides, instruction)
            }
            /// Initial must be a scalar
            guard case .tensor([], .bool) = initial.type.canonical else {
                throw VerificationError.unexpectedShape(initial, .scalar, instruction)
            }

        case let .reduceWindow(.function(f), v1, initial: initial, dimensions: dims,
                               strides: strides, padding: _):
            guard case let .tensor(s1, t1) = v1.type.unaliased, t1.isNumeric else {
                throw VerificationError.dataTypeNotNumeric(v1, instruction)
            }
            /// Window must have same rank as operand, window dims must be positive
            guard dims.count <= s1.rank, dims.forAll({ $0 > 0 }) else {
                throw VerificationError.invalidReductionDimensions(dims, v1, instruction)
            }
            /// Check expected function type
            let expectedFuncType: Type = .function([.tensor([], t1)], .tensor([], t1))
            guard expectedFuncType == f.type.unaliased else {
                throw VerificationError.unexpectedType(f, expectedFuncType, instruction)
            }
            /// Strides must be greater than one
            guard strides.forAll({ $0 >= 1 }) else {
                throw VerificationError.windowInvalidStrides(strides, instruction)
            }
            /// Initial must be a scalar
            guard case .tensor([], t1) = initial.type.canonical else {
                throw VerificationError.unexpectedShape(initial, .scalar, instruction)
            }

        case let .convolve(lhs, kernel: rhs, strides: strides, padding: padding,
                           leftDilation: ld, rightDilation: rd):
            guard case let .tensor(s1, t1) = lhs.type.unaliased else {
                throw VerificationError.notTensor(lhs, instruction)
            }
            guard case let .tensor(s2, t2) = rhs.type.unaliased else {
                throw VerificationError.notTensor(rhs, instruction)
            }
            /// Rank and datatypes must match
            guard s1.rank == s2.rank, t1 == t2 else {
                throw VerificationError.convolveTypeMismatch(lhs, rhs, instruction)
            }
            /// Rank must be at least 3
            guard s1.rank >= 3 else {
                throw VerificationError.convolveInvalidShape(lhs, instruction)
            }
            /// Input channel dimensions must match
            guard s1[1] == s2[1] else {
                throw VerificationError.convolveInputChannelMismatch(lhs, rhs, instruction)
            }
            /// Set argument defaults
            let n = s1.rank - 2
            let strides = strides ?? Array(repeating: 1, count: n)
            let padding = padding ?? Array(repeating: (low: 0, high: 0), count: n)
            let ld = ld ?? Array(repeating: 1, count: n)
            let rd = rd ?? Array(repeating: 1, count: n)
            /// Strides/padding/dilation factors must have rank equal to n
            guard strides.count == n else {
                throw VerificationError.windowInvalidStridesRank(strides, n, instruction)
            }
            guard padding.count == n else {
                throw VerificationError.windowInvalidPaddingRank(padding, n, instruction)
            }
            guard ld.count == n else {
                throw VerificationError.convolveInvalidDilationRank(ld, n, instruction)
            }
            guard rd.count == n else {
                throw VerificationError.convolveInvalidDilationRank(rd, n, instruction)
            }
            /// Strides must be greater than one
            guard strides.forAll({ $0 >= 1 }) else {
                throw VerificationError.windowInvalidStrides(strides, instruction)
            }
            /// Padding must be non-negative
            guard padding.forAll({ $0.low >= 0 && $0.high >= 0 }) else {
                throw VerificationError.windowInvalidPadding(padding, instruction)
            }
            /// Dilation factors must be positive
            guard ld.forAll({ $0 > 0 }) else {
                throw VerificationError.convolveInvalidDilation(ld, instruction)
            }
            guard rd.forAll({ $0 > 0 }) else {
                throw VerificationError.convolveInvalidDilation(rd, instruction)
            }

        case let .padShape(v1, at: index):
            guard case let .tensor(s1, _) = v1.type.unaliased else {
                throw VerificationError.notTensor(v1, instruction)
            }
            guard s1.indices.contains(index) || s1.endIndex == index else {
                throw VerificationError.invalidIndex(v1, index, instruction)
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

        case .createStack: break

        case let .destroyStack(stack):
            guard case .stack = stack.type else {
                throw VerificationError.notStack(stack, instruction)
            }

        case let .push(_, to: stack):
            guard case .stack = stack.type else {
                throw VerificationError.notStack(stack, instruction)
            }

        case let .pop(_, from: stack):
            guard case .stack = stack.type else {
                throw VerificationError.notStack(stack, instruction)
            }

        case let .random(_, from: lo, upTo: hi):
            /// Bounds must be scalar
            guard case let .tensor([], dt1) = lo.type.unaliased else {
                throw VerificationError.randomBoundNotScalar(lo, instruction)
            }
            guard case let .tensor([], dt2) = hi.type.unaliased else {
                throw VerificationError.randomBoundNotScalar(hi, instruction)
            }
            /// Same data type
            guard dt1 == dt2 else {
                throw VerificationError.dataTypeMismatch(lo, hi, instruction)
            }
            /// Numeric
            guard dt1.isNumeric else {
                throw VerificationError.dataTypeNotNumeric(lo, instruction)
            }
        
        case let .select(left, right, by: flags):
            guard case let .tensor(s1, dt1) = left.type.unaliased else {
                throw VerificationError.notTensor(left, instruction)
            }
            guard case let .tensor(s2, dt2) = right.type.unaliased else {
                throw VerificationError.notTensor(right, instruction)
            }
            guard case let .tensor(s3, dt3) = flags.type.unaliased else {
                throw VerificationError.notTensor(flags, instruction)
            }
            guard dt1 == dt2 else {
                throw VerificationError.dataTypeMismatch(left, right, instruction)
            }
            guard let _ = broadcast(s1, s2, s3) else {
                throw VerificationError.unbroadcastableMismatch([left, right, flags], instruction)
            }
            guard dt3.isBool else {
                throw VerificationError.unexpectedDataType(flags, .bool, instruction)
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
