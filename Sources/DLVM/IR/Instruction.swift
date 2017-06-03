//
//  Instruction.swift
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

public enum InstructionKind {
    /** Control flow **/
    /// Unconditionally branch to basic block
    case branch(BasicBlock, [Use])
    /// Conditional branch depending on the value
    case conditional(Use, BasicBlock, [Use], BasicBlock, [Use])
    /// Return
    case `return`(Use?)

    /** Tensor operations **/
    /// Monomorphic unary operation (map)
    case map(UnaryOp, Use)
    /// Monomorphic binary operation (zipWith)
    case zipWith(BinaryOp, Use, Use, BroadcastingConfig?)
    /// Data type cast operation
    case dataTypeCast(Use, DataType)
    /// Scan operation
    case scan(ReductionCombinator, Use, [Int])
    /// Reduction operation
    case reduce(ReductionCombinator, Use, [Int])
    /// Matrix multiplication operation
    case matrixMultiply(Use, Use)
    /// Concatenation operation
    case concatenate([Use], axis: Int)
    /// Transpose
    case transpose(Use)

    /** Cost-free casts **/
    /// Shape cast operation
    case shapeCast(Use, TensorShape)
    /// Bitcast
    case bitCast(Use, Type)

    /** Aggregate operation **/
    /// Extract an element from tensor, tuple, or array
    case extract(from: Use, at: [ElementKey])
    /// Insert an element to tensor, tuple, or array
    case insert(Use, to: Use, at: [ElementKey])

    /** Function invocation **/
    /// Function application
    case apply(Use, [Use])

    /** Heap memory of host and device **/
    /** Memory **/
    /// Allocate host stack memory, returning a pointer
    case allocateStack(Type, Int) /// => *T
    case allocateHeap(Type, count: Use) /// => *T
    /// Reference-counted box
    case allocateBox(Type) /// => box{T}
    case projectBox(Use) /// (box{T}) => *T
    /// Retain/release a box via reference counter
    case retain(Use)
    case release(Use)
    /// Dealloc any heap memory
    case deallocate(Use)
    /// Load value from pointer on the host
    case load(Use)
    /// Store value to pointer on the host
    case store(Use, to: Use)
    /// GEP (without leading index)
    case elementPointer(Use, [ElementKey])
    /// Memory copy
    case copy(from: Use, to: Use, count: Use)
    /// Trap
    case trap
}

public final class Instruction : IRUnit, MaybeNamed {
    public typealias Parent = BasicBlock
    public var name: String?
    public var kind: InstructionKind
    public unowned var parent: BasicBlock

    public init(name: String? = nil, kind: InstructionKind, parent: BasicBlock) {
        self.name = name
        self.kind = kind
        self.parent = parent
    }
}

extension Instruction : Value {
    public var type: Type {
        return kind.type
    }

    public func makeUse() -> Use {
        return .instruction(type, self)
    }
}

extension InstructionKind : Value {
    public func makeUse() -> Use {
        return .constant(type, self)
    }
}

// MARK: - Predicates
public extension InstructionKind {
    var isTerminator: Bool {
        switch self {
        case .branch, .conditional, .`return`:
            return true
        default:
            return false
        }
    }

    var isReturn: Bool {
        switch self {
        case .`return`: return true
        default: return false
        }
    }

    var isTrap: Bool {
        switch self {
        case .trap: return true
        default: return false
        }
    }

    var accessesMemory: Bool {
        switch self {
        case .allocateStack, .allocateHeap, .allocateBox,
             .projectBox, .load, .store, .deallocate:
            return true
        default:
            return false
        }
    }

    var mustWriteToMemory: Bool {
        switch self {
        case .store, .copy: return true
        default: return false
        }
    }
}

public extension BroadcastingConfig {
    func canBroadcast(_ lhs: TensorShape, _ rhs: TensorShape) -> Bool {
        switch direction {
        case .right where lhs.isBroadcastable(to: rhs, at: indices),
             .left where rhs.isBroadcastable(to: lhs, at: indices):
            return true
        default:
            return false
        }
    }
}

prefix operator <=
postfix operator =>

public prefix func <= (indices: [Int]) -> BroadcastingConfig {
    return BroadcastingConfig(indices: indices, direction: .right)
}

public postfix func => (indices: [Int]) -> BroadcastingConfig {
    return BroadcastingConfig(indices: indices, direction: .right)
}

public extension InstructionKind {

    var type: Type {
        switch self {
        case let .zipWith(.associative(assoc), v1, v2, nil):
            guard case let .tensor(s1, t1) = v1.type.unaliased,
                  case let .tensor(s2, t2) = v2.type.unaliased,
                  t1 == t2, s1 == s2 else {
                return .invalid
            }
            return .tensor(s1, assoc.isBoolean ? .bool : t1)

        case let .zipWith(.associative(assoc), v1, v2, bc?):
            guard case let .tensor(s1, t1) = v1.type.unaliased,
                  case let .tensor(s2, t2) = v2.type.unaliased,
                  t1 == t2, bc.canBroadcast(s1, s2) else {
                return .invalid
            }
            return .tensor(s1, assoc.isBoolean ? .bool : t1)

        case let .zipWith(.comparison(_), v1, v2, nil):
            guard case let .tensor(s1, t1) = v1.type.unaliased,
                  case let .tensor(s2, t2) = v2.type.unaliased,
                  t1 == t2 && s1 == s2 && t1.isNumeric else {
                return .invalid
            }
            return .tensor(s1, .bool)

        case let .zipWith(.comparison(_), v1, v2, bc?):
            guard case let .tensor(s1, t1) = v1.type.unaliased,
                  case let .tensor(s2, t2) = v2.type.unaliased,
                  t1.isNumeric, t1 == t2, bc.canBroadcast(s1, s2) else {
                return .invalid
            }
            return .tensor(s1, .bool)

        case let .matrixMultiply(v1, v2):
            guard case let .tensor(s1, t1) = v1.type.unaliased,
                  case let .tensor(s2, t2) = v2.type.unaliased,
                  let newShape = s1.matrixMultiplied(by: s2),
                  t1 == t2 else { return .invalid }
            return .tensor(newShape, t1)

        case let .map(_, v1):
            guard case .tensor(_, _) = v1.type.unaliased else { return .invalid }
            return v1.type

        case let .reduce(op, v1, dims):
            switch (op, v1.type.unaliased) {
            case let (.op(op), .tensor(s1, .bool))
                where op.isBoolean && dims.count <= s1.rank && dims.forAll({$0 < s1.rank}):
                return .tensor(dims.reduce(s1, { $0.droppingDimension($1) }), .bool)
            case let (.op(op), .tensor(s1, t1))
                where !op.isBoolean && t1.isNumeric && dims.count <= s1.rank && dims.forAll({$0 < s1.rank}):
                return .tensor(dims.reduce(s1, { $0.droppingDimension($1) }), t1)
            case let (.function(f), .tensor(s1, t1))
                where f.type.unaliased == .function([.tensor([], t1)], .tensor([], t1)):
                return .tensor(dims.reduce(s1, { $0.droppingDimension($1) }), t1)
            default:
                return .invalid
            }

        case let .scan(_, v1, _):
            guard case .tensor = v1.type.unaliased else { return .invalid }
            return v1.type

        case let .concatenate(vv, axis):
            guard let first = vv.first,
                  case let .tensor(s1, t1) = first.type.unaliased
                else { return .invalid }
            var accShape: TensorShape = s1
            for v in vv.dropFirst() {
                guard case let .tensor(shape, type) = v.type.unaliased,
                      type == t1,
                      let newShape = accShape.concatenating(with: shape, alongDimension: axis)
                    else { return .invalid }
                accShape = newShape
            }
            return .tensor(accShape, t1)

        case let .transpose(v1):
            guard case let .tensor(s1, t1) = v1.type.unaliased
                else { return .invalid }
            return .tensor(s1.transpose, t1)
            
        case let .dataTypeCast(v1, dt):
            guard case let .tensor(s1, t1) = v1.type.unaliased, t1.canCast(to: dt) else {
                return .invalid
            }
            return .tensor(s1, dt)

        case let .shapeCast(v1, s):
            switch v1.type.unaliased {
            case let .tensor(s1, t1) where s1.contiguousSize == s.contiguousSize:
                return .tensor(s, t1)
            case let .tensor(s1, t1)
                    where s1.contiguousSize == s.contiguousSize:
                return .tensor(s, t1)
            default: return .invalid
            }

        case let .apply(f, vv):
            switch f.type.unaliased {
            case let .pointer(.function(actual, ret)),
                 let .function(actual, ret):
                guard actual == vv.map({$0.type}) else { fallthrough }
                return ret
            default:
                return .invalid
            }

        case let .extract(from: v, at: indices):
            return v.type.subtype(at: indices) ?? .invalid

        case let .insert(src, to: dest, at: indices):
            guard let subtype = dest.type.subtype(at: indices), subtype == src.type else {
                return .invalid
            }
            return dest.type

        case let .allocateStack(type, n):
            guard n > 0 else { return .invalid }
            return .pointer(type)

        case let .load(v):
            guard case let .pointer(t) = v.type.unaliased else { return .invalid }
            return t

        case let .elementPointer(v, ii):
            guard case let .pointer(t) = v.type else { return .invalid }
            return t.subtype(at: ii).flatMap(Type.pointer) ?? .invalid

        case let .bitCast(_, t):
//            guard v.type.size == t.size else { return .invalid }
            return t

        case let .allocateBox(t):
            return .box(t)

        case let .allocateHeap(t, count: _):
            return .pointer(t)

        case let .projectBox(v):
            guard case let .box(t) = v.type.unaliased else { return .invalid }
            return .pointer(t)

        case .store, .copy, .deallocate,
             .branch, .conditional, .return, .retain, .release, .trap:
            return .void
        }
    }

    static var scope: Scope = .local

}

extension Instruction : User {
    public var operands: [Use] {
        return kind.operands
    }
}

extension InstructionKind {
    public var operands: [Use] {
        switch self {
        case let .zipWith(_, op1, op2, _),
             let .matrixMultiply(op1, op2),
             let .insert(op1, to: op2, at: _):
            return [op1, op2]
        case let .map(_, op), let .reduce(_, op, _), let .scan(_, op, _), let .transpose(op),
             let .shapeCast(op, _), let .dataTypeCast(op, _), let .bitCast(op, _), let .return(op?),
             let .extract(from: op, at: _),
             let .store(op, _), let .load(op), let .elementPointer(op, _),
             let .deallocate(op), let .allocateHeap(_, count: op),
             let .projectBox(op), let .release(op), let .retain(op):
            return [op]
        case .concatenate(let ops, _),
             .branch(_, let ops):
            return ops
        case let .conditional(cond, _, thenArgs, _, elseArgs):
            return [cond] + thenArgs + elseArgs
        case let .apply(f, args):
            return [f] + args
        case let .copy(from: op1, to: op2, count: op3):
            return [op1, op2, op3]
        case .return(nil), .allocateBox, .trap, .allocateStack:
            return []
        }
    }
}

public extension Instruction {
    func substitute(_ newUse: Use, for use: Use) {
        kind = kind.substituting(newUse, for: use)
    }
}

public extension InstructionKind {
    /// Substitutes new use for old use
    /// - Note: The current implementation is a vanilla tedious switch 
    /// matching all the permutations (a.k.a. very bad).
    func substituting(_ new: Use, for old: Use) -> InstructionKind {
        let condSubst = {$0 == old ? new : $0}
        switch self {
        case .branch(let dest, let args):
            return .branch(dest, args.map(condSubst))
        case let .conditional(cond, thenBB, thenArgs, elseBB, elseArgs):
            let newCond = cond == old ? new : cond
            return .conditional(newCond,
                                thenBB, thenArgs.map(condSubst),
                                elseBB, elseArgs.map(condSubst))
        case .return(old?):
            return .return(new)
        case .map(let fun, old):
            return .map(fun, new)
        case .zipWith(let fun, old, old, let bc):
            return .zipWith(fun, new, new, bc)
        case .zipWith(let fun, old, let use2, let bc):
            return .zipWith(fun, new, use2, bc)
        case .zipWith(let fun, let use1, old, let bc):
            return .zipWith(fun, use1, new, bc)
        case let .concatenate(uses, axis: axis):
            return .concatenate(uses.map(condSubst), axis: axis)
        case .transpose(old):
            return .transpose(new)
        case .reduce(.function(old), old, let dims):
            return .reduce(.function(new), new, dims)
        case .reduce(.function(old), let v1, let dims):
            return .reduce(.function(new), v1, dims)
        case .reduce(.function(let v1), old, let dims):
            return .reduce(.function(v1), new, dims)
        case .reduce(let fun, old, let dims):
            return .reduce(fun, new, dims)
        case .matrixMultiply(old, let use2):
            return .matrixMultiply(new, use2)
        case .matrixMultiply(let use1, old):
            return .matrixMultiply(use1, new)
        case .matrixMultiply(old, old):
            return .matrixMultiply(new, new)
        case .shapeCast(old, let shape):
            return .shapeCast(new, shape)
        case .dataTypeCast(old, let type):
            return .dataTypeCast(new, type)
        case let .apply(f, uses):
            return .apply(f, uses.map(condSubst))
        case .extract(from: old, at: let i):
            return .extract(from: new, at: i)
        case .insert(old, to: old, at: let indices):
            return .insert(new, to: new, at: indices)
        case .insert(old, to: let use1, at: let indices):
            return .insert(new, to: use1, at: indices)
        case .insert(let use1, to: old, at: let indices):
            return .insert(use1, to: new, at: indices)
        case .bitCast(old, let targetT):
            return .bitCast(new, targetT)
        case .elementPointer(old, let indices):
            return .elementPointer(new, indices)
        case .store(old, to: let dest):
            return .store(new, to: dest)
        case .store(let val, to: old):
            return .store(val, to: new)
        case .load(old):
            return .load(new)
        case .allocateHeap(let ty, count: old):
            return .allocateHeap(ty, count: new)
        case .deallocate(old):
            return .deallocate(new)
        case .copy(from: old, to: old, count: old):
            return .copy(from: new, to: new, count: new)
        case .copy(from: old, to: old, count: let v3):
            return .copy(from: new, to: new, count: v3)
        case .copy(from: old, to: let v2, count: old):
            return .copy(from: new, to: v2, count: new)
        case .copy(from: old, to: let v2, count: let v3):
            return .copy(from: new, to: v2, count: v3)
        case .copy(from: let v1, to: old, count: old):
            return .copy(from: v1, to: new, count: new)
        case .copy(from: let v1, to: old, count: let v3):
            return .copy(from: v1, to: new, count: v3)
        case .copy(from: let v1, to: let v2, count: old):
            return .copy(from: v1, to: v2, count: new)
        default:
            return self
        }
    }
}

/// Instruction ADT decomposition (opcodes, keywords, operands)
/// - Note: When adding a new instruction, you should insert its
/// corresponding opcode here
public extension InstructionKind {
    enum Opcode {
        case branch
        case conditional
        case `return`
        case dataTypeCast
        case scan
        case reduce
        case matrixMultiply
        case concatenate
        case transpose
        case shapeCast
        case bitCast
        case extract
        case insert
        case apply
        case allocateStack
        case allocateHeap
        case allocateBox
        case projectBox
        case retain
        case release
        case deallocate
        case load
        case store
        case elementPointer
        case copy
        case trap
        case binaryOp(BinaryOp)
        case unaryOp(UnaryOp)
    }

    var opcode: Opcode {
        switch self {
        case .branch: return .branch
        case .conditional: return .conditional
        case .return: return .return
        case .map(let op, _): return .unaryOp(op)
        case .zipWith(let op, _, _, _): return .binaryOp(op)
        case .dataTypeCast: return .dataTypeCast
        case .scan: return .scan
        case .reduce: return .reduce
        case .matrixMultiply: return .matrixMultiply
        case .concatenate: return .concatenate
        case .transpose: return .transpose
        case .shapeCast: return .shapeCast
        case .bitCast: return .bitCast
        case .extract: return .extract
        case .insert: return .insert
        case .apply: return .apply
        case .allocateStack: return .allocateStack
        case .allocateHeap: return .allocateHeap
        case .allocateBox: return .allocateBox
        case .projectBox: return .projectBox
        case .retain: return .retain
        case .release: return .release
        case .deallocate: return .deallocate
        case .load: return .load
        case .store: return .store
        case .elementPointer: return .elementPointer
        case .copy: return .copy
        case .trap: return .trap
        }
    }
}

extension InstructionKind.Opcode : Equatable {
    public static func == (lhs: InstructionKind.Opcode, rhs: InstructionKind.Opcode) -> Bool {
        switch (lhs, rhs) {
        case (.branch, .branch): return true
        case (.conditional, .conditional): return true
        case (.return, .return): return true
        case (.dataTypeCast, .dataTypeCast): return true
        case (.scan, .scan): return true
        case (.reduce, .reduce): return true
        case (.matrixMultiply, .matrixMultiply): return true
        case (.concatenate, .concatenate): return true
        case (.transpose, .transpose): return true
        case (.shapeCast, .shapeCast): return true
        case (.bitCast, .bitCast): return true
        case (.extract, .extract): return true
        case (.insert, .insert): return true
        case (.apply, .apply): return true
        case (.allocateStack, .allocateStack): return true
        case (.allocateHeap, .allocateHeap): return true
        case (.allocateBox, .allocateBox): return true
        case (.projectBox, .projectBox): return true
        case (.retain, .retain): return true
        case (.release, .release): return true
        case (.deallocate, .deallocate): return true
        case (.load, .load): return true
        case (.store, .store): return true
        case (.elementPointer, .elementPointer): return true
        case (.copy, .copy): return true
        case (.trap, .trap): return true
        case let (.binaryOp(o1), .binaryOp(o2)): return o1 == o2
        case let (.unaryOp(o1), .unaryOp(o2)): return o1 == o2
        default: return false
        }
    }
}
