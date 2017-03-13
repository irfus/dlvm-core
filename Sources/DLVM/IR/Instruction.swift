//
//  Instruction.swift
//  DLVM
//
//  Created by Richard Wei on 12/25/16.
//
//

public final class Instruction : IRSubUnit {
    public enum Kind {
        case control(Control)
        case operation(Def<Operation>)
    }
    public typealias Parent = BasicBlock
    public let kind: Kind
    public unowned var parent: BasicBlock
    public internal(set) var analysisManager: AnalysisManager<Instruction> = AnalysisManager()
    public internal(set) var transformManager: TransformManager<Instruction> = TransformManager()

    public required init(kind: Kind, parent: BasicBlock) {
        self.kind = kind
        self.parent = parent
    }

    public static func control(_ control: Control, parent: BasicBlock) -> Instruction {
        return self.init(kind: .control(control), parent: parent)
    }

    public static func operation(_ operation: Def<Operation>, parent: BasicBlock) -> Instruction {
        return self.init(kind: .operation(operation), parent: parent)
    }
}

public enum Control {
    /// Store use to global value
    case store(Use, to: Def<GlobalValue>)
    /// Yield value to output
    case yield(Use, to: Def<Output>)
    /// Unconditionally branch to basic block
    case branch(BasicBlock, [Use])
    /// Conditional branch depending on the value
    case conditional(Use, BasicBlock, BasicBlock)
    /// Return
    case `return`(Use?)
    /// Pull a value from the recurrent input batch in the current epoch
    /// and pass the value as an argument to the first basic block argument
    case pull(Def<Placeholder>, BasicBlock, BasicBlock)
}

public enum Operation {
    /// Get the value of the non-recurrent input
    /// - Precondition: placeholder must **not** be recurrent
    case get(Def<Placeholder>)
    /// Scan operation with optional axis
    /// If axis is not given, scan is performed on contiguous elements
    case scan(AssociativeOp, Use, axis: Int?)
    /// Reduction operation with optional axis
    /// If axis is not given, reduction is performed on contiguous elements
    case reduce(AssociativeOp, Use, axis: Int?)
    /// Monomorphic unary operation
    case unary(UnaryOp, Use)
    /// Monomorphic binary operation
    case binary(BinaryOp, Use, Use)
    /// Matrix multiplication operation
    case matrixMultiply(Use, Use)
    /// Concatenation operation
    case concatenate([Use], axis: Int)
    /// Transpose
    case transpose(Use)
    /// Type cast operation
    case dataTypeCast(Use, DataType)
    /// Shape cast operation
    case shapeCast(Use, TensorShape)
    /// Subtensor addressing
    case subtensor(Use, TensorIndex)
    /// Element in the immediate dimension
    case tupleElement(Use, Int)
    /// Create tuple
    case tuple([Use])
    /// Function call
    case call(Function, [Use])
    /// Gradient call
    case gradient(Function, [Use])
}

// MARK: - Instruction properties
public extension Instruction {
    var definition: Definition? {
        guard case let .operation(def) = kind else {
            return nil
        }
        return def
    }
}

extension Instruction : MaybeNamed {
    public var name: String? {
        guard case let .operation(def) = kind else { return nil }
        return def.name
    }
}

// MARK: - Control flow predicates
public extension Instruction {
    var isOperation: Bool {
        if case .operation = kind { return true }
        return false
    }
    
    var isControl: Bool {
        if case .control = kind { return true }
        return false
    }
    
    var isTerminator: Bool {
        switch kind {
        case .control(let ctrl): return ctrl.isTerminator
        default: return false
        }
    }

    var isReturn: Bool {
        switch kind {
        case .control(let ctrl): return ctrl.isReturn
        default: return false
        }
    }

    var isYield: Bool {
        switch kind {
        case .control(let ctrl): return ctrl.isYield
        default: return false
        }
    }

    var isExit: Bool {
        switch kind {
        case .control(let ctrl): return ctrl.isExit
        default: return false
        }
    }
}

public extension Control {
    var isTerminator: Bool {
        switch self {
        case .branch, .conditional, .`return`, .pull:
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

    var isYield: Bool {
        switch self {
        case .yield: return true
        default: return false
        }
    }

    var isExit: Bool {
        return isReturn
    }
}

infix operator <>

extension TensorShape {
    public static func <> (lhs: TensorShape, rhs: TensorShape) -> TensorShape? {
        return lhs.mutuallyBroadcasted(with: rhs)
    }
}

extension Operation : Value {

    public var type: Type {
        switch self {
        case let .binary(.associative(.arithmetic), v1, v2):
            guard case let .tensor(s1, t1) = v1.type,
                  case let .tensor(s2, t2) = v2.type,
                let shape = s1 <> s2, t1 == t2
                else { return .invalid }
            return .tensor(shape, t1)

        case let .binary(.associative(.boolean), v1, v2),
             let .binary(.comparison, v1, v2):
            guard case let .tensor(s1, t1) = v1.type,
                  case let .tensor(s2, t2) = v2.type,
                  let shape = s1 <> s2, t1 == t2
                else { return .invalid }
            return .tensor(shape, .bool)
            
        case let .matrixMultiply(v1, v2):
            guard case let .tensor(s1, t1) = v1.type,
                  case let .tensor(s2, t2) = v2.type,
                  let newShape = s1.matrixMultiplied(with: s2),
                  t1 == t2 else { return .invalid }
            return .tensor(newShape, t1)

        case let .unary(_, v1):
            return v1.type.isTensor ? v1.type : .invalid

        case let .reduce(_, v1, nil):
            guard case let .tensor(s1, t1) = v1.type else { return .invalid }
            return .tensor(s1.dropFirst(), t1)

        case let .reduce(_, v1, axis: axis?):
            guard case let .tensor(s1, t1) = v1.type,
                  axis < s1.rank
                else { return .invalid }
            return .tensor(s1.droppingDimension(axis), t1)

        case let .scan(_, v1, _):
            return v1.type.isTensor ? v1.type : .invalid

        case let .concatenate(vv, axis):
            guard let first = vv.first,
                  case let .tensor(s1, t1) = first.type
                else { return .invalid }
            /// Check simple, data type equality, and concatenability
            var accShape: TensorShape = s1
            for v in vv.dropFirst() {
                guard case let .tensor(shape, type) = v.type, type == t1,
                      let newShape = accShape.concatenating(with: shape,
                                                            alongDimension: axis)
                    else { return .invalid }
                accShape = newShape
            }
            return .tensor(accShape, t1)

        case let .transpose(v1):
            guard case let .tensor(s1, t1) = v1.type else { return .invalid }
            return .tensor(s1.transpose, t1)
            
        case let .dataTypeCast(v1, dt):
            guard case let .tensor(s1, t1) = v1.type,
                  t1.canCast(to: dt)
                else { return .invalid }
            return .tensor(s1, dt)
            
        case let .shapeCast(v1, s):
            guard case let .tensor(s1, t1) = v1.type,
                  s1.contiguousSize == s.contiguousSize
                else { return .invalid }
            return .tensor(s, t1)
            
        case let .get(ph):
            return ph.type

        case let .call(f, vv):
            guard f.acceptsArguments(vv.map{$0.type})
                else { return .invalid }
            return f.result

        case let .gradient(f, vv):
            guard f.isDifferentiable, f.acceptsArguments(vv.map{$0.type})
                else { return .invalid }
            if f.arguments.isEmpty { return .void }
            if f.arguments.count == 1 { return f.arguments[0].type }
            return .tuple(vv.map{$0.type})

        case let .tuple(vv):
            return .tuple(vv.map{$0.type})
            
        case let .tupleElement(v, i):
            guard case let .tuple(subtypes) = v.type,
                  subtypes.indices.contains(i)
                else { return .invalid }
            return subtypes[i]

        case let .subtensor(v, index):
            guard case let .tensor(s1, t1) = v.type,
                  index.count < s1.rank
                else { return .invalid }
            return .tensor(s1.dropFirst(index.count), t1)
            
        }
    }

    public static var scope: Scope = .local

}

extension Control : User {
    public var operands: [Use] {
        switch self {
        case .conditional(let op, _, _),
             .yield(let op, _),
             .store(let op, _),
             .`return`(let op?):
            return [op]
        default:
            return []
        }
    }
}

extension Operation : User {
    public var operands: [Use] {
        switch self {
        case let .binary(_, op1, op2),
             let .matrixMultiply(op1, op2):
            return [op1, op2]
        case .concatenate(let uses, _):
            return uses
        case let .transpose(op):
            return [op]
        case let .unary(_, op),
             let .reduce(_, op, _),
             let .scan(_, op, _),
             let .shapeCast(op, _),
             let .dataTypeCast(op, _):
            return [op]
        case .call(_, let ops), .gradient(_, let ops), .tuple(let ops):
            return ops
        case .get:
            return []
        case let .subtensor(op, _),
             let .tupleElement(op, _):
            return [op]
        }
    }
}

extension Instruction : User {
    public var operands: [Use] {
        switch kind {
        case .control(let ctrl): return ctrl.operands
        case .operation(let oper): return oper.operands
        }
    }
}

public extension Def where ValueType : User {
    public var operands: [Use] {
        return value.operands
    }
}

public extension Instruction {
    func substituting(_ actualUse: Use, for use: Use) -> Instruction {
        switch kind {
        case let .control(ctrl):
            return .control(ctrl.substituting(actualUse, for: use), parent: parent)
        case let .operation(def):
            let oper = def.value.substituting(actualUse, for: use)
            let newDef = Def<Operation>(name: def.name, value: oper)
            return .operation(newDef, parent: parent)
        }
    }

    var indexInParent: Int? {
        return parent.index(of: self)
    }

    func removeFromParent() {
        parent.remove(self)
    }
}

public extension Control {
    func substituting(_ newUse: Use, for use: Use) -> Control {
        switch self {
        case .store(use, to: let dest):
            return .store(newUse, to: dest)
        case .conditional(use, let thenBB, let elseBB):
            return .conditional(newUse, thenBB, elseBB)
        case .yield(use, to: let dest):
            return .yield(newUse, to: dest)
        case .`return`(use?):
            return .return(newUse)
        default:
            return self
        }
    }
}

public extension Operation {
    func substituting(_ newUse: Use, for old: Use) -> Operation {
        let condSubst = {$0 == old ? newUse : $0}
        switch self {
        case .unary(let fun, old):
            return .unary(fun, newUse)
        case .binary(let fun, old, let use2):
            return .binary(fun, newUse, use2)
        case .binary(let fun, let use1, old):
            return .binary(fun, use1, newUse)
        case .binary(let fun, old, old):
            return .binary(fun, newUse, newUse)
        case let .concatenate(uses, axis: axis):
            return .concatenate(uses.map(condSubst), axis: axis)
        case .transpose(old):
            return .transpose(newUse)
        case .reduce(let fun, old, axis: let axis):
            return .reduce(fun, newUse, axis: axis)
        case .matrixMultiply(old, let use2):
            return .matrixMultiply(newUse, use2)
        case .matrixMultiply(let use1, old):
            return .matrixMultiply(use1, newUse)
        case .matrixMultiply(old, old):
            return .matrixMultiply(newUse, newUse)
        case .shapeCast(old, let shape):
            return .shapeCast(newUse, shape)
        case .dataTypeCast(old, let type):
            return .dataTypeCast(newUse, type)
        case let .call(f, uses):
            return .call(f, uses.map(condSubst))
        case let .gradient(f, uses):
            return .gradient(f, uses.map(condSubst))
        case let .tuple(uses):
            return .tuple(uses.map(condSubst))
        case .tupleElement(old, let i):
            return .tupleElement(newUse, i)
        case .subtensor(old, let index):
            return .subtensor(newUse, index)
        default:
            return self
        }
    }
}

public extension Operation {
    var usedArguments: ObjectSet<Def<Argument>> {
        var arguments: ObjectSet<Def<Argument>> = []
        for case let .argument(arg) in operands.map({$0.kind}) {
            arguments.insert(arg)
        }
        return arguments
    }
}

public extension Instruction {
    var usedArguments: ObjectSet<Def<Argument>> {
        var arguments: ObjectSet<Def<Argument>> = []
        for case let .argument(arg) in operands.map({$0.kind}) {
            arguments.insert(arg)
        }
        return arguments
    }
}

public extension BasicBlock {
    var usedArguments: ObjectSet<Def<Argument>> {
        var arguments: ObjectSet<Def<Argument>> = []
        for inst in self {
            arguments.formUnion(inst.usedArguments)
        }
        return arguments
    }
}
