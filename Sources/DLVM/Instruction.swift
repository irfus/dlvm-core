//
//  Instruction.swift
//  DLVM
//
//  Created by Richard Wei on 12/25/16.
//
//

public class Instruction : IRObject {
    public enum ComparisonOperator {
        case lt, leq, gt, geq, eq
    }
    public enum Kind {
        case negate(Operand)
        case add(Operand, Operand)
        case mul(Operand, Operand)
        case min(Operand, Operand)
        case max(Operand, Operand)
        case compare(ComparisonOperator, Operand, Operand)
        case dotProduct(Tensor, Tensor)
        case random(Scalar, Scalar)
        case product(Tensor, Tensor)
        case activation(ActivationFunction, Tensor)
        case transfer(TransferFunction, Tensor)
        case concat([Tensor])
        case phi([Variable])
        case condBranch(Operand, then: BasicBlock, else: BasicBlock)
        case uncondBranch(BasicBlock)
        case output(Tensor)
    }
    public let kind: Kind
    public weak var parent: BasicBlock? = nil

    /// Initialize a standalone instruction by specifying its kind
    ///
    /// - Parameter kind: kind of instruction
    public init(kind: Kind) {
        self.kind = kind
    }
}

extension Instruction : VariableProducer {

    public func makeVariable(named name: String) -> Variable {
        switch kind {
        case let .negate(op as Tensor),
             let .add(op as Tensor, _), let .add(_, op as Tensor),
             let .mul(op as Tensor, _), let .mul(_, op as Tensor),
             let .min(op as Tensor, _), let .min(_, op as Tensor),
             let .max(op as Tensor, _), let .max(_, op as Tensor),
             let .dotProduct(op, _),
             let .activation(_, op),
             let .transfer(_, op):
            return Tensor(name: name, dataType: op.dataType,
                          shape: op.shape, definition: self)

        default: /// TODO
            return UnavailableVariable.shared
        }
    }

}

// MARK: - Hashable
extension Instruction : Hashable {

    /// Equatable by reference
    public static func == (lhs: Instruction, rhs: Instruction) -> Bool {
        return lhs === rhs
    }
    
    /// Hashable by object identifier
    public var hashValue: Int {
        return ObjectIdentifier(self).hashValue
    }
    
}
