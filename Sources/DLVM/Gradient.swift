//
//  Gradient.swift
//  DLVM
//
//  Created by Richard Wei on 12/20/16.
//
//

public protocol Differentiable {
    associatedtype Gradient
    var gradient: Gradient { get }
}

open class GradientPass : BasicBlockPass {

    open var basicBlock: BasicBlock
    open lazy var gradientBlock: BasicBlock = BasicBlock(name: self.basicBlock.name)
    open var changed: Bool = false

    public required init(basicBlock: BasicBlock) {
        self.basicBlock = basicBlock
    }

    open func run() {
        for inst in basicBlock.instructions {
//            switch inst {
//            case let inst as NegationInstruction:
//                
//            case let inst as TensorProductInstruction:
//                
//            case let inst as ArithmeticInstruction:
//                
//            case let inst as ComparisonInstruction:
//                
//            case let inst as ElementwiseCallInstruction:
//                
//            case let inst as AggregateCallInstruction:
//                
//            case let inst as StoreInstruction:
//                
//            case let inst as ConcatenationInstruction:
//
//            case let inst as ShapeCastInstruction:
//                
//            case let inst as TypeCastInstruction:
//                
//            }
        }
    }
    
}
