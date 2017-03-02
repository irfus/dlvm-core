//
//  CostFunctionGenerator.swift
//  DLVM
//
//  Created by Richard Wei on 2/21/17.
//
//

/*

public struct CostFunctionGenerator : GenerationPass {

    public typealias Body = (Use, BasicBlock)
    public typealias Result = Use

    public enum CostFunction {
        case meanSquaredError
        case custom((Use, BasicBlock) -> Result)
    }

    public static func run(on input: (Use, BasicBlock)) -> Result {
        let (output, bb) = input
        let builder = IRBuilder(basicBlock: bb)

        let diff = builder.buildOperation(.binary(.associative(.arithmetic(.subtract)), referenceOutput, output),
                                          name: "difference")
        let squared = builder.buildOperation(.binary(.associative(.arithmetic(.multiply)), diff, diff),
                                             name: "squared")
        let mean = builder.buildOperation(.reduce(.arithmetic(.mean), squared, axis: nil))
        result = .changed(error: mean)
        return result
    }

}

 */
