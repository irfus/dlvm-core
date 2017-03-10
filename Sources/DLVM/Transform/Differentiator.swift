//
//  Differentiator.swift
//  DLVM
//
//  Created by Richard Wei on 2/21/17.
//
//

public class Differentiator : TransformPass<Function> {

    open override class func run(on function: Function) throws -> Bool {
        var changed = false
        guard function.isDifferentiable else { return false }

        let entry = function.entry

        guard entry.isExit else {
            fatalError("Control flow not supported")
        }

        let builder = IRBuilder(module: function.parent)
//        builder.buildFunction(named: function.name + "_grad",
//                              arguments: <#T##[(String, Argument)]#>, result: <#T##Argument?#>, isDifferentiable: <#T##Bool#>)

        changed = true

        return changed
    }

    @discardableResult
    private static func differentiate(_ use: Use, using builder: IRBuilder, in function: Function) -> Use {
        
        switch use.kind {
            
        case .literal(let v):
            return builder.makeLiteral(v.makeScalarLiteral(0))
        case .global(let v):
            return builder.makeLiteral(v.makeScalarLiteral(0))
            
        case .argument(let argDef):
            let lit = builder.makeLiteral(argDef.makeScalarLiteral(1))
            return lit

        case let .local(def):
            let oper = def.value
            let result: Use
            switch oper {
            case .unary(.elementwise(.sigmoid), let arg):
                let sigmoidx = use
                let one = builder.makeLiteral(sigmoidx.value.makeScalarLiteral(1))
                let oneminus = builder.buildOperation(.binary(.associative(.arithmetic(.subtract)), one, sigmoidx))
                let dsig = builder.buildOperation(.binary(.associative(.arithmetic(.multiply)), oneminus, sigmoidx))
                let dg = differentiate(arg, using: builder, in: function)
                result = builder.buildOperation(.binary(.associative(.arithmetic(.multiply)), dsig, dg))

            case let .binary(.associative(.arithmetic(.multiply)), L, R):
                let dL = differentiate(L, using: builder, in: function)
                let dR = differentiate(R, using: builder, in: function)
                let dLxR = builder.buildOperation(.binary(.associative(.arithmetic(.multiply)), dL, R))
                let LxdR = builder.buildOperation(.binary(.associative(.arithmetic(.multiply)), L, dR))
                result = builder.buildOperation(.binary(.associative(.arithmetic(.add)), dLxR, LxdR))
                
            case let .binary(.associative(.arithmetic(.add)), L, R):
                let dL = differentiate(L, using: builder, in: function)
                let dR = differentiate(R, using: builder, in: function)
                result = builder.buildOperation(.binary(.associative(.arithmetic(.add)), dL, dR))

            case let .binary(.associative(.arithmetic(.subtract)), L, R):
                let dL = differentiate(L, using: builder, in: function)
                let dR = differentiate(R, using: builder, in: function)
                result = builder.buildOperation(.binary(.associative(.arithmetic(.subtract)), dL, dR))
                
            case let .binary(.associative(.arithmetic(.divide)), L, R):
                let dL = differentiate(L, using: builder, in: function)
                let dR = differentiate(R, using: builder, in: function)
                let dLxR = builder.buildOperation(.binary(.associative(.arithmetic(.multiply)), dL, R))
                let LxdR = builder.buildOperation(.binary(.associative(.arithmetic(.multiply)), L, dR))
                let ddiff = builder.buildOperation(.binary(.associative(.arithmetic(.divide)), dLxR, LxdR))
                let two = builder.makeLiteral(dR.value.makeScalarLiteral(2))
                let sqr = builder.buildOperation(.binary(.associative(.arithmetic(.power)), dR, two))
                result = builder.buildOperation(.binary(.associative(.arithmetic(.divide)), ddiff, sqr))
                
            case let .matMul(L, R):
                let dL = differentiate(L, using: builder, in: function)
                let dR = differentiate(R, using: builder, in: function)
                let dLxR = builder.buildOperation(.matMul(dL, R))
                let LxdR = builder.buildOperation(.matMul(L, dR))
                result = builder.buildOperation(.binary(.associative(.arithmetic(.add)), dLxR, LxdR))
                
            default:
                fatalError("Unhandled term \(def)")
                
            }

            return result
        }
    }
    
}
