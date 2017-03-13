//
// Created by Richard Wei on 3/12/17.
//

public struct GlobalGradientInfo {
    fileprivate var gradientMap: [Function : Function] = [:]
    fileprivate var antigradientMap: [Function : Function] = [:]
}

public extension GlobalGradientInfo {
    func gradient(of function: Function) -> Function? {
        return gradientMap[function]
    }

    func antigradient(of function: Function) -> Function? {
        return antigradientMap[function]
    }
}

public class GlobalGradientAnalysis: AnalysisPass<Module, GlobalGradientInfo> {

    public override class func run(on module: Module) -> GlobalGradientInfo {
        var ggi = GlobalGradientInfo()
        for grad in module where grad.name.characters.first == "∇" {
            let restName = String(grad.name.characters.dropFirst())
            if let antigrad: Function = module.element(named: restName) {
                ggi.gradientMap[antigrad] = grad
                ggi.antigradientMap[grad] = antigrad
            }
        }
        return ggi
    }

}
