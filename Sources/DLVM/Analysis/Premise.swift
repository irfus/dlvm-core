//
//  Premise.swift
//  DLVM
//
//  Created by Richard Wei on 3/3/17.
//
//

public protocol PremiseHolder {
    associatedtype Premise
    associatedtype PremiseVerifier // : Pass
    func premise() throws -> Premise
}

public extension PremiseHolder where Self : IRUnit, PremiseVerifier: AnalysisPass<Self, Premise> {
    func premise() throws -> Premise {
        return try analysis(from: PremiseVerifier.self)
    }
}

extension BasicBlock : PremiseHolder {

    public struct Premise { let terminator: Control, terminatorInstruction: Instruction }

    public class PremiseVerifier : AnalysisPass<BasicBlock, Premise> {
        public override class func run(on body: BasicBlock) throws -> Premise {
            guard let last = body.last, case .control(let ctrl) = last.kind else {
                throw VerificationError.missingTerminator(body)
            }
            return Premise(terminator: ctrl, terminatorInstruction: last)
        }
    }

}

extension Function : PremiseHolder {

    public struct Premise { let exits: [BasicBlock] }

    public class PremiseVerifier : AnalysisPass<Function, Premise> {
        public override class func run(on body: Function) throws -> Premise {
            var exits: [BasicBlock] = []
            for bb in body {
                let terminator = try bb.premise().terminator
                if terminator.isExit { exits.append(bb) }
            }
            return Premise(exits: exits)
        }
    }
    
}
