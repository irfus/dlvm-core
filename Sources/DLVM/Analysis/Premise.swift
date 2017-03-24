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

    public struct Premise { let terminator: Instruction }

    public class PremiseVerifier : AnalysisPass<BasicBlock, Premise> {
        public override class func run(on body: BasicBlock) throws -> Premise {
            guard let last = body.last, last.kind.isTerminator else {
                throw VerificationError.missingTerminator(body)
            }
            return Premise(terminator: last)
        }
    }

}

extension Function : PremiseHolder {

    public struct Premise { let exits: [(BasicBlock, Instruction)] }

    public class PremiseVerifier : AnalysisPass<Function, Premise> {
        public override class func run(on body: Function) throws -> Premise {
            var exits: [(BasicBlock, Instruction)] = []
            for bb in body {
                let terminator = try bb.premise().terminator
                if case .return = terminator.kind {
                    exits.append((bb, terminator))
                }
            }
            return Premise(exits: exits)
        }
    }
    
}
