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

    public struct Premise { let terminator: Control }

    public class PremiseVerifier : AnalysisPass<BasicBlock, Premise> {
        public override class func run(on body: BasicBlock) throws -> Premise {
            guard case .control(let ctrl)? = body.last?.kind else {
                throw VerificationError.missingTerminator(body)
            }
            return Premise(terminator: ctrl)
        }
    }

}

extension Section : PremiseHolder {

    public struct Premise {
        let exit: BasicBlock
    }

    public class PremiseVerifier : AnalysisPass<Section, Premise> {
        public override class func run(on body: Section) throws -> Premise {
            var exits: [BasicBlock] = []
            for bb in body {
                let terminator = try bb.premise().terminator
                if terminator.isExit { exits.append(bb) }
            }
            guard exits.count <= 1 else {
                throw VerificationError.multipleExits(exits, body)
            }
            guard let exit = exits.first else {
                throw VerificationError.noExit(body)
            }
            return Premise(exit: exit)
        }
    }
    
}
