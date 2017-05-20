//
//  Premise.swift
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

public protocol PremiseHolder : IRUnit {
    associatedtype Premise
    /// - TODO: Add constraints here (currently causing compiler crash)
    associatedtype PremiseVerifier : AnalysisPass
    func premise() throws -> Premise
}

public extension PremiseHolder
    where PremiseVerifier.Result == Premise, PremiseVerifier.Body == Self {
    func premise() throws -> Premise {
        return try analysis(from: PremiseVerifier.self)
    }
}

extension BasicBlock : PremiseHolder {

    public struct Premise { let terminator: Instruction }

    public enum PremiseVerifier : AnalysisPass {
        public typealias Body = BasicBlock

        public static func run(on body: BasicBlock) throws -> Premise {
            guard let last = body.last, last.kind.isTerminator else {
                throw VerificationError.missingTerminator(body)
            }
            return Premise(terminator: last)
        }
    }

}

extension Function : PremiseHolder {

    public struct Premise { let exits: [(BasicBlock, Instruction)] }

    public enum PremiseVerifier : AnalysisPass {
        public typealias Body = Function

        public static func run(on body: Function) throws -> Premise {
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
