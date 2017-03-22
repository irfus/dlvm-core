//
//  HPVM.swift
//  DLVM
//
//  Created by Richard Wei on 3/21/17.
//
//

import Foundation
import LLVM_C

public struct HPVM {

    public enum ReplicationMode {
        case allToAll
        case oneToOne
    }

    public enum Intrinsic {
        case createNode(function: LLVMValueRef)
        case createNode1D(function: LLVMValueRef, n1: Int)
        case createNode2D(function: LLVMValueRef, n1: Int, n2: Int)
        case createNode3D(function: LLVMValueRef, n1: Int, n2: Int, n3: Int)
        case createEdge(from: LLVMValueRef, to: LLVMValueRef,
                        output: LLVMValueRef, input: LLVMValueRef,
                        replication: ReplicationMode, streaming: Bool)
        case bindInput(node: LLVMValueRef,
                       parentInput: LLVMValueRef,
                       input: LLVMValueRef)
        case bindOutput(node: LLVMValueRef,
                        output: LLVMValueRef,
                        parentOutput: LLVMValueRef,
                        streaming: Bool)
        case getCurrentNode
        case getParentNode(LLVMValueRef)
        case getDimensionCount(LLVMValueRef)
        case getNodeInstanceIdX(LLVMValueRef)
        case getNodeInstanceIdY(LLVMValueRef)
        case getNodeInstanceIdZ(LLVMValueRef)
        case getNodeInstanceCountX(LLVMValueRef)
        case getNodeInstanceCountY(LLVMValueRef)
        case getNodeInstanceCountZ(LLVMValueRef)
        case getVectorLength(typeSize: LLVMValueRef)
        case malloc(count: LLVMValueRef)
        case barrier
        case launch(function: LLVMValueRef,
                    arguments: [LLVMValueRef],
                    streaming: Bool)
        case wait(id: LLVMValueRef)
        case push(id: LLVMValueRef, arguments: [LLVMValueRef])
        case pop(id: LLVMValueRef)
    }

    public let module: LLVMModuleRef
    fileprivate var functions: [Intrinsic : LLVMValueRef] = [:]
    
}

extension HPVM.Intrinsic : Equatable, Hashable {

    public static func == (lhs: HPVM.Intrinsic, rhs: HPVM.Intrinsic) -> Bool {
        switch (lhs, rhs) {
        case (.createNode, .createNode),
             (.createNode1D, .createNode1D),
             (.createNode2D, .createNode2D),
             (.createNode3D, .createNode3D),
             (.createEdge, .createEdge),
             (.bindInput, .bindInput),
             (.bindOutput, .bindOutput),
             (.getCurrentNode, .getCurrentNode),
             (.getParentNode, .getParentNode),
             (.getDimensionCount, .getDimensionCount),
             (.getNodeInstanceIdX, .getNodeInstanceIdX),
             (.getNodeInstanceIdY, .getNodeInstanceIdY),
             (.getNodeInstanceIdZ, .getNodeInstanceIdZ),
             (.getNodeInstanceCountX, .getNodeInstanceCountX),
             (.getNodeInstanceCountY, .getNodeInstanceCountY),
             (.getNodeInstanceCountZ, .getNodeInstanceCountZ),
             (.getVectorLength, .getVectorLength),
             (.malloc, .malloc),
             (.barrier, .barrier),
             (.launch, .launch),
             (.wait, .wait),
             (.push, .push),
             (.pop, .pop):
            return true
        default:
            return false
        }
    }

    public var hashValue: Int {
        switch self {
        case .createNode: return 1
        case .createNode1D: return 2
        case .createNode2D: return 3
        case .createNode3D: return 4
        case .createEdge: return 5
        case .bindInput: return 6
        case .bindOutput: return 7
        case .getCurrentNode: return 8
        case .getParentNode: return 9
        case .getDimensionCount: return 10
        case .getNodeInstanceIdX: return 11
        case .getNodeInstanceIdY: return 12
        case .getNodeInstanceIdZ: return 13
        case .getNodeInstanceCountX: return 14
        case .getNodeInstanceCountY: return 15
        case .getNodeInstanceCountZ: return 16
        case .getVectorLength: return 17
        case .malloc: return 18
        case .barrier: return 19
        case .launch: return 20
        case .wait: return 21
        case .push: return 22
        case .pop: return 23
        }
    }
    
}
