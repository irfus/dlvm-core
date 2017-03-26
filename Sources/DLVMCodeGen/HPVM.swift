//
//  HPVM.swift
//  DLVM
//
//  Created by Richard Wei on 3/21/17.
//
//

import LLVM

/// HPVM Target
public class HPVM : LLTarget, LLFunctionPrototypeCacheable {

    public enum ReplicationMode : Int, LLConstantConvertible {
        case allToAll = 0
        case oneToOne = 1

        public var constantType: IntType {
            return i1
        }
    }

    public enum Intrinsic {
        case createNode(Function)
        case createNode1D(Function, IRValue)
        case createNode2D(Function, IRValue, IRValue)
        case createNode3D(Function, IRValue, IRValue, IRValue)
        case createEdge(from: IRValue, to: IRValue,
                        output: IRValue, input: IRValue,
                        replication: ReplicationMode, streaming: Bool)
        case bindInput(node: IRValue,
                       parentInput: IRValue,
                       input: IRValue)
        case bindOutput(node: IRValue,
                        output: IRValue,
                        parentOutput: IRValue,
                        streaming: Bool)
        case getCurrentNode
        case getParentNode(IRValue)
        case getDimensionCount(IRValue)
        case getNodeInstanceIdX(IRValue)
        case getNodeInstanceIdY(IRValue)
        case getNodeInstanceIdZ(IRValue)
        case getNodeInstanceCountX(IRValue)
        case getNodeInstanceCountY(IRValue)
        case getNodeInstanceCountZ(IRValue)
        case getVectorLength(IRValue)
        case malloc(IRValue)
        case barrier
        case launch(Function, arguments: IRValue, streaming: Bool)
        case wait(id: IRValue)
        case push(id: IRValue, arguments: IRValue)
        case pop(id: IRValue)
    }

    public enum RuntimeFunction {
        case trackMemory(address: IRValue, size: IRValue)
        case requestMemory(address: IRValue, size: IRValue)
        case untrackMemory(address: IRValue)
    }

    public unowned let module: LLVM.Module
    public var functions: [AnyHashable : Function] = [:]

    public required init(module: Module) {
        self.module = module
    }
    
}

extension HPVM.Intrinsic : LLFunctionPrototype {

    public var name: StaticString {
        switch self {
        case .createNode: return "llvm.hpvm.createNode"
        case .createNode1D: return "llvm.hpvm.createNode1D"
        case .createNode2D: return "llvm.hpvm.createNode2D"
        case .createNode3D: return "llvm.hpvm.createNode3D"
        case .createEdge: return "llvm.hpvm.createEdge"
        case .bindInput: return "llvm.hpvm.bind.input"
        case .bindOutput: return "llvm.hpvm.bind.output"
        case .getCurrentNode: return "llvm.hpvm.getNode"
        case .getParentNode: return "llvm.hpvm.getParentNode"
        case .getDimensionCount: return "llvm.hpvm.getNumDims"
        case .getNodeInstanceIdX: return "llvm.hpvm.getNodeInstanceID.x"
        case .getNodeInstanceIdY: return "llvm.hpvm.getNodeInstanceID.y" 
        case .getNodeInstanceIdZ: return "llvm.hpvm.getNodeInstanceID.z" 
        case .getNodeInstanceCountX: return "llvm.hpvm.getNumNodeInstances.x"
        case .getNodeInstanceCountY: return "llvm.hpvm.getNumNodeInstances.y"
        case .getNodeInstanceCountZ: return "llvm.hpvm.getNumNodeInstances.z"
        case .getVectorLength: return "llvm.hpvm.getVectorLength"
        case .malloc: return "llvm.hpvm.malloc"
        case .barrier: return "llvm.hpvm.barrier"
        case .launch: return "llvm.hpvm.launch"
        case .wait: return "llvm.hpvm.wait"
        case .push: return "llvm.hpvm.push"
        case .pop: return "llvm.hpvm.pop"
        }
    }

    public var type: FunctionType {
        switch self {
        case .createNode:
            return [i8*] => i8*
        case .createNode1D:
            return [i8*, i32] => i8*
        case .createNode2D:
            return [i8*, i32, i32] => i8*
        case .createNode3D:
            return [i8*, i32, i32, i32] => i8*
        case .createEdge:
            return [i8*, i8*, i32, i32, i1, i1] => void
        case .bindInput:
            return [i8*, i32, i32, i1] => void
        case .bindOutput:
            return [i8*, i32, i32, i1] => void
        case .getCurrentNode:
            return [] => i8*
        case .getParentNode:
            return [i8*] => i8*
        case .getDimensionCount:
            return [i8*] => i32
        case .getNodeInstanceIdX,
             .getNodeInstanceIdY,
             .getNodeInstanceIdZ:
            return [i8*] => i32
        case .getNodeInstanceCountX,
             .getNodeInstanceCountY,
             .getNodeInstanceCountZ:
            return [i8*, i32] => i32
        case .getVectorLength:
            return [i32] => i32
        case .malloc:
            return [i32] => i8*
        case .barrier:
            return [] => void
        case .launch:
            return [i8*, i8*, i1*] => i8*
        case .wait:
            return [i8*] => void
        case .push:
            return [i8*, i8*] => void
        case .pop:
            return [i8*] => i8*
        }
    }

    public var arguments: [IRValue] {
        switch self {
        case .getCurrentNode: return []
        case .barrier: return []
        case let .wait(id: v1): return [v1]
        case let .push(v1, arguments: v2): return [v1, v2]
        case let .pop(v1): return [v1]
        case let .getParentNode(v1): return [v1]
        case let .getDimensionCount(v1): return [v1]
        case let .getNodeInstanceIdX(v1): return [v1]
        case let .getNodeInstanceIdY(v1): return [v1]
        case let .getNodeInstanceIdZ(v1): return [v1]
        case let .getNodeInstanceCountX(v1): return [v1]
        case let .getNodeInstanceCountY(v1): return [v1]
        case let .getNodeInstanceCountZ(v1): return [v1]
        case let .getVectorLength(v1): return [v1]
        case let .malloc(v1): return [v1]
        case let .createNode(v1): return [v1]
        case let .createNode1D(v1, v2): return [v1, v2]
        case let .createNode2D(v1, v2, v3): return [v1, v2, v3]
        case let .createNode3D(v1, v2, v3, v4): return [v1, v2, v3, v4]
        case let .createEdge(v1, v2, v3, v4, replication, v6):
            return [v1, v2, v3, v4, replication.constant, v6]
        case let .bindInput(node: v1, parentInput: v2, input: v3):
            return [v1, v2, v3]
        case let .bindOutput(node: v1, output: v2, parentOutput: v3,
                             streaming: streaming):
            return [v1, v2, v3, streaming.constant]
        case let .launch(v1, arguments: v2, streaming: streaming):
            return [v1, v2, streaming.constant]
        }
    }
    
}

extension HPVM.RuntimeFunction : LLFunctionPrototype {

    public var name: StaticString {
        switch self {
        case .trackMemory: return "llvm_hpvm_track_mem"
        case .requestMemory: return "llvm_hpvm_request_mem"
        case .untrackMemory: return "llvm_hpvm_untrack_mem"
        }
    }

    public var type: FunctionType {
        switch self {
        case .trackMemory: return [i8*, i64] => void
        case .requestMemory: return [i8*, i64] => void
        case .untrackMemory: return [i8*] => void
        }
    }

    public var arguments: [IRValue] {
        switch self {
        case let .trackMemory(address: v1, size: v2): return [v1, v2]
        case let .requestMemory(address: v1, size: v2): return [v1, v2]
        case let .untrackMemory(address: v1): return [v1]
        }
    }
    
}
