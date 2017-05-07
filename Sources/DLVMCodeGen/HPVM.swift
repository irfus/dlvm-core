//
//  HPVM.swift
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

import LLVM
import DLVM

/// HPVM Target
public final class HPVM : LLTarget, LLFunctionPrototypeCacheable {

    public enum ReplicationMode : Int, LLConstantConvertible {
        case allToAll = 0
        case oneToOne = 1

        public var constantType: IntType {
            return i1
        }
    }

    public enum Intrinsic {
        case createNode(LLVM.Function)
        case createNode1D(LLVM.Function, IRValue)
        case createNode2D(LLVM.Function, IRValue, IRValue)
        case createNode3D(LLVM.Function, IRValue, IRValue, IRValue)
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
        case currentNode
        case parentNode(IRValue)
        case dimensionCount(IRValue)
        case nodeInstanceIdX(IRValue)
        case nodeInstanceIdY(IRValue)
        case nodeInstanceIdZ(IRValue)
        case nodeInstanceCountX(IRValue)
        case nodeInstanceCountY(IRValue)
        case nodeInstanceCountZ(IRValue)
        case vectorLength(IRValue)
        case malloc(IRValue)
        case barrier
        case launch(LLVM.Function, arguments: IRValue, streaming: Bool)
        case wait(id: IRValue)
        case push(id: IRValue, arguments: IRValue)
        case pop(id: IRValue)
        /// Speical functions (non-intrinsics but handled by GenHPVM pass)
        case attributeIn(IRValue)
        case attributeOut(IRValue)
    }

    public enum RuntimeFunction {
        case trackMemory(address: IRValue, size: IRValue)
        case requestMemory(address: IRValue, size: IRValue)
        case untrackMemory(address: IRValue)
    }

    public unowned let module: LLVM.Module
    public var functions: [AnyHashable : LLVM.Function] = [:]

    public required init(module: LLVM.Module) {
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
        case .currentNode: return "llvm.hpvm.getNode"
        case .parentNode: return "llvm.hpvm.getParentNode"
        case .dimensionCount: return "llvm.hpvm.getNumDims"
        case .nodeInstanceIdX: return "llvm.hpvm.getNodeInstanceID.x"
        case .nodeInstanceIdY: return "llvm.hpvm.getNodeInstanceID.y" 
        case .nodeInstanceIdZ: return "llvm.hpvm.getNodeInstanceID.z" 
        case .nodeInstanceCountX: return "llvm.hpvm.getNumNodeInstances.x"
        case .nodeInstanceCountY: return "llvm.hpvm.getNumNodeInstances.y"
        case .nodeInstanceCountZ: return "llvm.hpvm.getNumNodeInstances.z"
        case .vectorLength: return "llvm.hpvm.getVectorLength"
        case .malloc: return "llvm.hpvm.malloc"
        case .barrier: return "llvm.hpvm.barrier"
        case .launch: return "llvm.hpvm.launch"
        case .wait: return "llvm.hpvm.wait"
        case .push: return "llvm.hpvm.push"
        case .pop: return "llvm.hpvm.pop"
        case .attributeIn: return "__hpvm__attributeIn"
        case .attributeOut: return "__hpvm__attributeOut"
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
        case .currentNode:
            return [] => i8*
        case .parentNode:
            return [i8*] => i8*
        case .dimensionCount:
            return [i8*] => i32
        case .nodeInstanceIdX,
             .nodeInstanceIdY,
             .nodeInstanceIdZ:
            return [i8*] => i32
        case .nodeInstanceCountX,
             .nodeInstanceCountY,
             .nodeInstanceCountZ:
            return [i8*, i32] => i32
        case .vectorLength:
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
        case .attributeIn:
            return [i8*] => void
        case .attributeOut:
            return [i8*] => void
        }
    }

    public var arguments: [IRValue] {
        switch self {
        case .currentNode: return []
        case .barrier: return []
        case let .wait(id: v1): return [v1]
        case let .push(v1, arguments: v2): return [v1, v2]
        case let .pop(v1): return [v1]
        case let .parentNode(v1): return [v1]
        case let .dimensionCount(v1): return [v1]
        case let .nodeInstanceIdX(v1): return [v1]
        case let .nodeInstanceIdY(v1): return [v1]
        case let .nodeInstanceIdZ(v1): return [v1]
        case let .nodeInstanceCountX(v1): return [v1]
        case let .nodeInstanceCountY(v1): return [v1]
        case let .nodeInstanceCountZ(v1): return [v1]
        case let .vectorLength(v1): return [v1]
        case let .malloc(v1): return [v1]
        case let .createNode(v1): return [v1]
        case let .createNode1D(v1, v2): return [v1, v2]
        case let .createNode2D(v1, v2, v3): return [v1, v2, v3]
        case let .createNode3D(v1, v2, v3, v4): return [v1, v2, v3, v4]
        case let .createEdge(v1, v2, v3, v4, replication, streaming):
            return [v1, v2, v3, v4, replication.constant, streaming.constant]
        case let .bindInput(node: v1, parentInput: v2, input: v3):
            return [v1, v2, v3]
        case let .bindOutput(node: v1, output: v2, parentOutput: v3,
                             streaming: streaming):
            return [v1, v2, v3, streaming.constant]
        case let .launch(v1, arguments: v2, streaming: streaming):
            return [v1, v2, streaming.constant]
        case let .attributeIn(v1):
            return [v1]
        case let .attributeOut(v1):
            return [v1]
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
