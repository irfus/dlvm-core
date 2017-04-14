//
//  conv.swift
//  DLVM Comptue Primitives
//
//  Copyright 2016-2017 Richard Wei.
//
//  Licensed under the Apache License, Version 2.0 (the "License")
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

#if os(Linux)
    import Glibc
#else
    import Darwin
#endif

@inline(never)
@_silgen_name("conv2d_float_root")
func Conv2DFloatRoot(I: UnsafePointer<Float>, // In 1
                     M: UnsafePointer<Float>, // In 2
                     P: UnsafeMutablePointer<Float>, // Out 3
                     /// Parameters
                     tileWidth: Int32, // 4
                     maskWidth: Int32, // 5
                     channelCount: Int32, // 6
                     width: Int32, // 7
                     height: Int32) // 8
{
    HPVM.hint(.cpu)
    HPVM.attributeIn(I)
    HPVM.attributeIn(M)
    HPVM.attributeOut(P)

    let workgroupNode = HPVM.createNode(bitCast(Conv2DFloatWorkgroup),
                                        x: width/tileWidth,
                                        y: height/tileWidth)

    HPVM.bindIn(workgroupNode, 0, 0, isStreaming: false)
    HPVM.bindIn(workgroupNode, 1, 1, isStreaming: false)
    HPVM.bindIn(workgroupNode, 2, 2, isStreaming: false)
    HPVM.bindIn(workgroupNode, 3, 3, isStreaming: false)
    HPVM.bindIn(workgroupNode, 4, 4, isStreaming: false)
    HPVM.bindIn(workgroupNode, 5, 5, isStreaming: false)
    HPVM.bindIn(workgroupNode, 6, 6, isStreaming: false)
    HPVM.bindIn(workgroupNode, 7, 7, isStreaming: false)
    HPVM.bindIn(workgroupNode, 8, 8, isStreaming: false)
}

@inline(never)
@_silgen_name("conv2d_float_allocate")
func Conv2DFloatAllocate(tileWidth: Int32, maskWidth: Int32) -> (UnsafeMutablePointer<Float>, Int32) {
    let count = tileWidth + maskWidth - 1
    let ptr = HPVM.malloc(count)
    return (ptr.assumingMemoryBound(to: Float.self), count)
}

@inline(never)
@_silgen_name("conv2d_float_workgroup")
func Conv2DFloatWorkgroup(I: UnsafePointer<Float>, // In 1
                          M: UnsafePointer<Float>, // In 2
                          P: UnsafeMutablePointer<Float>, // Out 3
                          /// Parameters
                          tileWidth: Int32, // 4
                          maskWidth: Int32, // 5
                          channelCount: Int32, // 6
                          width: Int32, // 7
                          height: Int32) // 8
{
    HPVM.hint(.cpu)
    HPVM.attributeIn(I)
    HPVM.attributeIn(M)
    HPVM.attributeOut(P)

    let allocationNode = HPVM.createNode(bitCast(Conv2DFloatAllocate))
    let leafNode = HPVM.createNode(bitCast(Conv2DFloatLeaf), x: tileWidth, y: tileWidth)

    HPVM.bindIn(leafNode, 0, 0, isStreaming: false)
    HPVM.bindIn(leafNode, 1, 1, isStreaming: false)
    HPVM.bindIn(leafNode, 2, 2, isStreaming: false)
    HPVM.bindIn(leafNode, 3, 3, isStreaming: false)
    HPVM.bindIn(leafNode, 4, 4, isStreaming: false)
    HPVM.bindIn(leafNode, 5, 5, isStreaming: false)
    HPVM.bindIn(leafNode, 6, 6, isStreaming: false)
    HPVM.bindIn(leafNode, 7, 7, isStreaming: false)
    HPVM.bindIn(leafNode, 8, 8, isStreaming: false)

    /// Shared
    HPVM.createEdge(allocationNode, bitCast(Conv2DFloatLeaf),
                    1, 0, 9, isStreaming: false)
    HPVM.createEdge(allocationNode, bitCast(Conv2DFloatLeaf),
                    1, 1, 10, isStreaming: false)
}

@inline(never)
@_silgen_name("conv2d_float_leaf")
func Conv2DFloatLeaf(I: UnsafePointer<Float>, // In 1
                     M: UnsafePointer<Float>, // In 2
                     P: UnsafeMutablePointer<Float>, // Out 3
                     /// Parameters
                     tileWidth: Int32, // 4
                     maskWidth: Int32, // 5
                     channelCount: Int32, // 6
                     width: Int32, // 7
                     height: Int32, // 8
                     /// Shared memory
                     N: UnsafeMutablePointer<Float>, // 9
                     countN: Int32) // 10
{
    HPVM.hint(.gpu)
    HPVM.attributeIn(I)
    HPVM.attributeIn(M)
    HPVM.attributeOut(P)
    HPVM.attributeIn(N) /// Shared memory

    let node = HPVM.currentNode()
    let parentNode = HPVM.parentNode(of: node)
    let idX = HPVM.nodeInstanceIdX(of: node)
    let idY = HPVM.nodeInstanceIdY(of: node)
    let blockIdX = HPVM.nodeInstanceIdX(of: parentNode)
    let blockIdY = HPVM.nodeInstanceIdY(of: parentNode)
    let maskRadius = maskWidth / 2
    let w = tileWidth + maskWidth - 1

    for k in 0..<channelCount {
        // First batch loading
        var dest = blockIdY * tileWidth + blockIdX
        var destY = dest / countN
        var destX = dest % countN
        var srcY = blockIdY * tileWidth + destY - maskRadius
        var srcX = blockIdX * tileWidth + destX - maskRadius
        var src = (srcY * width + srcX) * channelCount + k
        if srcY >= 0 && srcY < height && srcX >= 0 && srcX < width {
            N[Int(destY * w + destX)] = I[Int(src)]
        } else {
            N[Int(destY * w + destX)] = 0
        }

        // Second batch loading
        dest = blockIdY * tileWidth + idX + tileWidth * tileWidth
        destY = dest / countN
        destX = dest % countN
        srcY = blockIdY * tileWidth + destY - maskRadius
        srcX = blockIdX * tileWidth + destX - maskRadius
        src = (srcY * width + srcX) * channelCount + k
        if destY < countN {
            if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
                N[Int(destY * w + destX)] = I[Int(src)]
            } else {
                N[Int(destY * w + destX)] = 0
            }
        }
        HPVM.barrier()

        var accum: Float = 0
        var x: Int32 = 0
        var y: Int32 = 0
        repeat {
            repeat {
                accum += N[Int((idY + y) * w + idX + x)] * M[Int(y * maskWidth + x)]
                x += 1
            } while x < maskWidth
            y += 1
        } while y < maskWidth

        y = blockIdY * tileWidth + idY
        x = blockIdX * tileWidth + idX
        if y < height && x < width {
            P[Int((y * width + x) * channelCount + k)] = min(max(accum, 0.0), 1.0)
        }
        HPVM.barrier()
    }
}
