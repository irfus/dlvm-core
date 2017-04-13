//
//  mmul.swift
//  DLVM Comptue Primitives
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

@_silgen_name("mmul_f32")
public func mmul(A: UnsafePointer<CFloat>, lda: I32,
                 B: UnsafePointer<CFloat>, ldb: I32,
                 C: UnsafeMutablePointer<CFloat>, ldc: I32,
                 k: I32)
{
    HPVM.hint(.gpu)
    HPVM.attributeIn(A)
    HPVM.attributeIn(B)
    HPVM.attributeIn(C)
    HPVM.attributeOut(C)

    let thisNode = HPVM.currentNode()
    let m = HPVM.nodeInstanceIdX(of: thisNode)
    let n = HPVM.nodeInstanceIdY(of: thisNode)

    var i: I32 = 0
    var c: CFloat = 0.0
    repeat {
        let a = A[Int(m + i * lda)]
        let b = B[Int(n + i * ldb)]
        c += a * b
        i += 1
    } while i < k
    C[Int(m + n * ldc)] = c;
}

@_silgen_name("mmul_f64")
public func mmul(A: UnsafePointer<CDouble>, lda: I32,
                 B: UnsafePointer<CDouble>, ldb: I32,
                 C: UnsafeMutablePointer<CDouble>, ldc: I32,
                 k: I32)
{
    HPVM.hint(.gpu)
    HPVM.attributeIn(A)
    HPVM.attributeIn(B)
    HPVM.attributeIn(C)
    HPVM.attributeOut(C)

    let thisNode = HPVM.currentNode()
    let m = HPVM.nodeInstanceIdX(of: thisNode)
    let n = HPVM.nodeInstanceIdY(of: thisNode)

    var i: I32 = 0
    var c: CDouble = 0.0
    repeat {
        let a = A[Int(m + i * lda)]
        let b = B[Int(n + i * ldb)]
        c += a * b
        i += 1
    } while i < k
    C[Int(m + n * ldc)] = c;
}

@_silgen_name("mmul_i16")
public func mmul(A: UnsafePointer<Int16>, lda: I32,
                 B: UnsafePointer<Int16>, ldb: I32,
                 C: UnsafeMutablePointer<Int16>, ldc: I32,
                 k: I32)
{
    HPVM.hint(.gpu)
    HPVM.attributeIn(A)
    HPVM.attributeIn(B)
    HPVM.attributeIn(C)
    HPVM.attributeOut(C)

    let thisNode = HPVM.currentNode()
    let m = HPVM.nodeInstanceIdX(of: thisNode)
    let n = HPVM.nodeInstanceIdY(of: thisNode)

    var i: I32 = 0
    var c: Int16 = 0
    repeat {
        let a = A[Int(m + i * lda)]
        let b = B[Int(n + i * ldb)]
        c += a * b
        i += 1
    } while i < k
    C[Int(m + n * ldc)] = c;
}

@_silgen_name("mmul_i32")
public func mmul(A: UnsafePointer<Int32>, lda: I32,
                 B: UnsafePointer<Int32>, ldb: I32,
                 C: UnsafeMutablePointer<Int32>, ldc: I32,
                 k: I32)
{
    HPVM.hint(.gpu)
    HPVM.attributeIn(A)
    HPVM.attributeIn(B)
    HPVM.attributeIn(C)
    HPVM.attributeOut(C)

    let thisNode = HPVM.currentNode()
    let m = HPVM.nodeInstanceIdX(of: thisNode)
    let n = HPVM.nodeInstanceIdY(of: thisNode)

    var i: I32 = 0
    var c: Int32 = 0
    repeat {
        let a = A[Int(m + i * lda)]
        let b = B[Int(n + i * ldb)]
        c += a * b
        i += 1
    } while i < k
    C[Int(m + n * ldc)] = c;
}


@_silgen_name("mmul_i64")
public func mmul(A: UnsafePointer<Int64>, lda: I32,
                 B: UnsafePointer<Int64>, ldb: I32,
                 C: UnsafeMutablePointer<Int64>, ldc: I32,
                 k: I32)
{
    HPVM.hint(.gpu)
    HPVM.attributeIn(A)
    HPVM.attributeIn(B)
    HPVM.attributeIn(C)
    HPVM.attributeOut(C)

    let thisNode = HPVM.currentNode()
    let m = HPVM.nodeInstanceIdX(of: thisNode)
    let n = HPVM.nodeInstanceIdY(of: thisNode)

    var i: I32 = 0
    var c: Int64 = 0
    repeat {
        let a = A[Int(m + i * lda)]
        let b = B[Int(n + i * ldb)]
        c += a * b
        i += 1
    } while i < k
    C[Int(m + n * ldc)] = c;
}
