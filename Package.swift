//
//  Package.swift
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

import PackageDescription

let package = Package(
    name: "DLVM",
    targets: [
        Target(name: "DLVM"),
        Target(name: "DLVMCodeGen", dependencies: ["DLVM"]),
        Target(name: "DLParse", dependencies: ["DLVM"]),
//        Target(name: "TEL", dependencies: ["DLVM"]),
//        Target(name: "telc", dependencies: ["TEL", "DLVM", "DLVMCodeGen"])
    ],
    dependencies: [
      	.Package(url: "https://github.com/rxwei/LLVM_C", majorVersion: 2, minor: 1),
      	.Package(url: "https://github.com/rxwei/Parsey", majorVersion: 1, minor: 7),
      	.Package(url: "https://github.com/rxwei/CommandLine", majorVersion: 3, minor: 0),
      	.Package(url: "https://github.com/rxwei/CoreTensor", majorVersion: 0, minor: 4)
    ],
    exclude: [
        "Sources/TEL",
        "Sources/telc",
        "Sources/dlc",
        "Sources/dli",
        "Compute",
        "Runtime",
    ]
)
