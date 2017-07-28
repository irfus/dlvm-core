// swift-tools-version:4.0
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
    products: [
        .library(name: "DLVM", type: .dynamic,
                 targets: ["DLVM", "DLVMCodeGen", "DLParse"]),
        .library(name: "DLVMCore", type: .static,
                 targets: ["DLVM"]),
        .library(name: "DLVMCodeGen", type: .static,
                 targets: ["DLVMCodeGen"]),
        .library(name: "DLParse", type: .static,
                 targets: ["DLParse"]),
        .executable(name: "dlopt",
                    targets: ["dlopt"]),
        .executable(name: "dlc",
                    targets: ["dlc"]),
    ],
    dependencies: [
        .package(url: "https://github.com/rxwei/LLVM_C", from: "2.1.0"),
        .package(url: "https://github.com/rxwei/CommandLine", from: "3.0.0"),
        .package(url: "https://github.com/rxwei/CoreTensor", from: "0.5.0")
    ],
    targets: [
        .target(name: "DLVM", dependencies: ["CoreTensor"]),
        .target(name: "DLVMCodeGen", dependencies: ["DLVM"]),
        .target(name: "DLParse", dependencies: ["DLVM"]),
        .target(name: "DLCommandLineTools", dependencies: [
            "CommandLine", "DLVM", "DLVMCodeGen", "DLParse"
        ]),
        .target(name: "dlopt", dependencies: [
            "DLVM", "DLParse", "DLCommandLineTools"
        ]),
        .target(name: "dlc", dependencies: [
            "DLVM", "DLVMCodeGen", "DLParse", "DLCommandLineTools"
        ]),
    ],
    swiftLanguageVersions: [ 4 ]
)
