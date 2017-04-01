import PackageDescription

let package = Package(
    name: "DLVM",
    targets: [
        /// Core DLVM library: IR, BPGen (libDLVM)
        Target(name: "DLVM"),
        /// DLVM IR Reader
//        Target(name: "DLVMReader", dependencies: ["DLVM"]),
        /// DLVM compiler driver
//        Target(name: "dlc", dependencies: ["DLVM", "DLVMReader", "DLVMCodeGen"]),
        /// DLVM compiler primitives
        Target(name: "dlrt"),
        /// DLVM code generator
        Target(name: "DLVMCodeGen", dependencies: ["DLVM"]),
        /// DLVM interpreter
//        Target(name: "dli", dependencies: ["DLVM", "DLVMReader", "DLVMRuntime"]),
        /// TEL compiler library (libTEL)
        Target(name: "TEL", dependencies: ["DLVM"]),
        /// TEL compiler driver
        Target(name: "telc", dependencies: ["TEL", "DLVM", "DLVMCodeGen"])
    ],
    dependencies: [
      	.Package(url: "https://github.com/rxwei/Parsey", majorVersion: 1, minor: 7),
      	//.Package(url: "https://github.com/rxwei/LLVM_C", majorVersion: 2, minor: 1),
      	.Package(url: "https://github.com/trill-lang/LLVMSwift", majorVersion: 0),
      	.Package(url: "https://github.com/rxwei/CommandLine", majorVersion: 3, minor: 0),
        .Package(url: "https://github.com/rxwei/dlvm-tensor", majorVersion: 0)
    ],
    exclude: [
        "Sources/DLVMReader",
        "Sources/dlc",
        "Sources/dli",
        "Kernels"
    ]
)
