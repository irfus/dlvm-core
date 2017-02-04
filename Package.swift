import PackageDescription

let package = Package(
    name: "DLVM",
    targets: [
        /// Core DLVM library: IR, BPGen (libDLVM)
        Target(name: "DLVM"),
        /// DLVM IR Reader
        Target(name: "DLVMReader", dependencies: ["DLVM"]),
        /// DLVM compiler driver
        Target(name: "dlc", dependencies: ["DLVM", "DLVMReader"]),
        /// DLVM runtime
        Target(name: "DLVMRuntime", dependencies: ["DLVM", "DLVMReader"]),
        /// DLVM interpreter
        Target(name: "dli", dependencies: ["DLVM", "DLVMReader", "DLVMRuntime"]),
        /// TEL compiler library (libTEL)
        Target(name: "TEL", dependencies: ["DLVM"]),
        /// TEL compiler driver
        Target(name: "telc", dependencies: ["DLVM", "TEL"])
    ],
    dependencies: [
      	.Package(url: "https://github.com/rxwei/Parsey", majorVersion: 1, minor: 7)
    ]
)
