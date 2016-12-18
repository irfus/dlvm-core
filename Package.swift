import PackageDescription

let package = Package(
    name: "DLVM",
    targets: [
        /// Core DLVM library: IR, BPGen (libDLVM)
        Target(name: "DLVM"),
        /// DLVM compiler driver
        Target(name: "dlc"),
        /// DLVM bitcode disassembler
        Target(name: "dlvm-dis"),
        /// TEL compiler library (libTEL)
        Target(name: "TEL"),
        /// TEL compiler driver
        Target(name: "telc")
    ],
    dependencies: [
    	.Package(url: "https://github.com/rxwei/Parsey", majorVersion: 1, minor: 4)
    ]
)
