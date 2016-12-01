import PackageDescription

let package = Package(
    name: "DLVM",
    targets: [
        Target(name: "DLVM")
    ],
    dependencies: [
        .Package(url: "https://github.com/rxwei/cuda-swift", majorVersion: 1, minor: 3),
        .Package(url: "https://github.com/rxwei/CCUDA", majorVersion: 1, minor: 4)
    ]
)
