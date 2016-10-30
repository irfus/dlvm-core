import PackageDescription

let package = Package(
    name: "CuDNN",
    dependencies: [
        .Package(url: "https://github.com/rxwei/cuda-swift", majorVersion: 1),
        .Package(url: "https://github.com/rxwei/CCUDA", majorVersion: 1)
    ]
)
