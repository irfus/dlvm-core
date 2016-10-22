import PackageDescription

let package = Package(
    name: "CuDNN",
    dependencies: [
        .Package(url: "https://github.com/rxwei/cuda-swift", majorVersion: 0), // Test!
        .Package(url: "https://github.com/rxwei/CCUDA", majorVersion: 1)
    ]
)
