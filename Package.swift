import PackageDescription

let package = Package(
    name: "CuDNN",
    dependencies: [
        .Package(url: "../CUDA", majorVersion: 0), // Test!
        .Package(url: "https://github.com/rxwei/CCUDA", majorVersion: 1)
    ]
)
