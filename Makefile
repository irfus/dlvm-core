INCLUDEDIR=-I`llvm-config --includedir`
LIBDIR=-L`llvm-config --libdir`
SFLAGS=-Xcc $(INCLUDEDIR) -Xlinker -lLLVM -Xlinker $(LIBDIR)

all:
	swift build $(SFLAGS)

release:
	swift build -c release $(SFLAGS)

test:
	swift test $(SFLAGS)

update:
	swift package update

clean:
	swift package clean
