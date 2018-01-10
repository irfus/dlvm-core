all:
	swift build

release:
	swift build -c release

test:
	swift test

check: all
	lit FileCheck -v

testall:
	swift test
	lit FileCheck -v

update:
	swift package update

xcode:
	swift package generate-xcodeproj

clean:
	swift package clean
