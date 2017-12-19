all:
	swift build

release:
	swift build -c release

test:
	swift test

check: all
	lit FileCheck

update:
	swift package update

xcode:
	swift package generate-xcodeproj

clean:
	swift package clean
