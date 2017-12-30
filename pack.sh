#/bin/bash

if [ -z "$1" ]
then
	echo "Run the script with your login as the first parameter."
	exit 2
fi

cd doc
make
cd ..

mkdir "$1"
cp -t "$1" -r src/ CMakeLists.txt doc/doc.pdf
zip -r "$1".zip "$1"
rm -r "$1"
