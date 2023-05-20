if [ $(uname) = "Linux" ]; then
  export LD_LIBRARY_PATH=/home/jfcm/Repos/binaryTree/arma_libs/lib:$LD_LIBRARY_PATH
fi

if [ "$1" = "1" ]; then
  ./bin/main_1.exe
# elif [ "$1" = "2" ]; then
#   ./bin/main_2.exe
# elif [ "$1" = "3" ]; then
#   ./bin/main_3.exe
# elif [ "$1" = "4" ]; then
#   ./bin/main_4.exe
# elif [ "$1" = "5" ]; then
#   ./bin/main_5.exe
# elif [ "$1" = "6" ]; then
#   ./bin/main_6.exe
# elif [ "$1" = "6a" ]; then
#   ./bin/main_6a.exe
# elif [ "$1" = "7" ]; then
#   ./bin/main_7.exe
else
  echo "Invalid argument. Usage: $0 [1]"
  exit 1
fi
