
Single Core
main.cpp is single core quicksort 
build it >clang++ -O3 -o main main.cpp
run it >time ./main >/dev/null

Multi Core
main_mult is multi core non-recursive merge sort
build it >clang++ -O3 -o main_mult main_mult.cpp
run it >time ./main_mult >/dev/null