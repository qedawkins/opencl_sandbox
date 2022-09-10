clang++ -cc1 -triple=spir-unknown-unknown matrixMultiplyStatic.cl -O0 -finclude-default-header -emit-llvm-bc -o matrixMultiplyStatic.bc
llvm-spirv matrixMultiplyStatic.bc -o matrixMultiplyStatic.spv

