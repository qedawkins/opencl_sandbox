all:
	clang++ -std=c++14 -O0 -fpermissive -rdynamic -fPIC vectorAdd/main.cpp -o vectorAdd/run -l OpenCL
	clang++ -std=c++14 -O0 -fpermissive -rdynamic -fPIC staticMatmul/main.cpp -o staticMatmul/run -l OpenCL
	clang++ -std=c++14 -O0 -fpermissive -rdynamic -fPIC staticMatmul/spirv_main.cpp -o staticMatmul/srun -l OpenCL
