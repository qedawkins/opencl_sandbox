__kernel void forward_dispatch_0_matmul(const __global float* a, const __global float* b, __global float* c) {
    uint idx = get_global_id(0);
    uint jdx = get_global_id(1);

    int m = 2;
    int k = 4;
    int n = 3;

    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        float lhs = a[idx * k + i];
        float rhs = b[i * n + jdx];
        sum += lhs * rhs;
    }

    c[idx * n + jdx] = sum;
}
