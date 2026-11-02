#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 256
#define WP_NO_CRT
#include "builtin.h"
#include "deterministic.h"

// Map wp.breakpoint() to a device brkpt at the call site so cuda-gdb attributes the stop to the generated .cu line
#if defined(__CUDACC__) && !defined(_MSC_VER)
#define __debugbreak() __brkpt()
#endif

#define builtin_tid1d() wp::tid(_idx, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, _idx, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, _idx, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, _idx, dim)

#define builtin_block_dim() wp::block_dim()

// CUDA Thread Block Cluster shape declaration. Expands to __cluster_dims__
// only on devices that support clusters (compute capability 9.0+); otherwise
// expands to nothing so the same source compiles cleanly for any target arch.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define WP_CLUSTER_DIMS(x, y, z) __cluster_dims__(x, y, z)
#else
#define WP_CLUSTER_DIMS(x, y, z)
#endif

// Maximum registers per thread. __maxnreg__ was added in CUDA Toolkit 12.4;
// older toolkits ignore the opt-in so the same source remains compilable.
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 4))
#define WP_MAXNREG(n) __maxnreg__(n)
#else
#define WP_MAXNREG(n)
#endif

// Allow PTXAS to spill registers into shared memory. CUDA Toolkit 13.0
// introduced the pragma, which is unavailable in device-debug compilation.
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 13) && !defined(_DEBUG)
#define WP_ENABLE_SMEM_SPILLING() asm volatile(".pragma \"enable_smem_spilling\";");
#else
#define WP_ENABLE_SMEM_SPILLING()
#endif


// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)



extern "C" __global__ void _scatter_add_kernel_3267f473_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float32> var_forces,
    wp::array_t<wp::float32> var_computed_forces,
    wp::array_t<wp::uint32> var_indices,
    wp::array_t<wp::float32> var_output,
    wp::array_t<wp::float32> var_computed_output)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::uint32* var_1;
        wp::uint32 var_2;
        wp::uint32 var_3;
        wp::float32* var_4;
        wp::float32* var_5;
        wp::float32 var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        wp::float32* var_9;
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        //---------
        // forward
        // def _scatter_add_kernel(                                                               <L 30>
        // i = wp.tid()                                                                           <L 38>
        var_0 = builtin_tid1d();
        // idx = indices[i]                                                                       <L 39>
        var_1 = wp::address(var_indices, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // output[idx] = output[idx] + forces[i]                                                  <L 40>
        var_4 = wp::address(var_output, var_2);
        var_5 = wp::address(var_forces, var_0);
        var_7 = wp::load(var_4);
        var_8 = wp::load(var_5);
        var_6 = wp::add(var_7, var_8);
        wp::array_store(var_output, var_2, var_6);
        // if computed_output:                                                                    <L 41>
        if (var_computed_output) {
            // computed_output[idx] = computed_output[idx] + computed_forces[i]                   <L 42>
            var_9 = wp::address(var_computed_output, var_2);
            var_10 = wp::address(var_computed_forces, var_0);
            var_12 = wp::load(var_9);
            var_13 = wp::load(var_10);
            var_11 = wp::add(var_12, var_13);
            wp::array_store(var_computed_output, var_2, var_11);
        }
    }
}



extern "C" __global__ void _scatter_add_kernel_3267f473_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float32> var_forces,
    wp::array_t<wp::float32> var_computed_forces,
    wp::array_t<wp::uint32> var_indices,
    wp::array_t<wp::float32> var_output,
    wp::array_t<wp::float32> var_computed_output,
    wp::array_t<wp::float32> adj_forces,
    wp::array_t<wp::float32> adj_computed_forces,
    wp::array_t<wp::uint32> adj_indices,
    wp::array_t<wp::float32> adj_output,
    wp::array_t<wp::float32> adj_computed_output)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::uint32* var_1;
        wp::uint32 var_2;
        wp::uint32 var_3;
        wp::float32* var_4;
        wp::float32* var_5;
        wp::float32 var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        wp::float32* var_9;
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::uint32 adj_1 = {};
        wp::uint32 adj_2 = {};
        wp::uint32 adj_3 = {};
        wp::float32 adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        //---------
        // forward
        // def _scatter_add_kernel(                                                               <L 30>
        // i = wp.tid()                                                                           <L 38>
        var_0 = builtin_tid1d();
        // idx = indices[i]                                                                       <L 39>
        var_1 = wp::address(var_indices, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // output[idx] = output[idx] + forces[i]                                                  <L 40>
        var_4 = wp::address(var_output, var_2);
        var_5 = wp::address(var_forces, var_0);
        var_7 = wp::load(var_4);
        var_8 = wp::load(var_5);
        var_6 = wp::add(var_7, var_8);
        // wp::array_store(var_output, var_2, var_6);
        // if computed_output:                                                                    <L 41>
        if (var_computed_output) {
            // computed_output[idx] = computed_output[idx] + computed_forces[i]                   <L 42>
            var_9 = wp::address(var_computed_output, var_2);
            var_10 = wp::address(var_computed_forces, var_0);
            var_12 = wp::load(var_9);
            var_13 = wp::load(var_10);
            var_11 = wp::add(var_12, var_13);
            // wp::array_store(var_computed_output, var_2, var_11);
        }
        //---------
        // reverse
        if (var_computed_output) {
            wp::adj_array_store(var_computed_output, var_2, var_11, adj_computed_output, adj_2, adj_11);
            wp::adj_add(var_12, var_13, adj_9, adj_10, adj_11);
            wp::adj_address(var_computed_forces, var_0, adj_computed_forces, adj_0, adj_10);
            wp::adj_address(var_computed_output, var_2, adj_computed_output, adj_2, adj_9);
            // adj: computed_output[idx] = computed_output[idx] + computed_forces[i]              <L 42>
        }
        // adj: if computed_output:                                                               <L 41>
        wp::adj_array_store(var_output, var_2, var_6, adj_output, adj_2, adj_6);
        wp::adj_add(var_7, var_8, adj_4, adj_5, adj_6);
        wp::adj_address(var_forces, var_0, adj_forces, adj_0, adj_5);
        wp::adj_address(var_output, var_2, adj_output, adj_2, adj_4);
        // adj: output[idx] = output[idx] + forces[i]                                             <L 40>
        wp::adj_copy(var_3, adj_1, adj_2);
        wp::adj_address(var_indices, var_0, adj_indices, adj_0, adj_1);
        // adj: idx = indices[i]                                                                  <L 39>
        // adj: i = wp.tid()                                                                      <L 38>
        // adj: def _scatter_add_kernel(                                                          <L 30>
        continue;
    }
}

