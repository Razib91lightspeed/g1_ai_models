#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 64
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

extern "C" {
void potrs_31_31_1_121_64_1_1_5_x_x_0(wp::float32*, wp::float32*);
}


extern "C" __global__ void _tile_cholesky_solve_block__locals__kernel_bdfd039e_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_qLD_block_adr,
    wp::array_t<wp::int32> var_block_dof,
    wp::array_t<wp::float32> var_L_in,
    wp::array_t<wp::float32> var_y,
    wp::array_t<wp::float32> var_x)
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
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::array_t<wp::float32> var_5;
        const wp::int32 var_6 = 961;
        wp::tuple_t<wp::int32> var_7;
        wp::int32* var_8;
        wp::tuple_t<wp::int32> var_9;
        wp::int32 var_10;
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<961>, wp::tile_stride_t<1>>, true> var_11 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<961>,wp::tile_stride_t<1>,false>();
        wp::int32 var_12;
        const wp::int32 var_13 = 31;
        wp::tuple_t<wp::int32, wp::int32> var_14;
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<31,31>, wp::tile_stride_t<31,1>>, false> var_15 = nullptr;
        wp::array_t<wp::float32> var_16;
        wp::tuple_t<wp::int32> var_17;
        wp::tuple_t<wp::int32> var_18;
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<31>, wp::tile_stride_t<1>>, true> var_19 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<31>,wp::tile_stride_t<1>,false>();
        const wp::str var_20 = "upper";
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<31>, wp::tile_stride_t<1>>, true> var_21 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<31>,wp::tile_stride_t<1>,false>();
        wp::array_t<wp::float32> var_22;
        wp::tuple_t<wp::int32> var_23;
        //---------
        // forward
        // def kernel(                                                                            <L 3131>
        // worldid, blk = wp.tid()                                                                <L 3141>
        builtin_tid2d(var_0, var_1);
        // start = block_dof[blk]                                                                 <L 3142>
        var_2 = wp::address(var_block_dof, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // L = wp.tile_reshape(                                                                   <L 3144>
        // wp.tile_load(L_in[worldid], shape=(block_area,), offset=(qLD_block_adr[start],)), (block_size, block_size)       <L 3145>
        var_5 = wp::view(var_L_in, var_0);
        var_7 = wp::tuple(var_6);
        var_8 = wp::address(var_qLD_block_adr, var_3);
        var_10 = wp::load(var_8);
        var_9 = wp::tuple(var_10);
        var_12 = wp::load(var_8);
        var_11 = wp::tile_load<wp::float32, true, false, 961>(var_5, var_12);
        var_14 = wp::tuple(var_13, var_13);
        var_15 = wp::tile_reshape<wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<31,31>, wp::tile_stride_t<31,1>>, false>>(var_11);
        // rhs = wp.tile_load(y[worldid], shape=(block_size,), offset=(start,))                   <L 3147>
        var_16 = wp::view(var_y, var_0);
        var_17 = wp::tuple(var_13);
        var_18 = wp::tuple(var_3);
        var_19 = wp::tile_load<wp::float32, true, false, 31>(var_16, var_3);
        // sol = wp.tile_cholesky_solve(L, rhs, fill_mode="upper")                                <L 3148>
        var_21 = wp::tile_cholesky_solve<true>(potrs_31_31_1_121_64_1_1_5_x_x_0, var_15, var_19, var_21);
        // wp.tile_store(x[worldid], sol, offset=(start,))                                        <L 3149>
        var_22 = wp::view(var_x, var_0);
        var_23 = wp::tuple(var_3);
        wp::tile_store<wp::float32, true, false>(var_22, var_3, var_21);
    }
}

