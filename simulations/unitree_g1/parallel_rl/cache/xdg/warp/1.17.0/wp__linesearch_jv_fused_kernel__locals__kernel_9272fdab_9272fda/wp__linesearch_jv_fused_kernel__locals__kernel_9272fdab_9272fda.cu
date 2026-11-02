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



extern "C" __global__ void _linesearch_jv_fused_kernel__locals__kernel_8b1c4da6_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::array_t<wp::int32> var_efc_J_colind_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::int32> var_dof_cdof_in,
    wp::array_t<wp::float32> var_ctx_search_in,
    wp::array_t<bool> var_skip_in,
    wp::array_t<wp::float32> var_ctx_jv_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32 var_2;
        wp::int32* var_3;
        bool var_4;
        wp::int32 var_5;
        bool* var_6;
        bool var_7;
        bool var_8;
        const wp::float32 var_9 = 0.0;
        wp::float32 var_10;
        const bool var_11 = true;
        const bool var_12 = true;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::range_t var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        const wp::int32 var_22 = 0;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        const bool var_26 = false;
        const wp::int32 var_27 = 0;
        wp::float32* var_28;
        wp::float32* var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        //---------
        // forward
        // def kernel(                                                                            <L 1441>
        // worldid, efcid, dofstart = wp.tid()                                                    <L 1455>
        builtin_tid3d(var_0, var_1, var_2);
        // if efcid >= nefc_in[worldid]:                                                          <L 1457>
        var_3 = wp::address(var_nefc_in, var_0);
        var_5 = wp::load(var_3);
        var_4 = (var_1 >= var_5);
        if (var_4) {
            // return                                                                             <L 1458>
            return;
        }
        // if skip_in[worldid]:                                                                   <L 1460>
        var_6 = wp::address(var_skip_in, var_0);
        var_7 = wp::load(var_6);
        if (var_7) {
            // return                                                                             <L 1461>
            return;
        }
        var_8 = wp::load(var_6);
        // jv_out = float(0.0)                                                                    <L 1463>
        var_10 = wp::float(var_9);
        // if wp.static(dofs_per_thread >= nv):                                                   <L 1465>
        // if wp.static(is_sparse):                                                               <L 1466>
        // rownnz = efc_J_rownnz_in[worldid, efcid]                                               <L 1468>
        var_13 = wp::address(var_efc_J_rownnz_in, var_0, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // rowadr = efc_J_rowadr_in[worldid, efcid]                                               <L 1469>
        var_16 = wp::address(var_efc_J_rowadr_in, var_0, var_1);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // for k in range(rownnz):                                                                <L 1470>
        var_19 = wp::range(var_14);
        start_for_2:;
            if (iter_cmp(var_19) == 0) goto end_for_2;
            var_20 = wp::iter_next(var_19);
            // sparseid = rowadr + k                                                              <L 1471>
            var_21 = wp::add(var_17, var_20);
            // colind = efc_J_colind_in[worldid, 0, sparseid]                                     <L 1472>
            var_23 = wp::address(var_efc_J_colind_in, var_0, var_22, var_21);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // if wp.static(COMPACT):                                                             <L 1473>
            // jv_out += efc_J_in[worldid, 0, sparseid] * ctx_search_in[worldid, colind]          <L 1477>
            var_28 = wp::address(var_efc_J_in, var_0, var_27, var_21);
            var_29 = wp::address(var_ctx_search_in, var_0, var_24);
            var_31 = wp::load(var_28);
            var_32 = wp::load(var_29);
            var_30 = wp::mul(var_31, var_32);
            var_33 = wp::add(var_10, var_30);
            wp::assign(var_10, var_33);
            goto start_for_2;
        end_for_2:;
        // ctx_jv_out[worldid, efcid] = jv_out                                                    <L 1481>
        wp::array_store(var_ctx_jv_out, var_0, var_1, var_10);
}

