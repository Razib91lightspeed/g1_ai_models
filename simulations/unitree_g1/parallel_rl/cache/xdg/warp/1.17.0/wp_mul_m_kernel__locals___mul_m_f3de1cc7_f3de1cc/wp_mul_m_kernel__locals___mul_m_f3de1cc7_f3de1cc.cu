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



extern "C" __global__ void mul_m_kernel__locals___mul_m_9c0838e9_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_M_mulm_rowadr,
    wp::array_t<wp::int32> var_M_mulm_col,
    wp::array_t<wp::int32> var_M_mulm_madr,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<wp::float32> var_vec,
    wp::array_t<bool> var_skip,
    wp::array_t<wp::float32> var_res)
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
        const bool var_2 = true;
        bool* var_3;
        bool var_4;
        bool var_5;
        const wp::float32 var_6 = 0.0;
        wp::float32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        const wp::int32 var_11 = 1;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::range_t var_16;
        wp::int32 var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        wp::float32* var_24;
        wp::float32* var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        //---------
        // forward
        // def _mul_m(                                                                            <L 156>
        // worldid, dofid = wp.tid()                                                              <L 170>
        builtin_tid2d(var_0, var_1);
        // if wp.static(check_skip):                                                              <L 172>
        // if skip[worldid]:                                                                      <L 173>
        var_3 = wp::address(var_skip, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // return                                                                             <L 174>
            return;
        }
        var_5 = wp::load(var_3);
        // acc = float(0.0)                                                                       <L 177>
        var_7 = wp::float(var_6);
        // start = M_mulm_rowadr[dofid]                                                           <L 178>
        var_8 = wp::address(var_M_mulm_rowadr, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // end = M_mulm_rowadr[dofid + 1]                                                         <L 179>
        var_12 = wp::add(var_1, var_11);
        var_13 = wp::address(var_M_mulm_rowadr, var_12);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // for k in range(start, end):                                                            <L 180>
        var_16 = wp::range(var_9, var_14);
        start_for_1:;
            if (iter_cmp(var_16) == 0) goto end_for_1;
            var_17 = wp::iter_next(var_16);
            // col = M_mulm_col[k]                                                                <L 181>
            var_18 = wp::address(var_M_mulm_col, var_17);
            var_20 = wp::load(var_18);
            var_19 = wp::copy(var_20);
            // madr = M_mulm_madr[k]                                                              <L 182>
            var_21 = wp::address(var_M_mulm_madr, var_17);
            var_23 = wp::load(var_21);
            var_22 = wp::copy(var_23);
            // acc += M_in[worldid, madr] * vec[worldid, col]                                     <L 183>
            var_24 = wp::address(var_M_in, var_0, var_22);
            var_25 = wp::address(var_vec, var_0, var_19);
            var_27 = wp::load(var_24);
            var_28 = wp::load(var_25);
            var_26 = wp::mul(var_27, var_28);
            var_29 = wp::add(var_7, var_26);
            wp::assign(var_7, var_29);
            goto start_for_1;
        end_for_1:;
        // res[worldid, dofid] = acc                                                              <L 185>
        wp::array_store(var_res, var_0, var_1, var_7);
}



extern "C" __global__ void mul_m_kernel__locals___mul_m_9c0838e9_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_M_mulm_rowadr,
    wp::array_t<wp::int32> var_M_mulm_col,
    wp::array_t<wp::int32> var_M_mulm_madr,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<wp::float32> var_vec,
    wp::array_t<bool> var_skip,
    wp::array_t<wp::float32> var_res,
    wp::array_t<wp::int32> adj_M_mulm_rowadr,
    wp::array_t<wp::int32> adj_M_mulm_col,
    wp::array_t<wp::int32> adj_M_mulm_madr,
    wp::array_t<wp::float32> adj_M_in,
    wp::array_t<wp::float32> adj_vec,
    wp::array_t<bool> adj_skip,
    wp::array_t<wp::float32> adj_res)
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
        const bool var_2 = true;
        bool* var_3;
        bool var_4;
        bool var_5;
        const wp::float32 var_6 = 0.0;
        wp::float32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        const wp::int32 var_11 = 1;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::range_t var_16;
        wp::int32 var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        wp::float32* var_24;
        wp::float32* var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::range_t adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::float32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::float32 adj_29 = {};
        //---------
        // forward
        // def _mul_m(                                                                            <L 156>
        // worldid, dofid = wp.tid()                                                              <L 170>
        builtin_tid2d(var_0, var_1);
        // if wp.static(check_skip):                                                              <L 172>
        // if skip[worldid]:                                                                      <L 173>
        var_3 = wp::address(var_skip, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // return                                                                             <L 174>
            goto label0;
        }
        var_5 = wp::load(var_3);
        // acc = float(0.0)                                                                       <L 177>
        var_7 = wp::float(var_6);
        // start = M_mulm_rowadr[dofid]                                                           <L 178>
        var_8 = wp::address(var_M_mulm_rowadr, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // end = M_mulm_rowadr[dofid + 1]                                                         <L 179>
        var_12 = wp::add(var_1, var_11);
        var_13 = wp::address(var_M_mulm_rowadr, var_12);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // for k in range(start, end):                                                            <L 180>
        var_16 = wp::range(var_9, var_14);
        // res[worldid, dofid] = acc                                                              <L 185>
        // wp::array_store(var_res, var_0, var_1, var_7);
        //---------
        // reverse
        wp::adj_array_store(var_res, var_0, var_1, var_7, adj_res, adj_0, adj_1, adj_7);
        // adj: res[worldid, dofid] = acc                                                         <L 185>
        var_16 = wp::iter_reverse(var_16);
        start_for_1:;
            if (iter_cmp(var_16) == 0) goto end_for_1;
            var_17 = wp::iter_next(var_16);
        	adj_18 = {};
        	adj_19 = {};
        	adj_20 = {};
        	adj_21 = {};
        	adj_22 = {};
        	adj_23 = {};
        	adj_24 = {};
        	adj_25 = {};
        	adj_26 = {};
        	adj_27 = {};
        	adj_28 = {};
        	adj_29 = {};
            // col = M_mulm_col[k]                                                                <L 181>
            var_18 = wp::address(var_M_mulm_col, var_17);
            var_20 = wp::load(var_18);
            var_19 = wp::copy(var_20);
            // madr = M_mulm_madr[k]                                                              <L 182>
            var_21 = wp::address(var_M_mulm_madr, var_17);
            var_23 = wp::load(var_21);
            var_22 = wp::copy(var_23);
            // acc += M_in[worldid, madr] * vec[worldid, col]                                     <L 183>
            var_24 = wp::address(var_M_in, var_0, var_22);
            var_25 = wp::address(var_vec, var_0, var_19);
            var_27 = wp::load(var_24);
            var_28 = wp::load(var_25);
            var_26 = wp::mul(var_27, var_28);
            var_29 = wp::add(var_7, var_26);
            wp::assign(var_7, var_29);
            wp::adj_assign(var_7, var_29, adj_7, adj_29);
            wp::adj_add(var_7, var_26, adj_7, adj_26, adj_29);
            wp::adj_mul(var_27, var_28, adj_24, adj_25, adj_26);
            wp::adj_address(var_vec, var_0, var_19, adj_vec, adj_0, adj_19, adj_25);
            wp::adj_address(var_M_in, var_0, var_22, adj_M_in, adj_0, adj_22, adj_24);
            // adj: acc += M_in[worldid, madr] * vec[worldid, col]                                <L 183>
            wp::adj_copy(var_23, adj_21, adj_22);
            wp::adj_address(var_M_mulm_madr, var_17, adj_M_mulm_madr, adj_17, adj_21);
            // adj: madr = M_mulm_madr[k]                                                         <L 182>
            wp::adj_copy(var_20, adj_18, adj_19);
            wp::adj_address(var_M_mulm_col, var_17, adj_M_mulm_col, adj_17, adj_18);
            // adj: col = M_mulm_col[k]                                                           <L 181>
        	goto start_for_1;
        end_for_1:;
        // adj: for k in range(start, end):                                                       <L 180>
        wp::adj_copy(var_15, adj_13, adj_14);
        wp::adj_address(var_M_mulm_rowadr, var_12, adj_M_mulm_rowadr, adj_12, adj_13);
        wp::adj_add(var_1, var_11, adj_1, adj_11, adj_12);
        // adj: end = M_mulm_rowadr[dofid + 1]                                                    <L 179>
        wp::adj_copy(var_10, adj_8, adj_9);
        wp::adj_address(var_M_mulm_rowadr, var_1, adj_M_mulm_rowadr, adj_1, adj_8);
        // adj: start = M_mulm_rowadr[dofid]                                                      <L 178>
        wp::adj_float(var_6, adj_6, adj_7);
        // adj: acc = float(0.0)                                                                  <L 177>
        if (var_5) {
            label0:;
            // adj: return                                                                        <L 174>
        }
        wp::adj_address(var_skip, var_0, adj_skip, adj_0, adj_3);
        // adj: if skip[worldid]:                                                                 <L 173>
        // adj: if wp.static(check_skip):                                                         <L 172>
        // adj: worldid, dofid = wp.tid()                                                         <L 170>
        // adj: def _mul_m(                                                                       <L 156>
        return;
}

