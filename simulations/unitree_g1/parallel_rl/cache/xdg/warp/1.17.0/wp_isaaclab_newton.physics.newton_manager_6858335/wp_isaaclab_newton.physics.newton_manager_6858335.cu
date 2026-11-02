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



extern "C" __global__ void _or_world_reset_mask_from_mask_1e801e44_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_world_mask)
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
        bool* var_1;
        bool var_2;
        const bool var_3 = true;
        bool var_4;
        //---------
        // forward
        // def _or_world_reset_mask_from_mask(env_mask: wp.array(dtype=wp.bool), world_mask: wp.array(dtype=wp.bool)):       <L 140>
        // world = wp.tid()                                                                       <L 142>
        var_0 = builtin_tid1d();
        // if env_mask[world]:                                                                    <L 143>
        var_1 = wp::address(var_env_mask, var_0);
        var_2 = wp::load(var_1);
        if (var_2) {
            // world_mask[world] = True                                                           <L 144>
            wp::array_store(var_world_mask, var_0, var_3);
        }
        var_4 = wp::load(var_1);
    }
}



extern "C" __global__ void _scatter_reset_masks_from_ids_2405962b_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_articulation_ids,
    wp::array_t<bool> var_world_mask,
    wp::array_t<bool> var_fk_mask)
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
        const bool var_5 = true;
        const bool var_6 = true;
        wp::int32* var_7;
        wp::int32 var_8;
        //---------
        // forward
        // def _scatter_reset_masks_from_ids(                                                     <L 1>
        // i, arti = wp.tid()                                                                     <L 8>
        builtin_tid2d(var_0, var_1);
        // world = wp.int32(env_ids[i])                                                           <L 9>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // world_mask[world] = True                                                               <L 10>
        wp::array_store(var_world_mask, var_3, var_5);
        // fk_mask[articulation_ids[world, arti]] = True                                          <L 11>
        var_7 = wp::address(var_articulation_ids, var_3, var_1);
        var_8 = wp::load(var_7);
        wp::array_store(var_fk_mask, var_8, var_6);
    }
}



extern "C" __global__ void _or_reset_masks_from_mask_38791512_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<bool> var_env_mask,
    wp::array_t<wp::int32> var_articulation_ids,
    wp::array_t<bool> var_world_mask,
    wp::array_t<bool> var_fk_mask)
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
        bool* var_2;
        bool var_3;
        const bool var_4 = true;
        const bool var_5 = true;
        wp::int32* var_6;
        wp::int32 var_7;
        bool var_8;
        //---------
        // forward
        // def _or_reset_masks_from_mask(                                                         <L 104>
        // world, arti = wp.tid()                                                                 <L 111>
        builtin_tid2d(var_0, var_1);
        // if env_mask[world]:                                                                    <L 112>
        var_2 = wp::address(var_env_mask, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // world_mask[world] = True                                                           <L 113>
            wp::array_store(var_world_mask, var_0, var_4);
            // fk_mask[articulation_ids[world, arti]] = True                                      <L 114>
            var_6 = wp::address(var_articulation_ids, var_0, var_1);
            var_7 = wp::load(var_6);
            wp::array_store(var_fk_mask, var_7, var_5);
        }
        var_8 = wp::load(var_2);
    }
}



extern "C" __global__ void _scatter_world_reset_mask_from_ids_5f7037bd_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<bool> var_world_mask)
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
        const bool var_0 = true;
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        //---------
        // forward
        // def _scatter_world_reset_mask_from_ids(env_ids: wp.array(dtype=wp.int32), world_mask: wp.array(dtype=wp.bool)):       <L 148>
        // world_mask[env_ids[wp.tid()]] = True                                                   <L 150>
        var_1 = builtin_tid1d();
        var_2 = wp::address(var_env_ids, var_1);
        var_3 = wp::load(var_2);
        wp::array_store(var_world_mask, var_3, var_0);
    }
}



extern "C" __global__ void _scatter_reset_masks_from_ids_8e56df0c_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_articulation_ids,
    wp::array_t<bool> var_world_mask,
    wp::array_t<bool> var_fk_mask)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        const bool var_5 = true;
        const bool var_6 = true;
        wp::int32* var_7;
        wp::int32 var_8;
        //---------
        // forward
        // def _scatter_reset_masks_from_ids(                                                     <L 1>
        // i, arti = wp.tid()                                                                     <L 8>
        builtin_tid2d(var_0, var_1);
        // world = wp.int32(env_ids[i])                                                           <L 9>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // world_mask[world] = True                                                               <L 10>
        wp::array_store(var_world_mask, var_3, var_5);
        // fk_mask[articulation_ids[world, arti]] = True                                          <L 11>
        var_7 = wp::address(var_articulation_ids, var_3, var_1);
        var_8 = wp::load(var_7);
        wp::array_store(var_fk_mask, var_8, var_6);
    }
}

