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



extern "C" __global__ void set_articulation_attribute_3d_kernel_006f4357_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::int32> var_values,
    wp::indexedarray_t<wp::int32> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_006f4357_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::int32> var_values,
    wp::indexedarray_t<wp::int32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::int32> adj_values,
    wp::array_t<wp::int32> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_3);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_00994c3b_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::int32> var_values,
    wp::array_t<wp::int32> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_00994c3b_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::int32> var_values,
    wp::array_t<wp::int32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::int32> adj_values,
    wp::array_t<wp::int32> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_4);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_01732a08_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_values,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::vec_t<6, wp::float32>* var_5;
        wp::vec_t<6, wp::float32> var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_01732a08_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_values,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_values,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::vec_t<6, wp::float32>* var_5;
        wp::vec_t<6, wp::float32> var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::vec_t<6, wp::float32> adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_3);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_04741c08_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::float32> var_values,
    wp::indexedarray_t<wp::float32> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_04741c08_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::float32> var_values,
    wp::indexedarray_t<wp::float32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::float32> adj_values,
    wp::array_t<wp::float32> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_4);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_05ce2a8e_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_values,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::vec_t<6, wp::float32>* var_6;
        wp::vec_t<6, wp::float32> var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_05ce2a8e_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_values,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_values,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::vec_t<6, wp::float32>* var_6;
        wp::vec_t<6, wp::float32> var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        wp::vec_t<6, wp::float32> adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_4);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_0a2d285b_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::int32> var_values,
    wp::indexedarray_t<wp::int32> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_0a2d285b_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::int32> var_values,
    wp::indexedarray_t<wp::int32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::int32> adj_values,
    wp::array_t<wp::int32> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_3);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void _gather_indexed_3d_kernel_0b9b0b84_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_src,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::transform_t<wp::float32>> var_dst)
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
        wp::int32 var_2;
        wp::int32* var_3;
        wp::transform_t<wp::float32>* var_4;
        wp::int32 var_5;
        wp::transform_t<wp::float32> var_6;
        //---------
        // forward
        // def _gather_indexed_3d_kernel(                                                         <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // dst[i, j, k] = src[i, j, indices[k]]                                                   <L 7>
        var_3 = wp::address(var_indices, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::address(var_src, var_0, var_1, var_5);
        var_6 = wp::load(var_4);
        wp::array_store(var_dst, var_0, var_1, var_2, var_6);
    }
}



extern "C" __global__ void _gather_indexed_3d_kernel_0b9b0b84_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_src,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::transform_t<wp::float32>> var_dst,
    wp::array_t<wp::transform_t<wp::float32>> adj_src,
    wp::array_t<wp::int32> adj_indices,
    wp::array_t<wp::transform_t<wp::float32>> adj_dst)
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
        wp::int32 var_2;
        wp::int32* var_3;
        wp::transform_t<wp::float32>* var_4;
        wp::int32 var_5;
        wp::transform_t<wp::float32> var_6;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::transform_t<wp::float32> adj_4 = {};
        wp::int32 adj_5 = {};
        wp::transform_t<wp::float32> adj_6 = {};
        //---------
        // forward
        // def _gather_indexed_3d_kernel(                                                         <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // dst[i, j, k] = src[i, j, indices[k]]                                                   <L 7>
        var_3 = wp::address(var_indices, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::address(var_src, var_0, var_1, var_5);
        var_6 = wp::load(var_4);
        // wp::array_store(var_dst, var_0, var_1, var_2, var_6);
        //---------
        // reverse
        wp::adj_array_store(var_dst, var_0, var_1, var_2, var_6, adj_dst, adj_0, adj_1, adj_2, adj_4);
        wp::adj_address(var_src, var_0, var_1, var_5, adj_src, adj_0, adj_1, adj_3, adj_4);
        wp::adj_address(var_indices, var_2, adj_indices, adj_2, adj_3);
        // adj: dst[i, j, k] = src[i, j, indices[k]]                                              <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def _gather_indexed_3d_kernel(                                                    <L 1>
        continue;
    }
}



extern "C" __global__ void _gather_indexed_4d_kernel_0d24cf2e_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_src,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::vec_t<6, wp::float32>> var_dst)
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
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32* var_4;
        wp::vec_t<6, wp::float32>* var_5;
        wp::int32 var_6;
        wp::vec_t<6, wp::float32> var_7;
        //---------
        // forward
        // def _gather_indexed_4d_kernel(                                                         <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // dst[i, j, k, l] = src[i, j, indices[k], l]                                             <L 7>
        var_4 = wp::address(var_indices, var_2);
        var_6 = wp::load(var_4);
        var_5 = wp::address(var_src, var_0, var_1, var_6, var_3);
        var_7 = wp::load(var_5);
        wp::array_store(var_dst, var_0, var_1, var_2, var_3, var_7);
    }
}



extern "C" __global__ void _gather_indexed_4d_kernel_0d24cf2e_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_src,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::vec_t<6, wp::float32>> var_dst,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_src,
    wp::array_t<wp::int32> adj_indices,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_dst)
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
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32* var_4;
        wp::vec_t<6, wp::float32>* var_5;
        wp::int32 var_6;
        wp::vec_t<6, wp::float32> var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::vec_t<6, wp::float32> adj_5 = {};
        wp::int32 adj_6 = {};
        wp::vec_t<6, wp::float32> adj_7 = {};
        //---------
        // forward
        // def _gather_indexed_4d_kernel(                                                         <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // dst[i, j, k, l] = src[i, j, indices[k], l]                                             <L 7>
        var_4 = wp::address(var_indices, var_2);
        var_6 = wp::load(var_4);
        var_5 = wp::address(var_src, var_0, var_1, var_6, var_3);
        var_7 = wp::load(var_5);
        // wp::array_store(var_dst, var_0, var_1, var_2, var_3, var_7);
        //---------
        // reverse
        wp::adj_array_store(var_dst, var_0, var_1, var_2, var_3, var_7, adj_dst, adj_0, adj_1, adj_2, adj_3, adj_5);
        wp::adj_address(var_src, var_0, var_1, var_6, var_3, adj_src, adj_0, adj_1, adj_4, adj_3, adj_5);
        wp::adj_address(var_indices, var_2, adj_indices, adj_2, adj_4);
        // adj: dst[i, j, k, l] = src[i, j, indices[k], l]                                        <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def _gather_indexed_4d_kernel(                                                    <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_17962656_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::int32> var_values,
    wp::array_t<wp::int32> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_17962656_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::int32> var_values,
    wp::array_t<wp::int32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::int32> adj_values,
    wp::array_t<wp::int32> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_4);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_1cdefa79_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_values,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::vec_t<6, wp::float32>* var_6;
        wp::vec_t<6, wp::float32> var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_1cdefa79_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_values,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_values,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::vec_t<6, wp::float32>* var_6;
        wp::vec_t<6, wp::float32> var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        wp::vec_t<6, wp::float32> adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_4);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_1fab217d_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_values,
    wp::array_t<wp::transform_t<wp::float32>> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::transform_t<wp::float32>* var_5;
        wp::transform_t<wp::float32> var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_1fab217d_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_values,
    wp::array_t<wp::transform_t<wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_values,
    wp::array_t<wp::transform_t<wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::transform_t<wp::float32>* var_5;
        wp::transform_t<wp::float32> var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::transform_t<wp::float32> adj_5 = {};
        wp::transform_t<wp::float32> adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_3);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_1ffbf2c6_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_values,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::vec_t<6, wp::float32>* var_6;
        wp::vec_t<6, wp::float32> var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_1ffbf2c6_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_values,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_values,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::vec_t<6, wp::float32>* var_6;
        wp::vec_t<6, wp::float32> var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        wp::vec_t<6, wp::float32> adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_4);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_28335d73_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_values,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::vec_t<6, wp::float32>* var_5;
        wp::vec_t<6, wp::float32> var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_28335d73_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_values,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_values,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::vec_t<6, wp::float32>* var_5;
        wp::vec_t<6, wp::float32> var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::vec_t<6, wp::float32> adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_3);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_2c446cd6_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::int32> var_values,
    wp::array_t<wp::int32> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_2c446cd6_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::int32> var_values,
    wp::array_t<wp::int32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::int32> adj_values,
    wp::array_t<wp::int32> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_3);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_2c73de48_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::int32> var_values,
    wp::indexedarray_t<wp::int32> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_2c73de48_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::int32> var_values,
    wp::indexedarray_t<wp::int32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::int32> adj_values,
    wp::array_t<wp::int32> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_3);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_3019137c_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::int32> var_values,
    wp::array_t<wp::int32> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_3019137c_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::int32> var_values,
    wp::array_t<wp::int32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::int32> adj_values,
    wp::array_t<wp::int32> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_3);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_318ab171_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::int32> var_values,
    wp::indexedarray_t<wp::int32> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_318ab171_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::int32> var_values,
    wp::indexedarray_t<wp::int32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::int32> adj_values,
    wp::array_t<wp::int32> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_3);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_36fa1461_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_values,
    wp::array_t<wp::vec_t<6, wp::float32>> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::vec_t<6, wp::float32>* var_5;
        wp::vec_t<6, wp::float32> var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_36fa1461_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_values,
    wp::array_t<wp::vec_t<6, wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_values,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::vec_t<6, wp::float32>* var_5;
        wp::vec_t<6, wp::float32> var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::vec_t<6, wp::float32> adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_3);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_3a9a4045_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_values,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::transform_t<wp::float32>* var_6;
        wp::transform_t<wp::float32> var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_3a9a4045_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_values,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_values,
    wp::array_t<wp::transform_t<wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::transform_t<wp::float32>* var_6;
        wp::transform_t<wp::float32> var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::transform_t<wp::float32> adj_6 = {};
        wp::transform_t<wp::float32> adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_4);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_3fb68b00_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::int32> var_values,
    wp::array_t<wp::int32> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_3fb68b00_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::int32> var_values,
    wp::array_t<wp::int32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::int32> adj_values,
    wp::array_t<wp::int32> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_4);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_4640421e_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::float32> var_values,
    wp::array_t<wp::float32> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_4640421e_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::float32> var_values,
    wp::array_t<wp::float32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::float32> adj_values,
    wp::array_t<wp::float32> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_4);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_4c162086_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_values,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::transform_t<wp::float32>* var_5;
        wp::transform_t<wp::float32> var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_4c162086_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_values,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_values,
    wp::array_t<wp::transform_t<wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::transform_t<wp::float32>* var_5;
        wp::transform_t<wp::float32> var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::transform_t<wp::float32> adj_5 = {};
        wp::transform_t<wp::float32> adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_3);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_model_articulation_mask_kernel_4dc4c10c_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<bool> var_world_arti_mask,
    wp::array_t<wp::int32> var_view_to_model_map,
    wp::array_t<bool> var_model_articulation_mask)
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
        wp::int32* var_5;
        wp::int32 var_6;
        bool var_7;
        //---------
        // forward
        // def set_model_articulation_mask_kernel(                                                <L 35>
        // world, arti = wp.tid()                                                                 <L 43>
        builtin_tid2d(var_0, var_1);
        // if world_arti_mask[world, arti]:                                                       <L 44>
        var_2 = wp::address(var_world_arti_mask, var_0, var_1);
        var_3 = wp::load(var_2);
        if (var_3) {
            // model_articulation_mask[view_to_model_map[world, arti]] = True                     <L 45>
            var_5 = wp::address(var_view_to_model_map, var_0, var_1);
            var_6 = wp::load(var_5);
            wp::array_store(var_model_articulation_mask, var_6, var_4);
        }
        var_7 = wp::load(var_2);
    }
}



extern "C" __global__ void set_model_articulation_mask_kernel_4dc4c10c_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<bool> var_world_arti_mask,
    wp::array_t<wp::int32> var_view_to_model_map,
    wp::array_t<bool> var_model_articulation_mask,
    wp::array_t<bool> adj_world_arti_mask,
    wp::array_t<wp::int32> adj_view_to_model_map,
    wp::array_t<bool> adj_model_articulation_mask)
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
        wp::int32* var_5;
        wp::int32 var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_model_articulation_mask_kernel(                                                <L 35>
        // world, arti = wp.tid()                                                                 <L 43>
        builtin_tid2d(var_0, var_1);
        // if world_arti_mask[world, arti]:                                                       <L 44>
        var_2 = wp::address(var_world_arti_mask, var_0, var_1);
        var_3 = wp::load(var_2);
        if (var_3) {
            // model_articulation_mask[view_to_model_map[world, arti]] = True                     <L 45>
            var_5 = wp::address(var_view_to_model_map, var_0, var_1);
            var_6 = wp::load(var_5);
            // wp::array_store(var_model_articulation_mask, var_6, var_4);
        }
        var_7 = wp::load(var_2);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_model_articulation_mask, var_6, var_4, adj_model_articulation_mask, adj_5, adj_4);
            wp::adj_address(var_view_to_model_map, var_0, var_1, adj_view_to_model_map, adj_0, adj_1, adj_5);
            // adj: model_articulation_mask[view_to_model_map[world, arti]] = True                <L 45>
        }
        wp::adj_address(var_world_arti_mask, var_0, var_1, adj_world_arti_mask, adj_0, adj_1, adj_2);
        // adj: if world_arti_mask[world, arti]:                                                  <L 44>
        // adj: world, arti = wp.tid()                                                            <L 43>
        // adj: def set_model_articulation_mask_kernel(                                           <L 35>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_5200d26a_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_values,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::transform_t<wp::float32>* var_5;
        wp::transform_t<wp::float32> var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_5200d26a_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_values,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_values,
    wp::array_t<wp::transform_t<wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::transform_t<wp::float32>* var_5;
        wp::transform_t<wp::float32> var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::transform_t<wp::float32> adj_5 = {};
        wp::transform_t<wp::float32> adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_3);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_533d47fc_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_values,
    wp::array_t<wp::vec_t<6, wp::float32>> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::vec_t<6, wp::float32>* var_6;
        wp::vec_t<6, wp::float32> var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_533d47fc_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_values,
    wp::array_t<wp::vec_t<6, wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_values,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::vec_t<6, wp::float32>* var_6;
        wp::vec_t<6, wp::float32> var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        wp::vec_t<6, wp::float32> adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_4);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_58dfdb20_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::float32> var_values,
    wp::indexedarray_t<wp::float32> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::float32* var_5;
        wp::float32 var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_58dfdb20_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::float32> var_values,
    wp::indexedarray_t<wp::float32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::float32> adj_values,
    wp::array_t<wp::float32> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::float32* var_5;
        wp::float32 var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_3);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void _gather_indexed_3d_kernel_5c69daf1_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::float32> var_src,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::float32> var_dst)
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
        wp::int32 var_2;
        wp::int32* var_3;
        wp::float32* var_4;
        wp::int32 var_5;
        wp::float32 var_6;
        //---------
        // forward
        // def _gather_indexed_3d_kernel(                                                         <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // dst[i, j, k] = src[i, j, indices[k]]                                                   <L 7>
        var_3 = wp::address(var_indices, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::address(var_src, var_0, var_1, var_5);
        var_6 = wp::load(var_4);
        wp::array_store(var_dst, var_0, var_1, var_2, var_6);
    }
}



extern "C" __global__ void _gather_indexed_3d_kernel_5c69daf1_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::float32> var_src,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::float32> var_dst,
    wp::array_t<wp::float32> adj_src,
    wp::array_t<wp::int32> adj_indices,
    wp::array_t<wp::float32> adj_dst)
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
        wp::int32 var_2;
        wp::int32* var_3;
        wp::float32* var_4;
        wp::int32 var_5;
        wp::float32 var_6;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::float32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::float32 adj_6 = {};
        //---------
        // forward
        // def _gather_indexed_3d_kernel(                                                         <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // dst[i, j, k] = src[i, j, indices[k]]                                                   <L 7>
        var_3 = wp::address(var_indices, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::address(var_src, var_0, var_1, var_5);
        var_6 = wp::load(var_4);
        // wp::array_store(var_dst, var_0, var_1, var_2, var_6);
        //---------
        // reverse
        wp::adj_array_store(var_dst, var_0, var_1, var_2, var_6, adj_dst, adj_0, adj_1, adj_2, adj_4);
        wp::adj_address(var_src, var_0, var_1, var_5, adj_src, adj_0, adj_1, adj_3, adj_4);
        wp::adj_address(var_indices, var_2, adj_indices, adj_2, adj_3);
        // adj: dst[i, j, k] = src[i, j, indices[k]]                                              <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def _gather_indexed_3d_kernel(                                                    <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_5d553ce5_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_values,
    wp::array_t<wp::transform_t<wp::float32>> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::transform_t<wp::float32>* var_6;
        wp::transform_t<wp::float32> var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_5d553ce5_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_values,
    wp::array_t<wp::transform_t<wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_values,
    wp::array_t<wp::transform_t<wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::transform_t<wp::float32>* var_6;
        wp::transform_t<wp::float32> var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::transform_t<wp::float32> adj_6 = {};
        wp::transform_t<wp::float32> adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_4);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_5f68bb29_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::int32> var_values,
    wp::array_t<wp::int32> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_5f68bb29_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::int32> var_values,
    wp::array_t<wp::int32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::int32> adj_values,
    wp::array_t<wp::int32> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_4);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_64638d25_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_values,
    wp::array_t<wp::vec_t<6, wp::float32>> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::vec_t<6, wp::float32>* var_6;
        wp::vec_t<6, wp::float32> var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_64638d25_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_values,
    wp::array_t<wp::vec_t<6, wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_values,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::vec_t<6, wp::float32>* var_6;
        wp::vec_t<6, wp::float32> var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        wp::vec_t<6, wp::float32> adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_4);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_7090e890_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_values,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::vec_t<6, wp::float32>* var_5;
        wp::vec_t<6, wp::float32> var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_7090e890_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_values,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_values,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::vec_t<6, wp::float32>* var_5;
        wp::vec_t<6, wp::float32> var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::vec_t<6, wp::float32> adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_3);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_71ae6bae_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::float32> var_values,
    wp::indexedarray_t<wp::float32> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_71ae6bae_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::float32> var_values,
    wp::indexedarray_t<wp::float32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::float32> adj_values,
    wp::array_t<wp::float32> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_4);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_72da8708_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::float32> var_values,
    wp::indexedarray_t<wp::float32> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::float32* var_5;
        wp::float32 var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_72da8708_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::float32> var_values,
    wp::indexedarray_t<wp::float32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::float32> adj_values,
    wp::array_t<wp::float32> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::float32* var_5;
        wp::float32 var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_3);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_7343721f_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::float32> var_values,
    wp::array_t<wp::float32> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::float32* var_5;
        wp::float32 var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_7343721f_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::float32> var_values,
    wp::array_t<wp::float32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::float32> adj_values,
    wp::array_t<wp::float32> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::float32* var_5;
        wp::float32 var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_3);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void build_actuator_dof_mapping_indices_kernel_740cad9b_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::uint32> var_actuator_input_indices,
    wp::array_t<wp::int32> var_view_dof_indices,
    wp::int32 var_base_offset,
    wp::int32 var_stride_within_worlds,
    wp::int32 var_count_per_world,
    wp::int32 var_actuators_per_world,
    wp::int32 var_dofs_per_arti,
    wp::int32 var_dofs_per_world,
    wp::int32 var_num_worlds,
    wp::array_t<wp::int32> var_mapping)
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
        wp::int32 var_2;
        wp::uint32 var_3;
        wp::range_t var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::range_t var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        bool var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::range_t var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        //---------
        // forward
        // def build_actuator_dof_mapping_indices_kernel(                                         <L 258>
        // local_idx = wp.tid()  # 0 to actuators_per_world-1                                     <L 275>
        var_0 = builtin_tid1d();
        // global_dof = int(actuator_input_indices[local_idx])                                    <L 277>
        var_1 = wp::address(var_actuator_input_indices, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::int(var_3);
        // for arti_idx in range(count_per_world):                                                <L 279>
        var_4 = wp::range(var_count_per_world);
        start_for_0:;
            if (iter_cmp(var_4) == 0) goto end_for_0;
            var_5 = wp::iter_next(var_4);
            // arti_base = base_offset + arti_idx * stride_within_worlds                          <L 280>
            var_6 = wp::mul(var_5, var_stride_within_worlds);
            var_7 = wp::add(var_base_offset, var_6);
            // for i in range(dofs_per_arti):                                                     <L 281>
            var_8 = wp::range(var_dofs_per_arti);
            start_for_2:;
                if (iter_cmp(var_8) == 0) goto end_for_2;
                var_9 = wp::iter_next(var_8);
                // if arti_base + view_dof_indices[i] == global_dof:                              <L 283>
                var_10 = wp::address(var_view_dof_indices, var_9);
                var_12 = wp::load(var_10);
                var_11 = wp::add(var_7, var_12);
                var_13 = (var_11 == var_2);
                if (var_13) {
                    // view_local_pos = arti_idx * dofs_per_arti + i                              <L 284>
                    var_14 = wp::mul(var_5, var_dofs_per_arti);
                    var_15 = wp::add(var_14, var_9);
                    // for world_idx in range(num_worlds):                                        <L 287>
                    var_16 = wp::range(var_num_worlds);
                    start_for_4:;
                        if (iter_cmp(var_16) == 0) goto end_for_4;
                        var_17 = wp::iter_next(var_16);
                        // view_pos = world_idx * dofs_per_world + view_local_pos                 <L 288>
                        var_18 = wp::mul(var_17, var_dofs_per_world);
                        var_19 = wp::add(var_18, var_15);
                        // actuator_idx = world_idx * actuators_per_world + local_idx             <L 289>
                        var_20 = wp::mul(var_17, var_actuators_per_world);
                        var_21 = wp::add(var_20, var_0);
                        // mapping[view_pos] = actuator_idx                                       <L 290>
                        wp::array_store(var_mapping, var_19, var_21);
                        goto start_for_4;
                    end_for_4:;
                    // break                                                                      <L 291>
                    goto end_for_2;
                }
                goto start_for_2;
            end_for_2:;
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void build_actuator_dof_mapping_indices_kernel_740cad9b_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::uint32> var_actuator_input_indices,
    wp::array_t<wp::int32> var_view_dof_indices,
    wp::int32 var_base_offset,
    wp::int32 var_stride_within_worlds,
    wp::int32 var_count_per_world,
    wp::int32 var_actuators_per_world,
    wp::int32 var_dofs_per_arti,
    wp::int32 var_dofs_per_world,
    wp::int32 var_num_worlds,
    wp::array_t<wp::int32> var_mapping,
    wp::array_t<wp::uint32> adj_actuator_input_indices,
    wp::array_t<wp::int32> adj_view_dof_indices,
    wp::int32 adj_base_offset,
    wp::int32 adj_stride_within_worlds,
    wp::int32 adj_count_per_world,
    wp::int32 adj_actuators_per_world,
    wp::int32 adj_dofs_per_arti,
    wp::int32 adj_dofs_per_world,
    wp::int32 adj_num_worlds,
    wp::array_t<wp::int32> adj_mapping)
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
        wp::int32 var_2;
        wp::uint32 var_3;
        wp::range_t var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::range_t var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        bool var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::range_t var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::uint32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::uint32 adj_3 = {};
        wp::range_t adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::range_t adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        bool adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::range_t adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        //---------
        // forward
        // def build_actuator_dof_mapping_indices_kernel(                                         <L 258>
        // local_idx = wp.tid()  # 0 to actuators_per_world-1                                     <L 275>
        var_0 = builtin_tid1d();
        // global_dof = int(actuator_input_indices[local_idx])                                    <L 277>
        var_1 = wp::address(var_actuator_input_indices, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::int(var_3);
        // for arti_idx in range(count_per_world):                                                <L 279>
        var_4 = wp::range(var_count_per_world);
        //---------
        // reverse
        var_4 = wp::iter_reverse(var_4);
        start_for_0:;
            if (iter_cmp(var_4) == 0) goto end_for_0;
            var_5 = wp::iter_next(var_4);
        	adj_6 = {};
        	adj_7 = {};
        	adj_8 = {};
            // arti_base = base_offset + arti_idx * stride_within_worlds                          <L 280>
            var_6 = wp::mul(var_5, var_stride_within_worlds);
            var_7 = wp::add(var_base_offset, var_6);
            // for i in range(dofs_per_arti):                                                     <L 281>
            var_8 = wp::range(var_dofs_per_arti);
            var_8 = wp::iter_reverse(var_8);
            start_for_2:;
                if (iter_cmp(var_8) == 0) goto end_for_2;
                var_9 = wp::iter_next(var_8);
            	adj_10 = {};
            	adj_11 = {};
            	adj_12 = {};
            	adj_13 = {};
            	adj_14 = {};
            	adj_15 = {};
            	adj_16 = {};
                // if arti_base + view_dof_indices[i] == global_dof:                              <L 283>
                var_10 = wp::address(var_view_dof_indices, var_9);
                var_12 = wp::load(var_10);
                var_11 = wp::add(var_7, var_12);
                var_13 = (var_11 == var_2);
                if (var_13) {
                    // view_local_pos = arti_idx * dofs_per_arti + i                              <L 284>
                    var_14 = wp::mul(var_5, var_dofs_per_arti);
                    var_15 = wp::add(var_14, var_9);
                    // for world_idx in range(num_worlds):                                        <L 287>
                    var_16 = wp::range(var_num_worlds);
                    // break                                                                      <L 291>
                    goto end_for_2;
                }
                if (var_13) {
                    // adj: break                                                                 <L 291>
                    var_16 = wp::iter_reverse(var_16);
                    start_for_4:;
                        if (iter_cmp(var_16) == 0) goto end_for_4;
                        var_17 = wp::iter_next(var_16);
                    	adj_18 = {};
                    	adj_19 = {};
                    	adj_20 = {};
                    	adj_21 = {};
                        // view_pos = world_idx * dofs_per_world + view_local_pos                 <L 288>
                        var_18 = wp::mul(var_17, var_dofs_per_world);
                        var_19 = wp::add(var_18, var_15);
                        // actuator_idx = world_idx * actuators_per_world + local_idx             <L 289>
                        var_20 = wp::mul(var_17, var_actuators_per_world);
                        var_21 = wp::add(var_20, var_0);
                        // mapping[view_pos] = actuator_idx                                       <L 290>
                        // wp::array_store(var_mapping, var_19, var_21);
                        wp::adj_array_store(var_mapping, var_19, var_21, adj_mapping, adj_19, adj_21);
                        // adj: mapping[view_pos] = actuator_idx                                  <L 290>
                        wp::adj_add(var_20, var_0, adj_20, adj_0, adj_21);
                        wp::adj_mul(var_17, var_actuators_per_world, adj_17, adj_actuators_per_world, adj_20);
                        // adj: actuator_idx = world_idx * actuators_per_world + local_idx        <L 289>
                        wp::adj_add(var_18, var_15, adj_18, adj_15, adj_19);
                        wp::adj_mul(var_17, var_dofs_per_world, adj_17, adj_dofs_per_world, adj_18);
                        // adj: view_pos = world_idx * dofs_per_world + view_local_pos            <L 288>
                    	goto start_for_4;
                    end_for_4:;
                    // adj: for world_idx in range(num_worlds):                                   <L 287>
                    wp::adj_add(var_14, var_9, adj_14, adj_9, adj_15);
                    wp::adj_mul(var_5, var_dofs_per_arti, adj_5, adj_dofs_per_arti, adj_14);
                    // adj: view_local_pos = arti_idx * dofs_per_arti + i                         <L 284>
                }
                wp::adj_add(var_7, var_12, adj_7, adj_10, adj_11);
                wp::adj_address(var_view_dof_indices, var_9, adj_view_dof_indices, adj_9, adj_10);
                // adj: if arti_base + view_dof_indices[i] == global_dof:                         <L 283>
            	goto start_for_2;
            end_for_2:;
            // adj: for i in range(dofs_per_arti):                                                <L 281>
            wp::adj_add(var_base_offset, var_6, adj_base_offset, adj_6, adj_7);
            wp::adj_mul(var_5, var_stride_within_worlds, adj_5, adj_stride_within_worlds, adj_6);
            // adj: arti_base = base_offset + arti_idx * stride_within_worlds                     <L 280>
        	goto start_for_0;
        end_for_0:;
        // adj: for arti_idx in range(count_per_world):                                           <L 279>
        wp::adj_address(var_actuator_input_indices, var_0, adj_actuator_input_indices, adj_0, adj_1);
        // adj: global_dof = int(actuator_input_indices[local_idx])                               <L 277>
        // adj: local_idx = wp.tid()  # 0 to actuators_per_world-1                                <L 275>
        // adj: def build_actuator_dof_mapping_indices_kernel(                                    <L 258>
        continue;
    }
}



extern "C" __global__ void set_model_articulation_mask_per_world_kernel_7b6292eb_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<bool> var_world_mask,
    wp::array_t<wp::int32> var_view_to_model_map,
    wp::array_t<bool> var_model_articulation_mask)
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
        wp::int32* var_5;
        wp::int32 var_6;
        bool var_7;
        //---------
        // forward
        // def set_model_articulation_mask_per_world_kernel(                                      <L 49>
        // world, arti = wp.tid()                                                                 <L 57>
        builtin_tid2d(var_0, var_1);
        // if world_mask[world]:                                                                  <L 58>
        var_2 = wp::address(var_world_mask, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // model_articulation_mask[view_to_model_map[world, arti]] = True                     <L 59>
            var_5 = wp::address(var_view_to_model_map, var_0, var_1);
            var_6 = wp::load(var_5);
            wp::array_store(var_model_articulation_mask, var_6, var_4);
        }
        var_7 = wp::load(var_2);
    }
}



extern "C" __global__ void set_model_articulation_mask_per_world_kernel_7b6292eb_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<bool> var_world_mask,
    wp::array_t<wp::int32> var_view_to_model_map,
    wp::array_t<bool> var_model_articulation_mask,
    wp::array_t<bool> adj_world_mask,
    wp::array_t<wp::int32> adj_view_to_model_map,
    wp::array_t<bool> adj_model_articulation_mask)
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
        wp::int32* var_5;
        wp::int32 var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_model_articulation_mask_per_world_kernel(                                      <L 49>
        // world, arti = wp.tid()                                                                 <L 57>
        builtin_tid2d(var_0, var_1);
        // if world_mask[world]:                                                                  <L 58>
        var_2 = wp::address(var_world_mask, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // model_articulation_mask[view_to_model_map[world, arti]] = True                     <L 59>
            var_5 = wp::address(var_view_to_model_map, var_0, var_1);
            var_6 = wp::load(var_5);
            // wp::array_store(var_model_articulation_mask, var_6, var_4);
        }
        var_7 = wp::load(var_2);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_model_articulation_mask, var_6, var_4, adj_model_articulation_mask, adj_5, adj_4);
            wp::adj_address(var_view_to_model_map, var_0, var_1, adj_view_to_model_map, adj_0, adj_1, adj_5);
            // adj: model_articulation_mask[view_to_model_map[world, arti]] = True                <L 59>
        }
        wp::adj_address(var_world_mask, var_0, adj_world_mask, adj_0, adj_2);
        // adj: if world_mask[world]:                                                             <L 58>
        // adj: world, arti = wp.tid()                                                            <L 57>
        // adj: def set_model_articulation_mask_per_world_kernel(                                 <L 49>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_7c6ca662_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_values,
    wp::array_t<wp::vec_t<6, wp::float32>> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::vec_t<6, wp::float32>* var_6;
        wp::vec_t<6, wp::float32> var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_7c6ca662_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_values,
    wp::array_t<wp::vec_t<6, wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_values,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::vec_t<6, wp::float32>* var_6;
        wp::vec_t<6, wp::float32> var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        wp::vec_t<6, wp::float32> adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_4);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_7fec7a65_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::float32> var_values,
    wp::indexedarray_t<wp::float32> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::float32* var_5;
        wp::float32 var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_7fec7a65_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::float32> var_values,
    wp::indexedarray_t<wp::float32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::float32> adj_values,
    wp::array_t<wp::float32> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::float32* var_5;
        wp::float32 var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_3);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_856586dc_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_values,
    wp::array_t<wp::vec_t<6, wp::float32>> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::vec_t<6, wp::float32>* var_6;
        wp::vec_t<6, wp::float32> var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_856586dc_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_values,
    wp::array_t<wp::vec_t<6, wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_values,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::vec_t<6, wp::float32>* var_6;
        wp::vec_t<6, wp::float32> var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        wp::vec_t<6, wp::float32> adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_4);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_95d1473f_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::float32> var_values,
    wp::array_t<wp::float32> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::float32* var_5;
        wp::float32 var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_95d1473f_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::float32> var_values,
    wp::array_t<wp::float32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::float32> adj_values,
    wp::array_t<wp::float32> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::float32* var_5;
        wp::float32 var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_3);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_99052ce7_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_values,
    wp::array_t<wp::transform_t<wp::float32>> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::transform_t<wp::float32>* var_5;
        wp::transform_t<wp::float32> var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_99052ce7_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_values,
    wp::array_t<wp::transform_t<wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_values,
    wp::array_t<wp::transform_t<wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::transform_t<wp::float32>* var_5;
        wp::transform_t<wp::float32> var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::transform_t<wp::float32> adj_5 = {};
        wp::transform_t<wp::float32> adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_3);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_9bb40342_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_values,
    wp::array_t<wp::vec_t<6, wp::float32>> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::vec_t<6, wp::float32>* var_5;
        wp::vec_t<6, wp::float32> var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_9bb40342_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_values,
    wp::array_t<wp::vec_t<6, wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_values,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::vec_t<6, wp::float32>* var_5;
        wp::vec_t<6, wp::float32> var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::vec_t<6, wp::float32> adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_3);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_9c7ce31f_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::float32> var_values,
    wp::array_t<wp::float32> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::float32* var_5;
        wp::float32 var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_9c7ce31f_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::float32> var_values,
    wp::array_t<wp::float32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::float32> adj_values,
    wp::array_t<wp::float32> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::float32* var_5;
        wp::float32 var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_3);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_9d9ded2f_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::int32> var_values,
    wp::indexedarray_t<wp::int32> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_9d9ded2f_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::int32> var_values,
    wp::indexedarray_t<wp::int32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::int32> adj_values,
    wp::array_t<wp::int32> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_4);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void _gather_indexed_3d_kernel_9f4f1c16_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_src,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::vec_t<6, wp::float32>> var_dst)
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
        wp::int32 var_2;
        wp::int32* var_3;
        wp::vec_t<6, wp::float32>* var_4;
        wp::int32 var_5;
        wp::vec_t<6, wp::float32> var_6;
        //---------
        // forward
        // def _gather_indexed_3d_kernel(                                                         <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // dst[i, j, k] = src[i, j, indices[k]]                                                   <L 7>
        var_3 = wp::address(var_indices, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::address(var_src, var_0, var_1, var_5);
        var_6 = wp::load(var_4);
        wp::array_store(var_dst, var_0, var_1, var_2, var_6);
    }
}



extern "C" __global__ void _gather_indexed_3d_kernel_9f4f1c16_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_src,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::vec_t<6, wp::float32>> var_dst,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_src,
    wp::array_t<wp::int32> adj_indices,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_dst)
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
        wp::int32 var_2;
        wp::int32* var_3;
        wp::vec_t<6, wp::float32>* var_4;
        wp::int32 var_5;
        wp::vec_t<6, wp::float32> var_6;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::vec_t<6, wp::float32> adj_4 = {};
        wp::int32 adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        //---------
        // forward
        // def _gather_indexed_3d_kernel(                                                         <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // dst[i, j, k] = src[i, j, indices[k]]                                                   <L 7>
        var_3 = wp::address(var_indices, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::address(var_src, var_0, var_1, var_5);
        var_6 = wp::load(var_4);
        // wp::array_store(var_dst, var_0, var_1, var_2, var_6);
        //---------
        // reverse
        wp::adj_array_store(var_dst, var_0, var_1, var_2, var_6, adj_dst, adj_0, adj_1, adj_2, adj_4);
        wp::adj_address(var_src, var_0, var_1, var_5, adj_src, adj_0, adj_1, adj_3, adj_4);
        wp::adj_address(var_indices, var_2, adj_indices, adj_2, adj_3);
        // adj: dst[i, j, k] = src[i, j, indices[k]]                                              <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def _gather_indexed_3d_kernel(                                                    <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_a35a69f2_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::float32> var_values,
    wp::indexedarray_t<wp::float32> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_a35a69f2_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::float32> var_values,
    wp::indexedarray_t<wp::float32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::float32> adj_values,
    wp::array_t<wp::float32> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_4);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_a7961451_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_values,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::transform_t<wp::float32>* var_6;
        wp::transform_t<wp::float32> var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_a7961451_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_values,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_values,
    wp::array_t<wp::transform_t<wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::transform_t<wp::float32>* var_6;
        wp::transform_t<wp::float32> var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::transform_t<wp::float32> adj_6 = {};
        wp::transform_t<wp::float32> adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_4);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_a82a0762_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::float32> var_values,
    wp::indexedarray_t<wp::float32> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_a82a0762_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::float32> var_values,
    wp::indexedarray_t<wp::float32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::float32> adj_values,
    wp::array_t<wp::float32> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_4);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_a9d50e30_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::float32> var_values,
    wp::indexedarray_t<wp::float32> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::float32* var_5;
        wp::float32 var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_a9d50e30_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::float32> var_values,
    wp::indexedarray_t<wp::float32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::float32> adj_values,
    wp::array_t<wp::float32> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::float32* var_5;
        wp::float32 var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_3);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_acde1040_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::float32> var_values,
    wp::array_t<wp::float32> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::float32* var_5;
        wp::float32 var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_acde1040_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::float32> var_values,
    wp::array_t<wp::float32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::float32> adj_values,
    wp::array_t<wp::float32> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::float32* var_5;
        wp::float32 var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_3);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_b00d42fa_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::int32> var_values,
    wp::indexedarray_t<wp::int32> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_b00d42fa_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::int32> var_values,
    wp::indexedarray_t<wp::int32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::int32> adj_values,
    wp::array_t<wp::int32> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_4);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_b151199d_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_values,
    wp::array_t<wp::transform_t<wp::float32>> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::transform_t<wp::float32>* var_6;
        wp::transform_t<wp::float32> var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_b151199d_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_values,
    wp::array_t<wp::transform_t<wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_values,
    wp::array_t<wp::transform_t<wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::transform_t<wp::float32>* var_6;
        wp::transform_t<wp::float32> var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::transform_t<wp::float32> adj_6 = {};
        wp::transform_t<wp::float32> adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_4);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_b5cf8f8c_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::int32> var_values,
    wp::array_t<wp::int32> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_b5cf8f8c_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::int32> var_values,
    wp::array_t<wp::int32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::int32> adj_values,
    wp::array_t<wp::int32> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_3);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_b81fa072_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_values,
    wp::array_t<wp::vec_t<6, wp::float32>> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::vec_t<6, wp::float32>* var_5;
        wp::vec_t<6, wp::float32> var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_b81fa072_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_values,
    wp::array_t<wp::vec_t<6, wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_values,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::vec_t<6, wp::float32>* var_5;
        wp::vec_t<6, wp::float32> var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::vec_t<6, wp::float32> adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_3);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_bbddc7df_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_values,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::transform_t<wp::float32>* var_5;
        wp::transform_t<wp::float32> var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_bbddc7df_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_values,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_values,
    wp::array_t<wp::transform_t<wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::transform_t<wp::float32>* var_5;
        wp::transform_t<wp::float32> var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::transform_t<wp::float32> adj_5 = {};
        wp::transform_t<wp::float32> adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_3);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void _gather_indexed_4d_kernel_bcfa0a94_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_src,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::transform_t<wp::float32>> var_dst)
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
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32* var_4;
        wp::transform_t<wp::float32>* var_5;
        wp::int32 var_6;
        wp::transform_t<wp::float32> var_7;
        //---------
        // forward
        // def _gather_indexed_4d_kernel(                                                         <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // dst[i, j, k, l] = src[i, j, indices[k], l]                                             <L 7>
        var_4 = wp::address(var_indices, var_2);
        var_6 = wp::load(var_4);
        var_5 = wp::address(var_src, var_0, var_1, var_6, var_3);
        var_7 = wp::load(var_5);
        wp::array_store(var_dst, var_0, var_1, var_2, var_3, var_7);
    }
}



extern "C" __global__ void _gather_indexed_4d_kernel_bcfa0a94_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_src,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::transform_t<wp::float32>> var_dst,
    wp::array_t<wp::transform_t<wp::float32>> adj_src,
    wp::array_t<wp::int32> adj_indices,
    wp::array_t<wp::transform_t<wp::float32>> adj_dst)
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
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32* var_4;
        wp::transform_t<wp::float32>* var_5;
        wp::int32 var_6;
        wp::transform_t<wp::float32> var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::transform_t<wp::float32> adj_5 = {};
        wp::int32 adj_6 = {};
        wp::transform_t<wp::float32> adj_7 = {};
        //---------
        // forward
        // def _gather_indexed_4d_kernel(                                                         <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // dst[i, j, k, l] = src[i, j, indices[k], l]                                             <L 7>
        var_4 = wp::address(var_indices, var_2);
        var_6 = wp::load(var_4);
        var_5 = wp::address(var_src, var_0, var_1, var_6, var_3);
        var_7 = wp::load(var_5);
        // wp::array_store(var_dst, var_0, var_1, var_2, var_3, var_7);
        //---------
        // reverse
        wp::adj_array_store(var_dst, var_0, var_1, var_2, var_3, var_7, adj_dst, adj_0, adj_1, adj_2, adj_3, adj_5);
        wp::adj_address(var_src, var_0, var_1, var_6, var_3, adj_src, adj_0, adj_1, adj_4, adj_3, adj_5);
        wp::adj_address(var_indices, var_2, adj_indices, adj_2, adj_4);
        // adj: dst[i, j, k, l] = src[i, j, indices[k], l]                                        <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def _gather_indexed_4d_kernel(                                                    <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_becb1f3e_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_values,
    wp::array_t<wp::transform_t<wp::float32>> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::transform_t<wp::float32>* var_5;
        wp::transform_t<wp::float32> var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_becb1f3e_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_values,
    wp::array_t<wp::transform_t<wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_values,
    wp::array_t<wp::transform_t<wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::transform_t<wp::float32>* var_5;
        wp::transform_t<wp::float32> var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::transform_t<wp::float32> adj_5 = {};
        wp::transform_t<wp::float32> adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_3);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_cae328f7_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_values,
    wp::array_t<wp::transform_t<wp::float32>> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::transform_t<wp::float32>* var_6;
        wp::transform_t<wp::float32> var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_cae328f7_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_values,
    wp::array_t<wp::transform_t<wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_values,
    wp::array_t<wp::transform_t<wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::transform_t<wp::float32>* var_6;
        wp::transform_t<wp::float32> var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::transform_t<wp::float32> adj_6 = {};
        wp::transform_t<wp::float32> adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_4);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_cd3f016c_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_values,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::transform_t<wp::float32>* var_6;
        wp::transform_t<wp::float32> var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_cd3f016c_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_values,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_values,
    wp::array_t<wp::transform_t<wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::transform_t<wp::float32>* var_6;
        wp::transform_t<wp::float32> var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::transform_t<wp::float32> adj_6 = {};
        wp::transform_t<wp::float32> adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_4);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_cdd0f47d_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::int32> var_values,
    wp::indexedarray_t<wp::int32> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_cdd0f47d_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::int32> var_values,
    wp::indexedarray_t<wp::int32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::int32> adj_values,
    wp::array_t<wp::int32> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_4);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_d01bb931_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::float32> var_values,
    wp::array_t<wp::float32> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_per_world_kernel_d01bb931_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::float32> var_values,
    wp::array_t<wp::float32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::float32> adj_values,
    wp::array_t<wp::float32> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_per_world_kernel(                                    <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i]:                                                                       <L 7>
        var_4 = wp::address(var_view_mask, var_0);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_4);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void _gather_indexed_4d_kernel_d13c49a4_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<wp::float32> var_src,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::float32> var_dst)
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
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32* var_4;
        wp::float32* var_5;
        wp::int32 var_6;
        wp::float32 var_7;
        //---------
        // forward
        // def _gather_indexed_4d_kernel(                                                         <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // dst[i, j, k, l] = src[i, j, indices[k], l]                                             <L 7>
        var_4 = wp::address(var_indices, var_2);
        var_6 = wp::load(var_4);
        var_5 = wp::address(var_src, var_0, var_1, var_6, var_3);
        var_7 = wp::load(var_5);
        wp::array_store(var_dst, var_0, var_1, var_2, var_3, var_7);
    }
}



extern "C" __global__ void _gather_indexed_4d_kernel_d13c49a4_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<wp::float32> var_src,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::float32> var_dst,
    wp::array_t<wp::float32> adj_src,
    wp::array_t<wp::int32> adj_indices,
    wp::array_t<wp::float32> adj_dst)
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
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32* var_4;
        wp::float32* var_5;
        wp::int32 var_6;
        wp::float32 var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::float32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::float32 adj_7 = {};
        //---------
        // forward
        // def _gather_indexed_4d_kernel(                                                         <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // dst[i, j, k, l] = src[i, j, indices[k], l]                                             <L 7>
        var_4 = wp::address(var_indices, var_2);
        var_6 = wp::load(var_4);
        var_5 = wp::address(var_src, var_0, var_1, var_6, var_3);
        var_7 = wp::load(var_5);
        // wp::array_store(var_dst, var_0, var_1, var_2, var_3, var_7);
        //---------
        // reverse
        wp::adj_array_store(var_dst, var_0, var_1, var_2, var_3, var_7, adj_dst, adj_0, adj_1, adj_2, adj_3, adj_5);
        wp::adj_address(var_src, var_0, var_1, var_6, var_3, adj_src, adj_0, adj_1, adj_4, adj_3, adj_5);
        wp::adj_address(var_indices, var_2, adj_indices, adj_2, adj_4);
        // adj: dst[i, j, k, l] = src[i, j, indices[k], l]                                        <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def _gather_indexed_4d_kernel(                                                    <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_d402dc8d_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_values,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::transform_t<wp::float32>* var_5;
        wp::transform_t<wp::float32> var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_d402dc8d_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_values,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_values,
    wp::array_t<wp::transform_t<wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::transform_t<wp::float32>* var_5;
        wp::transform_t<wp::float32> var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::transform_t<wp::float32> adj_5 = {};
        wp::transform_t<wp::float32> adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_3);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_d56eba30_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_values,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::transform_t<wp::float32>* var_6;
        wp::transform_t<wp::float32> var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_d56eba30_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_values,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_values,
    wp::array_t<wp::transform_t<wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::transform_t<wp::float32>* var_6;
        wp::transform_t<wp::float32> var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::transform_t<wp::float32> adj_6 = {};
        wp::transform_t<wp::float32> adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_4);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_dd5effbb_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_values,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::vec_t<6, wp::float32>* var_6;
        wp::vec_t<6, wp::float32> var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_dd5effbb_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_values,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_values,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::vec_t<6, wp::float32>* var_6;
        wp::vec_t<6, wp::float32> var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        wp::vec_t<6, wp::float32> adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_4);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_de88c807_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::float32> var_values,
    wp::array_t<wp::float32> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_de88c807_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::float32> var_values,
    wp::array_t<wp::float32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::float32> adj_values,
    wp::array_t<wp::float32> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_4);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_deaafa15_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_values,
    wp::array_t<wp::transform_t<wp::float32>> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::transform_t<wp::float32>* var_5;
        wp::transform_t<wp::float32> var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_kernel_deaafa15_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::transform_t<wp::float32>> var_values,
    wp::array_t<wp::transform_t<wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_values,
    wp::array_t<wp::transform_t<wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::transform_t<wp::float32>* var_5;
        wp::transform_t<wp::float32> var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::transform_t<wp::float32> adj_5 = {};
        wp::transform_t<wp::float32> adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_kernel(                                              <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i, j]:                                                                    <L 7>
        var_3 = wp::address(var_view_mask, var_0, var_1);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_3);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_df274588_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::int32> var_values,
    wp::array_t<wp::int32> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_df274588_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::int32> var_values,
    wp::array_t<wp::int32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::int32> adj_values,
    wp::array_t<wp::int32> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_3);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_dfba9fd8_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_values,
    wp::array_t<wp::vec_t<6, wp::float32>> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::vec_t<6, wp::float32>* var_5;
        wp::vec_t<6, wp::float32> var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_dfba9fd8_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_values,
    wp::array_t<wp::vec_t<6, wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_values,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::vec_t<6, wp::float32>* var_5;
        wp::vec_t<6, wp::float32> var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::vec_t<6, wp::float32> adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_3);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_e4e10536_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::int32> var_values,
    wp::indexedarray_t<wp::int32> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_e4e10536_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::int32> var_values,
    wp::indexedarray_t<wp::int32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::int32> adj_values,
    wp::array_t<wp::int32> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_4);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_eaa399f8_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_values,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::vec_t<6, wp::float32>* var_5;
        wp::vec_t<6, wp::float32> var_6;
        bool var_7;
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
    }
}



extern "C" __global__ void set_articulation_attribute_3d_per_world_kernel_eaa399f8_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_values,
    wp::indexedarray_t<wp::vec_t<6, wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_values,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        wp::vec_t<6, wp::float32>* var_5;
        wp::vec_t<6, wp::float32> var_6;
        bool var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::vec_t<6, wp::float32> adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        bool adj_7 = {};
        //---------
        // forward
        // def set_articulation_attribute_3d_per_world_kernel(                                    <L 1>
        // i, j, k = wp.tid()                                                                     <L 6>
        builtin_tid3d(var_0, var_1, var_2);
        // if view_mask[i]:                                                                       <L 7>
        var_3 = wp::address(var_view_mask, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // attrib[i, j, k] = values[i, j, k]                                                  <L 8>
            var_5 = wp::address(var_values, var_0, var_1, var_2);
            var_6 = wp::load(var_5);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_6);
        }
        var_7 = wp::load(var_3);
        //---------
        // reverse
        if (var_7) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_6, adj_attrib, adj_0, adj_1, adj_2, adj_5);
            wp::adj_address(var_values, var_0, var_1, var_2, adj_values, adj_0, adj_1, adj_2, adj_5);
            // adj: attrib[i, j, k] = values[i, j, k]                                             <L 8>
        }
        wp::adj_address(var_view_mask, var_0, adj_view_mask, adj_0, adj_3);
        // adj: if view_mask[i]:                                                                  <L 7>
        // adj: i, j, k = wp.tid()                                                                <L 6>
        // adj: def set_articulation_attribute_3d_per_world_kernel(                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_eb655c1f_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_values,
    wp::array_t<wp::transform_t<wp::float32>> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::transform_t<wp::float32>* var_6;
        wp::transform_t<wp::float32> var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_eb655c1f_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_values,
    wp::array_t<wp::transform_t<wp::float32>> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_values,
    wp::array_t<wp::transform_t<wp::float32>> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::transform_t<wp::float32>* var_6;
        wp::transform_t<wp::float32> var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::transform_t<wp::float32> adj_6 = {};
        wp::transform_t<wp::float32> adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_4);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_f6e7d986_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::float32> var_values,
    wp::array_t<wp::float32> var_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        bool var_8;
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
    }
}



extern "C" __global__ void set_articulation_attribute_4d_kernel_f6e7d986_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<bool> var_view_mask,
    wp::indexedarray_t<wp::float32> var_values,
    wp::array_t<wp::float32> var_attrib,
    wp::array_t<bool> adj_view_mask,
    wp::array_t<wp::float32> adj_values,
    wp::array_t<wp::float32> adj_attrib)
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
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        bool var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        bool adj_8 = {};
        //---------
        // forward
        // def set_articulation_attribute_4d_kernel(                                              <L 1>
        // i, j, k, l = wp.tid()                                                                  <L 6>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // if view_mask[i, j]:                                                                    <L 7>
        var_4 = wp::address(var_view_mask, var_0, var_1);
        var_5 = wp::load(var_4);
        if (var_5) {
            // attrib[i, j, k, l] = values[i, j, k, l]                                            <L 8>
            var_6 = wp::address(var_values, var_0, var_1, var_2, var_3);
            var_7 = wp::load(var_6);
            // wp::array_store(var_attrib, var_0, var_1, var_2, var_3, var_7);
        }
        var_8 = wp::load(var_4);
        //---------
        // reverse
        if (var_8) {
            wp::adj_array_store(var_attrib, var_0, var_1, var_2, var_3, var_7, adj_attrib, adj_0, adj_1, adj_2, adj_3, adj_6);
            wp::adj_address(var_values, var_0, var_1, var_2, var_3, adj_values, adj_0, adj_1, adj_2, adj_3, adj_6);
            // adj: attrib[i, j, k, l] = values[i, j, k, l]                                       <L 8>
        }
        wp::adj_address(var_view_mask, var_0, var_1, adj_view_mask, adj_0, adj_1, adj_4);
        // adj: if view_mask[i, j]:                                                               <L 7>
        // adj: i, j, k, l = wp.tid()                                                             <L 6>
        // adj: def set_articulation_attribute_4d_kernel(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void build_actuator_dof_mapping_slice_kernel_ff9df6fa_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::uint32> var_actuator_input_indices,
    wp::int32 var_actuators_per_world,
    wp::int32 var_base_offset,
    wp::int32 var_slice_start,
    wp::int32 var_slice_stop,
    wp::int32 var_stride_within_worlds,
    wp::int32 var_count_per_world,
    wp::int32 var_dofs_per_arti,
    wp::int32 var_dofs_per_world,
    wp::int32 var_num_worlds,
    wp::array_t<wp::int32> var_mapping)
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
        wp::int32 var_2;
        wp::uint32 var_3;
        wp::range_t var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        bool var_12;
        bool var_13;
        bool var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::range_t var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        //---------
        // forward
        // def build_actuator_dof_mapping_slice_kernel(                                           <L 220>
        // local_idx = wp.tid()  # 0 to actuators_per_world-1                                     <L 238>
        var_0 = builtin_tid1d();
        // global_dof = int(actuator_input_indices[local_idx])                                    <L 241>
        var_1 = wp::address(var_actuator_input_indices, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::int(var_3);
        // for arti_idx in range(count_per_world):                                                <L 243>
        var_4 = wp::range(var_count_per_world);
        start_for_0:;
            if (iter_cmp(var_4) == 0) goto end_for_0;
            var_5 = wp::iter_next(var_4);
            // arti_global_start = base_offset + arti_idx * stride_within_worlds + slice_start       <L 244>
            var_6 = wp::mul(var_5, var_stride_within_worlds);
            var_7 = wp::add(var_base_offset, var_6);
            var_8 = wp::add(var_7, var_slice_start);
            // arti_global_stop = base_offset + arti_idx * stride_within_worlds + slice_stop       <L 245>
            var_9 = wp::mul(var_5, var_stride_within_worlds);
            var_10 = wp::add(var_base_offset, var_9);
            var_11 = wp::add(var_10, var_slice_stop);
            // if global_dof >= arti_global_start and global_dof < arti_global_stop:              <L 246>
            var_13 = (var_2 >= var_8);
            var_12 = var_13;
            if (var_12) {
                var_14 = (var_2 < var_11);
                var_12 = var_12 && var_14;
            }
            if (var_12) {
                // view_local_pos = arti_idx * dofs_per_arti + (global_dof - arti_global_start)       <L 247>
                var_15 = wp::mul(var_5, var_dofs_per_arti);
                var_16 = wp::sub(var_2, var_8);
                var_17 = wp::add(var_15, var_16);
                // for world_idx in range(num_worlds):                                            <L 250>
                var_18 = wp::range(var_num_worlds);
                start_for_2:;
                    if (iter_cmp(var_18) == 0) goto end_for_2;
                    var_19 = wp::iter_next(var_18);
                    // view_pos = world_idx * dofs_per_world + view_local_pos                     <L 251>
                    var_20 = wp::mul(var_19, var_dofs_per_world);
                    var_21 = wp::add(var_20, var_17);
                    // actuator_idx = world_idx * actuators_per_world + local_idx                 <L 252>
                    var_22 = wp::mul(var_19, var_actuators_per_world);
                    var_23 = wp::add(var_22, var_0);
                    // mapping[view_pos] = actuator_idx                                           <L 253>
                    wp::array_store(var_mapping, var_21, var_23);
                    goto start_for_2;
                end_for_2:;
                // break                                                                          <L 254>
                goto end_for_0;
            }
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void build_actuator_dof_mapping_slice_kernel_ff9df6fa_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::uint32> var_actuator_input_indices,
    wp::int32 var_actuators_per_world,
    wp::int32 var_base_offset,
    wp::int32 var_slice_start,
    wp::int32 var_slice_stop,
    wp::int32 var_stride_within_worlds,
    wp::int32 var_count_per_world,
    wp::int32 var_dofs_per_arti,
    wp::int32 var_dofs_per_world,
    wp::int32 var_num_worlds,
    wp::array_t<wp::int32> var_mapping,
    wp::array_t<wp::uint32> adj_actuator_input_indices,
    wp::int32 adj_actuators_per_world,
    wp::int32 adj_base_offset,
    wp::int32 adj_slice_start,
    wp::int32 adj_slice_stop,
    wp::int32 adj_stride_within_worlds,
    wp::int32 adj_count_per_world,
    wp::int32 adj_dofs_per_arti,
    wp::int32 adj_dofs_per_world,
    wp::int32 adj_num_worlds,
    wp::array_t<wp::int32> adj_mapping)
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
        wp::int32 var_2;
        wp::uint32 var_3;
        wp::range_t var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        bool var_12;
        bool var_13;
        bool var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::range_t var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::uint32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::uint32 adj_3 = {};
        wp::range_t adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        bool adj_12 = {};
        bool adj_13 = {};
        bool adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::range_t adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::int32 adj_23 = {};
        //---------
        // forward
        // def build_actuator_dof_mapping_slice_kernel(                                           <L 220>
        // local_idx = wp.tid()  # 0 to actuators_per_world-1                                     <L 238>
        var_0 = builtin_tid1d();
        // global_dof = int(actuator_input_indices[local_idx])                                    <L 241>
        var_1 = wp::address(var_actuator_input_indices, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::int(var_3);
        // for arti_idx in range(count_per_world):                                                <L 243>
        var_4 = wp::range(var_count_per_world);
        //---------
        // reverse
        var_4 = wp::iter_reverse(var_4);
        start_for_0:;
            if (iter_cmp(var_4) == 0) goto end_for_0;
            var_5 = wp::iter_next(var_4);
        	adj_6 = {};
        	adj_7 = {};
        	adj_8 = {};
        	adj_9 = {};
        	adj_10 = {};
        	adj_11 = {};
        	adj_12 = {};
        	adj_13 = {};
        	adj_14 = {};
        	adj_15 = {};
        	adj_16 = {};
        	adj_17 = {};
        	adj_18 = {};
            // arti_global_start = base_offset + arti_idx * stride_within_worlds + slice_start       <L 244>
            var_6 = wp::mul(var_5, var_stride_within_worlds);
            var_7 = wp::add(var_base_offset, var_6);
            var_8 = wp::add(var_7, var_slice_start);
            // arti_global_stop = base_offset + arti_idx * stride_within_worlds + slice_stop       <L 245>
            var_9 = wp::mul(var_5, var_stride_within_worlds);
            var_10 = wp::add(var_base_offset, var_9);
            var_11 = wp::add(var_10, var_slice_stop);
            // if global_dof >= arti_global_start and global_dof < arti_global_stop:              <L 246>
            var_13 = (var_2 >= var_8);
            var_12 = var_13;
            if (var_12) {
                var_14 = (var_2 < var_11);
                var_12 = var_12 && var_14;
            }
            if (var_12) {
                // view_local_pos = arti_idx * dofs_per_arti + (global_dof - arti_global_start)       <L 247>
                var_15 = wp::mul(var_5, var_dofs_per_arti);
                var_16 = wp::sub(var_2, var_8);
                var_17 = wp::add(var_15, var_16);
                // for world_idx in range(num_worlds):                                            <L 250>
                var_18 = wp::range(var_num_worlds);
                // break                                                                          <L 254>
                goto end_for_0;
            }
            if (var_12) {
                // adj: break                                                                     <L 254>
                var_18 = wp::iter_reverse(var_18);
                start_for_2:;
                    if (iter_cmp(var_18) == 0) goto end_for_2;
                    var_19 = wp::iter_next(var_18);
                	adj_20 = {};
                	adj_21 = {};
                	adj_22 = {};
                	adj_23 = {};
                    // view_pos = world_idx * dofs_per_world + view_local_pos                     <L 251>
                    var_20 = wp::mul(var_19, var_dofs_per_world);
                    var_21 = wp::add(var_20, var_17);
                    // actuator_idx = world_idx * actuators_per_world + local_idx                 <L 252>
                    var_22 = wp::mul(var_19, var_actuators_per_world);
                    var_23 = wp::add(var_22, var_0);
                    // mapping[view_pos] = actuator_idx                                           <L 253>
                    // wp::array_store(var_mapping, var_21, var_23);
                    wp::adj_array_store(var_mapping, var_21, var_23, adj_mapping, adj_21, adj_23);
                    // adj: mapping[view_pos] = actuator_idx                                      <L 253>
                    wp::adj_add(var_22, var_0, adj_22, adj_0, adj_23);
                    wp::adj_mul(var_19, var_actuators_per_world, adj_19, adj_actuators_per_world, adj_22);
                    // adj: actuator_idx = world_idx * actuators_per_world + local_idx            <L 252>
                    wp::adj_add(var_20, var_17, adj_20, adj_17, adj_21);
                    wp::adj_mul(var_19, var_dofs_per_world, adj_19, adj_dofs_per_world, adj_20);
                    // adj: view_pos = world_idx * dofs_per_world + view_local_pos                <L 251>
                	goto start_for_2;
                end_for_2:;
                // adj: for world_idx in range(num_worlds):                                       <L 250>
                wp::adj_add(var_15, var_16, adj_15, adj_16, adj_17);
                wp::adj_sub(var_2, var_8, adj_2, adj_8, adj_16);
                wp::adj_mul(var_5, var_dofs_per_arti, adj_5, adj_dofs_per_arti, adj_15);
                // adj: view_local_pos = arti_idx * dofs_per_arti + (global_dof - arti_global_start)  <L 247>
            }
            if (var_12) {
            }
            // adj: if global_dof >= arti_global_start and global_dof < arti_global_stop:         <L 246>
            wp::adj_add(var_10, var_slice_stop, adj_10, adj_slice_stop, adj_11);
            wp::adj_add(var_base_offset, var_9, adj_base_offset, adj_9, adj_10);
            wp::adj_mul(var_5, var_stride_within_worlds, adj_5, adj_stride_within_worlds, adj_9);
            // adj: arti_global_stop = base_offset + arti_idx * stride_within_worlds + slice_stop  <L 245>
            wp::adj_add(var_7, var_slice_start, adj_7, adj_slice_start, adj_8);
            wp::adj_add(var_base_offset, var_6, adj_base_offset, adj_6, adj_7);
            wp::adj_mul(var_5, var_stride_within_worlds, adj_5, adj_stride_within_worlds, adj_6);
            // adj: arti_global_start = base_offset + arti_idx * stride_within_worlds + slice_start  <L 244>
        	goto start_for_0;
        end_for_0:;
        // adj: for arti_idx in range(count_per_world):                                           <L 243>
        wp::adj_address(var_actuator_input_indices, var_0, adj_actuator_input_indices, adj_0, adj_1);
        // adj: global_dof = int(actuator_input_indices[local_idx])                               <L 241>
        // adj: local_idx = wp.tid()  # 0 to actuators_per_world-1                                <L 238>
        // adj: def build_actuator_dof_mapping_slice_kernel(                                      <L 220>
        continue;
    }
}

