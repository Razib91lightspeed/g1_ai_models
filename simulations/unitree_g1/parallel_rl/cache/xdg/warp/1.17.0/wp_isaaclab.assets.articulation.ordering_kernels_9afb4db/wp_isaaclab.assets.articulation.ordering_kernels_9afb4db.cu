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



extern "C" __global__ void _write_scalar_user_to_backend_with_indices_and_sim_ids_02e07aef_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_input_value,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> var_backend_data,
    wp::array_t<wp::int32> var_sim_env_ids)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        //---------
        // forward
        // def _write_scalar_user_to_backend_with_indices_and_sim_ids(                            <L 1>
        // i, j = wp.tid()                                                                        <L 11>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 12>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 13>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if j == 0:                                                                             <L 14>
        var_9 = (var_1 == var_8);
        if (var_9) {
            // sim_env_ids[i] = env_id                                                            <L 15>
            wp::array_store(var_sim_env_ids, var_0, var_3);
        }
        // user_data[env_id, user_id] = input_value                                               <L 16>
        wp::array_store(var_user_data, var_3, var_6, var_input_value);
        // if has_ordering:                                                                       <L 17>
        if (var_has_ordering) {
            // backend_data[env_id, user_to_backend[user_id]] = input_value                       <L 18>
            var_10 = wp::address(var_user_to_backend, var_6);
            var_11 = wp::load(var_10);
            wp::array_store(var_backend_data, var_3, var_11, var_input_value);
        }
    }
}



extern "C" __global__ void _write_scalar_user_to_backend_with_indices_and_sim_ids_02e07aef_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_input_value,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> var_backend_data,
    wp::array_t<wp::int32> var_sim_env_ids,
    wp::float32 adj_input_value,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int64> adj_user_ids,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    wp::array_t<wp::float32> adj_user_data,
    wp::array_t<wp::float32> adj_backend_data,
    wp::array_t<wp::int32> adj_sim_env_ids)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int64 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int64 adj_4 = {};
        wp::int64 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int64 adj_7 = {};
        wp::int32 adj_8 = {};
        bool adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        //---------
        // forward
        // def _write_scalar_user_to_backend_with_indices_and_sim_ids(                            <L 1>
        // i, j = wp.tid()                                                                        <L 11>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 12>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 13>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if j == 0:                                                                             <L 14>
        var_9 = (var_1 == var_8);
        if (var_9) {
            // sim_env_ids[i] = env_id                                                            <L 15>
            // wp::array_store(var_sim_env_ids, var_0, var_3);
        }
        // user_data[env_id, user_id] = input_value                                               <L 16>
        // wp::array_store(var_user_data, var_3, var_6, var_input_value);
        // if has_ordering:                                                                       <L 17>
        if (var_has_ordering) {
            // backend_data[env_id, user_to_backend[user_id]] = input_value                       <L 18>
            var_10 = wp::address(var_user_to_backend, var_6);
            var_11 = wp::load(var_10);
            // wp::array_store(var_backend_data, var_3, var_11, var_input_value);
        }
        //---------
        // reverse
        if (var_has_ordering) {
            wp::adj_array_store(var_backend_data, var_3, var_11, var_input_value, adj_backend_data, adj_3, adj_10, adj_input_value);
            wp::adj_address(var_user_to_backend, var_6, adj_user_to_backend, adj_6, adj_10);
            // adj: backend_data[env_id, user_to_backend[user_id]] = input_value                  <L 18>
        }
        // adj: if has_ordering:                                                                  <L 17>
        wp::adj_array_store(var_user_data, var_3, var_6, var_input_value, adj_user_data, adj_3, adj_6, adj_input_value);
        // adj: user_data[env_id, user_id] = input_value                                          <L 16>
        if (var_9) {
            wp::adj_array_store(var_sim_env_ids, var_0, var_3, adj_sim_env_ids, adj_0, adj_3);
            // adj: sim_env_ids[i] = env_id                                                       <L 15>
        }
        // adj: if j == 0:                                                                        <L 14>
        wp::adj_address(var_user_ids, var_1, adj_user_ids, adj_1, adj_5);
        // adj: user_id = wp.int32(user_ids[j])                                                   <L 13>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 12>
        // adj: i, j = wp.tid()                                                                   <L 11>
        // adj: def _write_scalar_user_to_backend_with_indices_and_sim_ids(                       <L 1>
        continue;
    }
}



extern "C" __global__ void reorder_body_state_backend_to_user_08b8f13f_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_backend_pose,
    wp::array_t<wp::vec_t<6, wp::float32>> var_backend_vel,
    wp::array_t<wp::int32> var_user_to_backend,
    wp::array_t<wp::transform_t<wp::float32>> var_user_pose,
    wp::array_t<wp::vec_t<6, wp::float32>> var_user_vel)
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
        wp::transform_t<wp::float32>* var_5;
        wp::transform_t<wp::float32> var_6;
        wp::vec_t<6, wp::float32>* var_7;
        wp::vec_t<6, wp::float32> var_8;
        //---------
        // forward
        // def reorder_body_state_backend_to_user(                                                <L 273>
        // env_id, user_id = wp.tid()                                                             <L 300>
        builtin_tid2d(var_0, var_1);
        // backend_id = user_to_backend[user_id]                                                  <L 301>
        var_2 = wp::address(var_user_to_backend, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // user_pose[env_id, user_id] = backend_pose[env_id, backend_id]                          <L 302>
        var_5 = wp::address(var_backend_pose, var_0, var_3);
        var_6 = wp::load(var_5);
        wp::array_store(var_user_pose, var_0, var_1, var_6);
        // user_vel[env_id, user_id] = backend_vel[env_id, backend_id]                            <L 303>
        var_7 = wp::address(var_backend_vel, var_0, var_3);
        var_8 = wp::load(var_7);
        wp::array_store(var_user_vel, var_0, var_1, var_8);
    }
}



extern "C" __global__ void reorder_body_state_backend_to_user_08b8f13f_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_backend_pose,
    wp::array_t<wp::vec_t<6, wp::float32>> var_backend_vel,
    wp::array_t<wp::int32> var_user_to_backend,
    wp::array_t<wp::transform_t<wp::float32>> var_user_pose,
    wp::array_t<wp::vec_t<6, wp::float32>> var_user_vel,
    wp::array_t<wp::transform_t<wp::float32>> adj_backend_pose,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_backend_vel,
    wp::array_t<wp::int32> adj_user_to_backend,
    wp::array_t<wp::transform_t<wp::float32>> adj_user_pose,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_user_vel)
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
        wp::transform_t<wp::float32>* var_5;
        wp::transform_t<wp::float32> var_6;
        wp::vec_t<6, wp::float32>* var_7;
        wp::vec_t<6, wp::float32> var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::transform_t<wp::float32> adj_5 = {};
        wp::transform_t<wp::float32> adj_6 = {};
        wp::vec_t<6, wp::float32> adj_7 = {};
        wp::vec_t<6, wp::float32> adj_8 = {};
        //---------
        // forward
        // def reorder_body_state_backend_to_user(                                                <L 273>
        // env_id, user_id = wp.tid()                                                             <L 300>
        builtin_tid2d(var_0, var_1);
        // backend_id = user_to_backend[user_id]                                                  <L 301>
        var_2 = wp::address(var_user_to_backend, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // user_pose[env_id, user_id] = backend_pose[env_id, backend_id]                          <L 302>
        var_5 = wp::address(var_backend_pose, var_0, var_3);
        var_6 = wp::load(var_5);
        // wp::array_store(var_user_pose, var_0, var_1, var_6);
        // user_vel[env_id, user_id] = backend_vel[env_id, backend_id]                            <L 303>
        var_7 = wp::address(var_backend_vel, var_0, var_3);
        var_8 = wp::load(var_7);
        // wp::array_store(var_user_vel, var_0, var_1, var_8);
        //---------
        // reverse
        wp::adj_array_store(var_user_vel, var_0, var_1, var_8, adj_user_vel, adj_0, adj_1, adj_7);
        wp::adj_address(var_backend_vel, var_0, var_3, adj_backend_vel, adj_0, adj_3, adj_7);
        // adj: user_vel[env_id, user_id] = backend_vel[env_id, backend_id]                       <L 303>
        wp::adj_array_store(var_user_pose, var_0, var_1, var_6, adj_user_pose, adj_0, adj_1, adj_5);
        wp::adj_address(var_backend_pose, var_0, var_3, adj_backend_pose, adj_0, adj_3, adj_5);
        // adj: user_pose[env_id, user_id] = backend_pose[env_id, backend_id]                     <L 302>
        wp::adj_copy(var_4, adj_2, adj_3);
        wp::adj_address(var_user_to_backend, var_1, adj_user_to_backend, adj_1, adj_2);
        // adj: backend_id = user_to_backend[user_id]                                             <L 301>
        // adj: env_id, user_id = wp.tid()                                                        <L 300>
        // adj: def reorder_body_state_backend_to_user(                                           <L 273>
        continue;
    }
}



extern "C" __global__ void reorder_joint_state_backend_to_user_08f448ef_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_backend_pos,
    wp::array_t<wp::float32> var_backend_vel,
    wp::array_t<wp::int32> var_user_to_backend,
    wp::array_t<wp::float32> var_user_pos,
    wp::array_t<wp::float32> var_user_vel)
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
        wp::float32* var_5;
        wp::float32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        //---------
        // forward
        // def reorder_joint_state_backend_to_user(                                               <L 210>
        // env_id, user_id = wp.tid()                                                             <L 237>
        builtin_tid2d(var_0, var_1);
        // backend_id = user_to_backend[user_id]                                                  <L 238>
        var_2 = wp::address(var_user_to_backend, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // user_pos[env_id, user_id] = backend_pos[env_id, backend_id]                            <L 239>
        var_5 = wp::address(var_backend_pos, var_0, var_3);
        var_6 = wp::load(var_5);
        wp::array_store(var_user_pos, var_0, var_1, var_6);
        // user_vel[env_id, user_id] = backend_vel[env_id, backend_id]                            <L 240>
        var_7 = wp::address(var_backend_vel, var_0, var_3);
        var_8 = wp::load(var_7);
        wp::array_store(var_user_vel, var_0, var_1, var_8);
    }
}



extern "C" __global__ void reorder_joint_state_backend_to_user_08f448ef_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_backend_pos,
    wp::array_t<wp::float32> var_backend_vel,
    wp::array_t<wp::int32> var_user_to_backend,
    wp::array_t<wp::float32> var_user_pos,
    wp::array_t<wp::float32> var_user_vel,
    wp::array_t<wp::float32> adj_backend_pos,
    wp::array_t<wp::float32> adj_backend_vel,
    wp::array_t<wp::int32> adj_user_to_backend,
    wp::array_t<wp::float32> adj_user_pos,
    wp::array_t<wp::float32> adj_user_vel)
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
        wp::float32* var_5;
        wp::float32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::float32 adj_8 = {};
        //---------
        // forward
        // def reorder_joint_state_backend_to_user(                                               <L 210>
        // env_id, user_id = wp.tid()                                                             <L 237>
        builtin_tid2d(var_0, var_1);
        // backend_id = user_to_backend[user_id]                                                  <L 238>
        var_2 = wp::address(var_user_to_backend, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // user_pos[env_id, user_id] = backend_pos[env_id, backend_id]                            <L 239>
        var_5 = wp::address(var_backend_pos, var_0, var_3);
        var_6 = wp::load(var_5);
        // wp::array_store(var_user_pos, var_0, var_1, var_6);
        // user_vel[env_id, user_id] = backend_vel[env_id, backend_id]                            <L 240>
        var_7 = wp::address(var_backend_vel, var_0, var_3);
        var_8 = wp::load(var_7);
        // wp::array_store(var_user_vel, var_0, var_1, var_8);
        //---------
        // reverse
        wp::adj_array_store(var_user_vel, var_0, var_1, var_8, adj_user_vel, adj_0, adj_1, adj_7);
        wp::adj_address(var_backend_vel, var_0, var_3, adj_backend_vel, adj_0, adj_3, adj_7);
        // adj: user_vel[env_id, user_id] = backend_vel[env_id, backend_id]                       <L 240>
        wp::adj_array_store(var_user_pos, var_0, var_1, var_6, adj_user_pos, adj_0, adj_1, adj_5);
        wp::adj_address(var_backend_pos, var_0, var_3, adj_backend_pos, adj_0, adj_3, adj_5);
        // adj: user_pos[env_id, user_id] = backend_pos[env_id, backend_id]                       <L 239>
        wp::adj_copy(var_4, adj_2, adj_3);
        wp::adj_address(var_user_to_backend, var_1, adj_user_to_backend, adj_1, adj_2);
        // adj: backend_id = user_to_backend[user_id]                                             <L 238>
        // adj: env_id, user_id = wp.tid()                                                        <L 237>
        // adj: def reorder_joint_state_backend_to_user(                                          <L 210>
        continue;
    }
}



extern "C" __global__ void _write_scalar_user_to_backend_with_indices_0f1d78e7_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_input_value,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> var_backend_data)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        //---------
        // forward
        // def _write_scalar_user_to_backend_with_indices(                                        <L 1>
        // i, j = wp.tid()                                                                        <L 25>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 26>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 27>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // user_data[env_id, user_id] = input_value                                               <L 29>
        wp::array_store(var_user_data, var_3, var_6, var_input_value);
        // if has_ordering:                                                                       <L 30>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 31>
            var_8 = wp::address(var_user_to_backend, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
            // backend_data[env_id, backend_id] = input_value                                     <L 32>
            wp::array_store(var_backend_data, var_3, var_9, var_input_value);
        }
    }
}



extern "C" __global__ void _write_scalar_user_to_backend_with_indices_0f1d78e7_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_input_value,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> var_backend_data,
    wp::float32 adj_input_value,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int64> adj_user_ids,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    wp::array_t<wp::float32> adj_user_data,
    wp::array_t<wp::float32> adj_backend_data)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int64 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int64 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        //---------
        // forward
        // def _write_scalar_user_to_backend_with_indices(                                        <L 1>
        // i, j = wp.tid()                                                                        <L 25>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 26>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 27>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // user_data[env_id, user_id] = input_value                                               <L 29>
        // wp::array_store(var_user_data, var_3, var_6, var_input_value);
        // if has_ordering:                                                                       <L 30>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 31>
            var_8 = wp::address(var_user_to_backend, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
            // backend_data[env_id, backend_id] = input_value                                     <L 32>
            // wp::array_store(var_backend_data, var_3, var_9, var_input_value);
        }
        //---------
        // reverse
        if (var_has_ordering) {
            wp::adj_array_store(var_backend_data, var_3, var_9, var_input_value, adj_backend_data, adj_3, adj_9, adj_input_value);
            // adj: backend_data[env_id, backend_id] = input_value                                <L 32>
            wp::adj_copy(var_10, adj_8, adj_9);
            wp::adj_address(var_user_to_backend, var_6, adj_user_to_backend, adj_6, adj_8);
            // adj: backend_id = user_to_backend[user_id]                                         <L 31>
        }
        // adj: if has_ordering:                                                                  <L 30>
        wp::adj_array_store(var_user_data, var_3, var_6, var_input_value, adj_user_data, adj_3, adj_6, adj_input_value);
        // adj: user_data[env_id, user_id] = input_value                                          <L 29>
        wp::adj_address(var_user_ids, var_1, adj_user_ids, adj_1, adj_5);
        // adj: user_id = wp.int32(user_ids[j])                                                   <L 27>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 26>
        // adj: i, j = wp.tid()                                                                   <L 25>
        // adj: def _write_scalar_user_to_backend_with_indices(                                   <L 1>
        continue;
    }
}



extern "C" __global__ void _write_scalar_user_to_backend_with_indices_1ae9a241_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_input_value,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> var_backend_data)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        //---------
        // forward
        // def _write_scalar_user_to_backend_with_indices(                                        <L 1>
        // i, j = wp.tid()                                                                        <L 25>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 26>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 27>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // user_data[env_id, user_id] = input_value                                               <L 29>
        wp::array_store(var_user_data, var_3, var_6, var_input_value);
        // if has_ordering:                                                                       <L 30>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 31>
            var_8 = wp::address(var_user_to_backend, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
            // backend_data[env_id, backend_id] = input_value                                     <L 32>
            wp::array_store(var_backend_data, var_3, var_9, var_input_value);
        }
    }
}



extern "C" __global__ void _write_scalar_user_to_backend_with_indices_1ae9a241_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_input_value,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> var_backend_data,
    wp::float32 adj_input_value,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_user_ids,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    wp::array_t<wp::float32> adj_user_data,
    wp::array_t<wp::float32> adj_backend_data)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        //---------
        // forward
        // def _write_scalar_user_to_backend_with_indices(                                        <L 1>
        // i, j = wp.tid()                                                                        <L 25>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 26>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 27>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // user_data[env_id, user_id] = input_value                                               <L 29>
        // wp::array_store(var_user_data, var_3, var_6, var_input_value);
        // if has_ordering:                                                                       <L 30>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 31>
            var_8 = wp::address(var_user_to_backend, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
            // backend_data[env_id, backend_id] = input_value                                     <L 32>
            // wp::array_store(var_backend_data, var_3, var_9, var_input_value);
        }
        //---------
        // reverse
        if (var_has_ordering) {
            wp::adj_array_store(var_backend_data, var_3, var_9, var_input_value, adj_backend_data, adj_3, adj_9, adj_input_value);
            // adj: backend_data[env_id, backend_id] = input_value                                <L 32>
            wp::adj_copy(var_10, adj_8, adj_9);
            wp::adj_address(var_user_to_backend, var_6, adj_user_to_backend, adj_6, adj_8);
            // adj: backend_id = user_to_backend[user_id]                                         <L 31>
        }
        // adj: if has_ordering:                                                                  <L 30>
        wp::adj_array_store(var_user_data, var_3, var_6, var_input_value, adj_user_data, adj_3, adj_6, adj_input_value);
        // adj: user_data[env_id, user_id] = input_value                                          <L 29>
        wp::adj_address(var_user_ids, var_1, adj_user_ids, adj_1, adj_5);
        // adj: user_id = wp.int32(user_ids[j])                                                   <L 27>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 26>
        // adj: i, j = wp.tid()                                                                   <L 25>
        // adj: def _write_scalar_user_to_backend_with_indices(                                   <L 1>
        continue;
    }
}



extern "C" __global__ void write_joint_vel_user_to_backend_with_indices_236d0547_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    bool var_full_data,
    wp::array_t<wp::float32> var_user_vel,
    wp::array_t<wp::float32> var_user_prev_vel,
    wp::array_t<wp::float32> var_user_acc,
    wp::array_t<wp::float32> var_backend_vel)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        const wp::float32 var_15 = 0.0;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        //---------
        // forward
        // def write_joint_vel_user_to_backend_with_indices(                                      <L 1>
        // i, j = wp.tid()                                                                        <L 35>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 36>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 37>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if full_data:                                                                          <L 39>
        if (var_full_data) {
            // value = in_data[env_id, user_id]                                                   <L 40>
            var_8 = wp::address(var_in_data, var_3, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
        }
        if (!var_full_data) {
            // value = in_data[i, j]                                                              <L 42>
            var_11 = wp::address(var_in_data, var_0, var_1);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
        }
        var_14 = wp::where(var_full_data, var_9, var_12);
        // user_vel[env_id, user_id] = value                                                      <L 44>
        wp::array_store(var_user_vel, var_3, var_6, var_14);
        // user_prev_vel[env_id, user_id] = value                                                 <L 45>
        wp::array_store(var_user_prev_vel, var_3, var_6, var_14);
        // user_acc[env_id, user_id] = 0.0                                                        <L 46>
        wp::array_store(var_user_acc, var_3, var_6, var_15);
        // if has_ordering:                                                                       <L 47>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 48>
            var_16 = wp::address(var_user_to_backend, var_6);
            var_18 = wp::load(var_16);
            var_17 = wp::copy(var_18);
            // backend_vel[env_id, backend_id] = value                                            <L 49>
            wp::array_store(var_backend_vel, var_3, var_17, var_14);
        }
    }
}



extern "C" __global__ void write_joint_vel_user_to_backend_with_indices_236d0547_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    bool var_full_data,
    wp::array_t<wp::float32> var_user_vel,
    wp::array_t<wp::float32> var_user_prev_vel,
    wp::array_t<wp::float32> var_user_acc,
    wp::array_t<wp::float32> var_backend_vel,
    wp::array_t<wp::float32> adj_in_data,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int32> adj_user_ids,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    bool adj_full_data,
    wp::array_t<wp::float32> adj_user_vel,
    wp::array_t<wp::float32> adj_user_prev_vel,
    wp::array_t<wp::float32> adj_user_acc,
    wp::array_t<wp::float32> adj_backend_vel)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        const wp::float32 var_15 = 0.0;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int64 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int64 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        //---------
        // forward
        // def write_joint_vel_user_to_backend_with_indices(                                      <L 1>
        // i, j = wp.tid()                                                                        <L 35>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 36>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 37>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if full_data:                                                                          <L 39>
        if (var_full_data) {
            // value = in_data[env_id, user_id]                                                   <L 40>
            var_8 = wp::address(var_in_data, var_3, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
        }
        if (!var_full_data) {
            // value = in_data[i, j]                                                              <L 42>
            var_11 = wp::address(var_in_data, var_0, var_1);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
        }
        var_14 = wp::where(var_full_data, var_9, var_12);
        // user_vel[env_id, user_id] = value                                                      <L 44>
        // wp::array_store(var_user_vel, var_3, var_6, var_14);
        // user_prev_vel[env_id, user_id] = value                                                 <L 45>
        // wp::array_store(var_user_prev_vel, var_3, var_6, var_14);
        // user_acc[env_id, user_id] = 0.0                                                        <L 46>
        // wp::array_store(var_user_acc, var_3, var_6, var_15);
        // if has_ordering:                                                                       <L 47>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 48>
            var_16 = wp::address(var_user_to_backend, var_6);
            var_18 = wp::load(var_16);
            var_17 = wp::copy(var_18);
            // backend_vel[env_id, backend_id] = value                                            <L 49>
            // wp::array_store(var_backend_vel, var_3, var_17, var_14);
        }
        //---------
        // reverse
        if (var_has_ordering) {
            wp::adj_array_store(var_backend_vel, var_3, var_17, var_14, adj_backend_vel, adj_3, adj_17, adj_14);
            // adj: backend_vel[env_id, backend_id] = value                                       <L 49>
            wp::adj_copy(var_18, adj_16, adj_17);
            wp::adj_address(var_user_to_backend, var_6, adj_user_to_backend, adj_6, adj_16);
            // adj: backend_id = user_to_backend[user_id]                                         <L 48>
        }
        // adj: if has_ordering:                                                                  <L 47>
        wp::adj_array_store(var_user_acc, var_3, var_6, var_15, adj_user_acc, adj_3, adj_6, adj_15);
        // adj: user_acc[env_id, user_id] = 0.0                                                   <L 46>
        wp::adj_array_store(var_user_prev_vel, var_3, var_6, var_14, adj_user_prev_vel, adj_3, adj_6, adj_14);
        // adj: user_prev_vel[env_id, user_id] = value                                            <L 45>
        wp::adj_array_store(var_user_vel, var_3, var_6, var_14, adj_user_vel, adj_3, adj_6, adj_14);
        // adj: user_vel[env_id, user_id] = value                                                 <L 44>
        wp::adj_where(var_full_data, var_9, var_12, adj_full_data, adj_9, adj_12, adj_14);
        if (!var_full_data) {
            wp::adj_copy(var_13, adj_11, adj_12);
            wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_11);
            // adj: value = in_data[i, j]                                                         <L 42>
        }
        if (var_full_data) {
            wp::adj_copy(var_10, adj_8, adj_9);
            wp::adj_address(var_in_data, var_3, var_6, adj_in_data, adj_3, adj_6, adj_8);
            // adj: value = in_data[env_id, user_id]                                              <L 40>
        }
        // adj: if full_data:                                                                     <L 39>
        wp::adj_address(var_user_ids, var_1, adj_user_ids, adj_1, adj_5);
        // adj: user_id = wp.int32(user_ids[j])                                                   <L 37>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 36>
        // adj: i, j = wp.tid()                                                                   <L 35>
        // adj: def write_joint_vel_user_to_backend_with_indices(                                 <L 1>
        continue;
    }
}



extern "C" __global__ void write_joint_state_user_to_backend_with_indices_23c30898_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_pos_data,
    wp::array_t<wp::float32> var_vel_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    bool var_full_data,
    wp::array_t<wp::float32> var_user_pos,
    wp::array_t<wp::float32> var_user_vel,
    wp::array_t<wp::float32> var_user_prev_vel,
    wp::array_t<wp::float32> var_user_acc,
    wp::array_t<wp::float32> var_backend_pos,
    wp::array_t<wp::float32> var_backend_vel)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        const wp::float32 var_22 = 0.0;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        //---------
        // forward
        // def write_joint_state_user_to_backend_with_indices(                                    <L 1>
        // i, j = wp.tid()                                                                        <L 43>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 44>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 45>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if full_data:                                                                          <L 47>
        if (var_full_data) {
            // position = pos_data[env_id, user_id]                                               <L 48>
            var_8 = wp::address(var_pos_data, var_3, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
            // velocity = vel_data[env_id, user_id]                                               <L 49>
            var_11 = wp::address(var_vel_data, var_3, var_6);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
        }
        if (!var_full_data) {
            // position = pos_data[i, j]                                                          <L 51>
            var_14 = wp::address(var_pos_data, var_0, var_1);
            var_16 = wp::load(var_14);
            var_15 = wp::copy(var_16);
            // velocity = vel_data[i, j]                                                          <L 52>
            var_17 = wp::address(var_vel_data, var_0, var_1);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
        }
        var_20 = wp::where(var_full_data, var_9, var_15);
        var_21 = wp::where(var_full_data, var_12, var_18);
        // user_pos[env_id, user_id] = position                                                   <L 54>
        wp::array_store(var_user_pos, var_3, var_6, var_20);
        // user_vel[env_id, user_id] = velocity                                                   <L 55>
        wp::array_store(var_user_vel, var_3, var_6, var_21);
        // user_prev_vel[env_id, user_id] = velocity                                              <L 56>
        wp::array_store(var_user_prev_vel, var_3, var_6, var_21);
        // user_acc[env_id, user_id] = 0.0                                                        <L 57>
        wp::array_store(var_user_acc, var_3, var_6, var_22);
        // if has_ordering:                                                                       <L 58>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 59>
            var_23 = wp::address(var_user_to_backend, var_6);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // backend_pos[env_id, backend_id] = position                                         <L 60>
            wp::array_store(var_backend_pos, var_3, var_24, var_20);
            // backend_vel[env_id, backend_id] = velocity                                         <L 61>
            wp::array_store(var_backend_vel, var_3, var_24, var_21);
        }
    }
}



extern "C" __global__ void write_joint_state_user_to_backend_with_indices_23c30898_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_pos_data,
    wp::array_t<wp::float32> var_vel_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    bool var_full_data,
    wp::array_t<wp::float32> var_user_pos,
    wp::array_t<wp::float32> var_user_vel,
    wp::array_t<wp::float32> var_user_prev_vel,
    wp::array_t<wp::float32> var_user_acc,
    wp::array_t<wp::float32> var_backend_pos,
    wp::array_t<wp::float32> var_backend_vel,
    wp::array_t<wp::float32> adj_pos_data,
    wp::array_t<wp::float32> adj_vel_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_user_ids,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    bool adj_full_data,
    wp::array_t<wp::float32> adj_user_pos,
    wp::array_t<wp::float32> adj_user_vel,
    wp::array_t<wp::float32> adj_user_prev_vel,
    wp::array_t<wp::float32> adj_user_acc,
    wp::array_t<wp::float32> adj_backend_pos,
    wp::array_t<wp::float32> adj_backend_vel)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        const wp::float32 var_22 = 0.0;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        //---------
        // forward
        // def write_joint_state_user_to_backend_with_indices(                                    <L 1>
        // i, j = wp.tid()                                                                        <L 43>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 44>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 45>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if full_data:                                                                          <L 47>
        if (var_full_data) {
            // position = pos_data[env_id, user_id]                                               <L 48>
            var_8 = wp::address(var_pos_data, var_3, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
            // velocity = vel_data[env_id, user_id]                                               <L 49>
            var_11 = wp::address(var_vel_data, var_3, var_6);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
        }
        if (!var_full_data) {
            // position = pos_data[i, j]                                                          <L 51>
            var_14 = wp::address(var_pos_data, var_0, var_1);
            var_16 = wp::load(var_14);
            var_15 = wp::copy(var_16);
            // velocity = vel_data[i, j]                                                          <L 52>
            var_17 = wp::address(var_vel_data, var_0, var_1);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
        }
        var_20 = wp::where(var_full_data, var_9, var_15);
        var_21 = wp::where(var_full_data, var_12, var_18);
        // user_pos[env_id, user_id] = position                                                   <L 54>
        // wp::array_store(var_user_pos, var_3, var_6, var_20);
        // user_vel[env_id, user_id] = velocity                                                   <L 55>
        // wp::array_store(var_user_vel, var_3, var_6, var_21);
        // user_prev_vel[env_id, user_id] = velocity                                              <L 56>
        // wp::array_store(var_user_prev_vel, var_3, var_6, var_21);
        // user_acc[env_id, user_id] = 0.0                                                        <L 57>
        // wp::array_store(var_user_acc, var_3, var_6, var_22);
        // if has_ordering:                                                                       <L 58>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 59>
            var_23 = wp::address(var_user_to_backend, var_6);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // backend_pos[env_id, backend_id] = position                                         <L 60>
            // wp::array_store(var_backend_pos, var_3, var_24, var_20);
            // backend_vel[env_id, backend_id] = velocity                                         <L 61>
            // wp::array_store(var_backend_vel, var_3, var_24, var_21);
        }
        //---------
        // reverse
        if (var_has_ordering) {
            wp::adj_array_store(var_backend_vel, var_3, var_24, var_21, adj_backend_vel, adj_3, adj_24, adj_21);
            // adj: backend_vel[env_id, backend_id] = velocity                                    <L 61>
            wp::adj_array_store(var_backend_pos, var_3, var_24, var_20, adj_backend_pos, adj_3, adj_24, adj_20);
            // adj: backend_pos[env_id, backend_id] = position                                    <L 60>
            wp::adj_copy(var_25, adj_23, adj_24);
            wp::adj_address(var_user_to_backend, var_6, adj_user_to_backend, adj_6, adj_23);
            // adj: backend_id = user_to_backend[user_id]                                         <L 59>
        }
        // adj: if has_ordering:                                                                  <L 58>
        wp::adj_array_store(var_user_acc, var_3, var_6, var_22, adj_user_acc, adj_3, adj_6, adj_22);
        // adj: user_acc[env_id, user_id] = 0.0                                                   <L 57>
        wp::adj_array_store(var_user_prev_vel, var_3, var_6, var_21, adj_user_prev_vel, adj_3, adj_6, adj_21);
        // adj: user_prev_vel[env_id, user_id] = velocity                                         <L 56>
        wp::adj_array_store(var_user_vel, var_3, var_6, var_21, adj_user_vel, adj_3, adj_6, adj_21);
        // adj: user_vel[env_id, user_id] = velocity                                              <L 55>
        wp::adj_array_store(var_user_pos, var_3, var_6, var_20, adj_user_pos, adj_3, adj_6, adj_20);
        // adj: user_pos[env_id, user_id] = position                                              <L 54>
        wp::adj_where(var_full_data, var_12, var_18, adj_full_data, adj_12, adj_18, adj_21);
        wp::adj_where(var_full_data, var_9, var_15, adj_full_data, adj_9, adj_15, adj_20);
        if (!var_full_data) {
            wp::adj_copy(var_19, adj_17, adj_18);
            wp::adj_address(var_vel_data, var_0, var_1, adj_vel_data, adj_0, adj_1, adj_17);
            // adj: velocity = vel_data[i, j]                                                     <L 52>
            wp::adj_copy(var_16, adj_14, adj_15);
            wp::adj_address(var_pos_data, var_0, var_1, adj_pos_data, adj_0, adj_1, adj_14);
            // adj: position = pos_data[i, j]                                                     <L 51>
        }
        if (var_full_data) {
            wp::adj_copy(var_13, adj_11, adj_12);
            wp::adj_address(var_vel_data, var_3, var_6, adj_vel_data, adj_3, adj_6, adj_11);
            // adj: velocity = vel_data[env_id, user_id]                                          <L 49>
            wp::adj_copy(var_10, adj_8, adj_9);
            wp::adj_address(var_pos_data, var_3, var_6, adj_pos_data, adj_3, adj_6, adj_8);
            // adj: position = pos_data[env_id, user_id]                                          <L 48>
        }
        // adj: if full_data:                                                                     <L 47>
        wp::adj_address(var_user_ids, var_1, adj_user_ids, adj_1, adj_5);
        // adj: user_id = wp.int32(user_ids[j])                                                   <L 45>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 44>
        // adj: i, j = wp.tid()                                                                   <L 43>
        // adj: def write_joint_state_user_to_backend_with_indices(                               <L 1>
        continue;
    }
}



extern "C" __global__ void _write_scalar_user_to_backend_with_indices_and_sim_ids_3927cd46_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_input_value,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> var_backend_data,
    wp::array_t<wp::int32> var_sim_env_ids)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        //---------
        // forward
        // def _write_scalar_user_to_backend_with_indices_and_sim_ids(                            <L 1>
        // i, j = wp.tid()                                                                        <L 11>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 12>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 13>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if j == 0:                                                                             <L 14>
        var_9 = (var_1 == var_8);
        if (var_9) {
            // sim_env_ids[i] = env_id                                                            <L 15>
            wp::array_store(var_sim_env_ids, var_0, var_3);
        }
        // user_data[env_id, user_id] = input_value                                               <L 16>
        wp::array_store(var_user_data, var_3, var_6, var_input_value);
        // if has_ordering:                                                                       <L 17>
        if (var_has_ordering) {
            // backend_data[env_id, user_to_backend[user_id]] = input_value                       <L 18>
            var_10 = wp::address(var_user_to_backend, var_6);
            var_11 = wp::load(var_10);
            wp::array_store(var_backend_data, var_3, var_11, var_input_value);
        }
    }
}



extern "C" __global__ void _write_scalar_user_to_backend_with_indices_and_sim_ids_3927cd46_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_input_value,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> var_backend_data,
    wp::array_t<wp::int32> var_sim_env_ids,
    wp::float32 adj_input_value,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int64> adj_user_ids,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    wp::array_t<wp::float32> adj_user_data,
    wp::array_t<wp::float32> adj_backend_data,
    wp::array_t<wp::int32> adj_sim_env_ids)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int64 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int64 adj_7 = {};
        wp::int32 adj_8 = {};
        bool adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        //---------
        // forward
        // def _write_scalar_user_to_backend_with_indices_and_sim_ids(                            <L 1>
        // i, j = wp.tid()                                                                        <L 11>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 12>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 13>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if j == 0:                                                                             <L 14>
        var_9 = (var_1 == var_8);
        if (var_9) {
            // sim_env_ids[i] = env_id                                                            <L 15>
            // wp::array_store(var_sim_env_ids, var_0, var_3);
        }
        // user_data[env_id, user_id] = input_value                                               <L 16>
        // wp::array_store(var_user_data, var_3, var_6, var_input_value);
        // if has_ordering:                                                                       <L 17>
        if (var_has_ordering) {
            // backend_data[env_id, user_to_backend[user_id]] = input_value                       <L 18>
            var_10 = wp::address(var_user_to_backend, var_6);
            var_11 = wp::load(var_10);
            // wp::array_store(var_backend_data, var_3, var_11, var_input_value);
        }
        //---------
        // reverse
        if (var_has_ordering) {
            wp::adj_array_store(var_backend_data, var_3, var_11, var_input_value, adj_backend_data, adj_3, adj_10, adj_input_value);
            wp::adj_address(var_user_to_backend, var_6, adj_user_to_backend, adj_6, adj_10);
            // adj: backend_data[env_id, user_to_backend[user_id]] = input_value                  <L 18>
        }
        // adj: if has_ordering:                                                                  <L 17>
        wp::adj_array_store(var_user_data, var_3, var_6, var_input_value, adj_user_data, adj_3, adj_6, adj_input_value);
        // adj: user_data[env_id, user_id] = input_value                                          <L 16>
        if (var_9) {
            wp::adj_array_store(var_sim_env_ids, var_0, var_3, adj_sim_env_ids, adj_0, adj_3);
            // adj: sim_env_ids[i] = env_id                                                       <L 15>
        }
        // adj: if j == 0:                                                                        <L 14>
        wp::adj_address(var_user_ids, var_1, adj_user_ids, adj_1, adj_5);
        // adj: user_id = wp.int32(user_ids[j])                                                   <L 13>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 12>
        // adj: i, j = wp.tid()                                                                   <L 11>
        // adj: def _write_scalar_user_to_backend_with_indices_and_sim_ids(                       <L 1>
        continue;
    }
}



extern "C" __global__ void _write_scalar_user_to_backend_with_mask_3a68ec9b_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_input_value,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_user_mask,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> var_backend_data)
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
        bool var_2;
        bool* var_3;
        bool var_4;
        bool var_5;
        bool* var_6;
        bool var_7;
        bool var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        //---------
        // forward
        // def _write_scalar_user_to_backend_with_mask(                                           <L 1385>
        // env_id, user_id = wp.tid()                                                             <L 1406>
        builtin_tid2d(var_0, var_1);
        // if not env_mask[env_id] or not user_mask[user_id]:                                     <L 1407>
        var_3 = wp::address(var_env_mask, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::unot(var_5);
        var_2 = var_4;
        if (!var_2) {
            var_6 = wp::address(var_user_mask, var_1);
            var_8 = wp::load(var_6);
            var_7 = wp::unot(var_8);
            var_2 = var_2 || var_7;
        }
        if (var_2) {
            // return                                                                             <L 1408>
            continue;
        }
        // user_data[env_id, user_id] = input_value                                               <L 1410>
        wp::array_store(var_user_data, var_0, var_1, var_input_value);
        // if has_ordering:                                                                       <L 1411>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 1412>
            var_9 = wp::address(var_user_to_backend, var_1);
            var_11 = wp::load(var_9);
            var_10 = wp::copy(var_11);
            // backend_data[env_id, backend_id] = input_value                                     <L 1413>
            wp::array_store(var_backend_data, var_0, var_10, var_input_value);
        }
    }
}



extern "C" __global__ void _write_scalar_user_to_backend_with_mask_3a68ec9b_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_input_value,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_user_mask,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> var_backend_data,
    wp::float32 adj_input_value,
    wp::array_t<bool> adj_env_mask,
    wp::array_t<bool> adj_user_mask,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    wp::array_t<wp::float32> adj_user_data,
    wp::array_t<wp::float32> adj_backend_data)
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
        bool var_2;
        bool* var_3;
        bool var_4;
        bool var_5;
        bool* var_6;
        bool var_7;
        bool var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        bool adj_6 = {};
        bool adj_7 = {};
        bool adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        //---------
        // forward
        // def _write_scalar_user_to_backend_with_mask(                                           <L 1385>
        // env_id, user_id = wp.tid()                                                             <L 1406>
        builtin_tid2d(var_0, var_1);
        // if not env_mask[env_id] or not user_mask[user_id]:                                     <L 1407>
        var_3 = wp::address(var_env_mask, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::unot(var_5);
        var_2 = var_4;
        if (!var_2) {
            var_6 = wp::address(var_user_mask, var_1);
            var_8 = wp::load(var_6);
            var_7 = wp::unot(var_8);
            var_2 = var_2 || var_7;
        }
        if (var_2) {
            // return                                                                             <L 1408>
            goto label0;
        }
        // user_data[env_id, user_id] = input_value                                               <L 1410>
        // wp::array_store(var_user_data, var_0, var_1, var_input_value);
        // if has_ordering:                                                                       <L 1411>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 1412>
            var_9 = wp::address(var_user_to_backend, var_1);
            var_11 = wp::load(var_9);
            var_10 = wp::copy(var_11);
            // backend_data[env_id, backend_id] = input_value                                     <L 1413>
            // wp::array_store(var_backend_data, var_0, var_10, var_input_value);
        }
        //---------
        // reverse
        if (var_has_ordering) {
            wp::adj_array_store(var_backend_data, var_0, var_10, var_input_value, adj_backend_data, adj_0, adj_10, adj_input_value);
            // adj: backend_data[env_id, backend_id] = input_value                                <L 1413>
            wp::adj_copy(var_11, adj_9, adj_10);
            wp::adj_address(var_user_to_backend, var_1, adj_user_to_backend, adj_1, adj_9);
            // adj: backend_id = user_to_backend[user_id]                                         <L 1412>
        }
        // adj: if has_ordering:                                                                  <L 1411>
        wp::adj_array_store(var_user_data, var_0, var_1, var_input_value, adj_user_data, adj_0, adj_1, adj_input_value);
        // adj: user_data[env_id, user_id] = input_value                                          <L 1410>
        if (var_2) {
            label0:;
            // adj: return                                                                        <L 1408>
        }
        if (!var_2) {
            wp::adj_address(var_user_mask, var_1, adj_user_mask, adj_1, adj_6);
        }
        wp::adj_address(var_env_mask, var_0, adj_env_mask, adj_0, adj_3);
        // adj: if not env_mask[env_id] or not user_mask[user_id]:                                <L 1407>
        // adj: env_id, user_id = wp.tid()                                                        <L 1406>
        // adj: def _write_scalar_user_to_backend_with_mask(                                      <L 1385>
        continue;
    }
}



extern "C" __global__ void reorder_joint_targets_user_to_backend_4478c5b4_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_user_effort,
    wp::array_t<wp::float32> var_user_pos_target,
    wp::array_t<wp::float32> var_user_vel_target,
    wp::array_t<wp::int32> var_backend_to_user,
    bool var_write_effort,
    bool var_write_pos_target,
    bool var_write_vel_target,
    bool var_write_joint_act,
    wp::array_t<wp::float32> var_backend_effort,
    wp::array_t<wp::float32> var_backend_pos_target,
    wp::array_t<wp::float32> var_backend_vel_target,
    wp::array_t<wp::float32> var_backend_joint_act)
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
        bool var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        wp::float32* var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        //---------
        // forward
        // def reorder_joint_targets_user_to_backend(                                             <L 108>
        // env_id, backend_id = wp.tid()                                                          <L 157>
        builtin_tid2d(var_0, var_1);
        // user_id = backend_to_user[backend_id]                                                  <L 158>
        var_2 = wp::address(var_backend_to_user, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if write_effort or write_joint_act:                                                    <L 160>
        var_5 = var_write_effort;
        if (!var_5) {
            var_5 = var_5 || var_write_joint_act;
        }
        if (var_5) {
            // effort = user_effort[env_id, user_id]                                              <L 161>
            var_6 = wp::address(var_user_effort, var_0, var_3);
            var_8 = wp::load(var_6);
            var_7 = wp::copy(var_8);
            // if write_effort:                                                                   <L 162>
            if (var_write_effort) {
                // backend_effort[env_id, backend_id] = effort                                    <L 163>
                wp::array_store(var_backend_effort, var_0, var_1, var_7);
            }
            // if write_joint_act:                                                                <L 164>
            if (var_write_joint_act) {
                // backend_joint_act[env_id, backend_id] = effort                                 <L 165>
                wp::array_store(var_backend_joint_act, var_0, var_1, var_7);
            }
        }
        // if write_pos_target:                                                                   <L 167>
        if (var_write_pos_target) {
            // backend_pos_target[env_id, backend_id] = user_pos_target[env_id, user_id]          <L 168>
            var_9 = wp::address(var_user_pos_target, var_0, var_3);
            var_10 = wp::load(var_9);
            wp::array_store(var_backend_pos_target, var_0, var_1, var_10);
        }
        // if write_vel_target:                                                                   <L 170>
        if (var_write_vel_target) {
            // backend_vel_target[env_id, backend_id] = user_vel_target[env_id, user_id]          <L 171>
            var_11 = wp::address(var_user_vel_target, var_0, var_3);
            var_12 = wp::load(var_11);
            wp::array_store(var_backend_vel_target, var_0, var_1, var_12);
        }
    }
}



extern "C" __global__ void reorder_joint_targets_user_to_backend_4478c5b4_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_user_effort,
    wp::array_t<wp::float32> var_user_pos_target,
    wp::array_t<wp::float32> var_user_vel_target,
    wp::array_t<wp::int32> var_backend_to_user,
    bool var_write_effort,
    bool var_write_pos_target,
    bool var_write_vel_target,
    bool var_write_joint_act,
    wp::array_t<wp::float32> var_backend_effort,
    wp::array_t<wp::float32> var_backend_pos_target,
    wp::array_t<wp::float32> var_backend_vel_target,
    wp::array_t<wp::float32> var_backend_joint_act,
    wp::array_t<wp::float32> adj_user_effort,
    wp::array_t<wp::float32> adj_user_pos_target,
    wp::array_t<wp::float32> adj_user_vel_target,
    wp::array_t<wp::int32> adj_backend_to_user,
    bool adj_write_effort,
    bool adj_write_pos_target,
    bool adj_write_vel_target,
    bool adj_write_joint_act,
    wp::array_t<wp::float32> adj_backend_effort,
    wp::array_t<wp::float32> adj_backend_pos_target,
    wp::array_t<wp::float32> adj_backend_vel_target,
    wp::array_t<wp::float32> adj_backend_joint_act)
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
        bool var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        wp::float32* var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        bool adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        //---------
        // forward
        // def reorder_joint_targets_user_to_backend(                                             <L 108>
        // env_id, backend_id = wp.tid()                                                          <L 157>
        builtin_tid2d(var_0, var_1);
        // user_id = backend_to_user[backend_id]                                                  <L 158>
        var_2 = wp::address(var_backend_to_user, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if write_effort or write_joint_act:                                                    <L 160>
        var_5 = var_write_effort;
        if (!var_5) {
            var_5 = var_5 || var_write_joint_act;
        }
        if (var_5) {
            // effort = user_effort[env_id, user_id]                                              <L 161>
            var_6 = wp::address(var_user_effort, var_0, var_3);
            var_8 = wp::load(var_6);
            var_7 = wp::copy(var_8);
            // if write_effort:                                                                   <L 162>
            if (var_write_effort) {
                // backend_effort[env_id, backend_id] = effort                                    <L 163>
                // wp::array_store(var_backend_effort, var_0, var_1, var_7);
            }
            // if write_joint_act:                                                                <L 164>
            if (var_write_joint_act) {
                // backend_joint_act[env_id, backend_id] = effort                                 <L 165>
                // wp::array_store(var_backend_joint_act, var_0, var_1, var_7);
            }
        }
        // if write_pos_target:                                                                   <L 167>
        if (var_write_pos_target) {
            // backend_pos_target[env_id, backend_id] = user_pos_target[env_id, user_id]          <L 168>
            var_9 = wp::address(var_user_pos_target, var_0, var_3);
            var_10 = wp::load(var_9);
            // wp::array_store(var_backend_pos_target, var_0, var_1, var_10);
        }
        // if write_vel_target:                                                                   <L 170>
        if (var_write_vel_target) {
            // backend_vel_target[env_id, backend_id] = user_vel_target[env_id, user_id]          <L 171>
            var_11 = wp::address(var_user_vel_target, var_0, var_3);
            var_12 = wp::load(var_11);
            // wp::array_store(var_backend_vel_target, var_0, var_1, var_12);
        }
        //---------
        // reverse
        if (var_write_vel_target) {
            wp::adj_array_store(var_backend_vel_target, var_0, var_1, var_12, adj_backend_vel_target, adj_0, adj_1, adj_11);
            wp::adj_address(var_user_vel_target, var_0, var_3, adj_user_vel_target, adj_0, adj_3, adj_11);
            // adj: backend_vel_target[env_id, backend_id] = user_vel_target[env_id, user_id]     <L 171>
        }
        // adj: if write_vel_target:                                                              <L 170>
        if (var_write_pos_target) {
            wp::adj_array_store(var_backend_pos_target, var_0, var_1, var_10, adj_backend_pos_target, adj_0, adj_1, adj_9);
            wp::adj_address(var_user_pos_target, var_0, var_3, adj_user_pos_target, adj_0, adj_3, adj_9);
            // adj: backend_pos_target[env_id, backend_id] = user_pos_target[env_id, user_id]     <L 168>
        }
        // adj: if write_pos_target:                                                              <L 167>
        if (var_5) {
            if (var_write_joint_act) {
                wp::adj_array_store(var_backend_joint_act, var_0, var_1, var_7, adj_backend_joint_act, adj_0, adj_1, adj_7);
                // adj: backend_joint_act[env_id, backend_id] = effort                            <L 165>
            }
            // adj: if write_joint_act:                                                           <L 164>
            if (var_write_effort) {
                wp::adj_array_store(var_backend_effort, var_0, var_1, var_7, adj_backend_effort, adj_0, adj_1, adj_7);
                // adj: backend_effort[env_id, backend_id] = effort                               <L 163>
            }
            // adj: if write_effort:                                                              <L 162>
            wp::adj_copy(var_8, adj_6, adj_7);
            wp::adj_address(var_user_effort, var_0, var_3, adj_user_effort, adj_0, adj_3, adj_6);
            // adj: effort = user_effort[env_id, user_id]                                         <L 161>
        }
        if (!var_5) {
        }
        // adj: if write_effort or write_joint_act:                                               <L 160>
        wp::adj_copy(var_4, adj_2, adj_3);
        wp::adj_address(var_backend_to_user, var_1, adj_backend_to_user, adj_1, adj_2);
        // adj: user_id = backend_to_user[backend_id]                                             <L 158>
        // adj: env_id, backend_id = wp.tid()                                                     <L 157>
        // adj: def reorder_joint_targets_user_to_backend(                                        <L 108>
        continue;
    }
}



extern "C" __global__ void write_joint_vel_user_to_backend_with_indices_4fbc79b2_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    bool var_full_data,
    wp::array_t<wp::float32> var_user_vel,
    wp::array_t<wp::float32> var_user_prev_vel,
    wp::array_t<wp::float32> var_user_acc,
    wp::array_t<wp::float32> var_backend_vel)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        const wp::float32 var_15 = 0.0;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        //---------
        // forward
        // def write_joint_vel_user_to_backend_with_indices(                                      <L 1>
        // i, j = wp.tid()                                                                        <L 35>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 36>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 37>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if full_data:                                                                          <L 39>
        if (var_full_data) {
            // value = in_data[env_id, user_id]                                                   <L 40>
            var_8 = wp::address(var_in_data, var_3, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
        }
        if (!var_full_data) {
            // value = in_data[i, j]                                                              <L 42>
            var_11 = wp::address(var_in_data, var_0, var_1);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
        }
        var_14 = wp::where(var_full_data, var_9, var_12);
        // user_vel[env_id, user_id] = value                                                      <L 44>
        wp::array_store(var_user_vel, var_3, var_6, var_14);
        // user_prev_vel[env_id, user_id] = value                                                 <L 45>
        wp::array_store(var_user_prev_vel, var_3, var_6, var_14);
        // user_acc[env_id, user_id] = 0.0                                                        <L 46>
        wp::array_store(var_user_acc, var_3, var_6, var_15);
        // if has_ordering:                                                                       <L 47>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 48>
            var_16 = wp::address(var_user_to_backend, var_6);
            var_18 = wp::load(var_16);
            var_17 = wp::copy(var_18);
            // backend_vel[env_id, backend_id] = value                                            <L 49>
            wp::array_store(var_backend_vel, var_3, var_17, var_14);
        }
    }
}



extern "C" __global__ void write_joint_vel_user_to_backend_with_indices_4fbc79b2_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    bool var_full_data,
    wp::array_t<wp::float32> var_user_vel,
    wp::array_t<wp::float32> var_user_prev_vel,
    wp::array_t<wp::float32> var_user_acc,
    wp::array_t<wp::float32> var_backend_vel,
    wp::array_t<wp::float32> adj_in_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_user_ids,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    bool adj_full_data,
    wp::array_t<wp::float32> adj_user_vel,
    wp::array_t<wp::float32> adj_user_prev_vel,
    wp::array_t<wp::float32> adj_user_acc,
    wp::array_t<wp::float32> adj_backend_vel)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        const wp::float32 var_15 = 0.0;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        //---------
        // forward
        // def write_joint_vel_user_to_backend_with_indices(                                      <L 1>
        // i, j = wp.tid()                                                                        <L 35>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 36>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 37>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if full_data:                                                                          <L 39>
        if (var_full_data) {
            // value = in_data[env_id, user_id]                                                   <L 40>
            var_8 = wp::address(var_in_data, var_3, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
        }
        if (!var_full_data) {
            // value = in_data[i, j]                                                              <L 42>
            var_11 = wp::address(var_in_data, var_0, var_1);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
        }
        var_14 = wp::where(var_full_data, var_9, var_12);
        // user_vel[env_id, user_id] = value                                                      <L 44>
        // wp::array_store(var_user_vel, var_3, var_6, var_14);
        // user_prev_vel[env_id, user_id] = value                                                 <L 45>
        // wp::array_store(var_user_prev_vel, var_3, var_6, var_14);
        // user_acc[env_id, user_id] = 0.0                                                        <L 46>
        // wp::array_store(var_user_acc, var_3, var_6, var_15);
        // if has_ordering:                                                                       <L 47>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 48>
            var_16 = wp::address(var_user_to_backend, var_6);
            var_18 = wp::load(var_16);
            var_17 = wp::copy(var_18);
            // backend_vel[env_id, backend_id] = value                                            <L 49>
            // wp::array_store(var_backend_vel, var_3, var_17, var_14);
        }
        //---------
        // reverse
        if (var_has_ordering) {
            wp::adj_array_store(var_backend_vel, var_3, var_17, var_14, adj_backend_vel, adj_3, adj_17, adj_14);
            // adj: backend_vel[env_id, backend_id] = value                                       <L 49>
            wp::adj_copy(var_18, adj_16, adj_17);
            wp::adj_address(var_user_to_backend, var_6, adj_user_to_backend, adj_6, adj_16);
            // adj: backend_id = user_to_backend[user_id]                                         <L 48>
        }
        // adj: if has_ordering:                                                                  <L 47>
        wp::adj_array_store(var_user_acc, var_3, var_6, var_15, adj_user_acc, adj_3, adj_6, adj_15);
        // adj: user_acc[env_id, user_id] = 0.0                                                   <L 46>
        wp::adj_array_store(var_user_prev_vel, var_3, var_6, var_14, adj_user_prev_vel, adj_3, adj_6, adj_14);
        // adj: user_prev_vel[env_id, user_id] = value                                            <L 45>
        wp::adj_array_store(var_user_vel, var_3, var_6, var_14, adj_user_vel, adj_3, adj_6, adj_14);
        // adj: user_vel[env_id, user_id] = value                                                 <L 44>
        wp::adj_where(var_full_data, var_9, var_12, adj_full_data, adj_9, adj_12, adj_14);
        if (!var_full_data) {
            wp::adj_copy(var_13, adj_11, adj_12);
            wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_11);
            // adj: value = in_data[i, j]                                                         <L 42>
        }
        if (var_full_data) {
            wp::adj_copy(var_10, adj_8, adj_9);
            wp::adj_address(var_in_data, var_3, var_6, adj_in_data, adj_3, adj_6, adj_8);
            // adj: value = in_data[env_id, user_id]                                              <L 40>
        }
        // adj: if full_data:                                                                     <L 39>
        wp::adj_address(var_user_ids, var_1, adj_user_ids, adj_1, adj_5);
        // adj: user_id = wp.int32(user_ids[j])                                                   <L 37>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 36>
        // adj: i, j = wp.tid()                                                                   <L 35>
        // adj: def write_joint_vel_user_to_backend_with_indices(                                 <L 1>
        continue;
    }
}



extern "C" __global__ void reorder_body_wrench_user_to_backend_51f62090_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_user_force,
    wp::array_t<wp::vec_t<3, wp::float32>> var_user_torque,
    wp::array_t<wp::int32> var_backend_to_user,
    wp::array_t<wp::vec_t<3, wp::float32>> var_backend_force,
    wp::array_t<wp::vec_t<3, wp::float32>> var_backend_torque)
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
        wp::vec_t<3, wp::float32>* var_5;
        wp::vec_t<3, wp::float32> var_6;
        wp::vec_t<3, wp::float32>* var_7;
        wp::vec_t<3, wp::float32> var_8;
        //---------
        // forward
        // def reorder_body_wrench_user_to_backend(                                               <L 244>
        // env_id, backend_body_id = wp.tid()                                                     <L 266>
        builtin_tid2d(var_0, var_1);
        // user_body_id = backend_to_user[backend_body_id]                                        <L 267>
        var_2 = wp::address(var_backend_to_user, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // backend_force[env_id, backend_body_id] = user_force[env_id, user_body_id]              <L 268>
        var_5 = wp::address(var_user_force, var_0, var_3);
        var_6 = wp::load(var_5);
        wp::array_store(var_backend_force, var_0, var_1, var_6);
        // backend_torque[env_id, backend_body_id] = user_torque[env_id, user_body_id]            <L 269>
        var_7 = wp::address(var_user_torque, var_0, var_3);
        var_8 = wp::load(var_7);
        wp::array_store(var_backend_torque, var_0, var_1, var_8);
    }
}



extern "C" __global__ void reorder_body_wrench_user_to_backend_51f62090_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_user_force,
    wp::array_t<wp::vec_t<3, wp::float32>> var_user_torque,
    wp::array_t<wp::int32> var_backend_to_user,
    wp::array_t<wp::vec_t<3, wp::float32>> var_backend_force,
    wp::array_t<wp::vec_t<3, wp::float32>> var_backend_torque,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_user_force,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_user_torque,
    wp::array_t<wp::int32> adj_backend_to_user,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_backend_force,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_backend_torque)
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
        wp::vec_t<3, wp::float32>* var_5;
        wp::vec_t<3, wp::float32> var_6;
        wp::vec_t<3, wp::float32>* var_7;
        wp::vec_t<3, wp::float32> var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::vec_t<3, wp::float32> adj_5 = {};
        wp::vec_t<3, wp::float32> adj_6 = {};
        wp::vec_t<3, wp::float32> adj_7 = {};
        wp::vec_t<3, wp::float32> adj_8 = {};
        //---------
        // forward
        // def reorder_body_wrench_user_to_backend(                                               <L 244>
        // env_id, backend_body_id = wp.tid()                                                     <L 266>
        builtin_tid2d(var_0, var_1);
        // user_body_id = backend_to_user[backend_body_id]                                        <L 267>
        var_2 = wp::address(var_backend_to_user, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // backend_force[env_id, backend_body_id] = user_force[env_id, user_body_id]              <L 268>
        var_5 = wp::address(var_user_force, var_0, var_3);
        var_6 = wp::load(var_5);
        // wp::array_store(var_backend_force, var_0, var_1, var_6);
        // backend_torque[env_id, backend_body_id] = user_torque[env_id, user_body_id]            <L 269>
        var_7 = wp::address(var_user_torque, var_0, var_3);
        var_8 = wp::load(var_7);
        // wp::array_store(var_backend_torque, var_0, var_1, var_8);
        //---------
        // reverse
        wp::adj_array_store(var_backend_torque, var_0, var_1, var_8, adj_backend_torque, adj_0, adj_1, adj_7);
        wp::adj_address(var_user_torque, var_0, var_3, adj_user_torque, adj_0, adj_3, adj_7);
        // adj: backend_torque[env_id, backend_body_id] = user_torque[env_id, user_body_id]       <L 269>
        wp::adj_array_store(var_backend_force, var_0, var_1, var_6, adj_backend_force, adj_0, adj_1, adj_5);
        wp::adj_address(var_user_force, var_0, var_3, adj_user_force, adj_0, adj_3, adj_5);
        // adj: backend_force[env_id, backend_body_id] = user_force[env_id, user_body_id]         <L 268>
        wp::adj_copy(var_4, adj_2, adj_3);
        wp::adj_address(var_backend_to_user, var_1, adj_backend_to_user, adj_1, adj_2);
        // adj: user_body_id = backend_to_user[backend_body_id]                                   <L 267>
        // adj: env_id, backend_body_id = wp.tid()                                                <L 266>
        // adj: def reorder_body_wrench_user_to_backend(                                          <L 244>
        continue;
    }
}



extern "C" __global__ void reorder_jacobian_backend_to_user_66b93245_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<wp::float32> var_backend_data,
    wp::array_t<wp::int32> var_jacobian_body_user_to_backend,
    wp::array_t<wp::int32> var_joint_user_to_backend,
    wp::array_t<wp::int32> var_joint_dof_signs,
    wp::int32 var_num_base_dofs,
    bool var_has_body_ordering,
    bool var_has_joint_ordering,
    wp::array_t<wp::float32> var_user_data)
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
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        bool var_10;
        bool var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        const wp::float32 var_17 = 1.0;
        bool var_18;
        wp::int32 var_19;
        wp::int32* var_20;
        wp::float32 var_21;
        wp::int32 var_22;
        wp::float32 var_23;
        wp::float32* var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        //---------
        // forward
        // def reorder_jacobian_backend_to_user(                                                  <L 340>
        // env_id, user_body_id, spatial_id, user_dof_id = wp.tid()                               <L 371>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // backend_body_id = user_body_id                                                         <L 372>
        var_4 = wp::copy(var_1);
        // if has_body_ordering:                                                                  <L 373>
        if (var_has_body_ordering) {
            // backend_body_id = jacobian_body_user_to_backend[user_body_id]                      <L 374>
            var_5 = wp::address(var_jacobian_body_user_to_backend, var_1);
            var_7 = wp::load(var_5);
            var_6 = wp::copy(var_7);
        }
        var_8 = wp::where(var_has_body_ordering, var_6, var_4);
        // backend_dof_id = user_dof_id                                                           <L 376>
        var_9 = wp::copy(var_3);
        // if has_joint_ordering and user_dof_id >= num_base_dofs:                                <L 377>
        var_10 = var_has_joint_ordering;
        if (var_10) {
            var_11 = (var_3 >= var_num_base_dofs);
            var_10 = var_10 && var_11;
        }
        if (var_10) {
            // backend_dof_id = num_base_dofs + joint_user_to_backend[user_dof_id - num_base_dofs]       <L 378>
            var_12 = wp::sub(var_3, var_num_base_dofs);
            var_13 = wp::address(var_joint_user_to_backend, var_12);
            var_15 = wp::load(var_13);
            var_14 = wp::add(var_num_base_dofs, var_15);
        }
        var_16 = wp::where(var_10, var_14, var_9);
        // sign = 1.0                                                                             <L 380>
        // if user_dof_id >= num_base_dofs:                                                       <L 381>
        var_18 = (var_3 >= var_num_base_dofs);
        if (var_18) {
            // sign = wp.float32(joint_dof_signs[backend_dof_id - num_base_dofs])                 <L 382>
            var_19 = wp::sub(var_16, var_num_base_dofs);
            var_20 = wp::address(var_joint_dof_signs, var_19);
            var_22 = wp::load(var_20);
            var_21 = wp::float32(var_22);
        }
        var_23 = wp::where(var_18, var_21, var_17);
        // user_data[env_id, user_body_id, spatial_id, user_dof_id] = (                           <L 383>
        // sign * backend_data[env_id, backend_body_id, spatial_id, backend_dof_id]               <L 384>
        var_24 = wp::address(var_backend_data, var_0, var_8, var_2, var_16);
        var_26 = wp::load(var_24);
        var_25 = wp::mul(var_23, var_26);
        // user_data[env_id, user_body_id, spatial_id, user_dof_id] = (                           <L 383>
        wp::array_store(var_user_data, var_0, var_1, var_2, var_3, var_25);
    }
}



extern "C" __global__ void reorder_jacobian_backend_to_user_66b93245_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<wp::float32> var_backend_data,
    wp::array_t<wp::int32> var_jacobian_body_user_to_backend,
    wp::array_t<wp::int32> var_joint_user_to_backend,
    wp::array_t<wp::int32> var_joint_dof_signs,
    wp::int32 var_num_base_dofs,
    bool var_has_body_ordering,
    bool var_has_joint_ordering,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> adj_backend_data,
    wp::array_t<wp::int32> adj_jacobian_body_user_to_backend,
    wp::array_t<wp::int32> adj_joint_user_to_backend,
    wp::array_t<wp::int32> adj_joint_dof_signs,
    wp::int32 adj_num_base_dofs,
    bool adj_has_body_ordering,
    bool adj_has_joint_ordering,
    wp::array_t<wp::float32> adj_user_data)
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
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        bool var_10;
        bool var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        const wp::float32 var_17 = 1.0;
        bool var_18;
        wp::int32 var_19;
        wp::int32* var_20;
        wp::float32 var_21;
        wp::int32 var_22;
        wp::float32 var_23;
        wp::float32* var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        bool adj_10 = {};
        bool adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::float32 adj_17 = {};
        bool adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::float32 adj_25 = {};
        wp::float32 adj_26 = {};
        //---------
        // forward
        // def reorder_jacobian_backend_to_user(                                                  <L 340>
        // env_id, user_body_id, spatial_id, user_dof_id = wp.tid()                               <L 371>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // backend_body_id = user_body_id                                                         <L 372>
        var_4 = wp::copy(var_1);
        // if has_body_ordering:                                                                  <L 373>
        if (var_has_body_ordering) {
            // backend_body_id = jacobian_body_user_to_backend[user_body_id]                      <L 374>
            var_5 = wp::address(var_jacobian_body_user_to_backend, var_1);
            var_7 = wp::load(var_5);
            var_6 = wp::copy(var_7);
        }
        var_8 = wp::where(var_has_body_ordering, var_6, var_4);
        // backend_dof_id = user_dof_id                                                           <L 376>
        var_9 = wp::copy(var_3);
        // if has_joint_ordering and user_dof_id >= num_base_dofs:                                <L 377>
        var_10 = var_has_joint_ordering;
        if (var_10) {
            var_11 = (var_3 >= var_num_base_dofs);
            var_10 = var_10 && var_11;
        }
        if (var_10) {
            // backend_dof_id = num_base_dofs + joint_user_to_backend[user_dof_id - num_base_dofs]       <L 378>
            var_12 = wp::sub(var_3, var_num_base_dofs);
            var_13 = wp::address(var_joint_user_to_backend, var_12);
            var_15 = wp::load(var_13);
            var_14 = wp::add(var_num_base_dofs, var_15);
        }
        var_16 = wp::where(var_10, var_14, var_9);
        // sign = 1.0                                                                             <L 380>
        // if user_dof_id >= num_base_dofs:                                                       <L 381>
        var_18 = (var_3 >= var_num_base_dofs);
        if (var_18) {
            // sign = wp.float32(joint_dof_signs[backend_dof_id - num_base_dofs])                 <L 382>
            var_19 = wp::sub(var_16, var_num_base_dofs);
            var_20 = wp::address(var_joint_dof_signs, var_19);
            var_22 = wp::load(var_20);
            var_21 = wp::float32(var_22);
        }
        var_23 = wp::where(var_18, var_21, var_17);
        // user_data[env_id, user_body_id, spatial_id, user_dof_id] = (                           <L 383>
        // sign * backend_data[env_id, backend_body_id, spatial_id, backend_dof_id]               <L 384>
        var_24 = wp::address(var_backend_data, var_0, var_8, var_2, var_16);
        var_26 = wp::load(var_24);
        var_25 = wp::mul(var_23, var_26);
        // user_data[env_id, user_body_id, spatial_id, user_dof_id] = (                           <L 383>
        // wp::array_store(var_user_data, var_0, var_1, var_2, var_3, var_25);
        //---------
        // reverse
        wp::adj_array_store(var_user_data, var_0, var_1, var_2, var_3, var_25, adj_user_data, adj_0, adj_1, adj_2, adj_3, adj_25);
        // adj: user_data[env_id, user_body_id, spatial_id, user_dof_id] = (                      <L 383>
        wp::adj_mul(var_23, var_26, adj_23, adj_24, adj_25);
        wp::adj_address(var_backend_data, var_0, var_8, var_2, var_16, adj_backend_data, adj_0, adj_8, adj_2, adj_16, adj_24);
        // adj: sign * backend_data[env_id, backend_body_id, spatial_id, backend_dof_id]          <L 384>
        // adj: user_data[env_id, user_body_id, spatial_id, user_dof_id] = (                      <L 383>
        wp::adj_where(var_18, var_21, var_17, adj_18, adj_21, adj_17, adj_23);
        if (var_18) {
            wp::adj_float32(var_22, adj_20, adj_21);
            wp::adj_address(var_joint_dof_signs, var_19, adj_joint_dof_signs, adj_19, adj_20);
            wp::adj_sub(var_16, var_num_base_dofs, adj_16, adj_num_base_dofs, adj_19);
            // adj: sign = wp.float32(joint_dof_signs[backend_dof_id - num_base_dofs])            <L 382>
        }
        // adj: if user_dof_id >= num_base_dofs:                                                  <L 381>
        // adj: sign = 1.0                                                                        <L 380>
        wp::adj_where(var_10, var_14, var_9, adj_10, adj_14, adj_9, adj_16);
        if (var_10) {
            wp::adj_add(var_num_base_dofs, var_15, adj_num_base_dofs, adj_13, adj_14);
            wp::adj_address(var_joint_user_to_backend, var_12, adj_joint_user_to_backend, adj_12, adj_13);
            wp::adj_sub(var_3, var_num_base_dofs, adj_3, adj_num_base_dofs, adj_12);
            // adj: backend_dof_id = num_base_dofs + joint_user_to_backend[user_dof_id - num_base_dofs]  <L 378>
        }
        if (var_10) {
        }
        // adj: if has_joint_ordering and user_dof_id >= num_base_dofs:                           <L 377>
        wp::adj_copy(var_3, adj_3, adj_9);
        // adj: backend_dof_id = user_dof_id                                                      <L 376>
        wp::adj_where(var_has_body_ordering, var_6, var_4, adj_has_body_ordering, adj_6, adj_4, adj_8);
        if (var_has_body_ordering) {
            wp::adj_copy(var_7, adj_5, adj_6);
            wp::adj_address(var_jacobian_body_user_to_backend, var_1, adj_jacobian_body_user_to_backend, adj_1, adj_5);
            // adj: backend_body_id = jacobian_body_user_to_backend[user_body_id]                 <L 374>
        }
        // adj: if has_body_ordering:                                                             <L 373>
        wp::adj_copy(var_1, adj_1, adj_4);
        // adj: backend_body_id = user_body_id                                                    <L 372>
        // adj: env_id, user_body_id, spatial_id, user_dof_id = wp.tid()                          <L 371>
        // adj: def reorder_jacobian_backend_to_user(                                             <L 340>
        continue;
    }
}



extern "C" __global__ void _write_2d_user_to_backend_with_indices_69eedb69_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_input_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    bool var_full_data,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> var_backend_data)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        //---------
        // forward
        // def _write_2d_user_to_backend_with_indices(                                            <L 1>
        // i, j = wp.tid()                                                                        <L 26>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 27>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 28>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if full_data:                                                                          <L 29>
        if (var_full_data) {
            // value = input_data[env_id, user_id]                                                <L 30>
            var_8 = wp::address(var_input_data, var_3, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
        }
        if (!var_full_data) {
            // value = input_data[i, j]                                                           <L 32>
            var_11 = wp::address(var_input_data, var_0, var_1);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
        }
        var_14 = wp::where(var_full_data, var_9, var_12);
        // user_data[env_id, user_id] = value                                                     <L 34>
        wp::array_store(var_user_data, var_3, var_6, var_14);
        // if has_ordering:                                                                       <L 35>
        if (var_has_ordering) {
            // backend_data[env_id, user_to_backend[user_id]] = value                             <L 36>
            var_15 = wp::address(var_user_to_backend, var_6);
            var_16 = wp::load(var_15);
            wp::array_store(var_backend_data, var_3, var_16, var_14);
        }
    }
}



extern "C" __global__ void _write_2d_user_to_backend_with_indices_69eedb69_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_input_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    bool var_full_data,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> var_backend_data,
    wp::array_t<wp::float32> adj_input_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_user_ids,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    bool adj_full_data,
    wp::array_t<wp::float32> adj_user_data,
    wp::array_t<wp::float32> adj_backend_data)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        //---------
        // forward
        // def _write_2d_user_to_backend_with_indices(                                            <L 1>
        // i, j = wp.tid()                                                                        <L 26>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 27>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 28>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if full_data:                                                                          <L 29>
        if (var_full_data) {
            // value = input_data[env_id, user_id]                                                <L 30>
            var_8 = wp::address(var_input_data, var_3, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
        }
        if (!var_full_data) {
            // value = input_data[i, j]                                                           <L 32>
            var_11 = wp::address(var_input_data, var_0, var_1);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
        }
        var_14 = wp::where(var_full_data, var_9, var_12);
        // user_data[env_id, user_id] = value                                                     <L 34>
        // wp::array_store(var_user_data, var_3, var_6, var_14);
        // if has_ordering:                                                                       <L 35>
        if (var_has_ordering) {
            // backend_data[env_id, user_to_backend[user_id]] = value                             <L 36>
            var_15 = wp::address(var_user_to_backend, var_6);
            var_16 = wp::load(var_15);
            // wp::array_store(var_backend_data, var_3, var_16, var_14);
        }
        //---------
        // reverse
        if (var_has_ordering) {
            wp::adj_array_store(var_backend_data, var_3, var_16, var_14, adj_backend_data, adj_3, adj_15, adj_14);
            wp::adj_address(var_user_to_backend, var_6, adj_user_to_backend, adj_6, adj_15);
            // adj: backend_data[env_id, user_to_backend[user_id]] = value                        <L 36>
        }
        // adj: if has_ordering:                                                                  <L 35>
        wp::adj_array_store(var_user_data, var_3, var_6, var_14, adj_user_data, adj_3, adj_6, adj_14);
        // adj: user_data[env_id, user_id] = value                                                <L 34>
        wp::adj_where(var_full_data, var_9, var_12, adj_full_data, adj_9, adj_12, adj_14);
        if (!var_full_data) {
            wp::adj_copy(var_13, adj_11, adj_12);
            wp::adj_address(var_input_data, var_0, var_1, adj_input_data, adj_0, adj_1, adj_11);
            // adj: value = input_data[i, j]                                                      <L 32>
        }
        if (var_full_data) {
            wp::adj_copy(var_10, adj_8, adj_9);
            wp::adj_address(var_input_data, var_3, var_6, adj_input_data, adj_3, adj_6, adj_8);
            // adj: value = input_data[env_id, user_id]                                           <L 30>
        }
        // adj: if full_data:                                                                     <L 29>
        wp::adj_address(var_user_ids, var_1, adj_user_ids, adj_1, adj_5);
        // adj: user_id = wp.int32(user_ids[j])                                                   <L 28>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 27>
        // adj: i, j = wp.tid()                                                                   <L 26>
        // adj: def _write_2d_user_to_backend_with_indices(                                       <L 1>
        continue;
    }
}



extern "C" __global__ void _write_scalar_user_to_backend_with_indices_6f6ec8d6_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_input_value,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> var_backend_data)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        //---------
        // forward
        // def _write_scalar_user_to_backend_with_indices(                                        <L 1>
        // i, j = wp.tid()                                                                        <L 25>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 26>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 27>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // user_data[env_id, user_id] = input_value                                               <L 29>
        wp::array_store(var_user_data, var_3, var_6, var_input_value);
        // if has_ordering:                                                                       <L 30>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 31>
            var_8 = wp::address(var_user_to_backend, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
            // backend_data[env_id, backend_id] = input_value                                     <L 32>
            wp::array_store(var_backend_data, var_3, var_9, var_input_value);
        }
    }
}



extern "C" __global__ void _write_scalar_user_to_backend_with_indices_6f6ec8d6_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_input_value,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> var_backend_data,
    wp::float32 adj_input_value,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int64> adj_user_ids,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    wp::array_t<wp::float32> adj_user_data,
    wp::array_t<wp::float32> adj_backend_data)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int64 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int64 adj_4 = {};
        wp::int64 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int64 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        //---------
        // forward
        // def _write_scalar_user_to_backend_with_indices(                                        <L 1>
        // i, j = wp.tid()                                                                        <L 25>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 26>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 27>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // user_data[env_id, user_id] = input_value                                               <L 29>
        // wp::array_store(var_user_data, var_3, var_6, var_input_value);
        // if has_ordering:                                                                       <L 30>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 31>
            var_8 = wp::address(var_user_to_backend, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
            // backend_data[env_id, backend_id] = input_value                                     <L 32>
            // wp::array_store(var_backend_data, var_3, var_9, var_input_value);
        }
        //---------
        // reverse
        if (var_has_ordering) {
            wp::adj_array_store(var_backend_data, var_3, var_9, var_input_value, adj_backend_data, adj_3, adj_9, adj_input_value);
            // adj: backend_data[env_id, backend_id] = input_value                                <L 32>
            wp::adj_copy(var_10, adj_8, adj_9);
            wp::adj_address(var_user_to_backend, var_6, adj_user_to_backend, adj_6, adj_8);
            // adj: backend_id = user_to_backend[user_id]                                         <L 31>
        }
        // adj: if has_ordering:                                                                  <L 30>
        wp::adj_array_store(var_user_data, var_3, var_6, var_input_value, adj_user_data, adj_3, adj_6, adj_input_value);
        // adj: user_data[env_id, user_id] = input_value                                          <L 29>
        wp::adj_address(var_user_ids, var_1, adj_user_ids, adj_1, adj_5);
        // adj: user_id = wp.int32(user_ids[j])                                                   <L 27>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 26>
        // adj: i, j = wp.tid()                                                                   <L 25>
        // adj: def _write_scalar_user_to_backend_with_indices(                                   <L 1>
        continue;
    }
}



extern "C" __global__ void write_joint_state_user_to_backend_with_indices_759c1fec_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_pos_data,
    wp::array_t<wp::float32> var_vel_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    bool var_full_data,
    wp::array_t<wp::float32> var_user_pos,
    wp::array_t<wp::float32> var_user_vel,
    wp::array_t<wp::float32> var_user_prev_vel,
    wp::array_t<wp::float32> var_user_acc,
    wp::array_t<wp::float32> var_backend_pos,
    wp::array_t<wp::float32> var_backend_vel)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        const wp::float32 var_22 = 0.0;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        //---------
        // forward
        // def write_joint_state_user_to_backend_with_indices(                                    <L 1>
        // i, j = wp.tid()                                                                        <L 43>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 44>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 45>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if full_data:                                                                          <L 47>
        if (var_full_data) {
            // position = pos_data[env_id, user_id]                                               <L 48>
            var_8 = wp::address(var_pos_data, var_3, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
            // velocity = vel_data[env_id, user_id]                                               <L 49>
            var_11 = wp::address(var_vel_data, var_3, var_6);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
        }
        if (!var_full_data) {
            // position = pos_data[i, j]                                                          <L 51>
            var_14 = wp::address(var_pos_data, var_0, var_1);
            var_16 = wp::load(var_14);
            var_15 = wp::copy(var_16);
            // velocity = vel_data[i, j]                                                          <L 52>
            var_17 = wp::address(var_vel_data, var_0, var_1);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
        }
        var_20 = wp::where(var_full_data, var_9, var_15);
        var_21 = wp::where(var_full_data, var_12, var_18);
        // user_pos[env_id, user_id] = position                                                   <L 54>
        wp::array_store(var_user_pos, var_3, var_6, var_20);
        // user_vel[env_id, user_id] = velocity                                                   <L 55>
        wp::array_store(var_user_vel, var_3, var_6, var_21);
        // user_prev_vel[env_id, user_id] = velocity                                              <L 56>
        wp::array_store(var_user_prev_vel, var_3, var_6, var_21);
        // user_acc[env_id, user_id] = 0.0                                                        <L 57>
        wp::array_store(var_user_acc, var_3, var_6, var_22);
        // if has_ordering:                                                                       <L 58>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 59>
            var_23 = wp::address(var_user_to_backend, var_6);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // backend_pos[env_id, backend_id] = position                                         <L 60>
            wp::array_store(var_backend_pos, var_3, var_24, var_20);
            // backend_vel[env_id, backend_id] = velocity                                         <L 61>
            wp::array_store(var_backend_vel, var_3, var_24, var_21);
        }
    }
}



extern "C" __global__ void write_joint_state_user_to_backend_with_indices_759c1fec_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_pos_data,
    wp::array_t<wp::float32> var_vel_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    bool var_full_data,
    wp::array_t<wp::float32> var_user_pos,
    wp::array_t<wp::float32> var_user_vel,
    wp::array_t<wp::float32> var_user_prev_vel,
    wp::array_t<wp::float32> var_user_acc,
    wp::array_t<wp::float32> var_backend_pos,
    wp::array_t<wp::float32> var_backend_vel,
    wp::array_t<wp::float32> adj_pos_data,
    wp::array_t<wp::float32> adj_vel_data,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int32> adj_user_ids,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    bool adj_full_data,
    wp::array_t<wp::float32> adj_user_pos,
    wp::array_t<wp::float32> adj_user_vel,
    wp::array_t<wp::float32> adj_user_prev_vel,
    wp::array_t<wp::float32> adj_user_acc,
    wp::array_t<wp::float32> adj_backend_pos,
    wp::array_t<wp::float32> adj_backend_vel)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        const wp::float32 var_22 = 0.0;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int64 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int64 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        //---------
        // forward
        // def write_joint_state_user_to_backend_with_indices(                                    <L 1>
        // i, j = wp.tid()                                                                        <L 43>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 44>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 45>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if full_data:                                                                          <L 47>
        if (var_full_data) {
            // position = pos_data[env_id, user_id]                                               <L 48>
            var_8 = wp::address(var_pos_data, var_3, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
            // velocity = vel_data[env_id, user_id]                                               <L 49>
            var_11 = wp::address(var_vel_data, var_3, var_6);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
        }
        if (!var_full_data) {
            // position = pos_data[i, j]                                                          <L 51>
            var_14 = wp::address(var_pos_data, var_0, var_1);
            var_16 = wp::load(var_14);
            var_15 = wp::copy(var_16);
            // velocity = vel_data[i, j]                                                          <L 52>
            var_17 = wp::address(var_vel_data, var_0, var_1);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
        }
        var_20 = wp::where(var_full_data, var_9, var_15);
        var_21 = wp::where(var_full_data, var_12, var_18);
        // user_pos[env_id, user_id] = position                                                   <L 54>
        // wp::array_store(var_user_pos, var_3, var_6, var_20);
        // user_vel[env_id, user_id] = velocity                                                   <L 55>
        // wp::array_store(var_user_vel, var_3, var_6, var_21);
        // user_prev_vel[env_id, user_id] = velocity                                              <L 56>
        // wp::array_store(var_user_prev_vel, var_3, var_6, var_21);
        // user_acc[env_id, user_id] = 0.0                                                        <L 57>
        // wp::array_store(var_user_acc, var_3, var_6, var_22);
        // if has_ordering:                                                                       <L 58>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 59>
            var_23 = wp::address(var_user_to_backend, var_6);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // backend_pos[env_id, backend_id] = position                                         <L 60>
            // wp::array_store(var_backend_pos, var_3, var_24, var_20);
            // backend_vel[env_id, backend_id] = velocity                                         <L 61>
            // wp::array_store(var_backend_vel, var_3, var_24, var_21);
        }
        //---------
        // reverse
        if (var_has_ordering) {
            wp::adj_array_store(var_backend_vel, var_3, var_24, var_21, adj_backend_vel, adj_3, adj_24, adj_21);
            // adj: backend_vel[env_id, backend_id] = velocity                                    <L 61>
            wp::adj_array_store(var_backend_pos, var_3, var_24, var_20, adj_backend_pos, adj_3, adj_24, adj_20);
            // adj: backend_pos[env_id, backend_id] = position                                    <L 60>
            wp::adj_copy(var_25, adj_23, adj_24);
            wp::adj_address(var_user_to_backend, var_6, adj_user_to_backend, adj_6, adj_23);
            // adj: backend_id = user_to_backend[user_id]                                         <L 59>
        }
        // adj: if has_ordering:                                                                  <L 58>
        wp::adj_array_store(var_user_acc, var_3, var_6, var_22, adj_user_acc, adj_3, adj_6, adj_22);
        // adj: user_acc[env_id, user_id] = 0.0                                                   <L 57>
        wp::adj_array_store(var_user_prev_vel, var_3, var_6, var_21, adj_user_prev_vel, adj_3, adj_6, adj_21);
        // adj: user_prev_vel[env_id, user_id] = velocity                                         <L 56>
        wp::adj_array_store(var_user_vel, var_3, var_6, var_21, adj_user_vel, adj_3, adj_6, adj_21);
        // adj: user_vel[env_id, user_id] = velocity                                              <L 55>
        wp::adj_array_store(var_user_pos, var_3, var_6, var_20, adj_user_pos, adj_3, adj_6, adj_20);
        // adj: user_pos[env_id, user_id] = position                                              <L 54>
        wp::adj_where(var_full_data, var_12, var_18, adj_full_data, adj_12, adj_18, adj_21);
        wp::adj_where(var_full_data, var_9, var_15, adj_full_data, adj_9, adj_15, adj_20);
        if (!var_full_data) {
            wp::adj_copy(var_19, adj_17, adj_18);
            wp::adj_address(var_vel_data, var_0, var_1, adj_vel_data, adj_0, adj_1, adj_17);
            // adj: velocity = vel_data[i, j]                                                     <L 52>
            wp::adj_copy(var_16, adj_14, adj_15);
            wp::adj_address(var_pos_data, var_0, var_1, adj_pos_data, adj_0, adj_1, adj_14);
            // adj: position = pos_data[i, j]                                                     <L 51>
        }
        if (var_full_data) {
            wp::adj_copy(var_13, adj_11, adj_12);
            wp::adj_address(var_vel_data, var_3, var_6, adj_vel_data, adj_3, adj_6, adj_11);
            // adj: velocity = vel_data[env_id, user_id]                                          <L 49>
            wp::adj_copy(var_10, adj_8, adj_9);
            wp::adj_address(var_pos_data, var_3, var_6, adj_pos_data, adj_3, adj_6, adj_8);
            // adj: position = pos_data[env_id, user_id]                                          <L 48>
        }
        // adj: if full_data:                                                                     <L 47>
        wp::adj_address(var_user_ids, var_1, adj_user_ids, adj_1, adj_5);
        // adj: user_id = wp.int32(user_ids[j])                                                   <L 45>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 44>
        // adj: i, j = wp.tid()                                                                   <L 43>
        // adj: def write_joint_state_user_to_backend_with_indices(                               <L 1>
        continue;
    }
}



extern "C" __global__ void _write_scalar_user_to_backend_with_indices_76069602_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_input_value,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> var_backend_data)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        //---------
        // forward
        // def _write_scalar_user_to_backend_with_indices(                                        <L 1>
        // i, j = wp.tid()                                                                        <L 25>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 26>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 27>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // user_data[env_id, user_id] = input_value                                               <L 29>
        wp::array_store(var_user_data, var_3, var_6, var_input_value);
        // if has_ordering:                                                                       <L 30>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 31>
            var_8 = wp::address(var_user_to_backend, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
            // backend_data[env_id, backend_id] = input_value                                     <L 32>
            wp::array_store(var_backend_data, var_3, var_9, var_input_value);
        }
    }
}



extern "C" __global__ void _write_scalar_user_to_backend_with_indices_76069602_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_input_value,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> var_backend_data,
    wp::float32 adj_input_value,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int32> adj_user_ids,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    wp::array_t<wp::float32> adj_user_data,
    wp::array_t<wp::float32> adj_backend_data)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int64 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int64 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        //---------
        // forward
        // def _write_scalar_user_to_backend_with_indices(                                        <L 1>
        // i, j = wp.tid()                                                                        <L 25>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 26>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 27>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // user_data[env_id, user_id] = input_value                                               <L 29>
        // wp::array_store(var_user_data, var_3, var_6, var_input_value);
        // if has_ordering:                                                                       <L 30>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 31>
            var_8 = wp::address(var_user_to_backend, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
            // backend_data[env_id, backend_id] = input_value                                     <L 32>
            // wp::array_store(var_backend_data, var_3, var_9, var_input_value);
        }
        //---------
        // reverse
        if (var_has_ordering) {
            wp::adj_array_store(var_backend_data, var_3, var_9, var_input_value, adj_backend_data, adj_3, adj_9, adj_input_value);
            // adj: backend_data[env_id, backend_id] = input_value                                <L 32>
            wp::adj_copy(var_10, adj_8, adj_9);
            wp::adj_address(var_user_to_backend, var_6, adj_user_to_backend, adj_6, adj_8);
            // adj: backend_id = user_to_backend[user_id]                                         <L 31>
        }
        // adj: if has_ordering:                                                                  <L 30>
        wp::adj_array_store(var_user_data, var_3, var_6, var_input_value, adj_user_data, adj_3, adj_6, adj_input_value);
        // adj: user_data[env_id, user_id] = input_value                                          <L 29>
        wp::adj_address(var_user_ids, var_1, adj_user_ids, adj_1, adj_5);
        // adj: user_id = wp.int32(user_ids[j])                                                   <L 27>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 26>
        // adj: i, j = wp.tid()                                                                   <L 25>
        // adj: def _write_scalar_user_to_backend_with_indices(                                   <L 1>
        continue;
    }
}



extern "C" __global__ void _write_scalar_user_to_backend_with_indices_and_sim_ids_7860a59e_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_input_value,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> var_backend_data,
    wp::array_t<wp::int32> var_sim_env_ids)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        //---------
        // forward
        // def _write_scalar_user_to_backend_with_indices_and_sim_ids(                            <L 1>
        // i, j = wp.tid()                                                                        <L 11>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 12>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 13>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if j == 0:                                                                             <L 14>
        var_9 = (var_1 == var_8);
        if (var_9) {
            // sim_env_ids[i] = env_id                                                            <L 15>
            wp::array_store(var_sim_env_ids, var_0, var_3);
        }
        // user_data[env_id, user_id] = input_value                                               <L 16>
        wp::array_store(var_user_data, var_3, var_6, var_input_value);
        // if has_ordering:                                                                       <L 17>
        if (var_has_ordering) {
            // backend_data[env_id, user_to_backend[user_id]] = input_value                       <L 18>
            var_10 = wp::address(var_user_to_backend, var_6);
            var_11 = wp::load(var_10);
            wp::array_store(var_backend_data, var_3, var_11, var_input_value);
        }
    }
}



extern "C" __global__ void _write_scalar_user_to_backend_with_indices_and_sim_ids_7860a59e_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_input_value,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> var_backend_data,
    wp::array_t<wp::int32> var_sim_env_ids,
    wp::float32 adj_input_value,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int32> adj_user_ids,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    wp::array_t<wp::float32> adj_user_data,
    wp::array_t<wp::float32> adj_backend_data,
    wp::array_t<wp::int32> adj_sim_env_ids)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int64 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int64 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        bool adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        //---------
        // forward
        // def _write_scalar_user_to_backend_with_indices_and_sim_ids(                            <L 1>
        // i, j = wp.tid()                                                                        <L 11>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 12>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 13>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if j == 0:                                                                             <L 14>
        var_9 = (var_1 == var_8);
        if (var_9) {
            // sim_env_ids[i] = env_id                                                            <L 15>
            // wp::array_store(var_sim_env_ids, var_0, var_3);
        }
        // user_data[env_id, user_id] = input_value                                               <L 16>
        // wp::array_store(var_user_data, var_3, var_6, var_input_value);
        // if has_ordering:                                                                       <L 17>
        if (var_has_ordering) {
            // backend_data[env_id, user_to_backend[user_id]] = input_value                       <L 18>
            var_10 = wp::address(var_user_to_backend, var_6);
            var_11 = wp::load(var_10);
            // wp::array_store(var_backend_data, var_3, var_11, var_input_value);
        }
        //---------
        // reverse
        if (var_has_ordering) {
            wp::adj_array_store(var_backend_data, var_3, var_11, var_input_value, adj_backend_data, adj_3, adj_10, adj_input_value);
            wp::adj_address(var_user_to_backend, var_6, adj_user_to_backend, adj_6, adj_10);
            // adj: backend_data[env_id, user_to_backend[user_id]] = input_value                  <L 18>
        }
        // adj: if has_ordering:                                                                  <L 17>
        wp::adj_array_store(var_user_data, var_3, var_6, var_input_value, adj_user_data, adj_3, adj_6, adj_input_value);
        // adj: user_data[env_id, user_id] = input_value                                          <L 16>
        if (var_9) {
            wp::adj_array_store(var_sim_env_ids, var_0, var_3, adj_sim_env_ids, adj_0, adj_3);
            // adj: sim_env_ids[i] = env_id                                                       <L 15>
        }
        // adj: if j == 0:                                                                        <L 14>
        wp::adj_address(var_user_ids, var_1, adj_user_ids, adj_1, adj_5);
        // adj: user_id = wp.int32(user_ids[j])                                                   <L 13>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 12>
        // adj: i, j = wp.tid()                                                                   <L 11>
        // adj: def _write_scalar_user_to_backend_with_indices_and_sim_ids(                       <L 1>
        continue;
    }
}



extern "C" __global__ void _write_2d_user_to_backend_with_indices_89adcc04_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_input_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    bool var_full_data,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> var_backend_data)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        //---------
        // forward
        // def _write_2d_user_to_backend_with_indices(                                            <L 1>
        // i, j = wp.tid()                                                                        <L 26>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 27>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 28>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if full_data:                                                                          <L 29>
        if (var_full_data) {
            // value = input_data[env_id, user_id]                                                <L 30>
            var_8 = wp::address(var_input_data, var_3, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
        }
        if (!var_full_data) {
            // value = input_data[i, j]                                                           <L 32>
            var_11 = wp::address(var_input_data, var_0, var_1);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
        }
        var_14 = wp::where(var_full_data, var_9, var_12);
        // user_data[env_id, user_id] = value                                                     <L 34>
        wp::array_store(var_user_data, var_3, var_6, var_14);
        // if has_ordering:                                                                       <L 35>
        if (var_has_ordering) {
            // backend_data[env_id, user_to_backend[user_id]] = value                             <L 36>
            var_15 = wp::address(var_user_to_backend, var_6);
            var_16 = wp::load(var_15);
            wp::array_store(var_backend_data, var_3, var_16, var_14);
        }
    }
}



extern "C" __global__ void _write_2d_user_to_backend_with_indices_89adcc04_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_input_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    bool var_full_data,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> var_backend_data,
    wp::array_t<wp::float32> adj_input_data,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int64> adj_user_ids,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    bool adj_full_data,
    wp::array_t<wp::float32> adj_user_data,
    wp::array_t<wp::float32> adj_backend_data)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int64 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int64 adj_4 = {};
        wp::int64 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int64 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        //---------
        // forward
        // def _write_2d_user_to_backend_with_indices(                                            <L 1>
        // i, j = wp.tid()                                                                        <L 26>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 27>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 28>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if full_data:                                                                          <L 29>
        if (var_full_data) {
            // value = input_data[env_id, user_id]                                                <L 30>
            var_8 = wp::address(var_input_data, var_3, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
        }
        if (!var_full_data) {
            // value = input_data[i, j]                                                           <L 32>
            var_11 = wp::address(var_input_data, var_0, var_1);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
        }
        var_14 = wp::where(var_full_data, var_9, var_12);
        // user_data[env_id, user_id] = value                                                     <L 34>
        // wp::array_store(var_user_data, var_3, var_6, var_14);
        // if has_ordering:                                                                       <L 35>
        if (var_has_ordering) {
            // backend_data[env_id, user_to_backend[user_id]] = value                             <L 36>
            var_15 = wp::address(var_user_to_backend, var_6);
            var_16 = wp::load(var_15);
            // wp::array_store(var_backend_data, var_3, var_16, var_14);
        }
        //---------
        // reverse
        if (var_has_ordering) {
            wp::adj_array_store(var_backend_data, var_3, var_16, var_14, adj_backend_data, adj_3, adj_15, adj_14);
            wp::adj_address(var_user_to_backend, var_6, adj_user_to_backend, adj_6, adj_15);
            // adj: backend_data[env_id, user_to_backend[user_id]] = value                        <L 36>
        }
        // adj: if has_ordering:                                                                  <L 35>
        wp::adj_array_store(var_user_data, var_3, var_6, var_14, adj_user_data, adj_3, adj_6, adj_14);
        // adj: user_data[env_id, user_id] = value                                                <L 34>
        wp::adj_where(var_full_data, var_9, var_12, adj_full_data, adj_9, adj_12, adj_14);
        if (!var_full_data) {
            wp::adj_copy(var_13, adj_11, adj_12);
            wp::adj_address(var_input_data, var_0, var_1, adj_input_data, adj_0, adj_1, adj_11);
            // adj: value = input_data[i, j]                                                      <L 32>
        }
        if (var_full_data) {
            wp::adj_copy(var_10, adj_8, adj_9);
            wp::adj_address(var_input_data, var_3, var_6, adj_input_data, adj_3, adj_6, adj_8);
            // adj: value = input_data[env_id, user_id]                                           <L 30>
        }
        // adj: if full_data:                                                                     <L 29>
        wp::adj_address(var_user_ids, var_1, adj_user_ids, adj_1, adj_5);
        // adj: user_id = wp.int32(user_ids[j])                                                   <L 28>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 27>
        // adj: i, j = wp.tid()                                                                   <L 26>
        // adj: def _write_2d_user_to_backend_with_indices(                                       <L 1>
        continue;
    }
}



extern "C" __global__ void _write_2d_user_to_backend_with_indices_8d1dd128_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_input_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    bool var_full_data,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> var_backend_data)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        //---------
        // forward
        // def _write_2d_user_to_backend_with_indices(                                            <L 1>
        // i, j = wp.tid()                                                                        <L 26>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 27>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 28>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if full_data:                                                                          <L 29>
        if (var_full_data) {
            // value = input_data[env_id, user_id]                                                <L 30>
            var_8 = wp::address(var_input_data, var_3, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
        }
        if (!var_full_data) {
            // value = input_data[i, j]                                                           <L 32>
            var_11 = wp::address(var_input_data, var_0, var_1);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
        }
        var_14 = wp::where(var_full_data, var_9, var_12);
        // user_data[env_id, user_id] = value                                                     <L 34>
        wp::array_store(var_user_data, var_3, var_6, var_14);
        // if has_ordering:                                                                       <L 35>
        if (var_has_ordering) {
            // backend_data[env_id, user_to_backend[user_id]] = value                             <L 36>
            var_15 = wp::address(var_user_to_backend, var_6);
            var_16 = wp::load(var_15);
            wp::array_store(var_backend_data, var_3, var_16, var_14);
        }
    }
}



extern "C" __global__ void _write_2d_user_to_backend_with_indices_8d1dd128_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_input_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    bool var_full_data,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> var_backend_data,
    wp::array_t<wp::float32> adj_input_data,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int32> adj_user_ids,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    bool adj_full_data,
    wp::array_t<wp::float32> adj_user_data,
    wp::array_t<wp::float32> adj_backend_data)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int64 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int64 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        //---------
        // forward
        // def _write_2d_user_to_backend_with_indices(                                            <L 1>
        // i, j = wp.tid()                                                                        <L 26>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 27>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 28>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if full_data:                                                                          <L 29>
        if (var_full_data) {
            // value = input_data[env_id, user_id]                                                <L 30>
            var_8 = wp::address(var_input_data, var_3, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
        }
        if (!var_full_data) {
            // value = input_data[i, j]                                                           <L 32>
            var_11 = wp::address(var_input_data, var_0, var_1);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
        }
        var_14 = wp::where(var_full_data, var_9, var_12);
        // user_data[env_id, user_id] = value                                                     <L 34>
        // wp::array_store(var_user_data, var_3, var_6, var_14);
        // if has_ordering:                                                                       <L 35>
        if (var_has_ordering) {
            // backend_data[env_id, user_to_backend[user_id]] = value                             <L 36>
            var_15 = wp::address(var_user_to_backend, var_6);
            var_16 = wp::load(var_15);
            // wp::array_store(var_backend_data, var_3, var_16, var_14);
        }
        //---------
        // reverse
        if (var_has_ordering) {
            wp::adj_array_store(var_backend_data, var_3, var_16, var_14, adj_backend_data, adj_3, adj_15, adj_14);
            wp::adj_address(var_user_to_backend, var_6, adj_user_to_backend, adj_6, adj_15);
            // adj: backend_data[env_id, user_to_backend[user_id]] = value                        <L 36>
        }
        // adj: if has_ordering:                                                                  <L 35>
        wp::adj_array_store(var_user_data, var_3, var_6, var_14, adj_user_data, adj_3, adj_6, adj_14);
        // adj: user_data[env_id, user_id] = value                                                <L 34>
        wp::adj_where(var_full_data, var_9, var_12, adj_full_data, adj_9, adj_12, adj_14);
        if (!var_full_data) {
            wp::adj_copy(var_13, adj_11, adj_12);
            wp::adj_address(var_input_data, var_0, var_1, adj_input_data, adj_0, adj_1, adj_11);
            // adj: value = input_data[i, j]                                                      <L 32>
        }
        if (var_full_data) {
            wp::adj_copy(var_10, adj_8, adj_9);
            wp::adj_address(var_input_data, var_3, var_6, adj_input_data, adj_3, adj_6, adj_8);
            // adj: value = input_data[env_id, user_id]                                           <L 30>
        }
        // adj: if full_data:                                                                     <L 29>
        wp::adj_address(var_user_ids, var_1, adj_user_ids, adj_1, adj_5);
        // adj: user_id = wp.int32(user_ids[j])                                                   <L 28>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 27>
        // adj: i, j = wp.tid()                                                                   <L 26>
        // adj: def _write_2d_user_to_backend_with_indices(                                       <L 1>
        continue;
    }
}



extern "C" __global__ void _write_2d_user_to_backend_with_indices_92fb52e6_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_input_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    bool var_full_data,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> var_backend_data)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        //---------
        // forward
        // def _write_2d_user_to_backend_with_indices(                                            <L 1>
        // i, j = wp.tid()                                                                        <L 26>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 27>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 28>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if full_data:                                                                          <L 29>
        if (var_full_data) {
            // value = input_data[env_id, user_id]                                                <L 30>
            var_8 = wp::address(var_input_data, var_3, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
        }
        if (!var_full_data) {
            // value = input_data[i, j]                                                           <L 32>
            var_11 = wp::address(var_input_data, var_0, var_1);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
        }
        var_14 = wp::where(var_full_data, var_9, var_12);
        // user_data[env_id, user_id] = value                                                     <L 34>
        wp::array_store(var_user_data, var_3, var_6, var_14);
        // if has_ordering:                                                                       <L 35>
        if (var_has_ordering) {
            // backend_data[env_id, user_to_backend[user_id]] = value                             <L 36>
            var_15 = wp::address(var_user_to_backend, var_6);
            var_16 = wp::load(var_15);
            wp::array_store(var_backend_data, var_3, var_16, var_14);
        }
    }
}



extern "C" __global__ void _write_2d_user_to_backend_with_indices_92fb52e6_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_input_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    bool var_full_data,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> var_backend_data,
    wp::array_t<wp::float32> adj_input_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int64> adj_user_ids,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    bool adj_full_data,
    wp::array_t<wp::float32> adj_user_data,
    wp::array_t<wp::float32> adj_backend_data)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int64 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int64 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        //---------
        // forward
        // def _write_2d_user_to_backend_with_indices(                                            <L 1>
        // i, j = wp.tid()                                                                        <L 26>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 27>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 28>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if full_data:                                                                          <L 29>
        if (var_full_data) {
            // value = input_data[env_id, user_id]                                                <L 30>
            var_8 = wp::address(var_input_data, var_3, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
        }
        if (!var_full_data) {
            // value = input_data[i, j]                                                           <L 32>
            var_11 = wp::address(var_input_data, var_0, var_1);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
        }
        var_14 = wp::where(var_full_data, var_9, var_12);
        // user_data[env_id, user_id] = value                                                     <L 34>
        // wp::array_store(var_user_data, var_3, var_6, var_14);
        // if has_ordering:                                                                       <L 35>
        if (var_has_ordering) {
            // backend_data[env_id, user_to_backend[user_id]] = value                             <L 36>
            var_15 = wp::address(var_user_to_backend, var_6);
            var_16 = wp::load(var_15);
            // wp::array_store(var_backend_data, var_3, var_16, var_14);
        }
        //---------
        // reverse
        if (var_has_ordering) {
            wp::adj_array_store(var_backend_data, var_3, var_16, var_14, adj_backend_data, adj_3, adj_15, adj_14);
            wp::adj_address(var_user_to_backend, var_6, adj_user_to_backend, adj_6, adj_15);
            // adj: backend_data[env_id, user_to_backend[user_id]] = value                        <L 36>
        }
        // adj: if has_ordering:                                                                  <L 35>
        wp::adj_array_store(var_user_data, var_3, var_6, var_14, adj_user_data, adj_3, adj_6, adj_14);
        // adj: user_data[env_id, user_id] = value                                                <L 34>
        wp::adj_where(var_full_data, var_9, var_12, adj_full_data, adj_9, adj_12, adj_14);
        if (!var_full_data) {
            wp::adj_copy(var_13, adj_11, adj_12);
            wp::adj_address(var_input_data, var_0, var_1, adj_input_data, adj_0, adj_1, adj_11);
            // adj: value = input_data[i, j]                                                      <L 32>
        }
        if (var_full_data) {
            wp::adj_copy(var_10, adj_8, adj_9);
            wp::adj_address(var_input_data, var_3, var_6, adj_input_data, adj_3, adj_6, adj_8);
            // adj: value = input_data[env_id, user_id]                                           <L 30>
        }
        // adj: if full_data:                                                                     <L 29>
        wp::adj_address(var_user_ids, var_1, adj_user_ids, adj_1, adj_5);
        // adj: user_id = wp.int32(user_ids[j])                                                   <L 28>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 27>
        // adj: i, j = wp.tid()                                                                   <L 26>
        // adj: def _write_2d_user_to_backend_with_indices(                                       <L 1>
        continue;
    }
}



extern "C" __global__ void reorder_generalized_vector_backend_to_user_97b75f77_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_backend_data,
    wp::array_t<wp::int32> var_joint_user_to_backend,
    wp::array_t<wp::int32> var_joint_dof_signs,
    wp::int32 var_num_base_dofs,
    bool var_has_joint_ordering,
    wp::array_t<wp::float32> var_user_data)
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
        bool var_3;
        bool var_4;
        wp::int32 var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        const wp::float32 var_10 = 1.0;
        bool var_11;
        bool var_12;
        wp::shape_t* var_13;
        const wp::int32 var_14 = 0;
        wp::int32 var_15;
        wp::shape_t var_16;
        const wp::int32 var_17 = 0;
        bool var_18;
        wp::int32 var_19;
        wp::int32* var_20;
        wp::float32 var_21;
        wp::int32 var_22;
        wp::float32 var_23;
        wp::float32* var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        //---------
        // forward
        // def reorder_generalized_vector_backend_to_user(                                        <L 433>
        // env_id, user_dof_id = wp.tid()                                                         <L 456>
        builtin_tid2d(var_0, var_1);
        // backend_dof_id = user_dof_id                                                           <L 457>
        var_2 = wp::copy(var_1);
        // if has_joint_ordering and user_dof_id >= num_base_dofs:                                <L 458>
        var_3 = var_has_joint_ordering;
        if (var_3) {
            var_4 = (var_1 >= var_num_base_dofs);
            var_3 = var_3 && var_4;
        }
        if (var_3) {
            // backend_dof_id = num_base_dofs + joint_user_to_backend[user_dof_id - num_base_dofs]       <L 459>
            var_5 = wp::sub(var_1, var_num_base_dofs);
            var_6 = wp::address(var_joint_user_to_backend, var_5);
            var_8 = wp::load(var_6);
            var_7 = wp::add(var_num_base_dofs, var_8);
        }
        var_9 = wp::where(var_3, var_7, var_2);
        // sign = 1.0                                                                             <L 461>
        // if user_dof_id >= num_base_dofs and joint_dof_signs.shape[0] > 0:                      <L 462>
        var_12 = (var_1 >= var_num_base_dofs);
        var_11 = var_12;
        if (var_11) {
            var_13 = &(var_joint_dof_signs.shape);
            var_16 = wp::load(var_13);
            var_15 = wp::extract(var_16, var_14);
            var_18 = (var_15 > var_17);
            var_11 = var_11 && var_18;
        }
        if (var_11) {
            // sign = wp.float32(joint_dof_signs[backend_dof_id - num_base_dofs])                 <L 463>
            var_19 = wp::sub(var_9, var_num_base_dofs);
            var_20 = wp::address(var_joint_dof_signs, var_19);
            var_22 = wp::load(var_20);
            var_21 = wp::float32(var_22);
        }
        var_23 = wp::where(var_11, var_21, var_10);
        // user_data[env_id, user_dof_id] = sign * backend_data[env_id, backend_dof_id]           <L 464>
        var_24 = wp::address(var_backend_data, var_0, var_9);
        var_26 = wp::load(var_24);
        var_25 = wp::mul(var_23, var_26);
        wp::array_store(var_user_data, var_0, var_1, var_25);
    }
}



extern "C" __global__ void reorder_generalized_vector_backend_to_user_97b75f77_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_backend_data,
    wp::array_t<wp::int32> var_joint_user_to_backend,
    wp::array_t<wp::int32> var_joint_dof_signs,
    wp::int32 var_num_base_dofs,
    bool var_has_joint_ordering,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> adj_backend_data,
    wp::array_t<wp::int32> adj_joint_user_to_backend,
    wp::array_t<wp::int32> adj_joint_dof_signs,
    wp::int32 adj_num_base_dofs,
    bool adj_has_joint_ordering,
    wp::array_t<wp::float32> adj_user_data)
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
        bool var_3;
        bool var_4;
        wp::int32 var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        const wp::float32 var_10 = 1.0;
        bool var_11;
        bool var_12;
        wp::shape_t* var_13;
        const wp::int32 var_14 = 0;
        wp::int32 var_15;
        wp::shape_t var_16;
        const wp::int32 var_17 = 0;
        bool var_18;
        wp::int32 var_19;
        wp::int32* var_20;
        wp::float32 var_21;
        wp::int32 var_22;
        wp::float32 var_23;
        wp::float32* var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::float32 adj_10 = {};
        bool adj_11 = {};
        bool adj_12 = {};
        wp::shape_t adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::shape_t adj_16 = {};
        wp::int32 adj_17 = {};
        bool adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::float32 adj_25 = {};
        wp::float32 adj_26 = {};
        //---------
        // forward
        // def reorder_generalized_vector_backend_to_user(                                        <L 433>
        // env_id, user_dof_id = wp.tid()                                                         <L 456>
        builtin_tid2d(var_0, var_1);
        // backend_dof_id = user_dof_id                                                           <L 457>
        var_2 = wp::copy(var_1);
        // if has_joint_ordering and user_dof_id >= num_base_dofs:                                <L 458>
        var_3 = var_has_joint_ordering;
        if (var_3) {
            var_4 = (var_1 >= var_num_base_dofs);
            var_3 = var_3 && var_4;
        }
        if (var_3) {
            // backend_dof_id = num_base_dofs + joint_user_to_backend[user_dof_id - num_base_dofs]       <L 459>
            var_5 = wp::sub(var_1, var_num_base_dofs);
            var_6 = wp::address(var_joint_user_to_backend, var_5);
            var_8 = wp::load(var_6);
            var_7 = wp::add(var_num_base_dofs, var_8);
        }
        var_9 = wp::where(var_3, var_7, var_2);
        // sign = 1.0                                                                             <L 461>
        // if user_dof_id >= num_base_dofs and joint_dof_signs.shape[0] > 0:                      <L 462>
        var_12 = (var_1 >= var_num_base_dofs);
        var_11 = var_12;
        if (var_11) {
            var_13 = &(var_joint_dof_signs.shape);
            var_16 = wp::load(var_13);
            var_15 = wp::extract(var_16, var_14);
            var_18 = (var_15 > var_17);
            var_11 = var_11 && var_18;
        }
        if (var_11) {
            // sign = wp.float32(joint_dof_signs[backend_dof_id - num_base_dofs])                 <L 463>
            var_19 = wp::sub(var_9, var_num_base_dofs);
            var_20 = wp::address(var_joint_dof_signs, var_19);
            var_22 = wp::load(var_20);
            var_21 = wp::float32(var_22);
        }
        var_23 = wp::where(var_11, var_21, var_10);
        // user_data[env_id, user_dof_id] = sign * backend_data[env_id, backend_dof_id]           <L 464>
        var_24 = wp::address(var_backend_data, var_0, var_9);
        var_26 = wp::load(var_24);
        var_25 = wp::mul(var_23, var_26);
        // wp::array_store(var_user_data, var_0, var_1, var_25);
        //---------
        // reverse
        wp::adj_array_store(var_user_data, var_0, var_1, var_25, adj_user_data, adj_0, adj_1, adj_25);
        wp::adj_mul(var_23, var_26, adj_23, adj_24, adj_25);
        wp::adj_address(var_backend_data, var_0, var_9, adj_backend_data, adj_0, adj_9, adj_24);
        // adj: user_data[env_id, user_dof_id] = sign * backend_data[env_id, backend_dof_id]      <L 464>
        wp::adj_where(var_11, var_21, var_10, adj_11, adj_21, adj_10, adj_23);
        if (var_11) {
            wp::adj_float32(var_22, adj_20, adj_21);
            wp::adj_address(var_joint_dof_signs, var_19, adj_joint_dof_signs, adj_19, adj_20);
            wp::adj_sub(var_9, var_num_base_dofs, adj_9, adj_num_base_dofs, adj_19);
            // adj: sign = wp.float32(joint_dof_signs[backend_dof_id - num_base_dofs])            <L 463>
        }
        if (var_11) {
            adj_joint_dof_signs.shape = adj_13;
        }
        // adj: if user_dof_id >= num_base_dofs and joint_dof_signs.shape[0] > 0:                 <L 462>
        // adj: sign = 1.0                                                                        <L 461>
        wp::adj_where(var_3, var_7, var_2, adj_3, adj_7, adj_2, adj_9);
        if (var_3) {
            wp::adj_add(var_num_base_dofs, var_8, adj_num_base_dofs, adj_6, adj_7);
            wp::adj_address(var_joint_user_to_backend, var_5, adj_joint_user_to_backend, adj_5, adj_6);
            wp::adj_sub(var_1, var_num_base_dofs, adj_1, adj_num_base_dofs, adj_5);
            // adj: backend_dof_id = num_base_dofs + joint_user_to_backend[user_dof_id - num_base_dofs]  <L 459>
        }
        if (var_3) {
        }
        // adj: if has_joint_ordering and user_dof_id >= num_base_dofs:                           <L 458>
        wp::adj_copy(var_1, adj_1, adj_2);
        // adj: backend_dof_id = user_dof_id                                                      <L 457>
        // adj: env_id, user_dof_id = wp.tid()                                                    <L 456>
        // adj: def reorder_generalized_vector_backend_to_user(                                   <L 433>
        continue;
    }
}



extern "C" __global__ void write_joint_vel_user_to_backend_with_indices_9ec33c91_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    bool var_full_data,
    wp::array_t<wp::float32> var_user_vel,
    wp::array_t<wp::float32> var_user_prev_vel,
    wp::array_t<wp::float32> var_user_acc,
    wp::array_t<wp::float32> var_backend_vel)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        const wp::float32 var_15 = 0.0;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        //---------
        // forward
        // def write_joint_vel_user_to_backend_with_indices(                                      <L 1>
        // i, j = wp.tid()                                                                        <L 35>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 36>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 37>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if full_data:                                                                          <L 39>
        if (var_full_data) {
            // value = in_data[env_id, user_id]                                                   <L 40>
            var_8 = wp::address(var_in_data, var_3, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
        }
        if (!var_full_data) {
            // value = in_data[i, j]                                                              <L 42>
            var_11 = wp::address(var_in_data, var_0, var_1);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
        }
        var_14 = wp::where(var_full_data, var_9, var_12);
        // user_vel[env_id, user_id] = value                                                      <L 44>
        wp::array_store(var_user_vel, var_3, var_6, var_14);
        // user_prev_vel[env_id, user_id] = value                                                 <L 45>
        wp::array_store(var_user_prev_vel, var_3, var_6, var_14);
        // user_acc[env_id, user_id] = 0.0                                                        <L 46>
        wp::array_store(var_user_acc, var_3, var_6, var_15);
        // if has_ordering:                                                                       <L 47>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 48>
            var_16 = wp::address(var_user_to_backend, var_6);
            var_18 = wp::load(var_16);
            var_17 = wp::copy(var_18);
            // backend_vel[env_id, backend_id] = value                                            <L 49>
            wp::array_store(var_backend_vel, var_3, var_17, var_14);
        }
    }
}



extern "C" __global__ void write_joint_vel_user_to_backend_with_indices_9ec33c91_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    bool var_full_data,
    wp::array_t<wp::float32> var_user_vel,
    wp::array_t<wp::float32> var_user_prev_vel,
    wp::array_t<wp::float32> var_user_acc,
    wp::array_t<wp::float32> var_backend_vel,
    wp::array_t<wp::float32> adj_in_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int64> adj_user_ids,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    bool adj_full_data,
    wp::array_t<wp::float32> adj_user_vel,
    wp::array_t<wp::float32> adj_user_prev_vel,
    wp::array_t<wp::float32> adj_user_acc,
    wp::array_t<wp::float32> adj_backend_vel)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        const wp::float32 var_15 = 0.0;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int64 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int64 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        //---------
        // forward
        // def write_joint_vel_user_to_backend_with_indices(                                      <L 1>
        // i, j = wp.tid()                                                                        <L 35>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 36>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 37>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if full_data:                                                                          <L 39>
        if (var_full_data) {
            // value = in_data[env_id, user_id]                                                   <L 40>
            var_8 = wp::address(var_in_data, var_3, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
        }
        if (!var_full_data) {
            // value = in_data[i, j]                                                              <L 42>
            var_11 = wp::address(var_in_data, var_0, var_1);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
        }
        var_14 = wp::where(var_full_data, var_9, var_12);
        // user_vel[env_id, user_id] = value                                                      <L 44>
        // wp::array_store(var_user_vel, var_3, var_6, var_14);
        // user_prev_vel[env_id, user_id] = value                                                 <L 45>
        // wp::array_store(var_user_prev_vel, var_3, var_6, var_14);
        // user_acc[env_id, user_id] = 0.0                                                        <L 46>
        // wp::array_store(var_user_acc, var_3, var_6, var_15);
        // if has_ordering:                                                                       <L 47>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 48>
            var_16 = wp::address(var_user_to_backend, var_6);
            var_18 = wp::load(var_16);
            var_17 = wp::copy(var_18);
            // backend_vel[env_id, backend_id] = value                                            <L 49>
            // wp::array_store(var_backend_vel, var_3, var_17, var_14);
        }
        //---------
        // reverse
        if (var_has_ordering) {
            wp::adj_array_store(var_backend_vel, var_3, var_17, var_14, adj_backend_vel, adj_3, adj_17, adj_14);
            // adj: backend_vel[env_id, backend_id] = value                                       <L 49>
            wp::adj_copy(var_18, adj_16, adj_17);
            wp::adj_address(var_user_to_backend, var_6, adj_user_to_backend, adj_6, adj_16);
            // adj: backend_id = user_to_backend[user_id]                                         <L 48>
        }
        // adj: if has_ordering:                                                                  <L 47>
        wp::adj_array_store(var_user_acc, var_3, var_6, var_15, adj_user_acc, adj_3, adj_6, adj_15);
        // adj: user_acc[env_id, user_id] = 0.0                                                   <L 46>
        wp::adj_array_store(var_user_prev_vel, var_3, var_6, var_14, adj_user_prev_vel, adj_3, adj_6, adj_14);
        // adj: user_prev_vel[env_id, user_id] = value                                            <L 45>
        wp::adj_array_store(var_user_vel, var_3, var_6, var_14, adj_user_vel, adj_3, adj_6, adj_14);
        // adj: user_vel[env_id, user_id] = value                                                 <L 44>
        wp::adj_where(var_full_data, var_9, var_12, adj_full_data, adj_9, adj_12, adj_14);
        if (!var_full_data) {
            wp::adj_copy(var_13, adj_11, adj_12);
            wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_11);
            // adj: value = in_data[i, j]                                                         <L 42>
        }
        if (var_full_data) {
            wp::adj_copy(var_10, adj_8, adj_9);
            wp::adj_address(var_in_data, var_3, var_6, adj_in_data, adj_3, adj_6, adj_8);
            // adj: value = in_data[env_id, user_id]                                              <L 40>
        }
        // adj: if full_data:                                                                     <L 39>
        wp::adj_address(var_user_ids, var_1, adj_user_ids, adj_1, adj_5);
        // adj: user_id = wp.int32(user_ids[j])                                                   <L 37>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 36>
        // adj: i, j = wp.tid()                                                                   <L 35>
        // adj: def write_joint_vel_user_to_backend_with_indices(                                 <L 1>
        continue;
    }
}



extern "C" __global__ void _write_scalar_user_to_backend_with_indices_and_sim_ids_a52b07bf_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_input_value,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> var_backend_data,
    wp::array_t<wp::int32> var_sim_env_ids)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        //---------
        // forward
        // def _write_scalar_user_to_backend_with_indices_and_sim_ids(                            <L 1>
        // i, j = wp.tid()                                                                        <L 11>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 12>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 13>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if j == 0:                                                                             <L 14>
        var_9 = (var_1 == var_8);
        if (var_9) {
            // sim_env_ids[i] = env_id                                                            <L 15>
            wp::array_store(var_sim_env_ids, var_0, var_3);
        }
        // user_data[env_id, user_id] = input_value                                               <L 16>
        wp::array_store(var_user_data, var_3, var_6, var_input_value);
        // if has_ordering:                                                                       <L 17>
        if (var_has_ordering) {
            // backend_data[env_id, user_to_backend[user_id]] = input_value                       <L 18>
            var_10 = wp::address(var_user_to_backend, var_6);
            var_11 = wp::load(var_10);
            wp::array_store(var_backend_data, var_3, var_11, var_input_value);
        }
    }
}



extern "C" __global__ void _write_scalar_user_to_backend_with_indices_and_sim_ids_a52b07bf_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_input_value,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> var_backend_data,
    wp::array_t<wp::int32> var_sim_env_ids,
    wp::float32 adj_input_value,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_user_ids,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    wp::array_t<wp::float32> adj_user_data,
    wp::array_t<wp::float32> adj_backend_data,
    wp::array_t<wp::int32> adj_sim_env_ids)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        bool adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        //---------
        // forward
        // def _write_scalar_user_to_backend_with_indices_and_sim_ids(                            <L 1>
        // i, j = wp.tid()                                                                        <L 11>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 12>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 13>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if j == 0:                                                                             <L 14>
        var_9 = (var_1 == var_8);
        if (var_9) {
            // sim_env_ids[i] = env_id                                                            <L 15>
            // wp::array_store(var_sim_env_ids, var_0, var_3);
        }
        // user_data[env_id, user_id] = input_value                                               <L 16>
        // wp::array_store(var_user_data, var_3, var_6, var_input_value);
        // if has_ordering:                                                                       <L 17>
        if (var_has_ordering) {
            // backend_data[env_id, user_to_backend[user_id]] = input_value                       <L 18>
            var_10 = wp::address(var_user_to_backend, var_6);
            var_11 = wp::load(var_10);
            // wp::array_store(var_backend_data, var_3, var_11, var_input_value);
        }
        //---------
        // reverse
        if (var_has_ordering) {
            wp::adj_array_store(var_backend_data, var_3, var_11, var_input_value, adj_backend_data, adj_3, adj_10, adj_input_value);
            wp::adj_address(var_user_to_backend, var_6, adj_user_to_backend, adj_6, adj_10);
            // adj: backend_data[env_id, user_to_backend[user_id]] = input_value                  <L 18>
        }
        // adj: if has_ordering:                                                                  <L 17>
        wp::adj_array_store(var_user_data, var_3, var_6, var_input_value, adj_user_data, adj_3, adj_6, adj_input_value);
        // adj: user_data[env_id, user_id] = input_value                                          <L 16>
        if (var_9) {
            wp::adj_array_store(var_sim_env_ids, var_0, var_3, adj_sim_env_ids, adj_0, adj_3);
            // adj: sim_env_ids[i] = env_id                                                       <L 15>
        }
        // adj: if j == 0:                                                                        <L 14>
        wp::adj_address(var_user_ids, var_1, adj_user_ids, adj_1, adj_5);
        // adj: user_id = wp.int32(user_ids[j])                                                   <L 13>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 12>
        // adj: i, j = wp.tid()                                                                   <L 11>
        // adj: def _write_scalar_user_to_backend_with_indices_and_sim_ids(                       <L 1>
        continue;
    }
}



extern "C" __global__ void write_joint_vel_user_to_backend_with_mask_b0d30cba_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_user_mask,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_vel,
    wp::array_t<wp::float32> var_user_prev_vel,
    wp::array_t<wp::float32> var_user_acc,
    wp::array_t<wp::float32> var_backend_vel)
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
        bool var_2;
        bool* var_3;
        bool var_4;
        bool var_5;
        bool* var_6;
        bool var_7;
        bool var_8;
        wp::float32* var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        const wp::float32 var_12 = 0.0;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        //---------
        // forward
        // def write_joint_vel_user_to_backend_with_mask(                                         <L 608>
        // env_id, user_id = wp.tid()                                                             <L 635>
        builtin_tid2d(var_0, var_1);
        // if not env_mask[env_id] or not user_mask[user_id]:                                     <L 636>
        var_3 = wp::address(var_env_mask, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::unot(var_5);
        var_2 = var_4;
        if (!var_2) {
            var_6 = wp::address(var_user_mask, var_1);
            var_8 = wp::load(var_6);
            var_7 = wp::unot(var_8);
            var_2 = var_2 || var_7;
        }
        if (var_2) {
            // return                                                                             <L 637>
            continue;
        }
        // value = in_data[env_id, user_id]                                                       <L 639>
        var_9 = wp::address(var_in_data, var_0, var_1);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // user_vel[env_id, user_id] = value                                                      <L 640>
        wp::array_store(var_user_vel, var_0, var_1, var_10);
        // user_prev_vel[env_id, user_id] = value                                                 <L 641>
        wp::array_store(var_user_prev_vel, var_0, var_1, var_10);
        // user_acc[env_id, user_id] = 0.0                                                        <L 642>
        wp::array_store(var_user_acc, var_0, var_1, var_12);
        // if has_ordering:                                                                       <L 643>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 644>
            var_13 = wp::address(var_user_to_backend, var_1);
            var_15 = wp::load(var_13);
            var_14 = wp::copy(var_15);
            // backend_vel[env_id, backend_id] = value                                            <L 645>
            wp::array_store(var_backend_vel, var_0, var_14, var_10);
        }
    }
}



extern "C" __global__ void write_joint_vel_user_to_backend_with_mask_b0d30cba_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_user_mask,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_vel,
    wp::array_t<wp::float32> var_user_prev_vel,
    wp::array_t<wp::float32> var_user_acc,
    wp::array_t<wp::float32> var_backend_vel,
    wp::array_t<wp::float32> adj_in_data,
    wp::array_t<bool> adj_env_mask,
    wp::array_t<bool> adj_user_mask,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    wp::array_t<wp::float32> adj_user_vel,
    wp::array_t<wp::float32> adj_user_prev_vel,
    wp::array_t<wp::float32> adj_user_acc,
    wp::array_t<wp::float32> adj_backend_vel)
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
        bool var_2;
        bool* var_3;
        bool var_4;
        bool var_5;
        bool* var_6;
        bool var_7;
        bool var_8;
        wp::float32* var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        const wp::float32 var_12 = 0.0;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        bool adj_6 = {};
        bool adj_7 = {};
        bool adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        //---------
        // forward
        // def write_joint_vel_user_to_backend_with_mask(                                         <L 608>
        // env_id, user_id = wp.tid()                                                             <L 635>
        builtin_tid2d(var_0, var_1);
        // if not env_mask[env_id] or not user_mask[user_id]:                                     <L 636>
        var_3 = wp::address(var_env_mask, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::unot(var_5);
        var_2 = var_4;
        if (!var_2) {
            var_6 = wp::address(var_user_mask, var_1);
            var_8 = wp::load(var_6);
            var_7 = wp::unot(var_8);
            var_2 = var_2 || var_7;
        }
        if (var_2) {
            // return                                                                             <L 637>
            goto label0;
        }
        // value = in_data[env_id, user_id]                                                       <L 639>
        var_9 = wp::address(var_in_data, var_0, var_1);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // user_vel[env_id, user_id] = value                                                      <L 640>
        // wp::array_store(var_user_vel, var_0, var_1, var_10);
        // user_prev_vel[env_id, user_id] = value                                                 <L 641>
        // wp::array_store(var_user_prev_vel, var_0, var_1, var_10);
        // user_acc[env_id, user_id] = 0.0                                                        <L 642>
        // wp::array_store(var_user_acc, var_0, var_1, var_12);
        // if has_ordering:                                                                       <L 643>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 644>
            var_13 = wp::address(var_user_to_backend, var_1);
            var_15 = wp::load(var_13);
            var_14 = wp::copy(var_15);
            // backend_vel[env_id, backend_id] = value                                            <L 645>
            // wp::array_store(var_backend_vel, var_0, var_14, var_10);
        }
        //---------
        // reverse
        if (var_has_ordering) {
            wp::adj_array_store(var_backend_vel, var_0, var_14, var_10, adj_backend_vel, adj_0, adj_14, adj_10);
            // adj: backend_vel[env_id, backend_id] = value                                       <L 645>
            wp::adj_copy(var_15, adj_13, adj_14);
            wp::adj_address(var_user_to_backend, var_1, adj_user_to_backend, adj_1, adj_13);
            // adj: backend_id = user_to_backend[user_id]                                         <L 644>
        }
        // adj: if has_ordering:                                                                  <L 643>
        wp::adj_array_store(var_user_acc, var_0, var_1, var_12, adj_user_acc, adj_0, adj_1, adj_12);
        // adj: user_acc[env_id, user_id] = 0.0                                                   <L 642>
        wp::adj_array_store(var_user_prev_vel, var_0, var_1, var_10, adj_user_prev_vel, adj_0, adj_1, adj_10);
        // adj: user_prev_vel[env_id, user_id] = value                                            <L 641>
        wp::adj_array_store(var_user_vel, var_0, var_1, var_10, adj_user_vel, adj_0, adj_1, adj_10);
        // adj: user_vel[env_id, user_id] = value                                                 <L 640>
        wp::adj_copy(var_11, adj_9, adj_10);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_9);
        // adj: value = in_data[env_id, user_id]                                                  <L 639>
        if (var_2) {
            label0:;
            // adj: return                                                                        <L 637>
        }
        if (!var_2) {
            wp::adj_address(var_user_mask, var_1, adj_user_mask, adj_1, adj_6);
        }
        wp::adj_address(var_env_mask, var_0, adj_env_mask, adj_0, adj_3);
        // adj: if not env_mask[env_id] or not user_mask[user_id]:                                <L 636>
        // adj: env_id, user_id = wp.tid()                                                        <L 635>
        // adj: def write_joint_vel_user_to_backend_with_mask(                                    <L 608>
        continue;
    }
}



extern "C" __global__ void reorder_mass_matrix_backend_to_user_df85939d_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::float32> var_backend_data,
    wp::array_t<wp::int32> var_joint_user_to_backend,
    wp::array_t<wp::int32> var_joint_dof_signs,
    wp::int32 var_num_base_dofs,
    bool var_has_joint_ordering,
    wp::array_t<wp::float32> var_user_data)
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
        wp::int32 var_4;
        bool var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        bool var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        const wp::float32 var_19 = 1.0;
        const wp::float32 var_20 = 1.0;
        bool var_21;
        wp::int32 var_22;
        wp::int32* var_23;
        wp::float32 var_24;
        wp::int32 var_25;
        wp::float32 var_26;
        bool var_27;
        wp::int32 var_28;
        wp::int32* var_29;
        wp::float32 var_30;
        wp::int32 var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::float32* var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        //---------
        // forward
        // def reorder_mass_matrix_backend_to_user(                                               <L 389>
        // env_id, user_row_id, user_col_id = wp.tid()                                            <L 412>
        builtin_tid3d(var_0, var_1, var_2);
        // backend_row_id = user_row_id                                                           <L 413>
        var_3 = wp::copy(var_1);
        // backend_col_id = user_col_id                                                           <L 414>
        var_4 = wp::copy(var_2);
        // if has_joint_ordering:                                                                 <L 415>
        if (var_has_joint_ordering) {
            // if user_row_id >= num_base_dofs:                                                   <L 416>
            var_5 = (var_1 >= var_num_base_dofs);
            if (var_5) {
                // backend_row_id = num_base_dofs + joint_user_to_backend[user_row_id - num_base_dofs]       <L 417>
                var_6 = wp::sub(var_1, var_num_base_dofs);
                var_7 = wp::address(var_joint_user_to_backend, var_6);
                var_9 = wp::load(var_7);
                var_8 = wp::add(var_num_base_dofs, var_9);
            }
            var_10 = wp::where(var_5, var_8, var_3);
            // if user_col_id >= num_base_dofs:                                                   <L 418>
            var_11 = (var_2 >= var_num_base_dofs);
            if (var_11) {
                // backend_col_id = num_base_dofs + joint_user_to_backend[user_col_id - num_base_dofs]       <L 419>
                var_12 = wp::sub(var_2, var_num_base_dofs);
                var_13 = wp::address(var_joint_user_to_backend, var_12);
                var_15 = wp::load(var_13);
                var_14 = wp::add(var_num_base_dofs, var_15);
            }
            var_16 = wp::where(var_11, var_14, var_4);
        }
        var_17 = wp::where(var_has_joint_ordering, var_10, var_3);
        var_18 = wp::where(var_has_joint_ordering, var_16, var_4);
        // row_sign = 1.0                                                                         <L 421>
        // col_sign = 1.0                                                                         <L 422>
        // if user_row_id >= num_base_dofs:                                                       <L 423>
        var_21 = (var_1 >= var_num_base_dofs);
        if (var_21) {
            // row_sign = wp.float32(joint_dof_signs[backend_row_id - num_base_dofs])             <L 424>
            var_22 = wp::sub(var_17, var_num_base_dofs);
            var_23 = wp::address(var_joint_dof_signs, var_22);
            var_25 = wp::load(var_23);
            var_24 = wp::float32(var_25);
        }
        var_26 = wp::where(var_21, var_24, var_19);
        // if user_col_id >= num_base_dofs:                                                       <L 425>
        var_27 = (var_2 >= var_num_base_dofs);
        if (var_27) {
            // col_sign = wp.float32(joint_dof_signs[backend_col_id - num_base_dofs])             <L 426>
            var_28 = wp::sub(var_18, var_num_base_dofs);
            var_29 = wp::address(var_joint_dof_signs, var_28);
            var_31 = wp::load(var_29);
            var_30 = wp::float32(var_31);
        }
        var_32 = wp::where(var_27, var_30, var_20);
        // user_data[env_id, user_row_id, user_col_id] = (                                        <L 427>
        // row_sign * col_sign * backend_data[env_id, backend_row_id, backend_col_id]             <L 428>
        var_33 = wp::mul(var_26, var_32);
        var_34 = wp::address(var_backend_data, var_0, var_17, var_18);
        var_36 = wp::load(var_34);
        var_35 = wp::mul(var_33, var_36);
        // user_data[env_id, user_row_id, user_col_id] = (                                        <L 427>
        wp::array_store(var_user_data, var_0, var_1, var_2, var_35);
    }
}



extern "C" __global__ void reorder_mass_matrix_backend_to_user_df85939d_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::float32> var_backend_data,
    wp::array_t<wp::int32> var_joint_user_to_backend,
    wp::array_t<wp::int32> var_joint_dof_signs,
    wp::int32 var_num_base_dofs,
    bool var_has_joint_ordering,
    wp::array_t<wp::float32> var_user_data,
    wp::array_t<wp::float32> adj_backend_data,
    wp::array_t<wp::int32> adj_joint_user_to_backend,
    wp::array_t<wp::int32> adj_joint_dof_signs,
    wp::int32 adj_num_base_dofs,
    bool adj_has_joint_ordering,
    wp::array_t<wp::float32> adj_user_data)
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
        wp::int32 var_4;
        bool var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        bool var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        const wp::float32 var_19 = 1.0;
        const wp::float32 var_20 = 1.0;
        bool var_21;
        wp::int32 var_22;
        wp::int32* var_23;
        wp::float32 var_24;
        wp::int32 var_25;
        wp::float32 var_26;
        bool var_27;
        wp::int32 var_28;
        wp::int32* var_29;
        wp::float32 var_30;
        wp::int32 var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::float32* var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        bool adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        bool adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        bool adj_21 = {};
        wp::int32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::float32 adj_26 = {};
        bool adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::float32 adj_36 = {};
        //---------
        // forward
        // def reorder_mass_matrix_backend_to_user(                                               <L 389>
        // env_id, user_row_id, user_col_id = wp.tid()                                            <L 412>
        builtin_tid3d(var_0, var_1, var_2);
        // backend_row_id = user_row_id                                                           <L 413>
        var_3 = wp::copy(var_1);
        // backend_col_id = user_col_id                                                           <L 414>
        var_4 = wp::copy(var_2);
        // if has_joint_ordering:                                                                 <L 415>
        if (var_has_joint_ordering) {
            // if user_row_id >= num_base_dofs:                                                   <L 416>
            var_5 = (var_1 >= var_num_base_dofs);
            if (var_5) {
                // backend_row_id = num_base_dofs + joint_user_to_backend[user_row_id - num_base_dofs]       <L 417>
                var_6 = wp::sub(var_1, var_num_base_dofs);
                var_7 = wp::address(var_joint_user_to_backend, var_6);
                var_9 = wp::load(var_7);
                var_8 = wp::add(var_num_base_dofs, var_9);
            }
            var_10 = wp::where(var_5, var_8, var_3);
            // if user_col_id >= num_base_dofs:                                                   <L 418>
            var_11 = (var_2 >= var_num_base_dofs);
            if (var_11) {
                // backend_col_id = num_base_dofs + joint_user_to_backend[user_col_id - num_base_dofs]       <L 419>
                var_12 = wp::sub(var_2, var_num_base_dofs);
                var_13 = wp::address(var_joint_user_to_backend, var_12);
                var_15 = wp::load(var_13);
                var_14 = wp::add(var_num_base_dofs, var_15);
            }
            var_16 = wp::where(var_11, var_14, var_4);
        }
        var_17 = wp::where(var_has_joint_ordering, var_10, var_3);
        var_18 = wp::where(var_has_joint_ordering, var_16, var_4);
        // row_sign = 1.0                                                                         <L 421>
        // col_sign = 1.0                                                                         <L 422>
        // if user_row_id >= num_base_dofs:                                                       <L 423>
        var_21 = (var_1 >= var_num_base_dofs);
        if (var_21) {
            // row_sign = wp.float32(joint_dof_signs[backend_row_id - num_base_dofs])             <L 424>
            var_22 = wp::sub(var_17, var_num_base_dofs);
            var_23 = wp::address(var_joint_dof_signs, var_22);
            var_25 = wp::load(var_23);
            var_24 = wp::float32(var_25);
        }
        var_26 = wp::where(var_21, var_24, var_19);
        // if user_col_id >= num_base_dofs:                                                       <L 425>
        var_27 = (var_2 >= var_num_base_dofs);
        if (var_27) {
            // col_sign = wp.float32(joint_dof_signs[backend_col_id - num_base_dofs])             <L 426>
            var_28 = wp::sub(var_18, var_num_base_dofs);
            var_29 = wp::address(var_joint_dof_signs, var_28);
            var_31 = wp::load(var_29);
            var_30 = wp::float32(var_31);
        }
        var_32 = wp::where(var_27, var_30, var_20);
        // user_data[env_id, user_row_id, user_col_id] = (                                        <L 427>
        // row_sign * col_sign * backend_data[env_id, backend_row_id, backend_col_id]             <L 428>
        var_33 = wp::mul(var_26, var_32);
        var_34 = wp::address(var_backend_data, var_0, var_17, var_18);
        var_36 = wp::load(var_34);
        var_35 = wp::mul(var_33, var_36);
        // user_data[env_id, user_row_id, user_col_id] = (                                        <L 427>
        // wp::array_store(var_user_data, var_0, var_1, var_2, var_35);
        //---------
        // reverse
        wp::adj_array_store(var_user_data, var_0, var_1, var_2, var_35, adj_user_data, adj_0, adj_1, adj_2, adj_35);
        // adj: user_data[env_id, user_row_id, user_col_id] = (                                   <L 427>
        wp::adj_mul(var_33, var_36, adj_33, adj_34, adj_35);
        wp::adj_address(var_backend_data, var_0, var_17, var_18, adj_backend_data, adj_0, adj_17, adj_18, adj_34);
        wp::adj_mul(var_26, var_32, adj_26, adj_32, adj_33);
        // adj: row_sign * col_sign * backend_data[env_id, backend_row_id, backend_col_id]        <L 428>
        // adj: user_data[env_id, user_row_id, user_col_id] = (                                   <L 427>
        wp::adj_where(var_27, var_30, var_20, adj_27, adj_30, adj_20, adj_32);
        if (var_27) {
            wp::adj_float32(var_31, adj_29, adj_30);
            wp::adj_address(var_joint_dof_signs, var_28, adj_joint_dof_signs, adj_28, adj_29);
            wp::adj_sub(var_18, var_num_base_dofs, adj_18, adj_num_base_dofs, adj_28);
            // adj: col_sign = wp.float32(joint_dof_signs[backend_col_id - num_base_dofs])        <L 426>
        }
        // adj: if user_col_id >= num_base_dofs:                                                  <L 425>
        wp::adj_where(var_21, var_24, var_19, adj_21, adj_24, adj_19, adj_26);
        if (var_21) {
            wp::adj_float32(var_25, adj_23, adj_24);
            wp::adj_address(var_joint_dof_signs, var_22, adj_joint_dof_signs, adj_22, adj_23);
            wp::adj_sub(var_17, var_num_base_dofs, adj_17, adj_num_base_dofs, adj_22);
            // adj: row_sign = wp.float32(joint_dof_signs[backend_row_id - num_base_dofs])        <L 424>
        }
        // adj: if user_row_id >= num_base_dofs:                                                  <L 423>
        // adj: col_sign = 1.0                                                                    <L 422>
        // adj: row_sign = 1.0                                                                    <L 421>
        wp::adj_where(var_has_joint_ordering, var_16, var_4, adj_has_joint_ordering, adj_16, adj_4, adj_18);
        wp::adj_where(var_has_joint_ordering, var_10, var_3, adj_has_joint_ordering, adj_10, adj_3, adj_17);
        if (var_has_joint_ordering) {
            wp::adj_where(var_11, var_14, var_4, adj_11, adj_14, adj_4, adj_16);
            if (var_11) {
                wp::adj_add(var_num_base_dofs, var_15, adj_num_base_dofs, adj_13, adj_14);
                wp::adj_address(var_joint_user_to_backend, var_12, adj_joint_user_to_backend, adj_12, adj_13);
                wp::adj_sub(var_2, var_num_base_dofs, adj_2, adj_num_base_dofs, adj_12);
                // adj: backend_col_id = num_base_dofs + joint_user_to_backend[user_col_id - num_base_dofs]  <L 419>
            }
            // adj: if user_col_id >= num_base_dofs:                                              <L 418>
            wp::adj_where(var_5, var_8, var_3, adj_5, adj_8, adj_3, adj_10);
            if (var_5) {
                wp::adj_add(var_num_base_dofs, var_9, adj_num_base_dofs, adj_7, adj_8);
                wp::adj_address(var_joint_user_to_backend, var_6, adj_joint_user_to_backend, adj_6, adj_7);
                wp::adj_sub(var_1, var_num_base_dofs, adj_1, adj_num_base_dofs, adj_6);
                // adj: backend_row_id = num_base_dofs + joint_user_to_backend[user_row_id - num_base_dofs]  <L 417>
            }
            // adj: if user_row_id >= num_base_dofs:                                              <L 416>
        }
        // adj: if has_joint_ordering:                                                            <L 415>
        wp::adj_copy(var_2, adj_2, adj_4);
        // adj: backend_col_id = user_col_id                                                      <L 414>
        wp::adj_copy(var_1, adj_1, adj_3);
        // adj: backend_row_id = user_row_id                                                      <L 413>
        // adj: env_id, user_row_id, user_col_id = wp.tid()                                       <L 412>
        // adj: def reorder_mass_matrix_backend_to_user(                                          <L 389>
        continue;
    }
}



extern "C" __global__ void write_joint_vel_user_to_backend_with_indices_e8fe6853_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    bool var_full_data,
    wp::array_t<wp::float32> var_user_vel,
    wp::array_t<wp::float32> var_user_prev_vel,
    wp::array_t<wp::float32> var_user_acc,
    wp::array_t<wp::float32> var_backend_vel)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        const wp::float32 var_15 = 0.0;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        //---------
        // forward
        // def write_joint_vel_user_to_backend_with_indices(                                      <L 1>
        // i, j = wp.tid()                                                                        <L 35>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 36>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 37>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if full_data:                                                                          <L 39>
        if (var_full_data) {
            // value = in_data[env_id, user_id]                                                   <L 40>
            var_8 = wp::address(var_in_data, var_3, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
        }
        if (!var_full_data) {
            // value = in_data[i, j]                                                              <L 42>
            var_11 = wp::address(var_in_data, var_0, var_1);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
        }
        var_14 = wp::where(var_full_data, var_9, var_12);
        // user_vel[env_id, user_id] = value                                                      <L 44>
        wp::array_store(var_user_vel, var_3, var_6, var_14);
        // user_prev_vel[env_id, user_id] = value                                                 <L 45>
        wp::array_store(var_user_prev_vel, var_3, var_6, var_14);
        // user_acc[env_id, user_id] = 0.0                                                        <L 46>
        wp::array_store(var_user_acc, var_3, var_6, var_15);
        // if has_ordering:                                                                       <L 47>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 48>
            var_16 = wp::address(var_user_to_backend, var_6);
            var_18 = wp::load(var_16);
            var_17 = wp::copy(var_18);
            // backend_vel[env_id, backend_id] = value                                            <L 49>
            wp::array_store(var_backend_vel, var_3, var_17, var_14);
        }
    }
}



extern "C" __global__ void write_joint_vel_user_to_backend_with_indices_e8fe6853_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    bool var_full_data,
    wp::array_t<wp::float32> var_user_vel,
    wp::array_t<wp::float32> var_user_prev_vel,
    wp::array_t<wp::float32> var_user_acc,
    wp::array_t<wp::float32> var_backend_vel,
    wp::array_t<wp::float32> adj_in_data,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int64> adj_user_ids,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    bool adj_full_data,
    wp::array_t<wp::float32> adj_user_vel,
    wp::array_t<wp::float32> adj_user_prev_vel,
    wp::array_t<wp::float32> adj_user_acc,
    wp::array_t<wp::float32> adj_backend_vel)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        const wp::float32 var_15 = 0.0;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int64 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int64 adj_4 = {};
        wp::int64 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int64 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        //---------
        // forward
        // def write_joint_vel_user_to_backend_with_indices(                                      <L 1>
        // i, j = wp.tid()                                                                        <L 35>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 36>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 37>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if full_data:                                                                          <L 39>
        if (var_full_data) {
            // value = in_data[env_id, user_id]                                                   <L 40>
            var_8 = wp::address(var_in_data, var_3, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
        }
        if (!var_full_data) {
            // value = in_data[i, j]                                                              <L 42>
            var_11 = wp::address(var_in_data, var_0, var_1);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
        }
        var_14 = wp::where(var_full_data, var_9, var_12);
        // user_vel[env_id, user_id] = value                                                      <L 44>
        // wp::array_store(var_user_vel, var_3, var_6, var_14);
        // user_prev_vel[env_id, user_id] = value                                                 <L 45>
        // wp::array_store(var_user_prev_vel, var_3, var_6, var_14);
        // user_acc[env_id, user_id] = 0.0                                                        <L 46>
        // wp::array_store(var_user_acc, var_3, var_6, var_15);
        // if has_ordering:                                                                       <L 47>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 48>
            var_16 = wp::address(var_user_to_backend, var_6);
            var_18 = wp::load(var_16);
            var_17 = wp::copy(var_18);
            // backend_vel[env_id, backend_id] = value                                            <L 49>
            // wp::array_store(var_backend_vel, var_3, var_17, var_14);
        }
        //---------
        // reverse
        if (var_has_ordering) {
            wp::adj_array_store(var_backend_vel, var_3, var_17, var_14, adj_backend_vel, adj_3, adj_17, adj_14);
            // adj: backend_vel[env_id, backend_id] = value                                       <L 49>
            wp::adj_copy(var_18, adj_16, adj_17);
            wp::adj_address(var_user_to_backend, var_6, adj_user_to_backend, adj_6, adj_16);
            // adj: backend_id = user_to_backend[user_id]                                         <L 48>
        }
        // adj: if has_ordering:                                                                  <L 47>
        wp::adj_array_store(var_user_acc, var_3, var_6, var_15, adj_user_acc, adj_3, adj_6, adj_15);
        // adj: user_acc[env_id, user_id] = 0.0                                                   <L 46>
        wp::adj_array_store(var_user_prev_vel, var_3, var_6, var_14, adj_user_prev_vel, adj_3, adj_6, adj_14);
        // adj: user_prev_vel[env_id, user_id] = value                                            <L 45>
        wp::adj_array_store(var_user_vel, var_3, var_6, var_14, adj_user_vel, adj_3, adj_6, adj_14);
        // adj: user_vel[env_id, user_id] = value                                                 <L 44>
        wp::adj_where(var_full_data, var_9, var_12, adj_full_data, adj_9, adj_12, adj_14);
        if (!var_full_data) {
            wp::adj_copy(var_13, adj_11, adj_12);
            wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_11);
            // adj: value = in_data[i, j]                                                         <L 42>
        }
        if (var_full_data) {
            wp::adj_copy(var_10, adj_8, adj_9);
            wp::adj_address(var_in_data, var_3, var_6, adj_in_data, adj_3, adj_6, adj_8);
            // adj: value = in_data[env_id, user_id]                                              <L 40>
        }
        // adj: if full_data:                                                                     <L 39>
        wp::adj_address(var_user_ids, var_1, adj_user_ids, adj_1, adj_5);
        // adj: user_id = wp.int32(user_ids[j])                                                   <L 37>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 36>
        // adj: i, j = wp.tid()                                                                   <L 35>
        // adj: def write_joint_vel_user_to_backend_with_indices(                                 <L 1>
        continue;
    }
}



extern "C" __global__ void write_joint_state_user_to_backend_with_indices_fa03d78b_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_pos_data,
    wp::array_t<wp::float32> var_vel_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    bool var_full_data,
    wp::array_t<wp::float32> var_user_pos,
    wp::array_t<wp::float32> var_user_vel,
    wp::array_t<wp::float32> var_user_prev_vel,
    wp::array_t<wp::float32> var_user_acc,
    wp::array_t<wp::float32> var_backend_pos,
    wp::array_t<wp::float32> var_backend_vel)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        const wp::float32 var_22 = 0.0;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        //---------
        // forward
        // def write_joint_state_user_to_backend_with_indices(                                    <L 1>
        // i, j = wp.tid()                                                                        <L 43>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 44>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 45>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if full_data:                                                                          <L 47>
        if (var_full_data) {
            // position = pos_data[env_id, user_id]                                               <L 48>
            var_8 = wp::address(var_pos_data, var_3, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
            // velocity = vel_data[env_id, user_id]                                               <L 49>
            var_11 = wp::address(var_vel_data, var_3, var_6);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
        }
        if (!var_full_data) {
            // position = pos_data[i, j]                                                          <L 51>
            var_14 = wp::address(var_pos_data, var_0, var_1);
            var_16 = wp::load(var_14);
            var_15 = wp::copy(var_16);
            // velocity = vel_data[i, j]                                                          <L 52>
            var_17 = wp::address(var_vel_data, var_0, var_1);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
        }
        var_20 = wp::where(var_full_data, var_9, var_15);
        var_21 = wp::where(var_full_data, var_12, var_18);
        // user_pos[env_id, user_id] = position                                                   <L 54>
        wp::array_store(var_user_pos, var_3, var_6, var_20);
        // user_vel[env_id, user_id] = velocity                                                   <L 55>
        wp::array_store(var_user_vel, var_3, var_6, var_21);
        // user_prev_vel[env_id, user_id] = velocity                                              <L 56>
        wp::array_store(var_user_prev_vel, var_3, var_6, var_21);
        // user_acc[env_id, user_id] = 0.0                                                        <L 57>
        wp::array_store(var_user_acc, var_3, var_6, var_22);
        // if has_ordering:                                                                       <L 58>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 59>
            var_23 = wp::address(var_user_to_backend, var_6);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // backend_pos[env_id, backend_id] = position                                         <L 60>
            wp::array_store(var_backend_pos, var_3, var_24, var_20);
            // backend_vel[env_id, backend_id] = velocity                                         <L 61>
            wp::array_store(var_backend_vel, var_3, var_24, var_21);
        }
    }
}



extern "C" __global__ void write_joint_state_user_to_backend_with_indices_fa03d78b_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_pos_data,
    wp::array_t<wp::float32> var_vel_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    bool var_full_data,
    wp::array_t<wp::float32> var_user_pos,
    wp::array_t<wp::float32> var_user_vel,
    wp::array_t<wp::float32> var_user_prev_vel,
    wp::array_t<wp::float32> var_user_acc,
    wp::array_t<wp::float32> var_backend_pos,
    wp::array_t<wp::float32> var_backend_vel,
    wp::array_t<wp::float32> adj_pos_data,
    wp::array_t<wp::float32> adj_vel_data,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int64> adj_user_ids,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    bool adj_full_data,
    wp::array_t<wp::float32> adj_user_pos,
    wp::array_t<wp::float32> adj_user_vel,
    wp::array_t<wp::float32> adj_user_prev_vel,
    wp::array_t<wp::float32> adj_user_acc,
    wp::array_t<wp::float32> adj_backend_pos,
    wp::array_t<wp::float32> adj_backend_vel)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        const wp::float32 var_22 = 0.0;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int64 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int64 adj_4 = {};
        wp::int64 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int64 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        //---------
        // forward
        // def write_joint_state_user_to_backend_with_indices(                                    <L 1>
        // i, j = wp.tid()                                                                        <L 43>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 44>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 45>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if full_data:                                                                          <L 47>
        if (var_full_data) {
            // position = pos_data[env_id, user_id]                                               <L 48>
            var_8 = wp::address(var_pos_data, var_3, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
            // velocity = vel_data[env_id, user_id]                                               <L 49>
            var_11 = wp::address(var_vel_data, var_3, var_6);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
        }
        if (!var_full_data) {
            // position = pos_data[i, j]                                                          <L 51>
            var_14 = wp::address(var_pos_data, var_0, var_1);
            var_16 = wp::load(var_14);
            var_15 = wp::copy(var_16);
            // velocity = vel_data[i, j]                                                          <L 52>
            var_17 = wp::address(var_vel_data, var_0, var_1);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
        }
        var_20 = wp::where(var_full_data, var_9, var_15);
        var_21 = wp::where(var_full_data, var_12, var_18);
        // user_pos[env_id, user_id] = position                                                   <L 54>
        // wp::array_store(var_user_pos, var_3, var_6, var_20);
        // user_vel[env_id, user_id] = velocity                                                   <L 55>
        // wp::array_store(var_user_vel, var_3, var_6, var_21);
        // user_prev_vel[env_id, user_id] = velocity                                              <L 56>
        // wp::array_store(var_user_prev_vel, var_3, var_6, var_21);
        // user_acc[env_id, user_id] = 0.0                                                        <L 57>
        // wp::array_store(var_user_acc, var_3, var_6, var_22);
        // if has_ordering:                                                                       <L 58>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 59>
            var_23 = wp::address(var_user_to_backend, var_6);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // backend_pos[env_id, backend_id] = position                                         <L 60>
            // wp::array_store(var_backend_pos, var_3, var_24, var_20);
            // backend_vel[env_id, backend_id] = velocity                                         <L 61>
            // wp::array_store(var_backend_vel, var_3, var_24, var_21);
        }
        //---------
        // reverse
        if (var_has_ordering) {
            wp::adj_array_store(var_backend_vel, var_3, var_24, var_21, adj_backend_vel, adj_3, adj_24, adj_21);
            // adj: backend_vel[env_id, backend_id] = velocity                                    <L 61>
            wp::adj_array_store(var_backend_pos, var_3, var_24, var_20, adj_backend_pos, adj_3, adj_24, adj_20);
            // adj: backend_pos[env_id, backend_id] = position                                    <L 60>
            wp::adj_copy(var_25, adj_23, adj_24);
            wp::adj_address(var_user_to_backend, var_6, adj_user_to_backend, adj_6, adj_23);
            // adj: backend_id = user_to_backend[user_id]                                         <L 59>
        }
        // adj: if has_ordering:                                                                  <L 58>
        wp::adj_array_store(var_user_acc, var_3, var_6, var_22, adj_user_acc, adj_3, adj_6, adj_22);
        // adj: user_acc[env_id, user_id] = 0.0                                                   <L 57>
        wp::adj_array_store(var_user_prev_vel, var_3, var_6, var_21, adj_user_prev_vel, adj_3, adj_6, adj_21);
        // adj: user_prev_vel[env_id, user_id] = velocity                                         <L 56>
        wp::adj_array_store(var_user_vel, var_3, var_6, var_21, adj_user_vel, adj_3, adj_6, adj_21);
        // adj: user_vel[env_id, user_id] = velocity                                              <L 55>
        wp::adj_array_store(var_user_pos, var_3, var_6, var_20, adj_user_pos, adj_3, adj_6, adj_20);
        // adj: user_pos[env_id, user_id] = position                                              <L 54>
        wp::adj_where(var_full_data, var_12, var_18, adj_full_data, adj_12, adj_18, adj_21);
        wp::adj_where(var_full_data, var_9, var_15, adj_full_data, adj_9, adj_15, adj_20);
        if (!var_full_data) {
            wp::adj_copy(var_19, adj_17, adj_18);
            wp::adj_address(var_vel_data, var_0, var_1, adj_vel_data, adj_0, adj_1, adj_17);
            // adj: velocity = vel_data[i, j]                                                     <L 52>
            wp::adj_copy(var_16, adj_14, adj_15);
            wp::adj_address(var_pos_data, var_0, var_1, adj_pos_data, adj_0, adj_1, adj_14);
            // adj: position = pos_data[i, j]                                                     <L 51>
        }
        if (var_full_data) {
            wp::adj_copy(var_13, adj_11, adj_12);
            wp::adj_address(var_vel_data, var_3, var_6, adj_vel_data, adj_3, adj_6, adj_11);
            // adj: velocity = vel_data[env_id, user_id]                                          <L 49>
            wp::adj_copy(var_10, adj_8, adj_9);
            wp::adj_address(var_pos_data, var_3, var_6, adj_pos_data, adj_3, adj_6, adj_8);
            // adj: position = pos_data[env_id, user_id]                                          <L 48>
        }
        // adj: if full_data:                                                                     <L 47>
        wp::adj_address(var_user_ids, var_1, adj_user_ids, adj_1, adj_5);
        // adj: user_id = wp.int32(user_ids[j])                                                   <L 45>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 44>
        // adj: i, j = wp.tid()                                                                   <L 43>
        // adj: def write_joint_state_user_to_backend_with_indices(                               <L 1>
        continue;
    }
}



extern "C" __global__ void write_joint_state_user_to_backend_with_mask_fc0fe9f0_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_pos_data,
    wp::array_t<wp::float32> var_vel_data,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_user_mask,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_pos,
    wp::array_t<wp::float32> var_user_vel,
    wp::array_t<wp::float32> var_user_prev_vel,
    wp::array_t<wp::float32> var_user_acc,
    wp::array_t<wp::float32> var_backend_pos,
    wp::array_t<wp::float32> var_backend_vel)
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
        bool var_2;
        bool* var_3;
        bool var_4;
        bool var_5;
        bool* var_6;
        bool var_7;
        bool var_8;
        wp::float32* var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        const wp::float32 var_15 = 0.0;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        //---------
        // forward
        // def write_joint_state_user_to_backend_with_mask(                                       <L 649>
        // env_id, user_id = wp.tid()                                                             <L 684>
        builtin_tid2d(var_0, var_1);
        // if not env_mask[env_id] or not user_mask[user_id]:                                     <L 685>
        var_3 = wp::address(var_env_mask, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::unot(var_5);
        var_2 = var_4;
        if (!var_2) {
            var_6 = wp::address(var_user_mask, var_1);
            var_8 = wp::load(var_6);
            var_7 = wp::unot(var_8);
            var_2 = var_2 || var_7;
        }
        if (var_2) {
            // return                                                                             <L 686>
            continue;
        }
        // position = pos_data[env_id, user_id]                                                   <L 688>
        var_9 = wp::address(var_pos_data, var_0, var_1);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // velocity = vel_data[env_id, user_id]                                                   <L 689>
        var_12 = wp::address(var_vel_data, var_0, var_1);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // user_pos[env_id, user_id] = position                                                   <L 690>
        wp::array_store(var_user_pos, var_0, var_1, var_10);
        // user_vel[env_id, user_id] = velocity                                                   <L 691>
        wp::array_store(var_user_vel, var_0, var_1, var_13);
        // user_prev_vel[env_id, user_id] = velocity                                              <L 692>
        wp::array_store(var_user_prev_vel, var_0, var_1, var_13);
        // user_acc[env_id, user_id] = 0.0                                                        <L 693>
        wp::array_store(var_user_acc, var_0, var_1, var_15);
        // if has_ordering:                                                                       <L 694>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 695>
            var_16 = wp::address(var_user_to_backend, var_1);
            var_18 = wp::load(var_16);
            var_17 = wp::copy(var_18);
            // backend_pos[env_id, backend_id] = position                                         <L 696>
            wp::array_store(var_backend_pos, var_0, var_17, var_10);
            // backend_vel[env_id, backend_id] = velocity                                         <L 697>
            wp::array_store(var_backend_vel, var_0, var_17, var_13);
        }
    }
}



extern "C" __global__ void write_joint_state_user_to_backend_with_mask_fc0fe9f0_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_pos_data,
    wp::array_t<wp::float32> var_vel_data,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_user_mask,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_pos,
    wp::array_t<wp::float32> var_user_vel,
    wp::array_t<wp::float32> var_user_prev_vel,
    wp::array_t<wp::float32> var_user_acc,
    wp::array_t<wp::float32> var_backend_pos,
    wp::array_t<wp::float32> var_backend_vel,
    wp::array_t<wp::float32> adj_pos_data,
    wp::array_t<wp::float32> adj_vel_data,
    wp::array_t<bool> adj_env_mask,
    wp::array_t<bool> adj_user_mask,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    wp::array_t<wp::float32> adj_user_pos,
    wp::array_t<wp::float32> adj_user_vel,
    wp::array_t<wp::float32> adj_user_prev_vel,
    wp::array_t<wp::float32> adj_user_acc,
    wp::array_t<wp::float32> adj_backend_pos,
    wp::array_t<wp::float32> adj_backend_vel)
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
        bool var_2;
        bool* var_3;
        bool var_4;
        bool var_5;
        bool* var_6;
        bool var_7;
        bool var_8;
        wp::float32* var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        const wp::float32 var_15 = 0.0;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        bool adj_6 = {};
        bool adj_7 = {};
        bool adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        //---------
        // forward
        // def write_joint_state_user_to_backend_with_mask(                                       <L 649>
        // env_id, user_id = wp.tid()                                                             <L 684>
        builtin_tid2d(var_0, var_1);
        // if not env_mask[env_id] or not user_mask[user_id]:                                     <L 685>
        var_3 = wp::address(var_env_mask, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::unot(var_5);
        var_2 = var_4;
        if (!var_2) {
            var_6 = wp::address(var_user_mask, var_1);
            var_8 = wp::load(var_6);
            var_7 = wp::unot(var_8);
            var_2 = var_2 || var_7;
        }
        if (var_2) {
            // return                                                                             <L 686>
            goto label0;
        }
        // position = pos_data[env_id, user_id]                                                   <L 688>
        var_9 = wp::address(var_pos_data, var_0, var_1);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // velocity = vel_data[env_id, user_id]                                                   <L 689>
        var_12 = wp::address(var_vel_data, var_0, var_1);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // user_pos[env_id, user_id] = position                                                   <L 690>
        // wp::array_store(var_user_pos, var_0, var_1, var_10);
        // user_vel[env_id, user_id] = velocity                                                   <L 691>
        // wp::array_store(var_user_vel, var_0, var_1, var_13);
        // user_prev_vel[env_id, user_id] = velocity                                              <L 692>
        // wp::array_store(var_user_prev_vel, var_0, var_1, var_13);
        // user_acc[env_id, user_id] = 0.0                                                        <L 693>
        // wp::array_store(var_user_acc, var_0, var_1, var_15);
        // if has_ordering:                                                                       <L 694>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 695>
            var_16 = wp::address(var_user_to_backend, var_1);
            var_18 = wp::load(var_16);
            var_17 = wp::copy(var_18);
            // backend_pos[env_id, backend_id] = position                                         <L 696>
            // wp::array_store(var_backend_pos, var_0, var_17, var_10);
            // backend_vel[env_id, backend_id] = velocity                                         <L 697>
            // wp::array_store(var_backend_vel, var_0, var_17, var_13);
        }
        //---------
        // reverse
        if (var_has_ordering) {
            wp::adj_array_store(var_backend_vel, var_0, var_17, var_13, adj_backend_vel, adj_0, adj_17, adj_13);
            // adj: backend_vel[env_id, backend_id] = velocity                                    <L 697>
            wp::adj_array_store(var_backend_pos, var_0, var_17, var_10, adj_backend_pos, adj_0, adj_17, adj_10);
            // adj: backend_pos[env_id, backend_id] = position                                    <L 696>
            wp::adj_copy(var_18, adj_16, adj_17);
            wp::adj_address(var_user_to_backend, var_1, adj_user_to_backend, adj_1, adj_16);
            // adj: backend_id = user_to_backend[user_id]                                         <L 695>
        }
        // adj: if has_ordering:                                                                  <L 694>
        wp::adj_array_store(var_user_acc, var_0, var_1, var_15, adj_user_acc, adj_0, adj_1, adj_15);
        // adj: user_acc[env_id, user_id] = 0.0                                                   <L 693>
        wp::adj_array_store(var_user_prev_vel, var_0, var_1, var_13, adj_user_prev_vel, adj_0, adj_1, adj_13);
        // adj: user_prev_vel[env_id, user_id] = velocity                                         <L 692>
        wp::adj_array_store(var_user_vel, var_0, var_1, var_13, adj_user_vel, adj_0, adj_1, adj_13);
        // adj: user_vel[env_id, user_id] = velocity                                              <L 691>
        wp::adj_array_store(var_user_pos, var_0, var_1, var_10, adj_user_pos, adj_0, adj_1, adj_10);
        // adj: user_pos[env_id, user_id] = position                                              <L 690>
        wp::adj_copy(var_14, adj_12, adj_13);
        wp::adj_address(var_vel_data, var_0, var_1, adj_vel_data, adj_0, adj_1, adj_12);
        // adj: velocity = vel_data[env_id, user_id]                                              <L 689>
        wp::adj_copy(var_11, adj_9, adj_10);
        wp::adj_address(var_pos_data, var_0, var_1, adj_pos_data, adj_0, adj_1, adj_9);
        // adj: position = pos_data[env_id, user_id]                                              <L 688>
        if (var_2) {
            label0:;
            // adj: return                                                                        <L 686>
        }
        if (!var_2) {
            wp::adj_address(var_user_mask, var_1, adj_user_mask, adj_1, adj_6);
        }
        wp::adj_address(var_env_mask, var_0, adj_env_mask, adj_0, adj_3);
        // adj: if not env_mask[env_id] or not user_mask[user_id]:                                <L 685>
        // adj: env_id, user_id = wp.tid()                                                        <L 684>
        // adj: def write_joint_state_user_to_backend_with_mask(                                  <L 649>
        continue;
    }
}



extern "C" __global__ void write_joint_state_user_to_backend_with_indices_ff0ae9f4_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_pos_data,
    wp::array_t<wp::float32> var_vel_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    bool var_full_data,
    wp::array_t<wp::float32> var_user_pos,
    wp::array_t<wp::float32> var_user_vel,
    wp::array_t<wp::float32> var_user_prev_vel,
    wp::array_t<wp::float32> var_user_acc,
    wp::array_t<wp::float32> var_backend_pos,
    wp::array_t<wp::float32> var_backend_vel)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        const wp::float32 var_22 = 0.0;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        //---------
        // forward
        // def write_joint_state_user_to_backend_with_indices(                                    <L 1>
        // i, j = wp.tid()                                                                        <L 43>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 44>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 45>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if full_data:                                                                          <L 47>
        if (var_full_data) {
            // position = pos_data[env_id, user_id]                                               <L 48>
            var_8 = wp::address(var_pos_data, var_3, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
            // velocity = vel_data[env_id, user_id]                                               <L 49>
            var_11 = wp::address(var_vel_data, var_3, var_6);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
        }
        if (!var_full_data) {
            // position = pos_data[i, j]                                                          <L 51>
            var_14 = wp::address(var_pos_data, var_0, var_1);
            var_16 = wp::load(var_14);
            var_15 = wp::copy(var_16);
            // velocity = vel_data[i, j]                                                          <L 52>
            var_17 = wp::address(var_vel_data, var_0, var_1);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
        }
        var_20 = wp::where(var_full_data, var_9, var_15);
        var_21 = wp::where(var_full_data, var_12, var_18);
        // user_pos[env_id, user_id] = position                                                   <L 54>
        wp::array_store(var_user_pos, var_3, var_6, var_20);
        // user_vel[env_id, user_id] = velocity                                                   <L 55>
        wp::array_store(var_user_vel, var_3, var_6, var_21);
        // user_prev_vel[env_id, user_id] = velocity                                              <L 56>
        wp::array_store(var_user_prev_vel, var_3, var_6, var_21);
        // user_acc[env_id, user_id] = 0.0                                                        <L 57>
        wp::array_store(var_user_acc, var_3, var_6, var_22);
        // if has_ordering:                                                                       <L 58>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 59>
            var_23 = wp::address(var_user_to_backend, var_6);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // backend_pos[env_id, backend_id] = position                                         <L 60>
            wp::array_store(var_backend_pos, var_3, var_24, var_20);
            // backend_vel[env_id, backend_id] = velocity                                         <L 61>
            wp::array_store(var_backend_vel, var_3, var_24, var_21);
        }
    }
}



extern "C" __global__ void write_joint_state_user_to_backend_with_indices_ff0ae9f4_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_pos_data,
    wp::array_t<wp::float32> var_vel_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    bool var_full_data,
    wp::array_t<wp::float32> var_user_pos,
    wp::array_t<wp::float32> var_user_vel,
    wp::array_t<wp::float32> var_user_prev_vel,
    wp::array_t<wp::float32> var_user_acc,
    wp::array_t<wp::float32> var_backend_pos,
    wp::array_t<wp::float32> var_backend_vel,
    wp::array_t<wp::float32> adj_pos_data,
    wp::array_t<wp::float32> adj_vel_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int64> adj_user_ids,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    bool adj_full_data,
    wp::array_t<wp::float32> adj_user_pos,
    wp::array_t<wp::float32> adj_user_vel,
    wp::array_t<wp::float32> adj_user_prev_vel,
    wp::array_t<wp::float32> adj_user_acc,
    wp::array_t<wp::float32> adj_backend_pos,
    wp::array_t<wp::float32> adj_backend_vel)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        const wp::float32 var_22 = 0.0;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int64 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int64 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        //---------
        // forward
        // def write_joint_state_user_to_backend_with_indices(                                    <L 1>
        // i, j = wp.tid()                                                                        <L 43>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 44>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 45>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if full_data:                                                                          <L 47>
        if (var_full_data) {
            // position = pos_data[env_id, user_id]                                               <L 48>
            var_8 = wp::address(var_pos_data, var_3, var_6);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
            // velocity = vel_data[env_id, user_id]                                               <L 49>
            var_11 = wp::address(var_vel_data, var_3, var_6);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
        }
        if (!var_full_data) {
            // position = pos_data[i, j]                                                          <L 51>
            var_14 = wp::address(var_pos_data, var_0, var_1);
            var_16 = wp::load(var_14);
            var_15 = wp::copy(var_16);
            // velocity = vel_data[i, j]                                                          <L 52>
            var_17 = wp::address(var_vel_data, var_0, var_1);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
        }
        var_20 = wp::where(var_full_data, var_9, var_15);
        var_21 = wp::where(var_full_data, var_12, var_18);
        // user_pos[env_id, user_id] = position                                                   <L 54>
        // wp::array_store(var_user_pos, var_3, var_6, var_20);
        // user_vel[env_id, user_id] = velocity                                                   <L 55>
        // wp::array_store(var_user_vel, var_3, var_6, var_21);
        // user_prev_vel[env_id, user_id] = velocity                                              <L 56>
        // wp::array_store(var_user_prev_vel, var_3, var_6, var_21);
        // user_acc[env_id, user_id] = 0.0                                                        <L 57>
        // wp::array_store(var_user_acc, var_3, var_6, var_22);
        // if has_ordering:                                                                       <L 58>
        if (var_has_ordering) {
            // backend_id = user_to_backend[user_id]                                              <L 59>
            var_23 = wp::address(var_user_to_backend, var_6);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // backend_pos[env_id, backend_id] = position                                         <L 60>
            // wp::array_store(var_backend_pos, var_3, var_24, var_20);
            // backend_vel[env_id, backend_id] = velocity                                         <L 61>
            // wp::array_store(var_backend_vel, var_3, var_24, var_21);
        }
        //---------
        // reverse
        if (var_has_ordering) {
            wp::adj_array_store(var_backend_vel, var_3, var_24, var_21, adj_backend_vel, adj_3, adj_24, adj_21);
            // adj: backend_vel[env_id, backend_id] = velocity                                    <L 61>
            wp::adj_array_store(var_backend_pos, var_3, var_24, var_20, adj_backend_pos, adj_3, adj_24, adj_20);
            // adj: backend_pos[env_id, backend_id] = position                                    <L 60>
            wp::adj_copy(var_25, adj_23, adj_24);
            wp::adj_address(var_user_to_backend, var_6, adj_user_to_backend, adj_6, adj_23);
            // adj: backend_id = user_to_backend[user_id]                                         <L 59>
        }
        // adj: if has_ordering:                                                                  <L 58>
        wp::adj_array_store(var_user_acc, var_3, var_6, var_22, adj_user_acc, adj_3, adj_6, adj_22);
        // adj: user_acc[env_id, user_id] = 0.0                                                   <L 57>
        wp::adj_array_store(var_user_prev_vel, var_3, var_6, var_21, adj_user_prev_vel, adj_3, adj_6, adj_21);
        // adj: user_prev_vel[env_id, user_id] = velocity                                         <L 56>
        wp::adj_array_store(var_user_vel, var_3, var_6, var_21, adj_user_vel, adj_3, adj_6, adj_21);
        // adj: user_vel[env_id, user_id] = velocity                                              <L 55>
        wp::adj_array_store(var_user_pos, var_3, var_6, var_20, adj_user_pos, adj_3, adj_6, adj_20);
        // adj: user_pos[env_id, user_id] = position                                              <L 54>
        wp::adj_where(var_full_data, var_12, var_18, adj_full_data, adj_12, adj_18, adj_21);
        wp::adj_where(var_full_data, var_9, var_15, adj_full_data, adj_9, adj_15, adj_20);
        if (!var_full_data) {
            wp::adj_copy(var_19, adj_17, adj_18);
            wp::adj_address(var_vel_data, var_0, var_1, adj_vel_data, adj_0, adj_1, adj_17);
            // adj: velocity = vel_data[i, j]                                                     <L 52>
            wp::adj_copy(var_16, adj_14, adj_15);
            wp::adj_address(var_pos_data, var_0, var_1, adj_pos_data, adj_0, adj_1, adj_14);
            // adj: position = pos_data[i, j]                                                     <L 51>
        }
        if (var_full_data) {
            wp::adj_copy(var_13, adj_11, adj_12);
            wp::adj_address(var_vel_data, var_3, var_6, adj_vel_data, adj_3, adj_6, adj_11);
            // adj: velocity = vel_data[env_id, user_id]                                          <L 49>
            wp::adj_copy(var_10, adj_8, adj_9);
            wp::adj_address(var_pos_data, var_3, var_6, adj_pos_data, adj_3, adj_6, adj_8);
            // adj: position = pos_data[env_id, user_id]                                          <L 48>
        }
        // adj: if full_data:                                                                     <L 47>
        wp::adj_address(var_user_ids, var_1, adj_user_ids, adj_1, adj_5);
        // adj: user_id = wp.int32(user_ids[j])                                                   <L 45>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 44>
        // adj: i, j = wp.tid()                                                                   <L 43>
        // adj: def write_joint_state_user_to_backend_with_indices(                               <L 1>
        continue;
    }
}

