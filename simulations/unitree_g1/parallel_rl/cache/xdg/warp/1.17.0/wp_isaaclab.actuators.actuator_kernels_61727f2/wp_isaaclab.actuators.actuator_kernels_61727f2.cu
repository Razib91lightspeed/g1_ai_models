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



extern "C" __global__ void write_2d_float_with_indices_089680a0_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_source,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    bool var_full_data,
    wp::array_t<wp::float32> var_target)
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
        wp::int64 var_6;
        wp::int64 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32* var_10;
        wp::float32 var_11;
        //---------
        // forward
        // def write_2d_float_with_indices(                                                       <L 1>
        // env_i, joint_i = wp.tid()                                                              <L 9>
        builtin_tid2d(var_0, var_1);
        // env_id = env_ids[env_i]                                                                <L 10>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // joint_id = joint_ids[joint_i]                                                          <L 11>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if full_data:                                                                          <L 12>
        if (var_full_data) {
            // target[env_id, joint_id] = source[env_id, joint_id]                                <L 13>
            var_8 = wp::address(var_source, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_target, var_3, var_6, var_9);
        }
        if (!var_full_data) {
            // target[env_id, joint_id] = source[env_i, joint_i]                                  <L 15>
            var_10 = wp::address(var_source, var_0, var_1);
            var_11 = wp::load(var_10);
            wp::array_store(var_target, var_3, var_6, var_11);
        }
    }
}



extern "C" __global__ void write_2d_float_with_mask_168cd667_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_source,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_joint_mask,
    wp::array_t<wp::float32> var_target)
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
        bool* var_5;
        bool var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        //---------
        // forward
        // def write_2d_float_with_mask(                                                          <L 55>
        // env_id, joint_id = wp.tid()                                                            <L 62>
        builtin_tid2d(var_0, var_1);
        // if env_mask[env_id] and joint_mask[joint_id]:                                          <L 63>
        var_3 = wp::address(var_env_mask, var_0);
        var_4 = wp::load(var_3);
        var_2 = var_4;
        if (var_2) {
            var_5 = wp::address(var_joint_mask, var_1);
            var_6 = wp::load(var_5);
            var_2 = var_2 && var_6;
        }
        if (var_2) {
            // target[env_id, joint_id] = source[env_id, joint_id]                                <L 64>
            var_7 = wp::address(var_source, var_0, var_1);
            var_8 = wp::load(var_7);
            wp::array_store(var_target, var_0, var_1, var_8);
        }
    }
}



extern "C" __global__ void compute_implicit_actuator_batch_25c8943e_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_command_pos,
    wp::array_t<wp::float32> var_command_vel,
    wp::array_t<wp::float32> var_command_effort,
    wp::array_t<wp::float32> var_joint_pos,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_joint_stiffness,
    wp::array_t<wp::float32> var_joint_damping,
    wp::array_t<wp::float32> var_joint_effort_limits,
    wp::array_t<wp::float32> var_velocity_limit,
    wp::array_t<wp::int32> var_joint_indices,
    wp::array_t<wp::float32> var_batch_computed_effort,
    wp::array_t<wp::float32> var_batch_applied_effort,
    wp::array_t<wp::float32> var_target_pos,
    wp::array_t<wp::float32> var_target_vel,
    wp::array_t<wp::float32> var_target_effort,
    wp::array_t<wp::float32> var_computed_effort,
    wp::array_t<wp::float32> var_applied_effort,
    wp::array_t<wp::float32> var_soft_velocity_limit)
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
        wp::float32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32* var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32* var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        wp::float32* var_28;
        wp::float32 var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        wp::float32* var_33;
        wp::float32 var_34;
        //---------
        // forward
        // def compute_implicit_actuator_batch(                                                   <L 104>
        // env_id, batch_joint_id = wp.tid()                                                      <L 125>
        builtin_tid2d(var_0, var_1);
        // joint_id = joint_indices[batch_joint_id]                                               <L 126>
        var_2 = wp::address(var_joint_indices, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // position_target = command_pos[env_id, joint_id]                                        <L 128>
        var_5 = wp::address(var_command_pos, var_0, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // velocity_target = command_vel[env_id, joint_id]                                        <L 129>
        var_8 = wp::address(var_command_vel, var_0, var_3);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // feedforward = command_effort[env_id, joint_id]                                         <L 130>
        var_11 = wp::address(var_command_effort, var_0, var_3);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // effort = (                                                                             <L 131>
        // joint_stiffness[env_id, joint_id] * (position_target - joint_pos[env_id, joint_id])       <L 132>
        var_14 = wp::address(var_joint_stiffness, var_0, var_3);
        var_15 = wp::address(var_joint_pos, var_0, var_3);
        var_17 = wp::load(var_15);
        var_16 = wp::sub(var_6, var_17);
        var_19 = wp::load(var_14);
        var_18 = wp::mul(var_19, var_16);
        // + joint_damping[env_id, joint_id] * (velocity_target - joint_vel[env_id, joint_id])       <L 133>
        var_20 = wp::address(var_joint_damping, var_0, var_3);
        var_21 = wp::address(var_joint_vel, var_0, var_3);
        var_23 = wp::load(var_21);
        var_22 = wp::sub(var_9, var_23);
        var_25 = wp::load(var_20);
        var_24 = wp::mul(var_25, var_22);
        var_26 = wp::add(var_18, var_24);
        // + feedforward                                                                          <L 134>
        var_27 = wp::add(var_26, var_12);
        // limit = joint_effort_limits[env_id, joint_id]                                          <L 136>
        var_28 = wp::address(var_joint_effort_limits, var_0, var_3);
        var_30 = wp::load(var_28);
        var_29 = wp::copy(var_30);
        // clamped_effort = wp.clamp(effort, -limit, limit)                                       <L 137>
        var_31 = wp::neg(var_29);
        var_32 = wp::clamp(var_27, var_31, var_29);
        // batch_computed_effort[env_id, batch_joint_id] = effort                                 <L 139>
        wp::array_store(var_batch_computed_effort, var_0, var_1, var_27);
        // batch_applied_effort[env_id, batch_joint_id] = clamped_effort                          <L 140>
        wp::array_store(var_batch_applied_effort, var_0, var_1, var_32);
        // target_pos[env_id, joint_id] = position_target                                         <L 141>
        wp::array_store(var_target_pos, var_0, var_3, var_6);
        // target_vel[env_id, joint_id] = velocity_target                                         <L 142>
        wp::array_store(var_target_vel, var_0, var_3, var_9);
        // target_effort[env_id, joint_id] = feedforward                                          <L 143>
        wp::array_store(var_target_effort, var_0, var_3, var_12);
        // computed_effort[env_id, joint_id] = effort                                             <L 144>
        wp::array_store(var_computed_effort, var_0, var_3, var_27);
        // applied_effort[env_id, joint_id] = clamped_effort                                      <L 145>
        wp::array_store(var_applied_effort, var_0, var_3, var_32);
        // soft_velocity_limit[env_id, joint_id] = velocity_limit[env_id, batch_joint_id]         <L 146>
        var_33 = wp::address(var_velocity_limit, var_0, var_1);
        var_34 = wp::load(var_33);
        wp::array_store(var_soft_velocity_limit, var_0, var_3, var_34);
    }
}



extern "C" __global__ void write_2d_float_with_indices_6e4471ae_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_source,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    bool var_full_data,
    wp::array_t<wp::float32> var_target)
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
        wp::int64 var_3;
        wp::int64 var_4;
        wp::int64* var_5;
        wp::int64 var_6;
        wp::int64 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32* var_10;
        wp::float32 var_11;
        //---------
        // forward
        // def write_2d_float_with_indices(                                                       <L 1>
        // env_i, joint_i = wp.tid()                                                              <L 9>
        builtin_tid2d(var_0, var_1);
        // env_id = env_ids[env_i]                                                                <L 10>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // joint_id = joint_ids[joint_i]                                                          <L 11>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if full_data:                                                                          <L 12>
        if (var_full_data) {
            // target[env_id, joint_id] = source[env_id, joint_id]                                <L 13>
            var_8 = wp::address(var_source, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_target, var_3, var_6, var_9);
        }
        if (!var_full_data) {
            // target[env_id, joint_id] = source[env_i, joint_i]                                  <L 15>
            var_10 = wp::address(var_source, var_0, var_1);
            var_11 = wp::load(var_10);
            wp::array_store(var_target, var_3, var_6, var_11);
        }
    }
}



extern "C" __global__ void write_2d_float_with_indices_c7885795_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_source,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    bool var_full_data,
    wp::array_t<wp::float32> var_target)
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
        wp::int64 var_3;
        wp::int64 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32* var_10;
        wp::float32 var_11;
        //---------
        // forward
        // def write_2d_float_with_indices(                                                       <L 1>
        // env_i, joint_i = wp.tid()                                                              <L 9>
        builtin_tid2d(var_0, var_1);
        // env_id = env_ids[env_i]                                                                <L 10>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // joint_id = joint_ids[joint_i]                                                          <L 11>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if full_data:                                                                          <L 12>
        if (var_full_data) {
            // target[env_id, joint_id] = source[env_id, joint_id]                                <L 13>
            var_8 = wp::address(var_source, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_target, var_3, var_6, var_9);
        }
        if (!var_full_data) {
            // target[env_id, joint_id] = source[env_i, joint_i]                                  <L 15>
            var_10 = wp::address(var_source, var_0, var_1);
            var_11 = wp::load(var_10);
            wp::array_store(var_target, var_3, var_6, var_11);
        }
    }
}



extern "C" __global__ void write_2d_float_with_indices_c8fc99ca_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_source,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    bool var_full_data,
    wp::array_t<wp::float32> var_target)
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
        wp::float32* var_10;
        wp::float32 var_11;
        //---------
        // forward
        // def write_2d_float_with_indices(                                                       <L 1>
        // env_i, joint_i = wp.tid()                                                              <L 9>
        builtin_tid2d(var_0, var_1);
        // env_id = env_ids[env_i]                                                                <L 10>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // joint_id = joint_ids[joint_i]                                                          <L 11>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if full_data:                                                                          <L 12>
        if (var_full_data) {
            // target[env_id, joint_id] = source[env_id, joint_id]                                <L 13>
            var_8 = wp::address(var_source, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_target, var_3, var_6, var_9);
        }
        if (!var_full_data) {
            // target[env_id, joint_id] = source[env_i, joint_i]                                  <L 15>
            var_10 = wp::address(var_source, var_0, var_1);
            var_11 = wp::load(var_10);
            wp::array_store(var_target, var_3, var_6, var_11);
        }
    }
}



extern "C" __global__ void scatter_explicit_actuator_outputs_ea7b523b_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_source_pos,
    wp::array_t<wp::float32> var_source_vel,
    wp::array_t<wp::float32> var_source_effort,
    wp::array_t<wp::float32> var_source_computed_effort,
    wp::array_t<wp::float32> var_source_applied_effort,
    wp::array_t<wp::float32> var_source_velocity_limit,
    wp::array_t<wp::int32> var_joint_indices,
    wp::array_t<wp::float32> var_target_pos,
    wp::array_t<wp::float32> var_target_vel,
    wp::array_t<wp::float32> var_target_effort,
    wp::array_t<wp::float32> var_target_computed_effort,
    wp::array_t<wp::float32> var_target_applied_effort,
    wp::array_t<wp::float32> var_target_velocity_limit)
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
        wp::float32* var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32* var_15;
        wp::float32 var_16;
        //---------
        // forward
        // def scatter_explicit_actuator_outputs(                                                 <L 68>
        // env_id, source_joint_id = wp.tid()                                                     <L 90>
        builtin_tid2d(var_0, var_1);
        // target_joint_id = joint_indices[source_joint_id]                                       <L 91>
        var_2 = wp::address(var_joint_indices, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if source_pos:                                                                         <L 92>
        if (var_source_pos) {
            // target_pos[env_id, target_joint_id] = source_pos[env_id, source_joint_id]          <L 93>
            var_5 = wp::address(var_source_pos, var_0, var_1);
            var_6 = wp::load(var_5);
            wp::array_store(var_target_pos, var_0, var_3, var_6);
        }
        // if source_vel:                                                                         <L 94>
        if (var_source_vel) {
            // target_vel[env_id, target_joint_id] = source_vel[env_id, source_joint_id]          <L 95>
            var_7 = wp::address(var_source_vel, var_0, var_1);
            var_8 = wp::load(var_7);
            wp::array_store(var_target_vel, var_0, var_3, var_8);
        }
        // if source_effort:                                                                      <L 96>
        if (var_source_effort) {
            // target_effort[env_id, target_joint_id] = source_effort[env_id, source_joint_id]       <L 97>
            var_9 = wp::address(var_source_effort, var_0, var_1);
            var_10 = wp::load(var_9);
            wp::array_store(var_target_effort, var_0, var_3, var_10);
        }
        // target_computed_effort[env_id, target_joint_id] = source_computed_effort[env_id, source_joint_id]       <L 98>
        var_11 = wp::address(var_source_computed_effort, var_0, var_1);
        var_12 = wp::load(var_11);
        wp::array_store(var_target_computed_effort, var_0, var_3, var_12);
        // target_applied_effort[env_id, target_joint_id] = source_applied_effort[env_id, source_joint_id]       <L 99>
        var_13 = wp::address(var_source_applied_effort, var_0, var_1);
        var_14 = wp::load(var_13);
        wp::array_store(var_target_applied_effort, var_0, var_3, var_14);
        // target_velocity_limit[env_id, target_joint_id] = source_velocity_limit[env_id, source_joint_id]       <L 100>
        var_15 = wp::address(var_source_velocity_limit, var_0, var_1);
        var_16 = wp::load(var_15);
        wp::array_store(var_target_velocity_limit, var_0, var_3, var_16);
    }
}

