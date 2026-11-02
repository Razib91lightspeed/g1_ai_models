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



extern "C" __global__ void build_per_dof_env_mask_kernel_614ca298_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::uint32> var_indices,
    wp::array_t<bool> var_env_mask,
    wp::int32 var_dof_offset,
    wp::int32 var_num_joints,
    wp::array_t<bool> var_out_mask)
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
        wp::int32 var_4;
        wp::int32 var_5;
        bool* var_6;
        bool var_7;
        //---------
        // forward
        // def build_per_dof_env_mask_kernel(                                                     <L 38>
        // i = wp.tid()                                                                           <L 53>
        var_0 = builtin_tid1d();
        // global_dof = int(indices[i]) - dof_offset                                              <L 54>
        var_1 = wp::address(var_indices, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::int(var_3);
        var_4 = wp::sub(var_2, var_dof_offset);
        // env = global_dof // num_joints                                                         <L 55>
        var_5 = wp::floordiv(var_4, var_num_joints);
        // out_mask[i] = env_mask[env]                                                            <L 56>
        var_6 = wp::address(var_env_mask, var_5);
        var_7 = wp::load(var_6);
        wp::array_store(var_out_mask, var_0, var_7);
    }
}



extern "C" __global__ void zero_at_indices_kernel_9063e296_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float32> var_data,
    wp::array_t<wp::uint32> var_indices)
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
        const wp::float32 var_1 = 0.0;
        wp::uint32* var_2;
        wp::uint32 var_3;
        //---------
        // forward
        // def zero_at_indices_kernel(data: wp.array(dtype=wp.float32), indices: wp.array(dtype=wp.uint32)):       <L 24>
        // i = wp.tid()                                                                           <L 26>
        var_0 = builtin_tid1d();
        // data[indices[i]] = 0.0                                                                 <L 27>
        var_2 = wp::address(var_indices, var_0);
        var_3 = wp::load(var_2);
        wp::array_store(var_data, var_3, var_1);
    }
}



extern "C" __global__ void sync_torque_telemetry_fdad2bc6_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_joint_pos_backend,
    wp::array_t<wp::float32> var_joint_vel_backend,
    wp::array_t<wp::float32> var_joint_pos_target,
    wp::array_t<wp::float32> var_joint_vel_target,
    wp::array_t<wp::float32> var_joint_stiffness,
    wp::array_t<wp::float32> var_joint_damping,
    wp::array_t<wp::float32> var_effort_limit,
    wp::array_t<wp::int32> var_joint_modes,
    wp::array_t<wp::float32> var_sim_bind_joint_effort,
    wp::array_t<wp::float32> var_actuator_computed_effort,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_sim_buffers_are_backend_order,
    wp::array_t<wp::float32> var_computed,
    wp::array_t<wp::float32> var_applied)
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
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        const wp::int32 var_8 = 1;
        bool var_9;
        wp::int32 var_10;
        wp::float32* var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::float32* var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::float32* var_24;
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
        wp::float32 var_35;
        wp::float32* var_36;
        wp::float32 var_37;
        wp::float32* var_38;
        wp::float32 var_39;
        //---------
        // forward
        // def sync_torque_telemetry(                                                             <L 65>
        // i, user_j = wp.tid()                                                                   <L 96>
        builtin_tid2d(var_0, var_1);
        // backend_j = user_j                                                                     <L 97>
        var_2 = wp::copy(var_1);
        // if sim_buffers_are_backend_order:                                                      <L 98>
        if (var_sim_buffers_are_backend_order) {
            // backend_j = user_to_backend[user_j]                                                <L 99>
            var_3 = wp::address(var_user_to_backend, var_1);
            var_5 = wp::load(var_3);
            var_4 = wp::copy(var_5);
        }
        var_6 = wp::where(var_sim_buffers_are_backend_order, var_4, var_2);
        // if joint_modes[user_j] == 1:                                                           <L 100>
        var_7 = wp::address(var_joint_modes, var_1);
        var_10 = wp::load(var_7);
        var_9 = (var_10 == var_8);
        if (var_9) {
            // err_p = joint_pos_target[i, user_j] - joint_pos_backend[i, backend_j]              <L 101>
            var_11 = wp::address(var_joint_pos_target, var_0, var_1);
            var_12 = wp::address(var_joint_pos_backend, var_0, var_6);
            var_14 = wp::load(var_11);
            var_15 = wp::load(var_12);
            var_13 = wp::sub(var_14, var_15);
            // err_v = joint_vel_target[i, user_j] - joint_vel_backend[i, backend_j]              <L 102>
            var_16 = wp::address(var_joint_vel_target, var_0, var_1);
            var_17 = wp::address(var_joint_vel_backend, var_0, var_6);
            var_19 = wp::load(var_16);
            var_20 = wp::load(var_17);
            var_18 = wp::sub(var_19, var_20);
            // pd = joint_stiffness[i, user_j] * err_p + joint_damping[i, user_j] * err_v         <L 103>
            var_21 = wp::address(var_joint_stiffness, var_0, var_1);
            var_23 = wp::load(var_21);
            var_22 = wp::mul(var_23, var_13);
            var_24 = wp::address(var_joint_damping, var_0, var_1);
            var_26 = wp::load(var_24);
            var_25 = wp::mul(var_26, var_18);
            var_27 = wp::add(var_22, var_25);
            // limit = effort_limit[i, user_j]                                                    <L 104>
            var_28 = wp::address(var_effort_limit, var_0, var_1);
            var_30 = wp::load(var_28);
            var_29 = wp::copy(var_30);
            // pd_clipped = wp.clamp(pd, -limit, limit)                                           <L 105>
            var_31 = wp::neg(var_29);
            var_32 = wp::clamp(var_27, var_31, var_29);
            // total = pd_clipped + sim_bind_joint_effort[i, backend_j]                           <L 106>
            var_33 = wp::address(var_sim_bind_joint_effort, var_0, var_6);
            var_35 = wp::load(var_33);
            var_34 = wp::add(var_32, var_35);
            // computed[i, user_j] = total                                                        <L 107>
            wp::array_store(var_computed, var_0, var_1, var_34);
            // applied[i, user_j] = total                                                         <L 108>
            wp::array_store(var_applied, var_0, var_1, var_34);
        }
        if (!var_9) {
            // computed[i, user_j] = actuator_computed_effort[i, backend_j]                       <L 110>
            var_36 = wp::address(var_actuator_computed_effort, var_0, var_6);
            var_37 = wp::load(var_36);
            wp::array_store(var_computed, var_0, var_1, var_37);
            // applied[i, user_j] = sim_bind_joint_effort[i, backend_j]                           <L 111>
            var_38 = wp::address(var_sim_bind_joint_effort, var_0, var_6);
            var_39 = wp::load(var_38);
            wp::array_store(var_applied, var_0, var_1, var_39);
        }
    }
}

