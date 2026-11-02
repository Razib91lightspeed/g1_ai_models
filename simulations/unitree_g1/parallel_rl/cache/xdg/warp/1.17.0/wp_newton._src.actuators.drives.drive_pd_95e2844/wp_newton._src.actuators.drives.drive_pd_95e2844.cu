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



extern "C" __global__ void _pd_effort_kernel_a501b73c_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float32> var_current_pos,
    wp::array_t<wp::float32> var_current_vel,
    wp::array_t<wp::float32> var_target_pos,
    wp::array_t<wp::float32> var_target_vel,
    wp::array_t<wp::float32> var_feedforward,
    wp::array_t<wp::uint32> var_pos_indices,
    wp::array_t<wp::uint32> var_vel_indices,
    wp::array_t<wp::uint32> var_target_pos_indices,
    wp::array_t<wp::uint32> var_target_vel_indices,
    wp::array_t<wp::float32> var_kp,
    wp::array_t<wp::float32> var_kd,
    wp::array_t<wp::float32> var_const_effort,
    wp::array_t<wp::float32> var_efforts)
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
        wp::uint32* var_4;
        wp::uint32 var_5;
        wp::uint32 var_6;
        wp::uint32* var_7;
        wp::uint32 var_8;
        wp::uint32 var_9;
        wp::uint32* var_10;
        wp::uint32 var_11;
        wp::uint32 var_12;
        wp::float32* var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::float32* var_18;
        wp::float32* var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        const wp::float32 var_23 = 0.0;
        wp::float32 var_24;
        wp::float32* var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        const wp::float32 var_29 = 0.0;
        wp::float32 var_30;
        wp::float32* var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32* var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        wp::float32* var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        //---------
        // forward
        // def _pd_effort_kernel(                                                                 <L 31>
        // i = wp.tid()                                                                           <L 47>
        var_0 = builtin_tid1d();
        // pos_idx = pos_indices[i]                                                               <L 48>
        var_1 = wp::address(var_pos_indices, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // vel_idx = vel_indices[i]                                                               <L 49>
        var_4 = wp::address(var_vel_indices, var_0);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // tgt_pos_idx = target_pos_indices[i]                                                    <L 50>
        var_7 = wp::address(var_target_pos_indices, var_0);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // tgt_vel_idx = target_vel_indices[i]                                                    <L 51>
        var_10 = wp::address(var_target_vel_indices, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // position_error = target_pos[tgt_pos_idx] - current_pos[pos_idx]                        <L 53>
        var_13 = wp::address(var_target_pos, var_8);
        var_14 = wp::address(var_current_pos, var_2);
        var_16 = wp::load(var_13);
        var_17 = wp::load(var_14);
        var_15 = wp::sub(var_16, var_17);
        // velocity_error = target_vel[tgt_vel_idx] - current_vel[vel_idx]                        <L 54>
        var_18 = wp::address(var_target_vel, var_11);
        var_19 = wp::address(var_current_vel, var_5);
        var_21 = wp::load(var_18);
        var_22 = wp::load(var_19);
        var_20 = wp::sub(var_21, var_22);
        // const_e = float(0.0)                                                                   <L 56>
        var_24 = wp::float(var_23);
        // if const_effort:                                                                       <L 57>
        if (var_const_effort) {
            // const_e = const_effort[i]                                                          <L 58>
            var_25 = wp::address(var_const_effort, var_0);
            var_27 = wp::load(var_25);
            var_26 = wp::copy(var_27);
        }
        var_28 = wp::where(var_const_effort, var_26, var_24);
        // ff = float(0.0)                                                                        <L 60>
        var_30 = wp::float(var_29);
        // if feedforward:                                                                        <L 61>
        if (var_feedforward) {
            // ff = feedforward[tgt_vel_idx]                                                      <L 62>
            var_31 = wp::address(var_feedforward, var_11);
            var_33 = wp::load(var_31);
            var_32 = wp::copy(var_33);
        }
        var_34 = wp::where(var_feedforward, var_32, var_30);
        // effort = const_e + ff + kp[i] * position_error + kd[i] * velocity_error                <L 64>
        var_35 = wp::add(var_28, var_34);
        var_36 = wp::address(var_kp, var_0);
        var_38 = wp::load(var_36);
        var_37 = wp::mul(var_38, var_15);
        var_39 = wp::add(var_35, var_37);
        var_40 = wp::address(var_kd, var_0);
        var_42 = wp::load(var_40);
        var_41 = wp::mul(var_42, var_20);
        var_43 = wp::add(var_39, var_41);
        // efforts[i] = effort                                                                    <L 65>
        wp::array_store(var_efforts, var_0, var_43);
    }
}



extern "C" __global__ void _pd_effort_kernel_a501b73c_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float32> var_current_pos,
    wp::array_t<wp::float32> var_current_vel,
    wp::array_t<wp::float32> var_target_pos,
    wp::array_t<wp::float32> var_target_vel,
    wp::array_t<wp::float32> var_feedforward,
    wp::array_t<wp::uint32> var_pos_indices,
    wp::array_t<wp::uint32> var_vel_indices,
    wp::array_t<wp::uint32> var_target_pos_indices,
    wp::array_t<wp::uint32> var_target_vel_indices,
    wp::array_t<wp::float32> var_kp,
    wp::array_t<wp::float32> var_kd,
    wp::array_t<wp::float32> var_const_effort,
    wp::array_t<wp::float32> var_efforts,
    wp::array_t<wp::float32> adj_current_pos,
    wp::array_t<wp::float32> adj_current_vel,
    wp::array_t<wp::float32> adj_target_pos,
    wp::array_t<wp::float32> adj_target_vel,
    wp::array_t<wp::float32> adj_feedforward,
    wp::array_t<wp::uint32> adj_pos_indices,
    wp::array_t<wp::uint32> adj_vel_indices,
    wp::array_t<wp::uint32> adj_target_pos_indices,
    wp::array_t<wp::uint32> adj_target_vel_indices,
    wp::array_t<wp::float32> adj_kp,
    wp::array_t<wp::float32> adj_kd,
    wp::array_t<wp::float32> adj_const_effort,
    wp::array_t<wp::float32> adj_efforts)
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
        wp::uint32* var_4;
        wp::uint32 var_5;
        wp::uint32 var_6;
        wp::uint32* var_7;
        wp::uint32 var_8;
        wp::uint32 var_9;
        wp::uint32* var_10;
        wp::uint32 var_11;
        wp::uint32 var_12;
        wp::float32* var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::float32* var_18;
        wp::float32* var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        const wp::float32 var_23 = 0.0;
        wp::float32 var_24;
        wp::float32* var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        const wp::float32 var_29 = 0.0;
        wp::float32 var_30;
        wp::float32* var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32* var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        wp::float32* var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::uint32 adj_1 = {};
        wp::uint32 adj_2 = {};
        wp::uint32 adj_3 = {};
        wp::uint32 adj_4 = {};
        wp::uint32 adj_5 = {};
        wp::uint32 adj_6 = {};
        wp::uint32 adj_7 = {};
        wp::uint32 adj_8 = {};
        wp::uint32 adj_9 = {};
        wp::uint32 adj_10 = {};
        wp::uint32 adj_11 = {};
        wp::uint32 adj_12 = {};
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
        wp::float32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::float32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::float32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::float32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::float32 adj_39 = {};
        wp::float32 adj_40 = {};
        wp::float32 adj_41 = {};
        wp::float32 adj_42 = {};
        wp::float32 adj_43 = {};
        //---------
        // forward
        // def _pd_effort_kernel(                                                                 <L 31>
        // i = wp.tid()                                                                           <L 47>
        var_0 = builtin_tid1d();
        // pos_idx = pos_indices[i]                                                               <L 48>
        var_1 = wp::address(var_pos_indices, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // vel_idx = vel_indices[i]                                                               <L 49>
        var_4 = wp::address(var_vel_indices, var_0);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // tgt_pos_idx = target_pos_indices[i]                                                    <L 50>
        var_7 = wp::address(var_target_pos_indices, var_0);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // tgt_vel_idx = target_vel_indices[i]                                                    <L 51>
        var_10 = wp::address(var_target_vel_indices, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // position_error = target_pos[tgt_pos_idx] - current_pos[pos_idx]                        <L 53>
        var_13 = wp::address(var_target_pos, var_8);
        var_14 = wp::address(var_current_pos, var_2);
        var_16 = wp::load(var_13);
        var_17 = wp::load(var_14);
        var_15 = wp::sub(var_16, var_17);
        // velocity_error = target_vel[tgt_vel_idx] - current_vel[vel_idx]                        <L 54>
        var_18 = wp::address(var_target_vel, var_11);
        var_19 = wp::address(var_current_vel, var_5);
        var_21 = wp::load(var_18);
        var_22 = wp::load(var_19);
        var_20 = wp::sub(var_21, var_22);
        // const_e = float(0.0)                                                                   <L 56>
        var_24 = wp::float(var_23);
        // if const_effort:                                                                       <L 57>
        if (var_const_effort) {
            // const_e = const_effort[i]                                                          <L 58>
            var_25 = wp::address(var_const_effort, var_0);
            var_27 = wp::load(var_25);
            var_26 = wp::copy(var_27);
        }
        var_28 = wp::where(var_const_effort, var_26, var_24);
        // ff = float(0.0)                                                                        <L 60>
        var_30 = wp::float(var_29);
        // if feedforward:                                                                        <L 61>
        if (var_feedforward) {
            // ff = feedforward[tgt_vel_idx]                                                      <L 62>
            var_31 = wp::address(var_feedforward, var_11);
            var_33 = wp::load(var_31);
            var_32 = wp::copy(var_33);
        }
        var_34 = wp::where(var_feedforward, var_32, var_30);
        // effort = const_e + ff + kp[i] * position_error + kd[i] * velocity_error                <L 64>
        var_35 = wp::add(var_28, var_34);
        var_36 = wp::address(var_kp, var_0);
        var_38 = wp::load(var_36);
        var_37 = wp::mul(var_38, var_15);
        var_39 = wp::add(var_35, var_37);
        var_40 = wp::address(var_kd, var_0);
        var_42 = wp::load(var_40);
        var_41 = wp::mul(var_42, var_20);
        var_43 = wp::add(var_39, var_41);
        // efforts[i] = effort                                                                    <L 65>
        // wp::array_store(var_efforts, var_0, var_43);
        //---------
        // reverse
        wp::adj_array_store(var_efforts, var_0, var_43, adj_efforts, adj_0, adj_43);
        // adj: efforts[i] = effort                                                               <L 65>
        wp::adj_add(var_39, var_41, adj_39, adj_41, adj_43);
        wp::adj_mul(var_42, var_20, adj_40, adj_20, adj_41);
        wp::adj_address(var_kd, var_0, adj_kd, adj_0, adj_40);
        wp::adj_add(var_35, var_37, adj_35, adj_37, adj_39);
        wp::adj_mul(var_38, var_15, adj_36, adj_15, adj_37);
        wp::adj_address(var_kp, var_0, adj_kp, adj_0, adj_36);
        wp::adj_add(var_28, var_34, adj_28, adj_34, adj_35);
        // adj: effort = const_e + ff + kp[i] * position_error + kd[i] * velocity_error           <L 64>
        wp::adj_where(var_feedforward, var_32, var_30, adj_feedforward, adj_32, adj_30, adj_34);
        if (var_feedforward) {
            wp::adj_copy(var_33, adj_31, adj_32);
            wp::adj_address(var_feedforward, var_11, adj_feedforward, adj_11, adj_31);
            // adj: ff = feedforward[tgt_vel_idx]                                                 <L 62>
        }
        // adj: if feedforward:                                                                   <L 61>
        wp::adj_float(var_29, adj_29, adj_30);
        // adj: ff = float(0.0)                                                                   <L 60>
        wp::adj_where(var_const_effort, var_26, var_24, adj_const_effort, adj_26, adj_24, adj_28);
        if (var_const_effort) {
            wp::adj_copy(var_27, adj_25, adj_26);
            wp::adj_address(var_const_effort, var_0, adj_const_effort, adj_0, adj_25);
            // adj: const_e = const_effort[i]                                                     <L 58>
        }
        // adj: if const_effort:                                                                  <L 57>
        wp::adj_float(var_23, adj_23, adj_24);
        // adj: const_e = float(0.0)                                                              <L 56>
        wp::adj_sub(var_21, var_22, adj_18, adj_19, adj_20);
        wp::adj_address(var_current_vel, var_5, adj_current_vel, adj_5, adj_19);
        wp::adj_address(var_target_vel, var_11, adj_target_vel, adj_11, adj_18);
        // adj: velocity_error = target_vel[tgt_vel_idx] - current_vel[vel_idx]                   <L 54>
        wp::adj_sub(var_16, var_17, adj_13, adj_14, adj_15);
        wp::adj_address(var_current_pos, var_2, adj_current_pos, adj_2, adj_14);
        wp::adj_address(var_target_pos, var_8, adj_target_pos, adj_8, adj_13);
        // adj: position_error = target_pos[tgt_pos_idx] - current_pos[pos_idx]                   <L 53>
        wp::adj_copy(var_12, adj_10, adj_11);
        wp::adj_address(var_target_vel_indices, var_0, adj_target_vel_indices, adj_0, adj_10);
        // adj: tgt_vel_idx = target_vel_indices[i]                                               <L 51>
        wp::adj_copy(var_9, adj_7, adj_8);
        wp::adj_address(var_target_pos_indices, var_0, adj_target_pos_indices, adj_0, adj_7);
        // adj: tgt_pos_idx = target_pos_indices[i]                                               <L 50>
        wp::adj_copy(var_6, adj_4, adj_5);
        wp::adj_address(var_vel_indices, var_0, adj_vel_indices, adj_0, adj_4);
        // adj: vel_idx = vel_indices[i]                                                          <L 49>
        wp::adj_copy(var_3, adj_1, adj_2);
        wp::adj_address(var_pos_indices, var_0, adj_pos_indices, adj_0, adj_1);
        // adj: pos_idx = pos_indices[i]                                                          <L 48>
        // adj: i = wp.tid()                                                                      <L 47>
        // adj: def _pd_effort_kernel(                                                            <L 31>
        continue;
    }
}

