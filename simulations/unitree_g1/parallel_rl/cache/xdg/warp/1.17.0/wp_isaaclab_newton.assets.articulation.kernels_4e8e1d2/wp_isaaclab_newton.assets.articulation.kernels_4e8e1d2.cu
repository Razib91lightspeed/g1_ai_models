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


// /workspace/isaaclab/source/isaaclab_newton/isaaclab_newton/assets/articulation/kernels.py:25
static CUDA_CALLABLE wp::vec_t<2, wp::float32> compute_soft_joint_pos_limits_func_0(
    wp::vec_t<2, wp::float32> var_joint_pos_limits,
    wp::float32 var_soft_limit_factor)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::float32 var_5 = 2.0;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    const wp::int32 var_9 = 0;
    wp::float32 var_10;
    wp::float32 var_11;
    const wp::float32 var_12 = 0.5;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::float32 var_16 = 0.5;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::vec_t<2, wp::float32> var_20;
    //---------
    // forward
    // def compute_soft_joint_pos_limits_func(                                                <L 26>
    // joint_pos_mean = (joint_pos_limits[0] + joint_pos_limits[1]) / 2.0                     <L 39>
    var_1 = wp::extract(var_joint_pos_limits, var_0);
    var_3 = wp::extract(var_joint_pos_limits, var_2);
    var_4 = wp::add(var_1, var_3);
    var_6 = wp::div(var_4, var_5);
    // joint_pos_range = joint_pos_limits[1] - joint_pos_limits[0]                            <L 40>
    var_8 = wp::extract(var_joint_pos_limits, var_7);
    var_10 = wp::extract(var_joint_pos_limits, var_9);
    var_11 = wp::sub(var_8, var_10);
    // return wp.vec2f(                                                                       <L 41>
    // joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor,                            <L 42>
    var_13 = wp::mul(var_12, var_11);
    var_14 = wp::mul(var_13, var_soft_limit_factor);
    var_15 = wp::sub(var_6, var_14);
    // joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor,                            <L 43>
    var_17 = wp::mul(var_16, var_11);
    var_18 = wp::mul(var_17, var_soft_limit_factor);
    var_19 = wp::add(var_6, var_18);
    var_20 = wp::vec_t<2, wp::float32>(var_15, var_19);
    return var_20;
}


// /workspace/isaaclab/source/isaaclab/isaaclab/assets/articulation/ordering_kernels.py:59
static CUDA_CALLABLE wp::int32 resolve_backend_index_0(
    wp::int32 var_user_id,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    //---------
    // forward
    // def resolve_backend_index(                                                             <L 60>
    // if has_ordering:                                                                       <L 82>
    if (var_has_ordering) {
        // return user_to_backend[user_id]                                                    <L 83>
        var_0 = wp::address(var_user_to_backend, var_user_id);
        var_2 = wp::load(var_0);
        var_1 = wp::copy(var_2);
        return var_1;
    }
    // return user_id                                                                         <L 84>
    return var_user_id;
}


// /workspace/isaaclab/source/isaaclab_newton/isaaclab_newton/assets/kernels.py:25
static CUDA_CALLABLE wp::vec_t<6, wp::float32> pack_body_wrench_to_world_0(
    wp::vec_t<3, wp::float32> var_force_b,
    wp::vec_t<3, wp::float32> var_torque_b,
    wp::quat_t<wp::float32> var_body_rot_w)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<6, wp::float32> var_2;
    //---------
    // forward
    // def pack_body_wrench_to_world(                                                         <L 26>
    // return wp.spatial_vector(                                                              <L 32>
    // wp.quat_rotate(body_rot_w, force_b),                                                   <L 33>
    var_0 = wp::quat_rotate(var_body_rot_w, var_force_b);
    // wp.quat_rotate(body_rot_w, torque_b),                                                  <L 34>
    var_1 = wp::quat_rotate(var_body_rot_w, var_torque_b);
    // wp.float32,                                                                            <L 35>
    var_2 = wp::vec_t<6, wp::float32>(var_0, var_1);
    return var_2;
}


// /workspace/isaaclab/source/isaaclab_newton/isaaclab_newton/assets/articulation/kernels.py:25
static CUDA_CALLABLE void adj_compute_soft_joint_pos_limits_func_0(
    wp::vec_t<2, wp::float32> var_joint_pos_limits,
    wp::float32 var_soft_limit_factor,
    wp::vec_t<2, wp::float32> & adj_joint_pos_limits,
    wp::float32 & adj_soft_limit_factor,
    wp::vec_t<2, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::float32 var_5 = 2.0;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    const wp::int32 var_9 = 0;
    wp::float32 var_10;
    wp::float32 var_11;
    const wp::float32 var_12 = 0.5;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::float32 var_16 = 0.5;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::vec_t<2, wp::float32> var_20;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::int32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::float32 adj_5 = {};
    wp::float32 adj_6 = {};
    wp::int32 adj_7 = {};
    wp::float32 adj_8 = {};
    wp::int32 adj_9 = {};
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
    wp::vec_t<2, wp::float32> adj_20 = {};
    //---------
    // forward
    // def compute_soft_joint_pos_limits_func(                                                <L 26>
    // joint_pos_mean = (joint_pos_limits[0] + joint_pos_limits[1]) / 2.0                     <L 39>
    var_1 = wp::extract(var_joint_pos_limits, var_0);
    var_3 = wp::extract(var_joint_pos_limits, var_2);
    var_4 = wp::add(var_1, var_3);
    var_6 = wp::div(var_4, var_5);
    // joint_pos_range = joint_pos_limits[1] - joint_pos_limits[0]                            <L 40>
    var_8 = wp::extract(var_joint_pos_limits, var_7);
    var_10 = wp::extract(var_joint_pos_limits, var_9);
    var_11 = wp::sub(var_8, var_10);
    // return wp.vec2f(                                                                       <L 41>
    // joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor,                            <L 42>
    var_13 = wp::mul(var_12, var_11);
    var_14 = wp::mul(var_13, var_soft_limit_factor);
    var_15 = wp::sub(var_6, var_14);
    // joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor,                            <L 43>
    var_17 = wp::mul(var_16, var_11);
    var_18 = wp::mul(var_17, var_soft_limit_factor);
    var_19 = wp::add(var_6, var_18);
    var_20 = wp::vec_t<2, wp::float32>(var_15, var_19);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_20 += adj_ret;
    wp::adj_vec_t(var_15, var_19, adj_15, adj_19, adj_20);
    wp::adj_add(var_6, var_18, adj_6, adj_18, adj_19);
    wp::adj_mul(var_17, var_soft_limit_factor, adj_17, adj_soft_limit_factor, adj_18);
    wp::adj_mul(var_16, var_11, adj_16, adj_11, adj_17);
    // adj: joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor,                       <L 43>
    wp::adj_sub(var_6, var_14, adj_6, adj_14, adj_15);
    wp::adj_mul(var_13, var_soft_limit_factor, adj_13, adj_soft_limit_factor, adj_14);
    wp::adj_mul(var_12, var_11, adj_12, adj_11, adj_13);
    // adj: joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor,                       <L 42>
    // adj: return wp.vec2f(                                                                  <L 41>
    wp::adj_sub(var_8, var_10, adj_8, adj_10, adj_11);
    wp::adj_extract(var_joint_pos_limits, var_9, adj_joint_pos_limits, adj_9, adj_10);
    wp::adj_extract(var_joint_pos_limits, var_7, adj_joint_pos_limits, adj_7, adj_8);
    // adj: joint_pos_range = joint_pos_limits[1] - joint_pos_limits[0]                       <L 40>
    wp::adj_div(var_4, var_5, var_6, adj_4, adj_5, adj_6);
    wp::adj_add(var_1, var_3, adj_1, adj_3, adj_4);
    wp::adj_extract(var_joint_pos_limits, var_2, adj_joint_pos_limits, adj_2, adj_3);
    wp::adj_extract(var_joint_pos_limits, var_0, adj_joint_pos_limits, adj_0, adj_1);
    // adj: joint_pos_mean = (joint_pos_limits[0] + joint_pos_limits[1]) / 2.0                <L 39>
    // adj: def compute_soft_joint_pos_limits_func(                                           <L 26>
    return;
}


// /workspace/isaaclab/source/isaaclab/isaaclab/assets/articulation/ordering_kernels.py:59
static CUDA_CALLABLE void adj_resolve_backend_index_0(
    wp::int32 var_user_id,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::int32 & adj_user_id,
    wp::array_t<wp::int32> & adj_user_to_backend,
    bool & adj_has_ordering,
    wp::int32 & adj_ret)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::int32 adj_1 = {};
    wp::int32 adj_2 = {};
    //---------
    // forward
    // def resolve_backend_index(                                                             <L 60>
    // if has_ordering:                                                                       <L 82>
    if (var_has_ordering) {
        // return user_to_backend[user_id]                                                    <L 83>
        var_0 = wp::address(var_user_to_backend, var_user_id);
        var_2 = wp::load(var_0);
        var_1 = wp::copy(var_2);
        goto label0;
    }
    // return user_id                                                                         <L 84>
    goto label1;
    //---------
    // reverse
    label1:;
    adj_user_id += adj_ret;
    // adj: return user_id                                                                    <L 84>
    if (var_has_ordering) {
        label0:;
        adj_1 += adj_ret;
        wp::adj_copy(var_2, adj_0, adj_1);
        wp::adj_address(var_user_to_backend, var_user_id, adj_user_to_backend, adj_user_id, adj_0);
        // adj: return user_to_backend[user_id]                                               <L 83>
    }
    // adj: if has_ordering:                                                                  <L 82>
    // adj: def resolve_backend_index(                                                        <L 60>
    return;
}


// /workspace/isaaclab/source/isaaclab_newton/isaaclab_newton/assets/kernels.py:25
static CUDA_CALLABLE void adj_pack_body_wrench_to_world_0(
    wp::vec_t<3, wp::float32> var_force_b,
    wp::vec_t<3, wp::float32> var_torque_b,
    wp::quat_t<wp::float32> var_body_rot_w,
    wp::vec_t<3, wp::float32> & adj_force_b,
    wp::vec_t<3, wp::float32> & adj_torque_b,
    wp::quat_t<wp::float32> & adj_body_rot_w,
    wp::vec_t<6, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<6, wp::float32> var_2;
    //---------
    // dual vars
    wp::vec_t<3, wp::float32> adj_0 = {};
    wp::vec_t<3, wp::float32> adj_1 = {};
    wp::vec_t<6, wp::float32> adj_2 = {};
    //---------
    // forward
    // def pack_body_wrench_to_world(                                                         <L 26>
    // return wp.spatial_vector(                                                              <L 32>
    // wp.quat_rotate(body_rot_w, force_b),                                                   <L 33>
    var_0 = wp::quat_rotate(var_body_rot_w, var_force_b);
    // wp.quat_rotate(body_rot_w, torque_b),                                                  <L 34>
    var_1 = wp::quat_rotate(var_body_rot_w, var_torque_b);
    // wp.float32,                                                                            <L 35>
    var_2 = wp::vec_t<6, wp::float32>(var_0, var_1);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_2 += adj_ret;
    wp::adj_vec_t(var_0, var_1, adj_0, adj_1, adj_2);
    // adj: wp.float32,                                                                       <L 35>
    wp::adj_quat_rotate(var_body_rot_w, var_torque_b, adj_body_rot_w, adj_torque_b, adj_1);
    // adj: wp.quat_rotate(body_rot_w, torque_b),                                             <L 34>
    wp::adj_quat_rotate(var_body_rot_w, var_force_b, adj_body_rot_w, adj_force_b, adj_0);
    // adj: wp.quat_rotate(body_rot_w, force_b),                                              <L 33>
    // adj: return wp.spatial_vector(                                                         <L 32>
    // adj: def pack_body_wrench_to_world(                                                    <L 26>
    return;
}



extern "C" __global__ void write_joint_limit_data_to_buffer_mask_02b07045_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_in_data,
    wp::float32 var_soft_limit_factor,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_joint_mask,
    wp::array_t<wp::float32> var_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_soft_joint_pos_limits,
    wp::array_t<wp::float32> var_default_joint_pos,
    wp::array_t<wp::int32> var_clamped_defaults)
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
        wp::vec_t<2, wp::float32>* var_7;
        const wp::int32 var_8 = 0;
        wp::float32 var_9;
        wp::vec_t<2, wp::float32> var_10;
        wp::vec_t<2, wp::float32>* var_11;
        const wp::int32 var_12 = 1;
        wp::float32 var_13;
        wp::vec_t<2, wp::float32> var_14;
        bool var_15;
        wp::float32* var_16;
        wp::float32* var_17;
        bool var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32* var_21;
        wp::float32* var_22;
        bool var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        const wp::int32 var_26 = 0;
        const wp::int32 var_27 = 1;
        wp::int32 var_28;
        wp::float32* var_29;
        wp::float32* var_30;
        wp::float32* var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32* var_36;
        wp::float32* var_37;
        wp::vec_t<2, wp::float32> var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        wp::vec_t<2, wp::float32> var_41;
        //---------
        // forward
        // def write_joint_limit_data_to_buffer_mask(                                             <L 206>
        // i, j = wp.tid()                                                                        <L 240>
        builtin_tid2d(var_0, var_1);
        // if env_mask[i] and joint_mask[j]:                                                      <L 241>
        var_3 = wp::address(var_env_mask, var_0);
        var_4 = wp::load(var_3);
        var_2 = var_4;
        if (var_2) {
            var_5 = wp::address(var_joint_mask, var_1);
            var_6 = wp::load(var_5);
            var_2 = var_2 && var_6;
        }
        if (var_2) {
            // joint_pos_limits_lower[i, j] = in_data[i, j][0]                                    <L 242>
            var_7 = wp::address(var_in_data, var_0, var_1);
            var_10 = wp::load(var_7);
            var_9 = wp::extract(var_10, var_8);
            wp::array_store(var_joint_pos_limits_lower, var_0, var_1, var_9);
            // joint_pos_limits_upper[i, j] = in_data[i, j][1]                                    <L 243>
            var_11 = wp::address(var_in_data, var_0, var_1);
            var_14 = wp::load(var_11);
            var_13 = wp::extract(var_14, var_12);
            wp::array_store(var_joint_pos_limits_upper, var_0, var_1, var_13);
            // if (default_joint_pos[i, j] < joint_pos_limits_lower[i, j]) or default_joint_pos[i, j] > joint_pos_limits_upper[       <L 244>
            var_16 = wp::address(var_default_joint_pos, var_0, var_1);
            var_17 = wp::address(var_joint_pos_limits_lower, var_0, var_1);
            var_19 = wp::load(var_16);
            var_20 = wp::load(var_17);
            var_18 = (var_19 < var_20);
            var_15 = var_18;
            if (!var_15) {
                var_21 = wp::address(var_default_joint_pos, var_0, var_1);
                // i, j                                                                           <L 245>
                var_22 = wp::address(var_joint_pos_limits_upper, var_0, var_1);
                var_24 = wp::load(var_21);
                var_25 = wp::load(var_22);
                var_23 = (var_24 > var_25);
                var_15 = var_15 || var_23;
            }
            if (var_15) {
                // wp.atomic_add(clamped_defaults, 0, 1)                                          <L 247>
                var_28 = wp::atomic_add(var_clamped_defaults, var_26, var_27);
                // default_joint_pos[i, j] = wp.clamp(                                            <L 248>
                // default_joint_pos[i, j],                                                       <L 249>
                var_29 = wp::address(var_default_joint_pos, var_0, var_1);
                // joint_pos_limits_lower[i, j],                                                  <L 250>
                var_30 = wp::address(var_joint_pos_limits_lower, var_0, var_1);
                // joint_pos_limits_upper[i, j],                                                  <L 251>
                var_31 = wp::address(var_joint_pos_limits_upper, var_0, var_1);
                var_33 = wp::load(var_29);
                var_34 = wp::load(var_30);
                var_35 = wp::load(var_31);
                var_32 = wp::clamp(var_33, var_34, var_35);
                // default_joint_pos[i, j] = wp.clamp(                                            <L 248>
                wp::array_store(var_default_joint_pos, var_0, var_1, var_32);
            }
            // soft_joint_pos_limits[i, j] = compute_soft_joint_pos_limits_func(                  <L 253>
            // wp.vec2f(joint_pos_limits_lower[i, j], joint_pos_limits_upper[i, j]), soft_limit_factor       <L 254>
            var_36 = wp::address(var_joint_pos_limits_lower, var_0, var_1);
            var_37 = wp::address(var_joint_pos_limits_upper, var_0, var_1);
            var_39 = wp::load(var_36);
            var_40 = wp::load(var_37);
            var_38 = wp::vec_t<2, wp::float32>(var_39, var_40);
            var_41 = compute_soft_joint_pos_limits_func_0(var_38, var_soft_limit_factor);
            // soft_joint_pos_limits[i, j] = compute_soft_joint_pos_limits_func(                  <L 253>
            wp::array_store(var_soft_joint_pos_limits, var_0, var_1, var_41);
        }
    }
}



extern "C" __global__ void write_joint_limit_data_to_buffer_mask_02b07045_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_in_data,
    wp::float32 var_soft_limit_factor,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_joint_mask,
    wp::array_t<wp::float32> var_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_soft_joint_pos_limits,
    wp::array_t<wp::float32> var_default_joint_pos,
    wp::array_t<wp::int32> var_clamped_defaults,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_in_data,
    wp::float32 adj_soft_limit_factor,
    wp::array_t<bool> adj_env_mask,
    wp::array_t<bool> adj_joint_mask,
    wp::array_t<wp::float32> adj_joint_pos_limits_lower,
    wp::array_t<wp::float32> adj_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_soft_joint_pos_limits,
    wp::array_t<wp::float32> adj_default_joint_pos,
    wp::array_t<wp::int32> adj_clamped_defaults)
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
        wp::vec_t<2, wp::float32>* var_7;
        const wp::int32 var_8 = 0;
        wp::float32 var_9;
        wp::vec_t<2, wp::float32> var_10;
        wp::vec_t<2, wp::float32>* var_11;
        const wp::int32 var_12 = 1;
        wp::float32 var_13;
        wp::vec_t<2, wp::float32> var_14;
        bool var_15;
        wp::float32* var_16;
        wp::float32* var_17;
        bool var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32* var_21;
        wp::float32* var_22;
        bool var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        const wp::int32 var_26 = 0;
        const wp::int32 var_27 = 1;
        wp::int32 var_28;
        wp::float32* var_29;
        wp::float32* var_30;
        wp::float32* var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32* var_36;
        wp::float32* var_37;
        wp::vec_t<2, wp::float32> var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        wp::vec_t<2, wp::float32> var_41;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        bool adj_6 = {};
        wp::vec_t<2, wp::float32> adj_7 = {};
        wp::int32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::vec_t<2, wp::float32> adj_10 = {};
        wp::vec_t<2, wp::float32> adj_11 = {};
        wp::int32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::vec_t<2, wp::float32> adj_14 = {};
        bool adj_15 = {};
        wp::float32 adj_16 = {};
        wp::float32 adj_17 = {};
        bool adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        bool adj_23 = {};
        wp::float32 adj_24 = {};
        wp::float32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::float32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::float32 adj_37 = {};
        wp::vec_t<2, wp::float32> adj_38 = {};
        wp::float32 adj_39 = {};
        wp::float32 adj_40 = {};
        wp::vec_t<2, wp::float32> adj_41 = {};
        //---------
        // forward
        // def write_joint_limit_data_to_buffer_mask(                                             <L 206>
        // i, j = wp.tid()                                                                        <L 240>
        builtin_tid2d(var_0, var_1);
        // if env_mask[i] and joint_mask[j]:                                                      <L 241>
        var_3 = wp::address(var_env_mask, var_0);
        var_4 = wp::load(var_3);
        var_2 = var_4;
        if (var_2) {
            var_5 = wp::address(var_joint_mask, var_1);
            var_6 = wp::load(var_5);
            var_2 = var_2 && var_6;
        }
        if (var_2) {
            // joint_pos_limits_lower[i, j] = in_data[i, j][0]                                    <L 242>
            var_7 = wp::address(var_in_data, var_0, var_1);
            var_10 = wp::load(var_7);
            var_9 = wp::extract(var_10, var_8);
            // wp::array_store(var_joint_pos_limits_lower, var_0, var_1, var_9);
            // joint_pos_limits_upper[i, j] = in_data[i, j][1]                                    <L 243>
            var_11 = wp::address(var_in_data, var_0, var_1);
            var_14 = wp::load(var_11);
            var_13 = wp::extract(var_14, var_12);
            // wp::array_store(var_joint_pos_limits_upper, var_0, var_1, var_13);
            // if (default_joint_pos[i, j] < joint_pos_limits_lower[i, j]) or default_joint_pos[i, j] > joint_pos_limits_upper[       <L 244>
            var_16 = wp::address(var_default_joint_pos, var_0, var_1);
            var_17 = wp::address(var_joint_pos_limits_lower, var_0, var_1);
            var_19 = wp::load(var_16);
            var_20 = wp::load(var_17);
            var_18 = (var_19 < var_20);
            var_15 = var_18;
            if (!var_15) {
                var_21 = wp::address(var_default_joint_pos, var_0, var_1);
                // i, j                                                                           <L 245>
                var_22 = wp::address(var_joint_pos_limits_upper, var_0, var_1);
                var_24 = wp::load(var_21);
                var_25 = wp::load(var_22);
                var_23 = (var_24 > var_25);
                var_15 = var_15 || var_23;
            }
            if (var_15) {
                // wp.atomic_add(clamped_defaults, 0, 1)                                          <L 247>
                // var_28 = wp::atomic_add(var_clamped_defaults, var_26, var_27);
                // default_joint_pos[i, j] = wp.clamp(                                            <L 248>
                // default_joint_pos[i, j],                                                       <L 249>
                var_29 = wp::address(var_default_joint_pos, var_0, var_1);
                // joint_pos_limits_lower[i, j],                                                  <L 250>
                var_30 = wp::address(var_joint_pos_limits_lower, var_0, var_1);
                // joint_pos_limits_upper[i, j],                                                  <L 251>
                var_31 = wp::address(var_joint_pos_limits_upper, var_0, var_1);
                var_33 = wp::load(var_29);
                var_34 = wp::load(var_30);
                var_35 = wp::load(var_31);
                var_32 = wp::clamp(var_33, var_34, var_35);
                // default_joint_pos[i, j] = wp.clamp(                                            <L 248>
                // wp::array_store(var_default_joint_pos, var_0, var_1, var_32);
            }
            // soft_joint_pos_limits[i, j] = compute_soft_joint_pos_limits_func(                  <L 253>
            // wp.vec2f(joint_pos_limits_lower[i, j], joint_pos_limits_upper[i, j]), soft_limit_factor       <L 254>
            var_36 = wp::address(var_joint_pos_limits_lower, var_0, var_1);
            var_37 = wp::address(var_joint_pos_limits_upper, var_0, var_1);
            var_39 = wp::load(var_36);
            var_40 = wp::load(var_37);
            var_38 = wp::vec_t<2, wp::float32>(var_39, var_40);
            var_41 = compute_soft_joint_pos_limits_func_0(var_38, var_soft_limit_factor);
            // soft_joint_pos_limits[i, j] = compute_soft_joint_pos_limits_func(                  <L 253>
            // wp::array_store(var_soft_joint_pos_limits, var_0, var_1, var_41);
        }
        //---------
        // reverse
        if (var_2) {
            wp::adj_array_store(var_soft_joint_pos_limits, var_0, var_1, var_41, adj_soft_joint_pos_limits, adj_0, adj_1, adj_41);
            // adj: soft_joint_pos_limits[i, j] = compute_soft_joint_pos_limits_func(             <L 253>
            adj_compute_soft_joint_pos_limits_func_0(var_38, var_soft_limit_factor, adj_38, adj_soft_limit_factor, adj_41);
            wp::adj_vec_t(var_39, var_40, adj_36, adj_37, adj_38);
            wp::adj_address(var_joint_pos_limits_upper, var_0, var_1, adj_joint_pos_limits_upper, adj_0, adj_1, adj_37);
            wp::adj_address(var_joint_pos_limits_lower, var_0, var_1, adj_joint_pos_limits_lower, adj_0, adj_1, adj_36);
            // adj: wp.vec2f(joint_pos_limits_lower[i, j], joint_pos_limits_upper[i, j]), soft_limit_factor  <L 254>
            // adj: soft_joint_pos_limits[i, j] = compute_soft_joint_pos_limits_func(             <L 253>
            if (var_15) {
                wp::adj_array_store(var_default_joint_pos, var_0, var_1, var_32, adj_default_joint_pos, adj_0, adj_1, adj_32);
                // adj: default_joint_pos[i, j] = wp.clamp(                                       <L 248>
                wp::adj_clamp(var_33, var_34, var_35, adj_29, adj_30, adj_31, adj_32);
                wp::adj_address(var_joint_pos_limits_upper, var_0, var_1, adj_joint_pos_limits_upper, adj_0, adj_1, adj_31);
                // adj: joint_pos_limits_upper[i, j],                                             <L 251>
                wp::adj_address(var_joint_pos_limits_lower, var_0, var_1, adj_joint_pos_limits_lower, adj_0, adj_1, adj_30);
                // adj: joint_pos_limits_lower[i, j],                                             <L 250>
                wp::adj_address(var_default_joint_pos, var_0, var_1, adj_default_joint_pos, adj_0, adj_1, adj_29);
                // adj: default_joint_pos[i, j],                                                  <L 249>
                // adj: default_joint_pos[i, j] = wp.clamp(                                       <L 248>
                wp::adj_atomic_add(var_clamped_defaults, var_26, var_27, adj_clamped_defaults, adj_26, adj_27, adj_28);
                // adj: wp.atomic_add(clamped_defaults, 0, 1)                                     <L 247>
            }
            if (!var_15) {
                wp::adj_address(var_joint_pos_limits_upper, var_0, var_1, adj_joint_pos_limits_upper, adj_0, adj_1, adj_22);
                // adj: i, j                                                                      <L 245>
                wp::adj_address(var_default_joint_pos, var_0, var_1, adj_default_joint_pos, adj_0, adj_1, adj_21);
            }
            wp::adj_address(var_joint_pos_limits_lower, var_0, var_1, adj_joint_pos_limits_lower, adj_0, adj_1, adj_17);
            wp::adj_address(var_default_joint_pos, var_0, var_1, adj_default_joint_pos, adj_0, adj_1, adj_16);
            // adj: if (default_joint_pos[i, j] < joint_pos_limits_lower[i, j]) or default_joint_pos[i, j] > joint_pos_limits_upper[  <L 244>
            wp::adj_array_store(var_joint_pos_limits_upper, var_0, var_1, var_13, adj_joint_pos_limits_upper, adj_0, adj_1, adj_13);
            wp::adj_extract(var_14, var_12, adj_11, adj_12, adj_13);
            wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_11);
            // adj: joint_pos_limits_upper[i, j] = in_data[i, j][1]                               <L 243>
            wp::adj_array_store(var_joint_pos_limits_lower, var_0, var_1, var_9, adj_joint_pos_limits_lower, adj_0, adj_1, adj_9);
            wp::adj_extract(var_10, var_8, adj_7, adj_8, adj_9);
            wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_7);
            // adj: joint_pos_limits_lower[i, j] = in_data[i, j][0]                               <L 242>
        }
        if (var_2) {
            wp::adj_address(var_joint_mask, var_1, adj_joint_mask, adj_1, adj_5);
        }
        wp::adj_address(var_env_mask, var_0, adj_env_mask, adj_0, adj_3);
        // adj: if env_mask[i] and joint_mask[j]:                                                 <L 241>
        // adj: i, j = wp.tid()                                                                   <L 240>
        // adj: def write_joint_limit_data_to_buffer_mask(                                        <L 206>
        continue;
    }
}



extern "C" __global__ void gather_dof_force_rows_04dc846d_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_src,
    wp::array_t<wp::int32> var_art_ids,
    wp::array_t<wp::int32> var_articulation_start,
    wp::array_t<wp::int32> var_joint_qd_start,
    wp::array_t<wp::int32> var_joint_user_to_backend,
    wp::int32 var_num_base_dofs,
    bool var_has_joint_ordering,
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
        bool var_3;
        bool var_4;
        wp::int32 var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        //---------
        // forward
        // def gather_dof_force_rows(                                                             <L 787>
        // i, user_dof = wp.tid()                                                                 <L 829>
        builtin_tid2d(var_0, var_1);
        // backend_dof = user_dof                                                                 <L 830>
        var_2 = wp::copy(var_1);
        // if has_joint_ordering and user_dof >= num_base_dofs:                                   <L 831>
        var_3 = var_has_joint_ordering;
        if (var_3) {
            var_4 = (var_1 >= var_num_base_dofs);
            var_3 = var_3 && var_4;
        }
        if (var_3) {
            // backend_dof = num_base_dofs + joint_user_to_backend[user_dof - num_base_dofs]       <L 832>
            var_5 = wp::sub(var_1, var_num_base_dofs);
            var_6 = wp::address(var_joint_user_to_backend, var_5);
            var_8 = wp::load(var_6);
            var_7 = wp::add(var_num_base_dofs, var_8);
        }
        var_9 = wp::where(var_3, var_7, var_2);
        // dof_start = joint_qd_start[articulation_start[art_ids[i]]]                             <L 834>
        var_10 = wp::address(var_art_ids, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::address(var_articulation_start, var_12);
        var_14 = wp::load(var_11);
        var_13 = wp::address(var_joint_qd_start, var_14);
        var_16 = wp::load(var_13);
        var_15 = wp::copy(var_16);
        // dst[i, user_dof] = src[dof_start + backend_dof]                                        <L 835>
        var_17 = wp::add(var_15, var_9);
        var_18 = wp::address(var_src, var_17);
        var_19 = wp::load(var_18);
        wp::array_store(var_dst, var_0, var_1, var_19);
    }
}



extern "C" __global__ void gather_dof_force_rows_04dc846d_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_src,
    wp::array_t<wp::int32> var_art_ids,
    wp::array_t<wp::int32> var_articulation_start,
    wp::array_t<wp::int32> var_joint_qd_start,
    wp::array_t<wp::int32> var_joint_user_to_backend,
    wp::int32 var_num_base_dofs,
    bool var_has_joint_ordering,
    wp::array_t<wp::float32> var_dst,
    wp::array_t<wp::float32> adj_src,
    wp::array_t<wp::int32> adj_art_ids,
    wp::array_t<wp::int32> adj_articulation_start,
    wp::array_t<wp::int32> adj_joint_qd_start,
    wp::array_t<wp::int32> adj_joint_user_to_backend,
    wp::int32 adj_num_base_dofs,
    bool adj_has_joint_ordering,
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
        bool var_3;
        bool var_4;
        wp::int32 var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
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
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        //---------
        // forward
        // def gather_dof_force_rows(                                                             <L 787>
        // i, user_dof = wp.tid()                                                                 <L 829>
        builtin_tid2d(var_0, var_1);
        // backend_dof = user_dof                                                                 <L 830>
        var_2 = wp::copy(var_1);
        // if has_joint_ordering and user_dof >= num_base_dofs:                                   <L 831>
        var_3 = var_has_joint_ordering;
        if (var_3) {
            var_4 = (var_1 >= var_num_base_dofs);
            var_3 = var_3 && var_4;
        }
        if (var_3) {
            // backend_dof = num_base_dofs + joint_user_to_backend[user_dof - num_base_dofs]       <L 832>
            var_5 = wp::sub(var_1, var_num_base_dofs);
            var_6 = wp::address(var_joint_user_to_backend, var_5);
            var_8 = wp::load(var_6);
            var_7 = wp::add(var_num_base_dofs, var_8);
        }
        var_9 = wp::where(var_3, var_7, var_2);
        // dof_start = joint_qd_start[articulation_start[art_ids[i]]]                             <L 834>
        var_10 = wp::address(var_art_ids, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::address(var_articulation_start, var_12);
        var_14 = wp::load(var_11);
        var_13 = wp::address(var_joint_qd_start, var_14);
        var_16 = wp::load(var_13);
        var_15 = wp::copy(var_16);
        // dst[i, user_dof] = src[dof_start + backend_dof]                                        <L 835>
        var_17 = wp::add(var_15, var_9);
        var_18 = wp::address(var_src, var_17);
        var_19 = wp::load(var_18);
        // wp::array_store(var_dst, var_0, var_1, var_19);
        //---------
        // reverse
        wp::adj_array_store(var_dst, var_0, var_1, var_19, adj_dst, adj_0, adj_1, adj_18);
        wp::adj_address(var_src, var_17, adj_src, adj_17, adj_18);
        wp::adj_add(var_15, var_9, adj_15, adj_9, adj_17);
        // adj: dst[i, user_dof] = src[dof_start + backend_dof]                                   <L 835>
        wp::adj_copy(var_16, adj_13, adj_15);
        wp::adj_address(var_joint_qd_start, var_14, adj_joint_qd_start, adj_11, adj_13);
        wp::adj_address(var_articulation_start, var_12, adj_articulation_start, adj_10, adj_11);
        wp::adj_address(var_art_ids, var_0, adj_art_ids, adj_0, adj_10);
        // adj: dof_start = joint_qd_start[articulation_start[art_ids[i]]]                        <L 834>
        wp::adj_where(var_3, var_7, var_2, adj_3, adj_7, adj_2, adj_9);
        if (var_3) {
            wp::adj_add(var_num_base_dofs, var_8, adj_num_base_dofs, adj_6, adj_7);
            wp::adj_address(var_joint_user_to_backend, var_5, adj_joint_user_to_backend, adj_5, adj_6);
            wp::adj_sub(var_1, var_num_base_dofs, adj_1, adj_num_base_dofs, adj_5);
            // adj: backend_dof = num_base_dofs + joint_user_to_backend[user_dof - num_base_dofs]  <L 832>
        }
        if (var_3) {
        }
        // adj: if has_joint_ordering and user_dof >= num_base_dofs:                              <L 831>
        wp::adj_copy(var_1, adj_1, adj_2);
        // adj: backend_dof = user_dof                                                            <L 830>
        // adj: i, user_dof = wp.tid()                                                            <L 829>
        // adj: def gather_dof_force_rows(                                                        <L 787>
        continue;
    }
}



extern "C" __global__ void float_data_to_buffer_with_indices_05f84586_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_in_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::array_t<wp::float32> var_out_data)
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
        //---------
        // forward
        // def float_data_to_buffer_with_indices(                                                 <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 19>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 20>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // out_data[env_id, joint_id] = in_data                                                   <L 21>
        wp::array_store(var_out_data, var_3, var_6, var_in_data);
    }
}



extern "C" __global__ void float_data_to_buffer_with_indices_05f84586_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_in_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::array_t<wp::float32> var_out_data,
    wp::float32 adj_in_data,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int32> adj_joint_ids,
    wp::array_t<wp::float32> adj_out_data)
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
        //---------
        // forward
        // def float_data_to_buffer_with_indices(                                                 <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 19>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 20>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // out_data[env_id, joint_id] = in_data                                                   <L 21>
        // wp::array_store(var_out_data, var_3, var_6, var_in_data);
        //---------
        // reverse
        wp::adj_array_store(var_out_data, var_3, var_6, var_in_data, adj_out_data, adj_3, adj_6, adj_in_data);
        // adj: out_data[env_id, joint_id] = in_data                                              <L 21>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 20>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 19>
        // adj: i, j = wp.tid()                                                                   <L 18>
        // adj: def float_data_to_buffer_with_indices(                                            <L 1>
        continue;
    }
}



extern "C" __global__ void write_joint_friction_param_to_buffer_080bc55a_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    wp::int32 var_buffer_index,
    bool var_from_mask,
    wp::array_t<wp::float32> var_out_data,
    wp::array_t<wp::float32> var_out_buffer)
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
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        //---------
        // forward
        // def write_joint_friction_param_to_buffer(                                              <L 1>
        // i, j = wp.tid()                                                                        <L 31>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 32>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 33>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 34>
        if (var_from_mask) {
            // out_data[env_id, joint_id] = in_data[env_id, joint_id]                             <L 35>
            var_8 = wp::address(var_in_data, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_out_data, var_3, var_6, var_9);
            // out_buffer[env_id, joint_id, buffer_index] = in_data[env_id, joint_id]             <L 36>
            var_10 = wp::address(var_in_data, var_3, var_6);
            var_11 = wp::load(var_10);
            wp::array_store(var_out_buffer, var_3, var_6, var_buffer_index, var_11);
        }
        if (!var_from_mask) {
            // out_data[env_id, joint_id] = in_data[i, j]                                         <L 38>
            var_12 = wp::address(var_in_data, var_0, var_1);
            var_13 = wp::load(var_12);
            wp::array_store(var_out_data, var_3, var_6, var_13);
            // out_buffer[env_id, joint_id, buffer_index] = in_data[i, j]                         <L 39>
            var_14 = wp::address(var_in_data, var_0, var_1);
            var_15 = wp::load(var_14);
            wp::array_store(var_out_buffer, var_3, var_6, var_buffer_index, var_15);
        }
    }
}



extern "C" __global__ void write_joint_friction_param_to_buffer_080bc55a_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    wp::int32 var_buffer_index,
    bool var_from_mask,
    wp::array_t<wp::float32> var_out_data,
    wp::array_t<wp::float32> var_out_buffer,
    wp::array_t<wp::float32> adj_in_data,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int64> adj_joint_ids,
    wp::int32 adj_buffer_index,
    bool adj_from_mask,
    wp::array_t<wp::float32> adj_out_data,
    wp::array_t<wp::float32> adj_out_buffer)
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
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
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
        //---------
        // forward
        // def write_joint_friction_param_to_buffer(                                              <L 1>
        // i, j = wp.tid()                                                                        <L 31>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 32>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 33>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 34>
        if (var_from_mask) {
            // out_data[env_id, joint_id] = in_data[env_id, joint_id]                             <L 35>
            var_8 = wp::address(var_in_data, var_3, var_6);
            var_9 = wp::load(var_8);
            // wp::array_store(var_out_data, var_3, var_6, var_9);
            // out_buffer[env_id, joint_id, buffer_index] = in_data[env_id, joint_id]             <L 36>
            var_10 = wp::address(var_in_data, var_3, var_6);
            var_11 = wp::load(var_10);
            // wp::array_store(var_out_buffer, var_3, var_6, var_buffer_index, var_11);
        }
        if (!var_from_mask) {
            // out_data[env_id, joint_id] = in_data[i, j]                                         <L 38>
            var_12 = wp::address(var_in_data, var_0, var_1);
            var_13 = wp::load(var_12);
            // wp::array_store(var_out_data, var_3, var_6, var_13);
            // out_buffer[env_id, joint_id, buffer_index] = in_data[i, j]                         <L 39>
            var_14 = wp::address(var_in_data, var_0, var_1);
            var_15 = wp::load(var_14);
            // wp::array_store(var_out_buffer, var_3, var_6, var_buffer_index, var_15);
        }
        //---------
        // reverse
        if (!var_from_mask) {
            wp::adj_array_store(var_out_buffer, var_3, var_6, var_buffer_index, var_15, adj_out_buffer, adj_3, adj_6, adj_buffer_index, adj_14);
            wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_14);
            // adj: out_buffer[env_id, joint_id, buffer_index] = in_data[i, j]                    <L 39>
            wp::adj_array_store(var_out_data, var_3, var_6, var_13, adj_out_data, adj_3, adj_6, adj_12);
            wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_12);
            // adj: out_data[env_id, joint_id] = in_data[i, j]                                    <L 38>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_out_buffer, var_3, var_6, var_buffer_index, var_11, adj_out_buffer, adj_3, adj_6, adj_buffer_index, adj_10);
            wp::adj_address(var_in_data, var_3, var_6, adj_in_data, adj_3, adj_6, adj_10);
            // adj: out_buffer[env_id, joint_id, buffer_index] = in_data[env_id, joint_id]        <L 36>
            wp::adj_array_store(var_out_data, var_3, var_6, var_9, adj_out_data, adj_3, adj_6, adj_8);
            wp::adj_address(var_in_data, var_3, var_6, adj_in_data, adj_3, adj_6, adj_8);
            // adj: out_data[env_id, joint_id] = in_data[env_id, joint_id]                        <L 35>
        }
        // adj: if from_mask:                                                                     <L 34>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 33>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 32>
        // adj: i, j = wp.tid()                                                                   <L 31>
        // adj: def write_joint_friction_param_to_buffer(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void float_data_to_buffer_with_indices_08343000_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_in_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    wp::array_t<wp::float32> var_out_data)
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
        //---------
        // forward
        // def float_data_to_buffer_with_indices(                                                 <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 19>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 20>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // out_data[env_id, joint_id] = in_data                                                   <L 21>
        wp::array_store(var_out_data, var_3, var_6, var_in_data);
    }
}



extern "C" __global__ void float_data_to_buffer_with_indices_08343000_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_in_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    wp::array_t<wp::float32> var_out_data,
    wp::float32 adj_in_data,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int64> adj_joint_ids,
    wp::array_t<wp::float32> adj_out_data)
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
        //---------
        // forward
        // def float_data_to_buffer_with_indices(                                                 <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 19>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 20>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // out_data[env_id, joint_id] = in_data                                                   <L 21>
        // wp::array_store(var_out_data, var_3, var_6, var_in_data);
        //---------
        // reverse
        wp::adj_array_store(var_out_data, var_3, var_6, var_in_data, adj_out_data, adj_3, adj_6, adj_in_data);
        // adj: out_data[env_id, joint_id] = in_data                                              <L 21>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 20>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 19>
        // adj: i, j = wp.tid()                                                                   <L 18>
        // adj: def float_data_to_buffer_with_indices(                                            <L 1>
        continue;
    }
}



extern "C" __global__ void float_data_to_buffer_with_indices_11497a97_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::array_t<wp::float32> var_out_data)
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
        //---------
        // forward
        // def float_data_to_buffer_with_indices(                                                 <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 19>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 20>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // out_data[env_id, joint_id] = in_data                                                   <L 21>
        wp::array_store(var_out_data, var_3, var_6, var_in_data);
    }
}



extern "C" __global__ void float_data_to_buffer_with_indices_11497a97_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::array_t<wp::float32> var_out_data,
    wp::float32 adj_in_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_joint_ids,
    wp::array_t<wp::float32> adj_out_data)
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
        //---------
        // forward
        // def float_data_to_buffer_with_indices(                                                 <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 19>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 20>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // out_data[env_id, joint_id] = in_data                                                   <L 21>
        // wp::array_store(var_out_data, var_3, var_6, var_in_data);
        //---------
        // reverse
        wp::adj_array_store(var_out_data, var_3, var_6, var_in_data, adj_out_data, adj_3, adj_6, adj_in_data);
        // adj: out_data[env_id, joint_id] = in_data                                              <L 21>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 20>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 19>
        // adj: i, j = wp.tid()                                                                   <L 18>
        // adj: def float_data_to_buffer_with_indices(                                            <L 1>
        continue;
    }
}



extern "C" __global__ void write_joint_friction_param_to_buffer_1454deaf_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::int32 var_buffer_index,
    bool var_from_mask,
    wp::array_t<wp::float32> var_out_data,
    wp::array_t<wp::float32> var_out_buffer)
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
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        //---------
        // forward
        // def write_joint_friction_param_to_buffer(                                              <L 1>
        // i, j = wp.tid()                                                                        <L 31>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 32>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 33>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 34>
        if (var_from_mask) {
            // out_data[env_id, joint_id] = in_data[env_id, joint_id]                             <L 35>
            var_8 = wp::address(var_in_data, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_out_data, var_3, var_6, var_9);
            // out_buffer[env_id, joint_id, buffer_index] = in_data[env_id, joint_id]             <L 36>
            var_10 = wp::address(var_in_data, var_3, var_6);
            var_11 = wp::load(var_10);
            wp::array_store(var_out_buffer, var_3, var_6, var_buffer_index, var_11);
        }
        if (!var_from_mask) {
            // out_data[env_id, joint_id] = in_data[i, j]                                         <L 38>
            var_12 = wp::address(var_in_data, var_0, var_1);
            var_13 = wp::load(var_12);
            wp::array_store(var_out_data, var_3, var_6, var_13);
            // out_buffer[env_id, joint_id, buffer_index] = in_data[i, j]                         <L 39>
            var_14 = wp::address(var_in_data, var_0, var_1);
            var_15 = wp::load(var_14);
            wp::array_store(var_out_buffer, var_3, var_6, var_buffer_index, var_15);
        }
    }
}



extern "C" __global__ void write_joint_friction_param_to_buffer_1454deaf_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::int32 var_buffer_index,
    bool var_from_mask,
    wp::array_t<wp::float32> var_out_data,
    wp::array_t<wp::float32> var_out_buffer,
    wp::array_t<wp::float32> adj_in_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_joint_ids,
    wp::int32 adj_buffer_index,
    bool adj_from_mask,
    wp::array_t<wp::float32> adj_out_data,
    wp::array_t<wp::float32> adj_out_buffer)
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
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
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
        //---------
        // forward
        // def write_joint_friction_param_to_buffer(                                              <L 1>
        // i, j = wp.tid()                                                                        <L 31>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 32>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 33>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 34>
        if (var_from_mask) {
            // out_data[env_id, joint_id] = in_data[env_id, joint_id]                             <L 35>
            var_8 = wp::address(var_in_data, var_3, var_6);
            var_9 = wp::load(var_8);
            // wp::array_store(var_out_data, var_3, var_6, var_9);
            // out_buffer[env_id, joint_id, buffer_index] = in_data[env_id, joint_id]             <L 36>
            var_10 = wp::address(var_in_data, var_3, var_6);
            var_11 = wp::load(var_10);
            // wp::array_store(var_out_buffer, var_3, var_6, var_buffer_index, var_11);
        }
        if (!var_from_mask) {
            // out_data[env_id, joint_id] = in_data[i, j]                                         <L 38>
            var_12 = wp::address(var_in_data, var_0, var_1);
            var_13 = wp::load(var_12);
            // wp::array_store(var_out_data, var_3, var_6, var_13);
            // out_buffer[env_id, joint_id, buffer_index] = in_data[i, j]                         <L 39>
            var_14 = wp::address(var_in_data, var_0, var_1);
            var_15 = wp::load(var_14);
            // wp::array_store(var_out_buffer, var_3, var_6, var_buffer_index, var_15);
        }
        //---------
        // reverse
        if (!var_from_mask) {
            wp::adj_array_store(var_out_buffer, var_3, var_6, var_buffer_index, var_15, adj_out_buffer, adj_3, adj_6, adj_buffer_index, adj_14);
            wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_14);
            // adj: out_buffer[env_id, joint_id, buffer_index] = in_data[i, j]                    <L 39>
            wp::adj_array_store(var_out_data, var_3, var_6, var_13, adj_out_data, adj_3, adj_6, adj_12);
            wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_12);
            // adj: out_data[env_id, joint_id] = in_data[i, j]                                    <L 38>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_out_buffer, var_3, var_6, var_buffer_index, var_11, adj_out_buffer, adj_3, adj_6, adj_buffer_index, adj_10);
            wp::adj_address(var_in_data, var_3, var_6, adj_in_data, adj_3, adj_6, adj_10);
            // adj: out_buffer[env_id, joint_id, buffer_index] = in_data[env_id, joint_id]        <L 36>
            wp::adj_array_store(var_out_data, var_3, var_6, var_9, adj_out_data, adj_3, adj_6, adj_8);
            wp::adj_address(var_in_data, var_3, var_6, adj_in_data, adj_3, adj_6, adj_8);
            // adj: out_data[env_id, joint_id] = in_data[env_id, joint_id]                        <L 35>
        }
        // adj: if from_mask:                                                                     <L 34>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 33>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 32>
        // adj: i, j = wp.tid()                                                                   <L 31>
        // adj: def write_joint_friction_param_to_buffer(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void write_joint_state_data_mask_14c28fd4_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_pos_data,
    wp::array_t<wp::float32> var_vel_data,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_joint_mask,
    wp::array_t<wp::float32> var_joint_pos,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::array_t<wp::float32> var_joint_acc)
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
        wp::float32* var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        const wp::float32 var_13 = 0.0;
        //---------
        // forward
        // def write_joint_state_data_mask(                                                       <L 1040>
        // i, j = wp.tid()                                                                        <L 1065>
        builtin_tid2d(var_0, var_1);
        // if env_mask[i] and joint_mask[j]:                                                      <L 1066>
        var_3 = wp::address(var_env_mask, var_0);
        var_4 = wp::load(var_3);
        var_2 = var_4;
        if (var_2) {
            var_5 = wp::address(var_joint_mask, var_1);
            var_6 = wp::load(var_5);
            var_2 = var_2 && var_6;
        }
        if (var_2) {
            // joint_pos[i, j] = pos_data[i, j]                                                   <L 1067>
            var_7 = wp::address(var_pos_data, var_0, var_1);
            var_8 = wp::load(var_7);
            wp::array_store(var_joint_pos, var_0, var_1, var_8);
            // joint_vel[i, j] = vel_data[i, j]                                                   <L 1068>
            var_9 = wp::address(var_vel_data, var_0, var_1);
            var_10 = wp::load(var_9);
            wp::array_store(var_joint_vel, var_0, var_1, var_10);
            // prev_joint_vel[i, j] = vel_data[i, j]                                              <L 1069>
            var_11 = wp::address(var_vel_data, var_0, var_1);
            var_12 = wp::load(var_11);
            wp::array_store(var_prev_joint_vel, var_0, var_1, var_12);
            // joint_acc[i, j] = 0.0                                                              <L 1070>
            wp::array_store(var_joint_acc, var_0, var_1, var_13);
        }
    }
}



extern "C" __global__ void write_joint_state_data_mask_14c28fd4_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_pos_data,
    wp::array_t<wp::float32> var_vel_data,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_joint_mask,
    wp::array_t<wp::float32> var_joint_pos,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::array_t<wp::float32> var_joint_acc,
    wp::array_t<wp::float32> adj_pos_data,
    wp::array_t<wp::float32> adj_vel_data,
    wp::array_t<bool> adj_env_mask,
    wp::array_t<bool> adj_joint_mask,
    wp::array_t<wp::float32> adj_joint_pos,
    wp::array_t<wp::float32> adj_joint_vel,
    wp::array_t<wp::float32> adj_prev_joint_vel,
    wp::array_t<wp::float32> adj_joint_acc)
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
        wp::float32* var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        const wp::float32 var_13 = 0.0;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        bool adj_6 = {};
        wp::float32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        //---------
        // forward
        // def write_joint_state_data_mask(                                                       <L 1040>
        // i, j = wp.tid()                                                                        <L 1065>
        builtin_tid2d(var_0, var_1);
        // if env_mask[i] and joint_mask[j]:                                                      <L 1066>
        var_3 = wp::address(var_env_mask, var_0);
        var_4 = wp::load(var_3);
        var_2 = var_4;
        if (var_2) {
            var_5 = wp::address(var_joint_mask, var_1);
            var_6 = wp::load(var_5);
            var_2 = var_2 && var_6;
        }
        if (var_2) {
            // joint_pos[i, j] = pos_data[i, j]                                                   <L 1067>
            var_7 = wp::address(var_pos_data, var_0, var_1);
            var_8 = wp::load(var_7);
            // wp::array_store(var_joint_pos, var_0, var_1, var_8);
            // joint_vel[i, j] = vel_data[i, j]                                                   <L 1068>
            var_9 = wp::address(var_vel_data, var_0, var_1);
            var_10 = wp::load(var_9);
            // wp::array_store(var_joint_vel, var_0, var_1, var_10);
            // prev_joint_vel[i, j] = vel_data[i, j]                                              <L 1069>
            var_11 = wp::address(var_vel_data, var_0, var_1);
            var_12 = wp::load(var_11);
            // wp::array_store(var_prev_joint_vel, var_0, var_1, var_12);
            // joint_acc[i, j] = 0.0                                                              <L 1070>
            // wp::array_store(var_joint_acc, var_0, var_1, var_13);
        }
        //---------
        // reverse
        if (var_2) {
            wp::adj_array_store(var_joint_acc, var_0, var_1, var_13, adj_joint_acc, adj_0, adj_1, adj_13);
            // adj: joint_acc[i, j] = 0.0                                                         <L 1070>
            wp::adj_array_store(var_prev_joint_vel, var_0, var_1, var_12, adj_prev_joint_vel, adj_0, adj_1, adj_11);
            wp::adj_address(var_vel_data, var_0, var_1, adj_vel_data, adj_0, adj_1, adj_11);
            // adj: prev_joint_vel[i, j] = vel_data[i, j]                                         <L 1069>
            wp::adj_array_store(var_joint_vel, var_0, var_1, var_10, adj_joint_vel, adj_0, adj_1, adj_9);
            wp::adj_address(var_vel_data, var_0, var_1, adj_vel_data, adj_0, adj_1, adj_9);
            // adj: joint_vel[i, j] = vel_data[i, j]                                              <L 1068>
            wp::adj_array_store(var_joint_pos, var_0, var_1, var_8, adj_joint_pos, adj_0, adj_1, adj_7);
            wp::adj_address(var_pos_data, var_0, var_1, adj_pos_data, adj_0, adj_1, adj_7);
            // adj: joint_pos[i, j] = pos_data[i, j]                                              <L 1067>
        }
        if (var_2) {
            wp::adj_address(var_joint_mask, var_1, adj_joint_mask, adj_1, adj_5);
        }
        wp::adj_address(var_env_mask, var_0, adj_env_mask, adj_0, adj_3);
        // adj: if env_mask[i] and joint_mask[j]:                                                 <L 1066>
        // adj: i, j = wp.tid()                                                                   <L 1065>
        // adj: def write_joint_state_data_mask(                                                  <L 1040>
        continue;
    }
}



extern "C" __global__ void write_joint_vel_data_index_1676bd68_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::array_t<wp::float32> var_joint_acc)
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
        wp::float32* var_10;
        wp::float32 var_11;
        const wp::float32 var_12 = 0.0;
        //---------
        // forward
        // def write_joint_vel_data_index(                                                        <L 1>
        // i, j = wp.tid()                                                                        <L 27>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 28>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 29>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // joint_vel[env_id, joint_id] = in_data[i, j]                                            <L 30>
        var_8 = wp::address(var_in_data, var_0, var_1);
        var_9 = wp::load(var_8);
        wp::array_store(var_joint_vel, var_3, var_6, var_9);
        // prev_joint_vel[env_id, joint_id] = in_data[i, j]                                       <L 31>
        var_10 = wp::address(var_in_data, var_0, var_1);
        var_11 = wp::load(var_10);
        wp::array_store(var_prev_joint_vel, var_3, var_6, var_11);
        // joint_acc[env_id, joint_id] = 0.0                                                      <L 32>
        wp::array_store(var_joint_acc, var_3, var_6, var_12);
    }
}



extern "C" __global__ void write_joint_vel_data_index_1676bd68_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::array_t<wp::float32> var_joint_acc,
    wp::array_t<wp::float32> adj_in_data,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int32> adj_joint_ids,
    wp::array_t<wp::float32> adj_joint_vel,
    wp::array_t<wp::float32> adj_prev_joint_vel,
    wp::array_t<wp::float32> adj_joint_acc)
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
        wp::float32* var_10;
        wp::float32 var_11;
        const wp::float32 var_12 = 0.0;
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
        //---------
        // forward
        // def write_joint_vel_data_index(                                                        <L 1>
        // i, j = wp.tid()                                                                        <L 27>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 28>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 29>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // joint_vel[env_id, joint_id] = in_data[i, j]                                            <L 30>
        var_8 = wp::address(var_in_data, var_0, var_1);
        var_9 = wp::load(var_8);
        // wp::array_store(var_joint_vel, var_3, var_6, var_9);
        // prev_joint_vel[env_id, joint_id] = in_data[i, j]                                       <L 31>
        var_10 = wp::address(var_in_data, var_0, var_1);
        var_11 = wp::load(var_10);
        // wp::array_store(var_prev_joint_vel, var_3, var_6, var_11);
        // joint_acc[env_id, joint_id] = 0.0                                                      <L 32>
        // wp::array_store(var_joint_acc, var_3, var_6, var_12);
        //---------
        // reverse
        wp::adj_array_store(var_joint_acc, var_3, var_6, var_12, adj_joint_acc, adj_3, adj_6, adj_12);
        // adj: joint_acc[env_id, joint_id] = 0.0                                                 <L 32>
        wp::adj_array_store(var_prev_joint_vel, var_3, var_6, var_11, adj_prev_joint_vel, adj_3, adj_6, adj_10);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_10);
        // adj: prev_joint_vel[env_id, joint_id] = in_data[i, j]                                  <L 31>
        wp::adj_array_store(var_joint_vel, var_3, var_6, var_9, adj_joint_vel, adj_3, adj_6, adj_8);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_8);
        // adj: joint_vel[env_id, joint_id] = in_data[i, j]                                       <L 30>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 29>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 28>
        // adj: i, j = wp.tid()                                                                   <L 27>
        // adj: def write_joint_vel_data_index(                                                   <L 1>
        continue;
    }
}



extern "C" __global__ void write_joint_limit_data_to_user_and_backend_index_195f3dc9_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_in_data,
    wp::float32 var_soft_limit_factor,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_user_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_user_joint_pos_limits,
    wp::array_t<wp::float32> var_backend_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_backend_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_soft_joint_pos_limits,
    wp::array_t<wp::float32> var_default_joint_pos,
    wp::array_t<wp::int32> var_clamped_defaults)
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
        wp::int32 var_8;
        wp::vec_t<2, wp::float32>* var_9;
        wp::vec_t<2, wp::float32> var_10;
        wp::vec_t<2, wp::float32> var_11;
        const wp::int32 var_12 = 0;
        wp::float32 var_13;
        const wp::int32 var_14 = 1;
        wp::float32 var_15;
        bool var_16;
        wp::float32* var_17;
        bool var_18;
        wp::float32 var_19;
        wp::float32* var_20;
        bool var_21;
        wp::float32 var_22;
        const wp::int32 var_23 = 0;
        const wp::int32 var_24 = 1;
        wp::int32 var_25;
        wp::float32* var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::vec_t<2, wp::float32> var_29;
        //---------
        // forward
        // def write_joint_limit_data_to_user_and_backend_index(                                  <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 19>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 20>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // backend_id = resolve_backend_index(user_id, user_to_backend, has_ordering)             <L 21>
        var_8 = resolve_backend_index_0(var_6, var_user_to_backend, var_has_ordering);
        // limits = in_data[i, j]                                                                 <L 22>
        var_9 = wp::address(var_in_data, var_0, var_1);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // lower = limits[0]                                                                      <L 23>
        var_13 = wp::extract(var_10, var_12);
        // upper = limits[1]                                                                      <L 24>
        var_15 = wp::extract(var_10, var_14);
        // user_joint_pos_limits_lower[env_id, user_id] = lower                                   <L 26>
        wp::array_store(var_user_joint_pos_limits_lower, var_3, var_6, var_13);
        // user_joint_pos_limits_upper[env_id, user_id] = upper                                   <L 27>
        wp::array_store(var_user_joint_pos_limits_upper, var_3, var_6, var_15);
        // user_joint_pos_limits[env_id, user_id] = limits                                        <L 28>
        wp::array_store(var_user_joint_pos_limits, var_3, var_6, var_10);
        // backend_joint_pos_limits_lower[env_id, backend_id] = lower                             <L 29>
        wp::array_store(var_backend_joint_pos_limits_lower, var_3, var_8, var_13);
        // backend_joint_pos_limits_upper[env_id, backend_id] = upper                             <L 30>
        wp::array_store(var_backend_joint_pos_limits_upper, var_3, var_8, var_15);
        // if (default_joint_pos[env_id, user_id] < lower) or default_joint_pos[env_id, user_id] > upper:       <L 32>
        var_17 = wp::address(var_default_joint_pos, var_3, var_6);
        var_19 = wp::load(var_17);
        var_18 = (var_19 < var_13);
        var_16 = var_18;
        if (!var_16) {
            var_20 = wp::address(var_default_joint_pos, var_3, var_6);
            var_22 = wp::load(var_20);
            var_21 = (var_22 > var_15);
            var_16 = var_16 || var_21;
        }
        if (var_16) {
            // wp.atomic_add(clamped_defaults, 0, 1)                                              <L 33>
            var_25 = wp::atomic_add(var_clamped_defaults, var_23, var_24);
            // default_joint_pos[env_id, user_id] = wp.clamp(default_joint_pos[env_id, user_id], lower, upper)       <L 34>
            var_26 = wp::address(var_default_joint_pos, var_3, var_6);
            var_28 = wp::load(var_26);
            var_27 = wp::clamp(var_28, var_13, var_15);
            wp::array_store(var_default_joint_pos, var_3, var_6, var_27);
        }
        // soft_joint_pos_limits[env_id, user_id] = compute_soft_joint_pos_limits_func(limits, soft_limit_factor)       <L 36>
        var_29 = compute_soft_joint_pos_limits_func_0(var_10, var_soft_limit_factor);
        wp::array_store(var_soft_joint_pos_limits, var_3, var_6, var_29);
    }
}



extern "C" __global__ void write_joint_limit_data_to_user_and_backend_index_195f3dc9_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_in_data,
    wp::float32 var_soft_limit_factor,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_user_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_user_joint_pos_limits,
    wp::array_t<wp::float32> var_backend_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_backend_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_soft_joint_pos_limits,
    wp::array_t<wp::float32> var_default_joint_pos,
    wp::array_t<wp::int32> var_clamped_defaults,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_in_data,
    wp::float32 adj_soft_limit_factor,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_user_ids,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    wp::array_t<wp::float32> adj_user_joint_pos_limits_lower,
    wp::array_t<wp::float32> adj_user_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_user_joint_pos_limits,
    wp::array_t<wp::float32> adj_backend_joint_pos_limits_lower,
    wp::array_t<wp::float32> adj_backend_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_soft_joint_pos_limits,
    wp::array_t<wp::float32> adj_default_joint_pos,
    wp::array_t<wp::int32> adj_clamped_defaults)
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
        wp::int32 var_8;
        wp::vec_t<2, wp::float32>* var_9;
        wp::vec_t<2, wp::float32> var_10;
        wp::vec_t<2, wp::float32> var_11;
        const wp::int32 var_12 = 0;
        wp::float32 var_13;
        const wp::int32 var_14 = 1;
        wp::float32 var_15;
        bool var_16;
        wp::float32* var_17;
        bool var_18;
        wp::float32 var_19;
        wp::float32* var_20;
        bool var_21;
        wp::float32 var_22;
        const wp::int32 var_23 = 0;
        const wp::int32 var_24 = 1;
        wp::int32 var_25;
        wp::float32* var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::vec_t<2, wp::float32> var_29;
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
        wp::vec_t<2, wp::float32> adj_9 = {};
        wp::vec_t<2, wp::float32> adj_10 = {};
        wp::vec_t<2, wp::float32> adj_11 = {};
        wp::int32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::float32 adj_15 = {};
        bool adj_16 = {};
        wp::float32 adj_17 = {};
        bool adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        bool adj_21 = {};
        wp::float32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::vec_t<2, wp::float32> adj_29 = {};
        //---------
        // forward
        // def write_joint_limit_data_to_user_and_backend_index(                                  <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 19>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 20>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // backend_id = resolve_backend_index(user_id, user_to_backend, has_ordering)             <L 21>
        var_8 = resolve_backend_index_0(var_6, var_user_to_backend, var_has_ordering);
        // limits = in_data[i, j]                                                                 <L 22>
        var_9 = wp::address(var_in_data, var_0, var_1);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // lower = limits[0]                                                                      <L 23>
        var_13 = wp::extract(var_10, var_12);
        // upper = limits[1]                                                                      <L 24>
        var_15 = wp::extract(var_10, var_14);
        // user_joint_pos_limits_lower[env_id, user_id] = lower                                   <L 26>
        // wp::array_store(var_user_joint_pos_limits_lower, var_3, var_6, var_13);
        // user_joint_pos_limits_upper[env_id, user_id] = upper                                   <L 27>
        // wp::array_store(var_user_joint_pos_limits_upper, var_3, var_6, var_15);
        // user_joint_pos_limits[env_id, user_id] = limits                                        <L 28>
        // wp::array_store(var_user_joint_pos_limits, var_3, var_6, var_10);
        // backend_joint_pos_limits_lower[env_id, backend_id] = lower                             <L 29>
        // wp::array_store(var_backend_joint_pos_limits_lower, var_3, var_8, var_13);
        // backend_joint_pos_limits_upper[env_id, backend_id] = upper                             <L 30>
        // wp::array_store(var_backend_joint_pos_limits_upper, var_3, var_8, var_15);
        // if (default_joint_pos[env_id, user_id] < lower) or default_joint_pos[env_id, user_id] > upper:       <L 32>
        var_17 = wp::address(var_default_joint_pos, var_3, var_6);
        var_19 = wp::load(var_17);
        var_18 = (var_19 < var_13);
        var_16 = var_18;
        if (!var_16) {
            var_20 = wp::address(var_default_joint_pos, var_3, var_6);
            var_22 = wp::load(var_20);
            var_21 = (var_22 > var_15);
            var_16 = var_16 || var_21;
        }
        if (var_16) {
            // wp.atomic_add(clamped_defaults, 0, 1)                                              <L 33>
            // var_25 = wp::atomic_add(var_clamped_defaults, var_23, var_24);
            // default_joint_pos[env_id, user_id] = wp.clamp(default_joint_pos[env_id, user_id], lower, upper)       <L 34>
            var_26 = wp::address(var_default_joint_pos, var_3, var_6);
            var_28 = wp::load(var_26);
            var_27 = wp::clamp(var_28, var_13, var_15);
            // wp::array_store(var_default_joint_pos, var_3, var_6, var_27);
        }
        // soft_joint_pos_limits[env_id, user_id] = compute_soft_joint_pos_limits_func(limits, soft_limit_factor)       <L 36>
        var_29 = compute_soft_joint_pos_limits_func_0(var_10, var_soft_limit_factor);
        // wp::array_store(var_soft_joint_pos_limits, var_3, var_6, var_29);
        //---------
        // reverse
        wp::adj_array_store(var_soft_joint_pos_limits, var_3, var_6, var_29, adj_soft_joint_pos_limits, adj_3, adj_6, adj_29);
        adj_compute_soft_joint_pos_limits_func_0(var_10, var_soft_limit_factor, adj_10, adj_soft_limit_factor, adj_29);
        // adj: soft_joint_pos_limits[env_id, user_id] = compute_soft_joint_pos_limits_func(limits, soft_limit_factor)  <L 36>
        if (var_16) {
            wp::adj_array_store(var_default_joint_pos, var_3, var_6, var_27, adj_default_joint_pos, adj_3, adj_6, adj_27);
            wp::adj_clamp(var_28, var_13, var_15, adj_26, adj_13, adj_15, adj_27);
            wp::adj_address(var_default_joint_pos, var_3, var_6, adj_default_joint_pos, adj_3, adj_6, adj_26);
            // adj: default_joint_pos[env_id, user_id] = wp.clamp(default_joint_pos[env_id, user_id], lower, upper)  <L 34>
            wp::adj_atomic_add(var_clamped_defaults, var_23, var_24, adj_clamped_defaults, adj_23, adj_24, adj_25);
            // adj: wp.atomic_add(clamped_defaults, 0, 1)                                         <L 33>
        }
        if (!var_16) {
            wp::adj_address(var_default_joint_pos, var_3, var_6, adj_default_joint_pos, adj_3, adj_6, adj_20);
        }
        wp::adj_address(var_default_joint_pos, var_3, var_6, adj_default_joint_pos, adj_3, adj_6, adj_17);
        // adj: if (default_joint_pos[env_id, user_id] < lower) or default_joint_pos[env_id, user_id] > upper:  <L 32>
        wp::adj_array_store(var_backend_joint_pos_limits_upper, var_3, var_8, var_15, adj_backend_joint_pos_limits_upper, adj_3, adj_8, adj_15);
        // adj: backend_joint_pos_limits_upper[env_id, backend_id] = upper                        <L 30>
        wp::adj_array_store(var_backend_joint_pos_limits_lower, var_3, var_8, var_13, adj_backend_joint_pos_limits_lower, adj_3, adj_8, adj_13);
        // adj: backend_joint_pos_limits_lower[env_id, backend_id] = lower                        <L 29>
        wp::adj_array_store(var_user_joint_pos_limits, var_3, var_6, var_10, adj_user_joint_pos_limits, adj_3, adj_6, adj_10);
        // adj: user_joint_pos_limits[env_id, user_id] = limits                                   <L 28>
        wp::adj_array_store(var_user_joint_pos_limits_upper, var_3, var_6, var_15, adj_user_joint_pos_limits_upper, adj_3, adj_6, adj_15);
        // adj: user_joint_pos_limits_upper[env_id, user_id] = upper                              <L 27>
        wp::adj_array_store(var_user_joint_pos_limits_lower, var_3, var_6, var_13, adj_user_joint_pos_limits_lower, adj_3, adj_6, adj_13);
        // adj: user_joint_pos_limits_lower[env_id, user_id] = lower                              <L 26>
        wp::adj_extract(var_10, var_14, adj_10, adj_14, adj_15);
        // adj: upper = limits[1]                                                                 <L 24>
        wp::adj_extract(var_10, var_12, adj_10, adj_12, adj_13);
        // adj: lower = limits[0]                                                                 <L 23>
        wp::adj_copy(var_11, adj_9, adj_10);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_9);
        // adj: limits = in_data[i, j]                                                            <L 22>
        adj_resolve_backend_index_0(var_6, var_user_to_backend, var_has_ordering, adj_6, adj_user_to_backend, adj_has_ordering, adj_8);
        // adj: backend_id = resolve_backend_index(user_id, user_to_backend, has_ordering)        <L 21>
        wp::adj_address(var_user_ids, var_1, adj_user_ids, adj_1, adj_5);
        // adj: user_id = wp.int32(user_ids[j])                                                   <L 20>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 19>
        // adj: i, j = wp.tid()                                                                   <L 18>
        // adj: def write_joint_limit_data_to_user_and_backend_index(                             <L 1>
        continue;
    }
}



extern "C" __global__ void write_joint_friction_param_to_buffer_1f013f0d_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::int32 var_buffer_index,
    bool var_from_mask,
    wp::array_t<wp::float32> var_out_data,
    wp::array_t<wp::float32> var_out_buffer)
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
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        //---------
        // forward
        // def write_joint_friction_param_to_buffer(                                              <L 1>
        // i, j = wp.tid()                                                                        <L 31>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 32>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 33>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 34>
        if (var_from_mask) {
            // out_data[env_id, joint_id] = in_data[env_id, joint_id]                             <L 35>
            var_8 = wp::address(var_in_data, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_out_data, var_3, var_6, var_9);
            // out_buffer[env_id, joint_id, buffer_index] = in_data[env_id, joint_id]             <L 36>
            var_10 = wp::address(var_in_data, var_3, var_6);
            var_11 = wp::load(var_10);
            wp::array_store(var_out_buffer, var_3, var_6, var_buffer_index, var_11);
        }
        if (!var_from_mask) {
            // out_data[env_id, joint_id] = in_data[i, j]                                         <L 38>
            var_12 = wp::address(var_in_data, var_0, var_1);
            var_13 = wp::load(var_12);
            wp::array_store(var_out_data, var_3, var_6, var_13);
            // out_buffer[env_id, joint_id, buffer_index] = in_data[i, j]                         <L 39>
            var_14 = wp::address(var_in_data, var_0, var_1);
            var_15 = wp::load(var_14);
            wp::array_store(var_out_buffer, var_3, var_6, var_buffer_index, var_15);
        }
    }
}



extern "C" __global__ void write_joint_friction_param_to_buffer_1f013f0d_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::int32 var_buffer_index,
    bool var_from_mask,
    wp::array_t<wp::float32> var_out_data,
    wp::array_t<wp::float32> var_out_buffer,
    wp::array_t<wp::float32> adj_in_data,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int32> adj_joint_ids,
    wp::int32 adj_buffer_index,
    bool adj_from_mask,
    wp::array_t<wp::float32> adj_out_data,
    wp::array_t<wp::float32> adj_out_buffer)
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
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
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
        //---------
        // forward
        // def write_joint_friction_param_to_buffer(                                              <L 1>
        // i, j = wp.tid()                                                                        <L 31>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 32>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 33>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 34>
        if (var_from_mask) {
            // out_data[env_id, joint_id] = in_data[env_id, joint_id]                             <L 35>
            var_8 = wp::address(var_in_data, var_3, var_6);
            var_9 = wp::load(var_8);
            // wp::array_store(var_out_data, var_3, var_6, var_9);
            // out_buffer[env_id, joint_id, buffer_index] = in_data[env_id, joint_id]             <L 36>
            var_10 = wp::address(var_in_data, var_3, var_6);
            var_11 = wp::load(var_10);
            // wp::array_store(var_out_buffer, var_3, var_6, var_buffer_index, var_11);
        }
        if (!var_from_mask) {
            // out_data[env_id, joint_id] = in_data[i, j]                                         <L 38>
            var_12 = wp::address(var_in_data, var_0, var_1);
            var_13 = wp::load(var_12);
            // wp::array_store(var_out_data, var_3, var_6, var_13);
            // out_buffer[env_id, joint_id, buffer_index] = in_data[i, j]                         <L 39>
            var_14 = wp::address(var_in_data, var_0, var_1);
            var_15 = wp::load(var_14);
            // wp::array_store(var_out_buffer, var_3, var_6, var_buffer_index, var_15);
        }
        //---------
        // reverse
        if (!var_from_mask) {
            wp::adj_array_store(var_out_buffer, var_3, var_6, var_buffer_index, var_15, adj_out_buffer, adj_3, adj_6, adj_buffer_index, adj_14);
            wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_14);
            // adj: out_buffer[env_id, joint_id, buffer_index] = in_data[i, j]                    <L 39>
            wp::adj_array_store(var_out_data, var_3, var_6, var_13, adj_out_data, adj_3, adj_6, adj_12);
            wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_12);
            // adj: out_data[env_id, joint_id] = in_data[i, j]                                    <L 38>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_out_buffer, var_3, var_6, var_buffer_index, var_11, adj_out_buffer, adj_3, adj_6, adj_buffer_index, adj_10);
            wp::adj_address(var_in_data, var_3, var_6, adj_in_data, adj_3, adj_6, adj_10);
            // adj: out_buffer[env_id, joint_id, buffer_index] = in_data[env_id, joint_id]        <L 36>
            wp::adj_array_store(var_out_data, var_3, var_6, var_9, adj_out_data, adj_3, adj_6, adj_8);
            wp::adj_address(var_in_data, var_3, var_6, adj_in_data, adj_3, adj_6, adj_8);
            // adj: out_data[env_id, joint_id] = in_data[env_id, joint_id]                        <L 35>
        }
        // adj: if from_mask:                                                                     <L 34>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 33>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 32>
        // adj: i, j = wp.tid()                                                                   <L 31>
        // adj: def write_joint_friction_param_to_buffer(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void write_joint_vel_data_index_235a34de_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::array_t<wp::float32> var_joint_acc)
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
        wp::float32* var_10;
        wp::float32 var_11;
        const wp::float32 var_12 = 0.0;
        //---------
        // forward
        // def write_joint_vel_data_index(                                                        <L 1>
        // i, j = wp.tid()                                                                        <L 27>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 28>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 29>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // joint_vel[env_id, joint_id] = in_data[i, j]                                            <L 30>
        var_8 = wp::address(var_in_data, var_0, var_1);
        var_9 = wp::load(var_8);
        wp::array_store(var_joint_vel, var_3, var_6, var_9);
        // prev_joint_vel[env_id, joint_id] = in_data[i, j]                                       <L 31>
        var_10 = wp::address(var_in_data, var_0, var_1);
        var_11 = wp::load(var_10);
        wp::array_store(var_prev_joint_vel, var_3, var_6, var_11);
        // joint_acc[env_id, joint_id] = 0.0                                                      <L 32>
        wp::array_store(var_joint_acc, var_3, var_6, var_12);
    }
}



extern "C" __global__ void write_joint_vel_data_index_235a34de_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::array_t<wp::float32> var_joint_acc,
    wp::array_t<wp::float32> adj_in_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int64> adj_joint_ids,
    wp::array_t<wp::float32> adj_joint_vel,
    wp::array_t<wp::float32> adj_prev_joint_vel,
    wp::array_t<wp::float32> adj_joint_acc)
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
        wp::float32* var_10;
        wp::float32 var_11;
        const wp::float32 var_12 = 0.0;
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
        //---------
        // forward
        // def write_joint_vel_data_index(                                                        <L 1>
        // i, j = wp.tid()                                                                        <L 27>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 28>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 29>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // joint_vel[env_id, joint_id] = in_data[i, j]                                            <L 30>
        var_8 = wp::address(var_in_data, var_0, var_1);
        var_9 = wp::load(var_8);
        // wp::array_store(var_joint_vel, var_3, var_6, var_9);
        // prev_joint_vel[env_id, joint_id] = in_data[i, j]                                       <L 31>
        var_10 = wp::address(var_in_data, var_0, var_1);
        var_11 = wp::load(var_10);
        // wp::array_store(var_prev_joint_vel, var_3, var_6, var_11);
        // joint_acc[env_id, joint_id] = 0.0                                                      <L 32>
        // wp::array_store(var_joint_acc, var_3, var_6, var_12);
        //---------
        // reverse
        wp::adj_array_store(var_joint_acc, var_3, var_6, var_12, adj_joint_acc, adj_3, adj_6, adj_12);
        // adj: joint_acc[env_id, joint_id] = 0.0                                                 <L 32>
        wp::adj_array_store(var_prev_joint_vel, var_3, var_6, var_11, adj_prev_joint_vel, adj_3, adj_6, adj_10);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_10);
        // adj: prev_joint_vel[env_id, joint_id] = in_data[i, j]                                  <L 31>
        wp::adj_array_store(var_joint_vel, var_3, var_6, var_9, adj_joint_vel, adj_3, adj_6, adj_8);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_8);
        // adj: joint_vel[env_id, joint_id] = in_data[i, j]                                       <L 30>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 29>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 28>
        // adj: i, j = wp.tid()                                                                   <L 27>
        // adj: def write_joint_vel_data_index(                                                   <L 1>
        continue;
    }
}



extern "C" __global__ void gather_jacobian_rows_29f53004_cuda_kernel_forward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<wp::float32> var_src,
    wp::array_t<wp::int32> var_art_ids,
    wp::array_t<wp::int32> var_jacobian_body_user_to_backend,
    wp::array_t<wp::int32> var_joint_user_to_backend,
    wp::int32 var_num_base_dofs,
    bool var_has_joint_ordering,
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
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        bool var_8;
        bool var_9;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32* var_15;
        wp::float32* var_16;
        wp::int32 var_17;
        wp::float32 var_18;
        //---------
        // forward
        // def gather_jacobian_rows(                                                              <L 680>
        // i, user_link, s, user_dof = wp.tid()                                                   <L 734>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // backend_link = jacobian_body_user_to_backend[user_link]                                <L 735>
        var_4 = wp::address(var_jacobian_body_user_to_backend, var_1);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // backend_dof = user_dof                                                                 <L 737>
        var_7 = wp::copy(var_3);
        // if has_joint_ordering and user_dof >= num_base_dofs:                                   <L 738>
        var_8 = var_has_joint_ordering;
        if (var_8) {
            var_9 = (var_3 >= var_num_base_dofs);
            var_8 = var_8 && var_9;
        }
        if (var_8) {
            // backend_dof = num_base_dofs + joint_user_to_backend[user_dof - num_base_dofs]       <L 739>
            var_10 = wp::sub(var_3, var_num_base_dofs);
            var_11 = wp::address(var_joint_user_to_backend, var_10);
            var_13 = wp::load(var_11);
            var_12 = wp::add(var_num_base_dofs, var_13);
        }
        var_14 = wp::where(var_8, var_12, var_7);
        // dst[i, user_link, s, user_dof] = src[art_ids[i], backend_link, s, backend_dof]         <L 741>
        var_15 = wp::address(var_art_ids, var_0);
        var_17 = wp::load(var_15);
        var_16 = wp::address(var_src, var_17, var_5, var_2, var_14);
        var_18 = wp::load(var_16);
        wp::array_store(var_dst, var_0, var_1, var_2, var_3, var_18);
    }
}



extern "C" __global__ void gather_jacobian_rows_29f53004_cuda_kernel_backward(
    wp::launch_bounds_t<4> dim,
    wp::array_t<wp::float32> var_src,
    wp::array_t<wp::int32> var_art_ids,
    wp::array_t<wp::int32> var_jacobian_body_user_to_backend,
    wp::array_t<wp::int32> var_joint_user_to_backend,
    wp::int32 var_num_base_dofs,
    bool var_has_joint_ordering,
    wp::array_t<wp::float32> var_dst,
    wp::array_t<wp::float32> adj_src,
    wp::array_t<wp::int32> adj_art_ids,
    wp::array_t<wp::int32> adj_jacobian_body_user_to_backend,
    wp::array_t<wp::int32> adj_joint_user_to_backend,
    wp::int32 adj_num_base_dofs,
    bool adj_has_joint_ordering,
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
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        bool var_8;
        bool var_9;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32* var_15;
        wp::float32* var_16;
        wp::int32 var_17;
        wp::float32 var_18;
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
        bool adj_8 = {};
        bool adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::float32 adj_18 = {};
        //---------
        // forward
        // def gather_jacobian_rows(                                                              <L 680>
        // i, user_link, s, user_dof = wp.tid()                                                   <L 734>
        builtin_tid4d(var_0, var_1, var_2, var_3);
        // backend_link = jacobian_body_user_to_backend[user_link]                                <L 735>
        var_4 = wp::address(var_jacobian_body_user_to_backend, var_1);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // backend_dof = user_dof                                                                 <L 737>
        var_7 = wp::copy(var_3);
        // if has_joint_ordering and user_dof >= num_base_dofs:                                   <L 738>
        var_8 = var_has_joint_ordering;
        if (var_8) {
            var_9 = (var_3 >= var_num_base_dofs);
            var_8 = var_8 && var_9;
        }
        if (var_8) {
            // backend_dof = num_base_dofs + joint_user_to_backend[user_dof - num_base_dofs]       <L 739>
            var_10 = wp::sub(var_3, var_num_base_dofs);
            var_11 = wp::address(var_joint_user_to_backend, var_10);
            var_13 = wp::load(var_11);
            var_12 = wp::add(var_num_base_dofs, var_13);
        }
        var_14 = wp::where(var_8, var_12, var_7);
        // dst[i, user_link, s, user_dof] = src[art_ids[i], backend_link, s, backend_dof]         <L 741>
        var_15 = wp::address(var_art_ids, var_0);
        var_17 = wp::load(var_15);
        var_16 = wp::address(var_src, var_17, var_5, var_2, var_14);
        var_18 = wp::load(var_16);
        // wp::array_store(var_dst, var_0, var_1, var_2, var_3, var_18);
        //---------
        // reverse
        wp::adj_array_store(var_dst, var_0, var_1, var_2, var_3, var_18, adj_dst, adj_0, adj_1, adj_2, adj_3, adj_16);
        wp::adj_address(var_src, var_17, var_5, var_2, var_14, adj_src, adj_15, adj_5, adj_2, adj_14, adj_16);
        wp::adj_address(var_art_ids, var_0, adj_art_ids, adj_0, adj_15);
        // adj: dst[i, user_link, s, user_dof] = src[art_ids[i], backend_link, s, backend_dof]    <L 741>
        wp::adj_where(var_8, var_12, var_7, adj_8, adj_12, adj_7, adj_14);
        if (var_8) {
            wp::adj_add(var_num_base_dofs, var_13, adj_num_base_dofs, adj_11, adj_12);
            wp::adj_address(var_joint_user_to_backend, var_10, adj_joint_user_to_backend, adj_10, adj_11);
            wp::adj_sub(var_3, var_num_base_dofs, adj_3, adj_num_base_dofs, adj_10);
            // adj: backend_dof = num_base_dofs + joint_user_to_backend[user_dof - num_base_dofs]  <L 739>
        }
        if (var_8) {
        }
        // adj: if has_joint_ordering and user_dof >= num_base_dofs:                              <L 738>
        wp::adj_copy(var_3, adj_3, adj_7);
        // adj: backend_dof = user_dof                                                            <L 737>
        wp::adj_copy(var_6, adj_4, adj_5);
        wp::adj_address(var_jacobian_body_user_to_backend, var_1, adj_jacobian_body_user_to_backend, adj_1, adj_4);
        // adj: backend_link = jacobian_body_user_to_backend[user_link]                           <L 735>
        // adj: i, user_link, s, user_dof = wp.tid()                                              <L 734>
        // adj: def gather_jacobian_rows(                                                         <L 680>
        continue;
    }
}



extern "C" __global__ void get_joint_acc_from_joint_vel_2ab71688_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::float32 var_dt,
    wp::array_t<wp::float32> var_joint_acc)
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
        wp::float32* var_2;
        wp::float32* var_3;
        wp::float32 var_4;
        wp::float32 var_5;
        wp::float32 var_6;
        wp::float32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        //---------
        // forward
        // def get_joint_acc_from_joint_vel(                                                      <L 95>
        // i, j = wp.tid()                                                                        <L 114>
        builtin_tid2d(var_0, var_1);
        // joint_acc[i, j] = (joint_vel[i, j] - prev_joint_vel[i, j]) / dt                        <L 115>
        var_2 = wp::address(var_joint_vel, var_0, var_1);
        var_3 = wp::address(var_prev_joint_vel, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = wp::sub(var_5, var_6);
        var_7 = wp::div(var_4, var_dt);
        wp::array_store(var_joint_acc, var_0, var_1, var_7);
        // prev_joint_vel[i, j] = joint_vel[i, j]                                                 <L 116>
        var_8 = wp::address(var_joint_vel, var_0, var_1);
        var_9 = wp::load(var_8);
        wp::array_store(var_prev_joint_vel, var_0, var_1, var_9);
    }
}



extern "C" __global__ void get_joint_acc_from_joint_vel_2ab71688_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::float32 var_dt,
    wp::array_t<wp::float32> var_joint_acc,
    wp::array_t<wp::float32> adj_joint_vel,
    wp::array_t<wp::float32> adj_prev_joint_vel,
    wp::float32 adj_dt,
    wp::array_t<wp::float32> adj_joint_acc)
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
        wp::float32* var_2;
        wp::float32* var_3;
        wp::float32 var_4;
        wp::float32 var_5;
        wp::float32 var_6;
        wp::float32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::float32 adj_2 = {};
        wp::float32 adj_3 = {};
        wp::float32 adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        //---------
        // forward
        // def get_joint_acc_from_joint_vel(                                                      <L 95>
        // i, j = wp.tid()                                                                        <L 114>
        builtin_tid2d(var_0, var_1);
        // joint_acc[i, j] = (joint_vel[i, j] - prev_joint_vel[i, j]) / dt                        <L 115>
        var_2 = wp::address(var_joint_vel, var_0, var_1);
        var_3 = wp::address(var_prev_joint_vel, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = wp::sub(var_5, var_6);
        var_7 = wp::div(var_4, var_dt);
        // wp::array_store(var_joint_acc, var_0, var_1, var_7);
        // prev_joint_vel[i, j] = joint_vel[i, j]                                                 <L 116>
        var_8 = wp::address(var_joint_vel, var_0, var_1);
        var_9 = wp::load(var_8);
        // wp::array_store(var_prev_joint_vel, var_0, var_1, var_9);
        //---------
        // reverse
        wp::adj_array_store(var_prev_joint_vel, var_0, var_1, var_9, adj_prev_joint_vel, adj_0, adj_1, adj_8);
        wp::adj_address(var_joint_vel, var_0, var_1, adj_joint_vel, adj_0, adj_1, adj_8);
        // adj: prev_joint_vel[i, j] = joint_vel[i, j]                                            <L 116>
        wp::adj_array_store(var_joint_acc, var_0, var_1, var_7, adj_joint_acc, adj_0, adj_1, adj_7);
        wp::adj_div(var_4, var_dt, var_7, adj_4, adj_dt, adj_7);
        wp::adj_sub(var_5, var_6, adj_2, adj_3, adj_4);
        wp::adj_address(var_prev_joint_vel, var_0, var_1, adj_prev_joint_vel, adj_0, adj_1, adj_3);
        wp::adj_address(var_joint_vel, var_0, var_1, adj_joint_vel, adj_0, adj_1, adj_2);
        // adj: joint_acc[i, j] = (joint_vel[i, j] - prev_joint_vel[i, j]) / dt                   <L 115>
        // adj: i, j = wp.tid()                                                                   <L 114>
        // adj: def get_joint_acc_from_joint_vel(                                                 <L 95>
        continue;
    }
}



extern "C" __global__ void write_joint_limit_data_to_user_and_backend_mask_2e0af177_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_in_data,
    wp::float32 var_soft_limit_factor,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_user_mask,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_user_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_user_joint_pos_limits,
    wp::array_t<wp::float32> var_backend_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_backend_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_soft_joint_pos_limits,
    wp::array_t<wp::float32> var_default_joint_pos,
    wp::array_t<wp::int32> var_clamped_defaults)
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
        wp::int32 var_9;
        wp::vec_t<2, wp::float32>* var_10;
        wp::vec_t<2, wp::float32> var_11;
        wp::vec_t<2, wp::float32> var_12;
        const wp::int32 var_13 = 0;
        wp::float32 var_14;
        const wp::int32 var_15 = 1;
        wp::float32 var_16;
        bool var_17;
        wp::float32* var_18;
        bool var_19;
        wp::float32 var_20;
        wp::float32* var_21;
        bool var_22;
        wp::float32 var_23;
        const wp::int32 var_24 = 0;
        const wp::int32 var_25 = 1;
        wp::int32 var_26;
        wp::float32* var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::vec_t<2, wp::float32> var_30;
        //---------
        // forward
        // def write_joint_limit_data_to_user_and_backend_mask(                                   <L 298>
        // env_id, user_id = wp.tid()                                                             <L 315>
        builtin_tid2d(var_0, var_1);
        // if not env_mask[env_id] or not user_mask[user_id]:                                     <L 316>
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
            // return                                                                             <L 317>
            continue;
        }
        // backend_id = resolve_backend_index(user_id, user_to_backend, has_ordering)             <L 319>
        var_9 = resolve_backend_index_0(var_1, var_user_to_backend, var_has_ordering);
        // limits = in_data[env_id, user_id]                                                      <L 320>
        var_10 = wp::address(var_in_data, var_0, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // lower = limits[0]                                                                      <L 321>
        var_14 = wp::extract(var_11, var_13);
        // upper = limits[1]                                                                      <L 322>
        var_16 = wp::extract(var_11, var_15);
        // user_joint_pos_limits_lower[env_id, user_id] = lower                                   <L 324>
        wp::array_store(var_user_joint_pos_limits_lower, var_0, var_1, var_14);
        // user_joint_pos_limits_upper[env_id, user_id] = upper                                   <L 325>
        wp::array_store(var_user_joint_pos_limits_upper, var_0, var_1, var_16);
        // user_joint_pos_limits[env_id, user_id] = limits                                        <L 326>
        wp::array_store(var_user_joint_pos_limits, var_0, var_1, var_11);
        // backend_joint_pos_limits_lower[env_id, backend_id] = lower                             <L 327>
        wp::array_store(var_backend_joint_pos_limits_lower, var_0, var_9, var_14);
        // backend_joint_pos_limits_upper[env_id, backend_id] = upper                             <L 328>
        wp::array_store(var_backend_joint_pos_limits_upper, var_0, var_9, var_16);
        // if (default_joint_pos[env_id, user_id] < lower) or default_joint_pos[env_id, user_id] > upper:       <L 330>
        var_18 = wp::address(var_default_joint_pos, var_0, var_1);
        var_20 = wp::load(var_18);
        var_19 = (var_20 < var_14);
        var_17 = var_19;
        if (!var_17) {
            var_21 = wp::address(var_default_joint_pos, var_0, var_1);
            var_23 = wp::load(var_21);
            var_22 = (var_23 > var_16);
            var_17 = var_17 || var_22;
        }
        if (var_17) {
            // wp.atomic_add(clamped_defaults, 0, 1)                                              <L 331>
            var_26 = wp::atomic_add(var_clamped_defaults, var_24, var_25);
            // default_joint_pos[env_id, user_id] = wp.clamp(default_joint_pos[env_id, user_id], lower, upper)       <L 332>
            var_27 = wp::address(var_default_joint_pos, var_0, var_1);
            var_29 = wp::load(var_27);
            var_28 = wp::clamp(var_29, var_14, var_16);
            wp::array_store(var_default_joint_pos, var_0, var_1, var_28);
        }
        // soft_joint_pos_limits[env_id, user_id] = compute_soft_joint_pos_limits_func(limits, soft_limit_factor)       <L 334>
        var_30 = compute_soft_joint_pos_limits_func_0(var_11, var_soft_limit_factor);
        wp::array_store(var_soft_joint_pos_limits, var_0, var_1, var_30);
    }
}



extern "C" __global__ void write_joint_limit_data_to_user_and_backend_mask_2e0af177_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_in_data,
    wp::float32 var_soft_limit_factor,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_user_mask,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_user_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_user_joint_pos_limits,
    wp::array_t<wp::float32> var_backend_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_backend_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_soft_joint_pos_limits,
    wp::array_t<wp::float32> var_default_joint_pos,
    wp::array_t<wp::int32> var_clamped_defaults,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_in_data,
    wp::float32 adj_soft_limit_factor,
    wp::array_t<bool> adj_env_mask,
    wp::array_t<bool> adj_user_mask,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    wp::array_t<wp::float32> adj_user_joint_pos_limits_lower,
    wp::array_t<wp::float32> adj_user_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_user_joint_pos_limits,
    wp::array_t<wp::float32> adj_backend_joint_pos_limits_lower,
    wp::array_t<wp::float32> adj_backend_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_soft_joint_pos_limits,
    wp::array_t<wp::float32> adj_default_joint_pos,
    wp::array_t<wp::int32> adj_clamped_defaults)
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
        wp::int32 var_9;
        wp::vec_t<2, wp::float32>* var_10;
        wp::vec_t<2, wp::float32> var_11;
        wp::vec_t<2, wp::float32> var_12;
        const wp::int32 var_13 = 0;
        wp::float32 var_14;
        const wp::int32 var_15 = 1;
        wp::float32 var_16;
        bool var_17;
        wp::float32* var_18;
        bool var_19;
        wp::float32 var_20;
        wp::float32* var_21;
        bool var_22;
        wp::float32 var_23;
        const wp::int32 var_24 = 0;
        const wp::int32 var_25 = 1;
        wp::int32 var_26;
        wp::float32* var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::vec_t<2, wp::float32> var_30;
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
        wp::vec_t<2, wp::float32> adj_10 = {};
        wp::vec_t<2, wp::float32> adj_11 = {};
        wp::vec_t<2, wp::float32> adj_12 = {};
        wp::int32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::float32 adj_16 = {};
        bool adj_17 = {};
        wp::float32 adj_18 = {};
        bool adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        bool adj_22 = {};
        wp::float32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::float32 adj_29 = {};
        wp::vec_t<2, wp::float32> adj_30 = {};
        //---------
        // forward
        // def write_joint_limit_data_to_user_and_backend_mask(                                   <L 298>
        // env_id, user_id = wp.tid()                                                             <L 315>
        builtin_tid2d(var_0, var_1);
        // if not env_mask[env_id] or not user_mask[user_id]:                                     <L 316>
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
            // return                                                                             <L 317>
            goto label0;
        }
        // backend_id = resolve_backend_index(user_id, user_to_backend, has_ordering)             <L 319>
        var_9 = resolve_backend_index_0(var_1, var_user_to_backend, var_has_ordering);
        // limits = in_data[env_id, user_id]                                                      <L 320>
        var_10 = wp::address(var_in_data, var_0, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // lower = limits[0]                                                                      <L 321>
        var_14 = wp::extract(var_11, var_13);
        // upper = limits[1]                                                                      <L 322>
        var_16 = wp::extract(var_11, var_15);
        // user_joint_pos_limits_lower[env_id, user_id] = lower                                   <L 324>
        // wp::array_store(var_user_joint_pos_limits_lower, var_0, var_1, var_14);
        // user_joint_pos_limits_upper[env_id, user_id] = upper                                   <L 325>
        // wp::array_store(var_user_joint_pos_limits_upper, var_0, var_1, var_16);
        // user_joint_pos_limits[env_id, user_id] = limits                                        <L 326>
        // wp::array_store(var_user_joint_pos_limits, var_0, var_1, var_11);
        // backend_joint_pos_limits_lower[env_id, backend_id] = lower                             <L 327>
        // wp::array_store(var_backend_joint_pos_limits_lower, var_0, var_9, var_14);
        // backend_joint_pos_limits_upper[env_id, backend_id] = upper                             <L 328>
        // wp::array_store(var_backend_joint_pos_limits_upper, var_0, var_9, var_16);
        // if (default_joint_pos[env_id, user_id] < lower) or default_joint_pos[env_id, user_id] > upper:       <L 330>
        var_18 = wp::address(var_default_joint_pos, var_0, var_1);
        var_20 = wp::load(var_18);
        var_19 = (var_20 < var_14);
        var_17 = var_19;
        if (!var_17) {
            var_21 = wp::address(var_default_joint_pos, var_0, var_1);
            var_23 = wp::load(var_21);
            var_22 = (var_23 > var_16);
            var_17 = var_17 || var_22;
        }
        if (var_17) {
            // wp.atomic_add(clamped_defaults, 0, 1)                                              <L 331>
            // var_26 = wp::atomic_add(var_clamped_defaults, var_24, var_25);
            // default_joint_pos[env_id, user_id] = wp.clamp(default_joint_pos[env_id, user_id], lower, upper)       <L 332>
            var_27 = wp::address(var_default_joint_pos, var_0, var_1);
            var_29 = wp::load(var_27);
            var_28 = wp::clamp(var_29, var_14, var_16);
            // wp::array_store(var_default_joint_pos, var_0, var_1, var_28);
        }
        // soft_joint_pos_limits[env_id, user_id] = compute_soft_joint_pos_limits_func(limits, soft_limit_factor)       <L 334>
        var_30 = compute_soft_joint_pos_limits_func_0(var_11, var_soft_limit_factor);
        // wp::array_store(var_soft_joint_pos_limits, var_0, var_1, var_30);
        //---------
        // reverse
        wp::adj_array_store(var_soft_joint_pos_limits, var_0, var_1, var_30, adj_soft_joint_pos_limits, adj_0, adj_1, adj_30);
        adj_compute_soft_joint_pos_limits_func_0(var_11, var_soft_limit_factor, adj_11, adj_soft_limit_factor, adj_30);
        // adj: soft_joint_pos_limits[env_id, user_id] = compute_soft_joint_pos_limits_func(limits, soft_limit_factor)  <L 334>
        if (var_17) {
            wp::adj_array_store(var_default_joint_pos, var_0, var_1, var_28, adj_default_joint_pos, adj_0, adj_1, adj_28);
            wp::adj_clamp(var_29, var_14, var_16, adj_27, adj_14, adj_16, adj_28);
            wp::adj_address(var_default_joint_pos, var_0, var_1, adj_default_joint_pos, adj_0, adj_1, adj_27);
            // adj: default_joint_pos[env_id, user_id] = wp.clamp(default_joint_pos[env_id, user_id], lower, upper)  <L 332>
            wp::adj_atomic_add(var_clamped_defaults, var_24, var_25, adj_clamped_defaults, adj_24, adj_25, adj_26);
            // adj: wp.atomic_add(clamped_defaults, 0, 1)                                         <L 331>
        }
        if (!var_17) {
            wp::adj_address(var_default_joint_pos, var_0, var_1, adj_default_joint_pos, adj_0, adj_1, adj_21);
        }
        wp::adj_address(var_default_joint_pos, var_0, var_1, adj_default_joint_pos, adj_0, adj_1, adj_18);
        // adj: if (default_joint_pos[env_id, user_id] < lower) or default_joint_pos[env_id, user_id] > upper:  <L 330>
        wp::adj_array_store(var_backend_joint_pos_limits_upper, var_0, var_9, var_16, adj_backend_joint_pos_limits_upper, adj_0, adj_9, adj_16);
        // adj: backend_joint_pos_limits_upper[env_id, backend_id] = upper                        <L 328>
        wp::adj_array_store(var_backend_joint_pos_limits_lower, var_0, var_9, var_14, adj_backend_joint_pos_limits_lower, adj_0, adj_9, adj_14);
        // adj: backend_joint_pos_limits_lower[env_id, backend_id] = lower                        <L 327>
        wp::adj_array_store(var_user_joint_pos_limits, var_0, var_1, var_11, adj_user_joint_pos_limits, adj_0, adj_1, adj_11);
        // adj: user_joint_pos_limits[env_id, user_id] = limits                                   <L 326>
        wp::adj_array_store(var_user_joint_pos_limits_upper, var_0, var_1, var_16, adj_user_joint_pos_limits_upper, adj_0, adj_1, adj_16);
        // adj: user_joint_pos_limits_upper[env_id, user_id] = upper                              <L 325>
        wp::adj_array_store(var_user_joint_pos_limits_lower, var_0, var_1, var_14, adj_user_joint_pos_limits_lower, adj_0, adj_1, adj_14);
        // adj: user_joint_pos_limits_lower[env_id, user_id] = lower                              <L 324>
        wp::adj_extract(var_11, var_15, adj_11, adj_15, adj_16);
        // adj: upper = limits[1]                                                                 <L 322>
        wp::adj_extract(var_11, var_13, adj_11, adj_13, adj_14);
        // adj: lower = limits[0]                                                                 <L 321>
        wp::adj_copy(var_12, adj_10, adj_11);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_10);
        // adj: limits = in_data[env_id, user_id]                                                 <L 320>
        adj_resolve_backend_index_0(var_1, var_user_to_backend, var_has_ordering, adj_1, adj_user_to_backend, adj_has_ordering, adj_9);
        // adj: backend_id = resolve_backend_index(user_id, user_to_backend, has_ordering)        <L 319>
        if (var_2) {
            label0:;
            // adj: return                                                                        <L 317>
        }
        if (!var_2) {
            wp::adj_address(var_user_mask, var_1, adj_user_mask, adj_1, adj_6);
        }
        wp::adj_address(var_env_mask, var_0, adj_env_mask, adj_0, adj_3);
        // adj: if not env_mask[env_id] or not user_mask[user_id]:                                <L 316>
        // adj: env_id, user_id = wp.tid()                                                        <L 315>
        // adj: def write_joint_limit_data_to_user_and_backend_mask(                              <L 298>
        continue;
    }
}



extern "C" __global__ void gather_mass_matrix_rows_38fe401d_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::float32> var_src,
    wp::array_t<wp::int32> var_art_ids,
    wp::array_t<wp::int32> var_joint_user_to_backend,
    wp::int32 var_num_base_dofs,
    bool var_has_joint_ordering,
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
        wp::int32* var_19;
        wp::float32* var_20;
        wp::int32 var_21;
        wp::float32 var_22;
        //---------
        // forward
        // def gather_mass_matrix_rows(                                                           <L 745>
        // i, user_row, user_col = wp.tid()                                                       <L 774>
        builtin_tid3d(var_0, var_1, var_2);
        // backend_row = user_row                                                                 <L 775>
        var_3 = wp::copy(var_1);
        // backend_col = user_col                                                                 <L 776>
        var_4 = wp::copy(var_2);
        // if has_joint_ordering:                                                                 <L 777>
        if (var_has_joint_ordering) {
            // if user_row >= num_base_dofs:                                                      <L 778>
            var_5 = (var_1 >= var_num_base_dofs);
            if (var_5) {
                // backend_row = num_base_dofs + joint_user_to_backend[user_row - num_base_dofs]       <L 779>
                var_6 = wp::sub(var_1, var_num_base_dofs);
                var_7 = wp::address(var_joint_user_to_backend, var_6);
                var_9 = wp::load(var_7);
                var_8 = wp::add(var_num_base_dofs, var_9);
            }
            var_10 = wp::where(var_5, var_8, var_3);
            // if user_col >= num_base_dofs:                                                      <L 780>
            var_11 = (var_2 >= var_num_base_dofs);
            if (var_11) {
                // backend_col = num_base_dofs + joint_user_to_backend[user_col - num_base_dofs]       <L 781>
                var_12 = wp::sub(var_2, var_num_base_dofs);
                var_13 = wp::address(var_joint_user_to_backend, var_12);
                var_15 = wp::load(var_13);
                var_14 = wp::add(var_num_base_dofs, var_15);
            }
            var_16 = wp::where(var_11, var_14, var_4);
        }
        var_17 = wp::where(var_has_joint_ordering, var_10, var_3);
        var_18 = wp::where(var_has_joint_ordering, var_16, var_4);
        // dst[i, user_row, user_col] = src[art_ids[i], backend_row, backend_col]                 <L 783>
        var_19 = wp::address(var_art_ids, var_0);
        var_21 = wp::load(var_19);
        var_20 = wp::address(var_src, var_21, var_17, var_18);
        var_22 = wp::load(var_20);
        wp::array_store(var_dst, var_0, var_1, var_2, var_22);
    }
}



extern "C" __global__ void gather_mass_matrix_rows_38fe401d_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::float32> var_src,
    wp::array_t<wp::int32> var_art_ids,
    wp::array_t<wp::int32> var_joint_user_to_backend,
    wp::int32 var_num_base_dofs,
    bool var_has_joint_ordering,
    wp::array_t<wp::float32> var_dst,
    wp::array_t<wp::float32> adj_src,
    wp::array_t<wp::int32> adj_art_ids,
    wp::array_t<wp::int32> adj_joint_user_to_backend,
    wp::int32 adj_num_base_dofs,
    bool adj_has_joint_ordering,
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
        wp::int32* var_19;
        wp::float32* var_20;
        wp::int32 var_21;
        wp::float32 var_22;
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
        wp::int32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::float32 adj_22 = {};
        //---------
        // forward
        // def gather_mass_matrix_rows(                                                           <L 745>
        // i, user_row, user_col = wp.tid()                                                       <L 774>
        builtin_tid3d(var_0, var_1, var_2);
        // backend_row = user_row                                                                 <L 775>
        var_3 = wp::copy(var_1);
        // backend_col = user_col                                                                 <L 776>
        var_4 = wp::copy(var_2);
        // if has_joint_ordering:                                                                 <L 777>
        if (var_has_joint_ordering) {
            // if user_row >= num_base_dofs:                                                      <L 778>
            var_5 = (var_1 >= var_num_base_dofs);
            if (var_5) {
                // backend_row = num_base_dofs + joint_user_to_backend[user_row - num_base_dofs]       <L 779>
                var_6 = wp::sub(var_1, var_num_base_dofs);
                var_7 = wp::address(var_joint_user_to_backend, var_6);
                var_9 = wp::load(var_7);
                var_8 = wp::add(var_num_base_dofs, var_9);
            }
            var_10 = wp::where(var_5, var_8, var_3);
            // if user_col >= num_base_dofs:                                                      <L 780>
            var_11 = (var_2 >= var_num_base_dofs);
            if (var_11) {
                // backend_col = num_base_dofs + joint_user_to_backend[user_col - num_base_dofs]       <L 781>
                var_12 = wp::sub(var_2, var_num_base_dofs);
                var_13 = wp::address(var_joint_user_to_backend, var_12);
                var_15 = wp::load(var_13);
                var_14 = wp::add(var_num_base_dofs, var_15);
            }
            var_16 = wp::where(var_11, var_14, var_4);
        }
        var_17 = wp::where(var_has_joint_ordering, var_10, var_3);
        var_18 = wp::where(var_has_joint_ordering, var_16, var_4);
        // dst[i, user_row, user_col] = src[art_ids[i], backend_row, backend_col]                 <L 783>
        var_19 = wp::address(var_art_ids, var_0);
        var_21 = wp::load(var_19);
        var_20 = wp::address(var_src, var_21, var_17, var_18);
        var_22 = wp::load(var_20);
        // wp::array_store(var_dst, var_0, var_1, var_2, var_22);
        //---------
        // reverse
        wp::adj_array_store(var_dst, var_0, var_1, var_2, var_22, adj_dst, adj_0, adj_1, adj_2, adj_20);
        wp::adj_address(var_src, var_21, var_17, var_18, adj_src, adj_19, adj_17, adj_18, adj_20);
        wp::adj_address(var_art_ids, var_0, adj_art_ids, adj_0, adj_19);
        // adj: dst[i, user_row, user_col] = src[art_ids[i], backend_row, backend_col]            <L 783>
        wp::adj_where(var_has_joint_ordering, var_16, var_4, adj_has_joint_ordering, adj_16, adj_4, adj_18);
        wp::adj_where(var_has_joint_ordering, var_10, var_3, adj_has_joint_ordering, adj_10, adj_3, adj_17);
        if (var_has_joint_ordering) {
            wp::adj_where(var_11, var_14, var_4, adj_11, adj_14, adj_4, adj_16);
            if (var_11) {
                wp::adj_add(var_num_base_dofs, var_15, adj_num_base_dofs, adj_13, adj_14);
                wp::adj_address(var_joint_user_to_backend, var_12, adj_joint_user_to_backend, adj_12, adj_13);
                wp::adj_sub(var_2, var_num_base_dofs, adj_2, adj_num_base_dofs, adj_12);
                // adj: backend_col = num_base_dofs + joint_user_to_backend[user_col - num_base_dofs]  <L 781>
            }
            // adj: if user_col >= num_base_dofs:                                                 <L 780>
            wp::adj_where(var_5, var_8, var_3, adj_5, adj_8, adj_3, adj_10);
            if (var_5) {
                wp::adj_add(var_num_base_dofs, var_9, adj_num_base_dofs, adj_7, adj_8);
                wp::adj_address(var_joint_user_to_backend, var_6, adj_joint_user_to_backend, adj_6, adj_7);
                wp::adj_sub(var_1, var_num_base_dofs, adj_1, adj_num_base_dofs, adj_6);
                // adj: backend_row = num_base_dofs + joint_user_to_backend[user_row - num_base_dofs]  <L 779>
            }
            // adj: if user_row >= num_base_dofs:                                                 <L 778>
        }
        // adj: if has_joint_ordering:                                                            <L 777>
        wp::adj_copy(var_2, adj_2, adj_4);
        // adj: backend_col = user_col                                                            <L 776>
        wp::adj_copy(var_1, adj_1, adj_3);
        // adj: backend_row = user_row                                                            <L 775>
        // adj: i, user_row, user_col = wp.tid()                                                  <L 774>
        // adj: def gather_mass_matrix_rows(                                                      <L 745>
        continue;
    }
}



extern "C" __global__ void write_joint_limit_data_to_buffer_index_443a7997_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_in_data,
    wp::float32 var_soft_limit_factor,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    wp::array_t<wp::float32> var_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_soft_joint_pos_limits,
    wp::array_t<wp::float32> var_default_joint_pos,
    wp::array_t<wp::int32> var_clamped_defaults)
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
        wp::vec_t<2, wp::float32>* var_8;
        const wp::int32 var_9 = 0;
        wp::float32 var_10;
        wp::vec_t<2, wp::float32> var_11;
        wp::vec_t<2, wp::float32>* var_12;
        const wp::int32 var_13 = 1;
        wp::float32 var_14;
        wp::vec_t<2, wp::float32> var_15;
        bool var_16;
        wp::float32* var_17;
        wp::float32* var_18;
        bool var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32* var_22;
        wp::float32* var_23;
        bool var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        const wp::int32 var_27 = 0;
        const wp::int32 var_28 = 1;
        wp::int32 var_29;
        wp::float32* var_30;
        wp::float32* var_31;
        wp::float32* var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32* var_37;
        wp::float32* var_38;
        wp::vec_t<2, wp::float32> var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::vec_t<2, wp::float32> var_42;
        //---------
        // forward
        // def write_joint_limit_data_to_buffer_index(                                            <L 1>
        // i, j = wp.tid()                                                                        <L 35>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 36>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 37>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // joint_pos_limits_lower[env_id, joint_id] = in_data[i, j][0]                            <L 38>
        var_8 = wp::address(var_in_data, var_0, var_1);
        var_11 = wp::load(var_8);
        var_10 = wp::extract(var_11, var_9);
        wp::array_store(var_joint_pos_limits_lower, var_3, var_6, var_10);
        // joint_pos_limits_upper[env_id, joint_id] = in_data[i, j][1]                            <L 39>
        var_12 = wp::address(var_in_data, var_0, var_1);
        var_15 = wp::load(var_12);
        var_14 = wp::extract(var_15, var_13);
        wp::array_store(var_joint_pos_limits_upper, var_3, var_6, var_14);
        // if (default_joint_pos[env_id, joint_id] < joint_pos_limits_lower[env_id, joint_id]) or default_joint_pos[       <L 40>
        var_17 = wp::address(var_default_joint_pos, var_3, var_6);
        var_18 = wp::address(var_joint_pos_limits_lower, var_3, var_6);
        var_20 = wp::load(var_17);
        var_21 = wp::load(var_18);
        var_19 = (var_20 < var_21);
        var_16 = var_19;
        if (!var_16) {
            // env_id, joint_id                                                                   <L 41>
            var_22 = wp::address(var_default_joint_pos, var_3, var_6);
            // ] > joint_pos_limits_upper[env_id, joint_id]:                                      <L 42>
            var_23 = wp::address(var_joint_pos_limits_upper, var_3, var_6);
            var_25 = wp::load(var_22);
            var_26 = wp::load(var_23);
            var_24 = (var_25 > var_26);
            var_16 = var_16 || var_24;
        }
        if (var_16) {
            // wp.atomic_add(clamped_defaults, 0, 1)                                              <L 43>
            var_29 = wp::atomic_add(var_clamped_defaults, var_27, var_28);
            // default_joint_pos[env_id, joint_id] = wp.clamp(                                    <L 44>
            // default_joint_pos[env_id, joint_id],                                               <L 45>
            var_30 = wp::address(var_default_joint_pos, var_3, var_6);
            // joint_pos_limits_lower[env_id, joint_id],                                          <L 46>
            var_31 = wp::address(var_joint_pos_limits_lower, var_3, var_6);
            // joint_pos_limits_upper[env_id, joint_id],                                          <L 47>
            var_32 = wp::address(var_joint_pos_limits_upper, var_3, var_6);
            var_34 = wp::load(var_30);
            var_35 = wp::load(var_31);
            var_36 = wp::load(var_32);
            var_33 = wp::clamp(var_34, var_35, var_36);
            // default_joint_pos[env_id, joint_id] = wp.clamp(                                    <L 44>
            wp::array_store(var_default_joint_pos, var_3, var_6, var_33);
        }
        // soft_joint_pos_limits[env_id, joint_id] = compute_soft_joint_pos_limits_func(          <L 49>
        // wp.vec2f(joint_pos_limits_lower[env_id, joint_id], joint_pos_limits_upper[env_id, joint_id]),       <L 50>
        var_37 = wp::address(var_joint_pos_limits_lower, var_3, var_6);
        var_38 = wp::address(var_joint_pos_limits_upper, var_3, var_6);
        var_40 = wp::load(var_37);
        var_41 = wp::load(var_38);
        var_39 = wp::vec_t<2, wp::float32>(var_40, var_41);
        // soft_limit_factor,                                                                     <L 51>
        var_42 = compute_soft_joint_pos_limits_func_0(var_39, var_soft_limit_factor);
        // soft_joint_pos_limits[env_id, joint_id] = compute_soft_joint_pos_limits_func(          <L 49>
        wp::array_store(var_soft_joint_pos_limits, var_3, var_6, var_42);
    }
}



extern "C" __global__ void write_joint_limit_data_to_buffer_index_443a7997_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_in_data,
    wp::float32 var_soft_limit_factor,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    wp::array_t<wp::float32> var_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_soft_joint_pos_limits,
    wp::array_t<wp::float32> var_default_joint_pos,
    wp::array_t<wp::int32> var_clamped_defaults,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_in_data,
    wp::float32 adj_soft_limit_factor,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int64> adj_joint_ids,
    wp::array_t<wp::float32> adj_joint_pos_limits_lower,
    wp::array_t<wp::float32> adj_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_soft_joint_pos_limits,
    wp::array_t<wp::float32> adj_default_joint_pos,
    wp::array_t<wp::int32> adj_clamped_defaults)
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
        wp::vec_t<2, wp::float32>* var_8;
        const wp::int32 var_9 = 0;
        wp::float32 var_10;
        wp::vec_t<2, wp::float32> var_11;
        wp::vec_t<2, wp::float32>* var_12;
        const wp::int32 var_13 = 1;
        wp::float32 var_14;
        wp::vec_t<2, wp::float32> var_15;
        bool var_16;
        wp::float32* var_17;
        wp::float32* var_18;
        bool var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32* var_22;
        wp::float32* var_23;
        bool var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        const wp::int32 var_27 = 0;
        const wp::int32 var_28 = 1;
        wp::int32 var_29;
        wp::float32* var_30;
        wp::float32* var_31;
        wp::float32* var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32* var_37;
        wp::float32* var_38;
        wp::vec_t<2, wp::float32> var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::vec_t<2, wp::float32> var_42;
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
        wp::vec_t<2, wp::float32> adj_8 = {};
        wp::int32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::vec_t<2, wp::float32> adj_11 = {};
        wp::vec_t<2, wp::float32> adj_12 = {};
        wp::int32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::vec_t<2, wp::float32> adj_15 = {};
        bool adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        bool adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        bool adj_24 = {};
        wp::float32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::float32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::vec_t<2, wp::float32> adj_39 = {};
        wp::float32 adj_40 = {};
        wp::float32 adj_41 = {};
        wp::vec_t<2, wp::float32> adj_42 = {};
        //---------
        // forward
        // def write_joint_limit_data_to_buffer_index(                                            <L 1>
        // i, j = wp.tid()                                                                        <L 35>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 36>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 37>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // joint_pos_limits_lower[env_id, joint_id] = in_data[i, j][0]                            <L 38>
        var_8 = wp::address(var_in_data, var_0, var_1);
        var_11 = wp::load(var_8);
        var_10 = wp::extract(var_11, var_9);
        // wp::array_store(var_joint_pos_limits_lower, var_3, var_6, var_10);
        // joint_pos_limits_upper[env_id, joint_id] = in_data[i, j][1]                            <L 39>
        var_12 = wp::address(var_in_data, var_0, var_1);
        var_15 = wp::load(var_12);
        var_14 = wp::extract(var_15, var_13);
        // wp::array_store(var_joint_pos_limits_upper, var_3, var_6, var_14);
        // if (default_joint_pos[env_id, joint_id] < joint_pos_limits_lower[env_id, joint_id]) or default_joint_pos[       <L 40>
        var_17 = wp::address(var_default_joint_pos, var_3, var_6);
        var_18 = wp::address(var_joint_pos_limits_lower, var_3, var_6);
        var_20 = wp::load(var_17);
        var_21 = wp::load(var_18);
        var_19 = (var_20 < var_21);
        var_16 = var_19;
        if (!var_16) {
            // env_id, joint_id                                                                   <L 41>
            var_22 = wp::address(var_default_joint_pos, var_3, var_6);
            // ] > joint_pos_limits_upper[env_id, joint_id]:                                      <L 42>
            var_23 = wp::address(var_joint_pos_limits_upper, var_3, var_6);
            var_25 = wp::load(var_22);
            var_26 = wp::load(var_23);
            var_24 = (var_25 > var_26);
            var_16 = var_16 || var_24;
        }
        if (var_16) {
            // wp.atomic_add(clamped_defaults, 0, 1)                                              <L 43>
            // var_29 = wp::atomic_add(var_clamped_defaults, var_27, var_28);
            // default_joint_pos[env_id, joint_id] = wp.clamp(                                    <L 44>
            // default_joint_pos[env_id, joint_id],                                               <L 45>
            var_30 = wp::address(var_default_joint_pos, var_3, var_6);
            // joint_pos_limits_lower[env_id, joint_id],                                          <L 46>
            var_31 = wp::address(var_joint_pos_limits_lower, var_3, var_6);
            // joint_pos_limits_upper[env_id, joint_id],                                          <L 47>
            var_32 = wp::address(var_joint_pos_limits_upper, var_3, var_6);
            var_34 = wp::load(var_30);
            var_35 = wp::load(var_31);
            var_36 = wp::load(var_32);
            var_33 = wp::clamp(var_34, var_35, var_36);
            // default_joint_pos[env_id, joint_id] = wp.clamp(                                    <L 44>
            // wp::array_store(var_default_joint_pos, var_3, var_6, var_33);
        }
        // soft_joint_pos_limits[env_id, joint_id] = compute_soft_joint_pos_limits_func(          <L 49>
        // wp.vec2f(joint_pos_limits_lower[env_id, joint_id], joint_pos_limits_upper[env_id, joint_id]),       <L 50>
        var_37 = wp::address(var_joint_pos_limits_lower, var_3, var_6);
        var_38 = wp::address(var_joint_pos_limits_upper, var_3, var_6);
        var_40 = wp::load(var_37);
        var_41 = wp::load(var_38);
        var_39 = wp::vec_t<2, wp::float32>(var_40, var_41);
        // soft_limit_factor,                                                                     <L 51>
        var_42 = compute_soft_joint_pos_limits_func_0(var_39, var_soft_limit_factor);
        // soft_joint_pos_limits[env_id, joint_id] = compute_soft_joint_pos_limits_func(          <L 49>
        // wp::array_store(var_soft_joint_pos_limits, var_3, var_6, var_42);
        //---------
        // reverse
        wp::adj_array_store(var_soft_joint_pos_limits, var_3, var_6, var_42, adj_soft_joint_pos_limits, adj_3, adj_6, adj_42);
        // adj: soft_joint_pos_limits[env_id, joint_id] = compute_soft_joint_pos_limits_func(     <L 49>
        adj_compute_soft_joint_pos_limits_func_0(var_39, var_soft_limit_factor, adj_39, adj_soft_limit_factor, adj_42);
        // adj: soft_limit_factor,                                                                <L 51>
        wp::adj_vec_t(var_40, var_41, adj_37, adj_38, adj_39);
        wp::adj_address(var_joint_pos_limits_upper, var_3, var_6, adj_joint_pos_limits_upper, adj_3, adj_6, adj_38);
        wp::adj_address(var_joint_pos_limits_lower, var_3, var_6, adj_joint_pos_limits_lower, adj_3, adj_6, adj_37);
        // adj: wp.vec2f(joint_pos_limits_lower[env_id, joint_id], joint_pos_limits_upper[env_id, joint_id]),  <L 50>
        // adj: soft_joint_pos_limits[env_id, joint_id] = compute_soft_joint_pos_limits_func(     <L 49>
        if (var_16) {
            wp::adj_array_store(var_default_joint_pos, var_3, var_6, var_33, adj_default_joint_pos, adj_3, adj_6, adj_33);
            // adj: default_joint_pos[env_id, joint_id] = wp.clamp(                               <L 44>
            wp::adj_clamp(var_34, var_35, var_36, adj_30, adj_31, adj_32, adj_33);
            wp::adj_address(var_joint_pos_limits_upper, var_3, var_6, adj_joint_pos_limits_upper, adj_3, adj_6, adj_32);
            // adj: joint_pos_limits_upper[env_id, joint_id],                                     <L 47>
            wp::adj_address(var_joint_pos_limits_lower, var_3, var_6, adj_joint_pos_limits_lower, adj_3, adj_6, adj_31);
            // adj: joint_pos_limits_lower[env_id, joint_id],                                     <L 46>
            wp::adj_address(var_default_joint_pos, var_3, var_6, adj_default_joint_pos, adj_3, adj_6, adj_30);
            // adj: default_joint_pos[env_id, joint_id],                                          <L 45>
            // adj: default_joint_pos[env_id, joint_id] = wp.clamp(                               <L 44>
            wp::adj_atomic_add(var_clamped_defaults, var_27, var_28, adj_clamped_defaults, adj_27, adj_28, adj_29);
            // adj: wp.atomic_add(clamped_defaults, 0, 1)                                         <L 43>
        }
        if (!var_16) {
            wp::adj_address(var_joint_pos_limits_upper, var_3, var_6, adj_joint_pos_limits_upper, adj_3, adj_6, adj_23);
            // adj: ] > joint_pos_limits_upper[env_id, joint_id]:                                 <L 42>
            wp::adj_address(var_default_joint_pos, var_3, var_6, adj_default_joint_pos, adj_3, adj_6, adj_22);
            // adj: env_id, joint_id                                                              <L 41>
        }
        wp::adj_address(var_joint_pos_limits_lower, var_3, var_6, adj_joint_pos_limits_lower, adj_3, adj_6, adj_18);
        wp::adj_address(var_default_joint_pos, var_3, var_6, adj_default_joint_pos, adj_3, adj_6, adj_17);
        // adj: if (default_joint_pos[env_id, joint_id] < joint_pos_limits_lower[env_id, joint_id]) or default_joint_pos[  <L 40>
        wp::adj_array_store(var_joint_pos_limits_upper, var_3, var_6, var_14, adj_joint_pos_limits_upper, adj_3, adj_6, adj_14);
        wp::adj_extract(var_15, var_13, adj_12, adj_13, adj_14);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_12);
        // adj: joint_pos_limits_upper[env_id, joint_id] = in_data[i, j][1]                       <L 39>
        wp::adj_array_store(var_joint_pos_limits_lower, var_3, var_6, var_10, adj_joint_pos_limits_lower, adj_3, adj_6, adj_10);
        wp::adj_extract(var_11, var_9, adj_8, adj_9, adj_10);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_8);
        // adj: joint_pos_limits_lower[env_id, joint_id] = in_data[i, j][0]                       <L 38>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 37>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 36>
        // adj: i, j = wp.tid()                                                                   <L 35>
        // adj: def write_joint_limit_data_to_buffer_index(                                       <L 1>
        continue;
    }
}



extern "C" __global__ void concat_joint_pos_limits_lower_and_upper_47a833bf_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_joint_pos_limits)
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
        wp::float32* var_2;
        wp::float32* var_3;
        wp::vec_t<2, wp::float32> var_4;
        wp::float32 var_5;
        wp::float32 var_6;
        //---------
        // forward
        // def concat_joint_pos_limits_lower_and_upper(                                           <L 663>
        // i, j = wp.tid()                                                                        <L 675>
        builtin_tid2d(var_0, var_1);
        // joint_pos_limits[i, j] = wp.vec2f(joint_pos_limits_lower[i, j], joint_pos_limits_upper[i, j])       <L 676>
        var_2 = wp::address(var_joint_pos_limits_lower, var_0, var_1);
        var_3 = wp::address(var_joint_pos_limits_upper, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = wp::vec_t<2, wp::float32>(var_5, var_6);
        wp::array_store(var_joint_pos_limits, var_0, var_1, var_4);
    }
}



extern "C" __global__ void concat_joint_pos_limits_lower_and_upper_47a833bf_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_joint_pos_limits,
    wp::array_t<wp::float32> adj_joint_pos_limits_lower,
    wp::array_t<wp::float32> adj_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_joint_pos_limits)
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
        wp::float32* var_2;
        wp::float32* var_3;
        wp::vec_t<2, wp::float32> var_4;
        wp::float32 var_5;
        wp::float32 var_6;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::float32 adj_2 = {};
        wp::float32 adj_3 = {};
        wp::vec_t<2, wp::float32> adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        //---------
        // forward
        // def concat_joint_pos_limits_lower_and_upper(                                           <L 663>
        // i, j = wp.tid()                                                                        <L 675>
        builtin_tid2d(var_0, var_1);
        // joint_pos_limits[i, j] = wp.vec2f(joint_pos_limits_lower[i, j], joint_pos_limits_upper[i, j])       <L 676>
        var_2 = wp::address(var_joint_pos_limits_lower, var_0, var_1);
        var_3 = wp::address(var_joint_pos_limits_upper, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = wp::vec_t<2, wp::float32>(var_5, var_6);
        // wp::array_store(var_joint_pos_limits, var_0, var_1, var_4);
        //---------
        // reverse
        wp::adj_array_store(var_joint_pos_limits, var_0, var_1, var_4, adj_joint_pos_limits, adj_0, adj_1, adj_4);
        wp::adj_vec_t(var_5, var_6, adj_2, adj_3, adj_4);
        wp::adj_address(var_joint_pos_limits_upper, var_0, var_1, adj_joint_pos_limits_upper, adj_0, adj_1, adj_3);
        wp::adj_address(var_joint_pos_limits_lower, var_0, var_1, adj_joint_pos_limits_lower, adj_0, adj_1, adj_2);
        // adj: joint_pos_limits[i, j] = wp.vec2f(joint_pos_limits_lower[i, j], joint_pos_limits_upper[i, j])  <L 676>
        // adj: i, j = wp.tid()                                                                   <L 675>
        // adj: def concat_joint_pos_limits_lower_and_upper(                                      <L 663>
        continue;
    }
}



extern "C" __global__ void write_joint_friction_data_to_buffer_569763d2_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_friction,
    wp::array_t<wp::float32> var_in_dynamic_friction,
    wp::array_t<wp::float32> var_in_viscous_friction,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    bool var_from_mask,
    wp::array_t<wp::float32> var_out_friction,
    wp::array_t<wp::float32> var_out_dynamic_friction,
    wp::array_t<wp::float32> var_out_viscous_friction,
    wp::array_t<wp::float32> var_friction_props)
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
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32* var_16;
        wp::float32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32* var_20;
        const wp::int32 var_21 = 0;
        wp::float32 var_22;
        wp::float32* var_23;
        const wp::int32 var_24 = 1;
        wp::float32 var_25;
        wp::float32* var_26;
        const wp::int32 var_27 = 2;
        wp::float32 var_28;
        //---------
        // forward
        // def write_joint_friction_data_to_buffer(                                               <L 1>
        // i, j = wp.tid()                                                                        <L 43>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 44>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 45>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 47>
        if (var_from_mask) {
            // out_friction[env_id, joint_id] = in_friction[env_id, joint_id]                     <L 48>
            var_8 = wp::address(var_in_friction, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_out_friction, var_3, var_6, var_9);
            // if in_dynamic_friction:                                                            <L 49>
            if (var_in_dynamic_friction) {
                // out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[env_id, joint_id]       <L 50>
                var_10 = wp::address(var_in_dynamic_friction, var_3, var_6);
                var_11 = wp::load(var_10);
                wp::array_store(var_out_dynamic_friction, var_3, var_6, var_11);
            }
            // if in_viscous_friction:                                                            <L 51>
            if (var_in_viscous_friction) {
                // out_viscous_friction[env_id, joint_id] = in_viscous_friction[env_id, joint_id]       <L 52>
                var_12 = wp::address(var_in_viscous_friction, var_3, var_6);
                var_13 = wp::load(var_12);
                wp::array_store(var_out_viscous_friction, var_3, var_6, var_13);
            }
        }
        if (!var_from_mask) {
            // out_friction[env_id, joint_id] = in_friction[i, j]                                 <L 54>
            var_14 = wp::address(var_in_friction, var_0, var_1);
            var_15 = wp::load(var_14);
            wp::array_store(var_out_friction, var_3, var_6, var_15);
            // if in_dynamic_friction:                                                            <L 55>
            if (var_in_dynamic_friction) {
                // out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[i, j]             <L 56>
                var_16 = wp::address(var_in_dynamic_friction, var_0, var_1);
                var_17 = wp::load(var_16);
                wp::array_store(var_out_dynamic_friction, var_3, var_6, var_17);
            }
            // if in_viscous_friction:                                                            <L 57>
            if (var_in_viscous_friction) {
                // out_viscous_friction[env_id, joint_id] = in_viscous_friction[i, j]             <L 58>
                var_18 = wp::address(var_in_viscous_friction, var_0, var_1);
                var_19 = wp::load(var_18);
                wp::array_store(var_out_viscous_friction, var_3, var_6, var_19);
            }
        }
        // friction_props[env_id, joint_id, 0] = out_friction[env_id, joint_id]                   <L 60>
        var_20 = wp::address(var_out_friction, var_3, var_6);
        var_22 = wp::load(var_20);
        wp::array_store(var_friction_props, var_3, var_6, var_21, var_22);
        // if in_dynamic_friction:                                                                <L 61>
        if (var_in_dynamic_friction) {
            // friction_props[env_id, joint_id, 1] = out_dynamic_friction[env_id, joint_id]       <L 62>
            var_23 = wp::address(var_out_dynamic_friction, var_3, var_6);
            var_25 = wp::load(var_23);
            wp::array_store(var_friction_props, var_3, var_6, var_24, var_25);
        }
        // if in_viscous_friction:                                                                <L 63>
        if (var_in_viscous_friction) {
            // friction_props[env_id, joint_id, 2] = out_viscous_friction[env_id, joint_id]       <L 64>
            var_26 = wp::address(var_out_viscous_friction, var_3, var_6);
            var_28 = wp::load(var_26);
            wp::array_store(var_friction_props, var_3, var_6, var_27, var_28);
        }
    }
}



extern "C" __global__ void write_joint_friction_data_to_buffer_569763d2_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_friction,
    wp::array_t<wp::float32> var_in_dynamic_friction,
    wp::array_t<wp::float32> var_in_viscous_friction,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    bool var_from_mask,
    wp::array_t<wp::float32> var_out_friction,
    wp::array_t<wp::float32> var_out_dynamic_friction,
    wp::array_t<wp::float32> var_out_viscous_friction,
    wp::array_t<wp::float32> var_friction_props,
    wp::array_t<wp::float32> adj_in_friction,
    wp::array_t<wp::float32> adj_in_dynamic_friction,
    wp::array_t<wp::float32> adj_in_viscous_friction,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_joint_ids,
    bool adj_from_mask,
    wp::array_t<wp::float32> adj_out_friction,
    wp::array_t<wp::float32> adj_out_dynamic_friction,
    wp::array_t<wp::float32> adj_out_viscous_friction,
    wp::array_t<wp::float32> adj_friction_props)
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
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32* var_16;
        wp::float32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32* var_20;
        const wp::int32 var_21 = 0;
        wp::float32 var_22;
        wp::float32* var_23;
        const wp::int32 var_24 = 1;
        wp::float32 var_25;
        wp::float32* var_26;
        const wp::int32 var_27 = 2;
        wp::float32 var_28;
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
        wp::int32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::float32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::float32 adj_28 = {};
        //---------
        // forward
        // def write_joint_friction_data_to_buffer(                                               <L 1>
        // i, j = wp.tid()                                                                        <L 43>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 44>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 45>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 47>
        if (var_from_mask) {
            // out_friction[env_id, joint_id] = in_friction[env_id, joint_id]                     <L 48>
            var_8 = wp::address(var_in_friction, var_3, var_6);
            var_9 = wp::load(var_8);
            // wp::array_store(var_out_friction, var_3, var_6, var_9);
            // if in_dynamic_friction:                                                            <L 49>
            if (var_in_dynamic_friction) {
                // out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[env_id, joint_id]       <L 50>
                var_10 = wp::address(var_in_dynamic_friction, var_3, var_6);
                var_11 = wp::load(var_10);
                // wp::array_store(var_out_dynamic_friction, var_3, var_6, var_11);
            }
            // if in_viscous_friction:                                                            <L 51>
            if (var_in_viscous_friction) {
                // out_viscous_friction[env_id, joint_id] = in_viscous_friction[env_id, joint_id]       <L 52>
                var_12 = wp::address(var_in_viscous_friction, var_3, var_6);
                var_13 = wp::load(var_12);
                // wp::array_store(var_out_viscous_friction, var_3, var_6, var_13);
            }
        }
        if (!var_from_mask) {
            // out_friction[env_id, joint_id] = in_friction[i, j]                                 <L 54>
            var_14 = wp::address(var_in_friction, var_0, var_1);
            var_15 = wp::load(var_14);
            // wp::array_store(var_out_friction, var_3, var_6, var_15);
            // if in_dynamic_friction:                                                            <L 55>
            if (var_in_dynamic_friction) {
                // out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[i, j]             <L 56>
                var_16 = wp::address(var_in_dynamic_friction, var_0, var_1);
                var_17 = wp::load(var_16);
                // wp::array_store(var_out_dynamic_friction, var_3, var_6, var_17);
            }
            // if in_viscous_friction:                                                            <L 57>
            if (var_in_viscous_friction) {
                // out_viscous_friction[env_id, joint_id] = in_viscous_friction[i, j]             <L 58>
                var_18 = wp::address(var_in_viscous_friction, var_0, var_1);
                var_19 = wp::load(var_18);
                // wp::array_store(var_out_viscous_friction, var_3, var_6, var_19);
            }
        }
        // friction_props[env_id, joint_id, 0] = out_friction[env_id, joint_id]                   <L 60>
        var_20 = wp::address(var_out_friction, var_3, var_6);
        var_22 = wp::load(var_20);
        // wp::array_store(var_friction_props, var_3, var_6, var_21, var_22);
        // if in_dynamic_friction:                                                                <L 61>
        if (var_in_dynamic_friction) {
            // friction_props[env_id, joint_id, 1] = out_dynamic_friction[env_id, joint_id]       <L 62>
            var_23 = wp::address(var_out_dynamic_friction, var_3, var_6);
            var_25 = wp::load(var_23);
            // wp::array_store(var_friction_props, var_3, var_6, var_24, var_25);
        }
        // if in_viscous_friction:                                                                <L 63>
        if (var_in_viscous_friction) {
            // friction_props[env_id, joint_id, 2] = out_viscous_friction[env_id, joint_id]       <L 64>
            var_26 = wp::address(var_out_viscous_friction, var_3, var_6);
            var_28 = wp::load(var_26);
            // wp::array_store(var_friction_props, var_3, var_6, var_27, var_28);
        }
        //---------
        // reverse
        if (var_in_viscous_friction) {
            wp::adj_array_store(var_friction_props, var_3, var_6, var_27, var_28, adj_friction_props, adj_3, adj_6, adj_27, adj_26);
            wp::adj_address(var_out_viscous_friction, var_3, var_6, adj_out_viscous_friction, adj_3, adj_6, adj_26);
            // adj: friction_props[env_id, joint_id, 2] = out_viscous_friction[env_id, joint_id]  <L 64>
        }
        // adj: if in_viscous_friction:                                                           <L 63>
        if (var_in_dynamic_friction) {
            wp::adj_array_store(var_friction_props, var_3, var_6, var_24, var_25, adj_friction_props, adj_3, adj_6, adj_24, adj_23);
            wp::adj_address(var_out_dynamic_friction, var_3, var_6, adj_out_dynamic_friction, adj_3, adj_6, adj_23);
            // adj: friction_props[env_id, joint_id, 1] = out_dynamic_friction[env_id, joint_id]  <L 62>
        }
        // adj: if in_dynamic_friction:                                                           <L 61>
        wp::adj_array_store(var_friction_props, var_3, var_6, var_21, var_22, adj_friction_props, adj_3, adj_6, adj_21, adj_20);
        wp::adj_address(var_out_friction, var_3, var_6, adj_out_friction, adj_3, adj_6, adj_20);
        // adj: friction_props[env_id, joint_id, 0] = out_friction[env_id, joint_id]              <L 60>
        if (!var_from_mask) {
            if (var_in_viscous_friction) {
                wp::adj_array_store(var_out_viscous_friction, var_3, var_6, var_19, adj_out_viscous_friction, adj_3, adj_6, adj_18);
                wp::adj_address(var_in_viscous_friction, var_0, var_1, adj_in_viscous_friction, adj_0, adj_1, adj_18);
                // adj: out_viscous_friction[env_id, joint_id] = in_viscous_friction[i, j]        <L 58>
            }
            // adj: if in_viscous_friction:                                                       <L 57>
            if (var_in_dynamic_friction) {
                wp::adj_array_store(var_out_dynamic_friction, var_3, var_6, var_17, adj_out_dynamic_friction, adj_3, adj_6, adj_16);
                wp::adj_address(var_in_dynamic_friction, var_0, var_1, adj_in_dynamic_friction, adj_0, adj_1, adj_16);
                // adj: out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[i, j]        <L 56>
            }
            // adj: if in_dynamic_friction:                                                       <L 55>
            wp::adj_array_store(var_out_friction, var_3, var_6, var_15, adj_out_friction, adj_3, adj_6, adj_14);
            wp::adj_address(var_in_friction, var_0, var_1, adj_in_friction, adj_0, adj_1, adj_14);
            // adj: out_friction[env_id, joint_id] = in_friction[i, j]                            <L 54>
        }
        if (var_from_mask) {
            if (var_in_viscous_friction) {
                wp::adj_array_store(var_out_viscous_friction, var_3, var_6, var_13, adj_out_viscous_friction, adj_3, adj_6, adj_12);
                wp::adj_address(var_in_viscous_friction, var_3, var_6, adj_in_viscous_friction, adj_3, adj_6, adj_12);
                // adj: out_viscous_friction[env_id, joint_id] = in_viscous_friction[env_id, joint_id]  <L 52>
            }
            // adj: if in_viscous_friction:                                                       <L 51>
            if (var_in_dynamic_friction) {
                wp::adj_array_store(var_out_dynamic_friction, var_3, var_6, var_11, adj_out_dynamic_friction, adj_3, adj_6, adj_10);
                wp::adj_address(var_in_dynamic_friction, var_3, var_6, adj_in_dynamic_friction, adj_3, adj_6, adj_10);
                // adj: out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[env_id, joint_id]  <L 50>
            }
            // adj: if in_dynamic_friction:                                                       <L 49>
            wp::adj_array_store(var_out_friction, var_3, var_6, var_9, adj_out_friction, adj_3, adj_6, adj_8);
            wp::adj_address(var_in_friction, var_3, var_6, adj_in_friction, adj_3, adj_6, adj_8);
            // adj: out_friction[env_id, joint_id] = in_friction[env_id, joint_id]                <L 48>
        }
        // adj: if from_mask:                                                                     <L 47>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 45>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 44>
        // adj: i, j = wp.tid()                                                                   <L 43>
        // adj: def write_joint_friction_data_to_buffer(                                          <L 1>
        continue;
    }
}



extern "C" __global__ void update_wrench_array_with_force_and_torque_ordered_588d5ed8_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> var_torques,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w,
    wp::array_t<wp::int32> var_user_to_backend,
    wp::array_t<wp::vec_t<6, wp::float32>> var_wrench,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_body_mask)
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
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::transform_t<wp::float32>* var_10;
        wp::quat_t<wp::float32> var_11;
        wp::transform_t<wp::float32> var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32>* var_14;
        wp::vec_t<6, wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32> var_17;
        //---------
        // forward
        // def update_wrench_array_with_force_and_torque_ordered(                                 <L 53>
        // env_id, user_body_id = wp.tid()                                                        <L 83>
        builtin_tid2d(var_0, var_1);
        // if env_mask[env_id] and body_mask[user_body_id]:                                       <L 84>
        var_3 = wp::address(var_env_mask, var_0);
        var_4 = wp::load(var_3);
        var_2 = var_4;
        if (var_2) {
            var_5 = wp::address(var_body_mask, var_1);
            var_6 = wp::load(var_5);
            var_2 = var_2 && var_6;
        }
        if (var_2) {
            // backend_body_id = user_to_backend[user_body_id]                                    <L 85>
            var_7 = wp::address(var_user_to_backend, var_1);
            var_9 = wp::load(var_7);
            var_8 = wp::copy(var_9);
            // body_rot_w = wp.transform_get_rotation(body_link_pose_w[env_id, user_body_id])       <L 86>
            var_10 = wp::address(var_body_link_pose_w, var_0, var_1);
            var_12 = wp::load(var_10);
            var_11 = wp::transform_get_rotation(var_12);
            // wrench[env_id, backend_body_id] = pack_body_wrench_to_world(                       <L 87>
            // forces[env_id, user_body_id],                                                      <L 88>
            var_13 = wp::address(var_forces, var_0, var_1);
            // torques[env_id, user_body_id],                                                     <L 89>
            var_14 = wp::address(var_torques, var_0, var_1);
            // body_rot_w,                                                                        <L 90>
            var_16 = wp::load(var_13);
            var_17 = wp::load(var_14);
            var_15 = pack_body_wrench_to_world_0(var_16, var_17, var_11);
            // wrench[env_id, backend_body_id] = pack_body_wrench_to_world(                       <L 87>
            wp::array_store(var_wrench, var_0, var_8, var_15);
        }
    }
}



extern "C" __global__ void update_wrench_array_with_force_and_torque_ordered_588d5ed8_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> var_torques,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w,
    wp::array_t<wp::int32> var_user_to_backend,
    wp::array_t<wp::vec_t<6, wp::float32>> var_wrench,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_body_mask,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_torques,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_link_pose_w,
    wp::array_t<wp::int32> adj_user_to_backend,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_wrench,
    wp::array_t<bool> adj_env_mask,
    wp::array_t<bool> adj_body_mask)
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
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::transform_t<wp::float32>* var_10;
        wp::quat_t<wp::float32> var_11;
        wp::transform_t<wp::float32> var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32>* var_14;
        wp::vec_t<6, wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32> var_17;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        bool adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::transform_t<wp::float32> adj_10 = {};
        wp::quat_t<wp::float32> adj_11 = {};
        wp::transform_t<wp::float32> adj_12 = {};
        wp::vec_t<3, wp::float32> adj_13 = {};
        wp::vec_t<3, wp::float32> adj_14 = {};
        wp::vec_t<6, wp::float32> adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::vec_t<3, wp::float32> adj_17 = {};
        //---------
        // forward
        // def update_wrench_array_with_force_and_torque_ordered(                                 <L 53>
        // env_id, user_body_id = wp.tid()                                                        <L 83>
        builtin_tid2d(var_0, var_1);
        // if env_mask[env_id] and body_mask[user_body_id]:                                       <L 84>
        var_3 = wp::address(var_env_mask, var_0);
        var_4 = wp::load(var_3);
        var_2 = var_4;
        if (var_2) {
            var_5 = wp::address(var_body_mask, var_1);
            var_6 = wp::load(var_5);
            var_2 = var_2 && var_6;
        }
        if (var_2) {
            // backend_body_id = user_to_backend[user_body_id]                                    <L 85>
            var_7 = wp::address(var_user_to_backend, var_1);
            var_9 = wp::load(var_7);
            var_8 = wp::copy(var_9);
            // body_rot_w = wp.transform_get_rotation(body_link_pose_w[env_id, user_body_id])       <L 86>
            var_10 = wp::address(var_body_link_pose_w, var_0, var_1);
            var_12 = wp::load(var_10);
            var_11 = wp::transform_get_rotation(var_12);
            // wrench[env_id, backend_body_id] = pack_body_wrench_to_world(                       <L 87>
            // forces[env_id, user_body_id],                                                      <L 88>
            var_13 = wp::address(var_forces, var_0, var_1);
            // torques[env_id, user_body_id],                                                     <L 89>
            var_14 = wp::address(var_torques, var_0, var_1);
            // body_rot_w,                                                                        <L 90>
            var_16 = wp::load(var_13);
            var_17 = wp::load(var_14);
            var_15 = pack_body_wrench_to_world_0(var_16, var_17, var_11);
            // wrench[env_id, backend_body_id] = pack_body_wrench_to_world(                       <L 87>
            // wp::array_store(var_wrench, var_0, var_8, var_15);
        }
        //---------
        // reverse
        if (var_2) {
            wp::adj_array_store(var_wrench, var_0, var_8, var_15, adj_wrench, adj_0, adj_8, adj_15);
            // adj: wrench[env_id, backend_body_id] = pack_body_wrench_to_world(                  <L 87>
            adj_pack_body_wrench_to_world_0(var_16, var_17, var_11, adj_13, adj_14, adj_11, adj_15);
            // adj: body_rot_w,                                                                   <L 90>
            wp::adj_address(var_torques, var_0, var_1, adj_torques, adj_0, adj_1, adj_14);
            // adj: torques[env_id, user_body_id],                                                <L 89>
            wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_13);
            // adj: forces[env_id, user_body_id],                                                 <L 88>
            // adj: wrench[env_id, backend_body_id] = pack_body_wrench_to_world(                  <L 87>
            wp::adj_transform_get_rotation(var_12, adj_10, adj_11);
            wp::adj_address(var_body_link_pose_w, var_0, var_1, adj_body_link_pose_w, adj_0, adj_1, adj_10);
            // adj: body_rot_w = wp.transform_get_rotation(body_link_pose_w[env_id, user_body_id])  <L 86>
            wp::adj_copy(var_9, adj_7, adj_8);
            wp::adj_address(var_user_to_backend, var_1, adj_user_to_backend, adj_1, adj_7);
            // adj: backend_body_id = user_to_backend[user_body_id]                               <L 85>
        }
        if (var_2) {
            wp::adj_address(var_body_mask, var_1, adj_body_mask, adj_1, adj_5);
        }
        wp::adj_address(var_env_mask, var_0, adj_env_mask, adj_0, adj_3);
        // adj: if env_mask[env_id] and body_mask[user_body_id]:                                  <L 84>
        // adj: env_id, user_body_id = wp.tid()                                                   <L 83>
        // adj: def update_wrench_array_with_force_and_torque_ordered(                            <L 53>
        continue;
    }
}



extern "C" __global__ void write_joint_friction_param_to_buffer_58c991d6_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    wp::int32 var_buffer_index,
    bool var_from_mask,
    wp::array_t<wp::float32> var_out_data,
    wp::array_t<wp::float32> var_out_buffer)
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
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        //---------
        // forward
        // def write_joint_friction_param_to_buffer(                                              <L 1>
        // i, j = wp.tid()                                                                        <L 31>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 32>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 33>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 34>
        if (var_from_mask) {
            // out_data[env_id, joint_id] = in_data[env_id, joint_id]                             <L 35>
            var_8 = wp::address(var_in_data, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_out_data, var_3, var_6, var_9);
            // out_buffer[env_id, joint_id, buffer_index] = in_data[env_id, joint_id]             <L 36>
            var_10 = wp::address(var_in_data, var_3, var_6);
            var_11 = wp::load(var_10);
            wp::array_store(var_out_buffer, var_3, var_6, var_buffer_index, var_11);
        }
        if (!var_from_mask) {
            // out_data[env_id, joint_id] = in_data[i, j]                                         <L 38>
            var_12 = wp::address(var_in_data, var_0, var_1);
            var_13 = wp::load(var_12);
            wp::array_store(var_out_data, var_3, var_6, var_13);
            // out_buffer[env_id, joint_id, buffer_index] = in_data[i, j]                         <L 39>
            var_14 = wp::address(var_in_data, var_0, var_1);
            var_15 = wp::load(var_14);
            wp::array_store(var_out_buffer, var_3, var_6, var_buffer_index, var_15);
        }
    }
}



extern "C" __global__ void write_joint_friction_param_to_buffer_58c991d6_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    wp::int32 var_buffer_index,
    bool var_from_mask,
    wp::array_t<wp::float32> var_out_data,
    wp::array_t<wp::float32> var_out_buffer,
    wp::array_t<wp::float32> adj_in_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int64> adj_joint_ids,
    wp::int32 adj_buffer_index,
    bool adj_from_mask,
    wp::array_t<wp::float32> adj_out_data,
    wp::array_t<wp::float32> adj_out_buffer)
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
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
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
        //---------
        // forward
        // def write_joint_friction_param_to_buffer(                                              <L 1>
        // i, j = wp.tid()                                                                        <L 31>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 32>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 33>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 34>
        if (var_from_mask) {
            // out_data[env_id, joint_id] = in_data[env_id, joint_id]                             <L 35>
            var_8 = wp::address(var_in_data, var_3, var_6);
            var_9 = wp::load(var_8);
            // wp::array_store(var_out_data, var_3, var_6, var_9);
            // out_buffer[env_id, joint_id, buffer_index] = in_data[env_id, joint_id]             <L 36>
            var_10 = wp::address(var_in_data, var_3, var_6);
            var_11 = wp::load(var_10);
            // wp::array_store(var_out_buffer, var_3, var_6, var_buffer_index, var_11);
        }
        if (!var_from_mask) {
            // out_data[env_id, joint_id] = in_data[i, j]                                         <L 38>
            var_12 = wp::address(var_in_data, var_0, var_1);
            var_13 = wp::load(var_12);
            // wp::array_store(var_out_data, var_3, var_6, var_13);
            // out_buffer[env_id, joint_id, buffer_index] = in_data[i, j]                         <L 39>
            var_14 = wp::address(var_in_data, var_0, var_1);
            var_15 = wp::load(var_14);
            // wp::array_store(var_out_buffer, var_3, var_6, var_buffer_index, var_15);
        }
        //---------
        // reverse
        if (!var_from_mask) {
            wp::adj_array_store(var_out_buffer, var_3, var_6, var_buffer_index, var_15, adj_out_buffer, adj_3, adj_6, adj_buffer_index, adj_14);
            wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_14);
            // adj: out_buffer[env_id, joint_id, buffer_index] = in_data[i, j]                    <L 39>
            wp::adj_array_store(var_out_data, var_3, var_6, var_13, adj_out_data, adj_3, adj_6, adj_12);
            wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_12);
            // adj: out_data[env_id, joint_id] = in_data[i, j]                                    <L 38>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_out_buffer, var_3, var_6, var_buffer_index, var_11, adj_out_buffer, adj_3, adj_6, adj_buffer_index, adj_10);
            wp::adj_address(var_in_data, var_3, var_6, adj_in_data, adj_3, adj_6, adj_10);
            // adj: out_buffer[env_id, joint_id, buffer_index] = in_data[env_id, joint_id]        <L 36>
            wp::adj_array_store(var_out_data, var_3, var_6, var_9, adj_out_data, adj_3, adj_6, adj_8);
            wp::adj_address(var_in_data, var_3, var_6, adj_in_data, adj_3, adj_6, adj_8);
            // adj: out_data[env_id, joint_id] = in_data[env_id, joint_id]                        <L 35>
        }
        // adj: if from_mask:                                                                     <L 34>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 33>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 32>
        // adj: i, j = wp.tid()                                                                   <L 31>
        // adj: def write_joint_friction_param_to_buffer(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void get_body_com_acc_from_body_com_vel_ordered_5ec3c72d_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_vel_backend,
    wp::array_t<wp::vec_t<6, wp::float32>> var_prev_body_com_vel_backend,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::float32 var_dt,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_acc_user)
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
        wp::vec_t<6, wp::float32>* var_3;
        wp::vec_t<6, wp::float32> var_4;
        wp::vec_t<6, wp::float32> var_5;
        wp::vec_t<6, wp::float32>* var_6;
        wp::vec_t<6, wp::float32> var_7;
        wp::vec_t<6, wp::float32> var_8;
        wp::vec_t<6, wp::float32> var_9;
        //---------
        // forward
        // def get_body_com_acc_from_body_com_vel_ordered(                                        <L 120>
        // env_id, user_body_id = wp.tid()                                                        <L 143>
        builtin_tid2d(var_0, var_1);
        // backend_body_id = resolve_backend_index(user_body_id, user_to_backend, has_ordering)       <L 144>
        var_2 = resolve_backend_index_0(var_1, var_user_to_backend, var_has_ordering);
        // body_com_vel = body_com_vel_backend[env_id, backend_body_id]                           <L 145>
        var_3 = wp::address(var_body_com_vel_backend, var_0, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // body_com_acc_user[env_id, user_body_id] = (body_com_vel - prev_body_com_vel_backend[env_id, backend_body_id]) / dt       <L 146>
        var_6 = wp::address(var_prev_body_com_vel_backend, var_0, var_2);
        var_8 = wp::load(var_6);
        var_7 = wp::sub(var_4, var_8);
        var_9 = wp::div(var_7, var_dt);
        wp::array_store(var_body_com_acc_user, var_0, var_1, var_9);
        // prev_body_com_vel_backend[env_id, backend_body_id] = body_com_vel                      <L 147>
        wp::array_store(var_prev_body_com_vel_backend, var_0, var_2, var_4);
    }
}



extern "C" __global__ void get_body_com_acc_from_body_com_vel_ordered_5ec3c72d_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_vel_backend,
    wp::array_t<wp::vec_t<6, wp::float32>> var_prev_body_com_vel_backend,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::float32 var_dt,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_acc_user,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_com_vel_backend,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_prev_body_com_vel_backend,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    wp::float32 adj_dt,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_com_acc_user)
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
        wp::vec_t<6, wp::float32>* var_3;
        wp::vec_t<6, wp::float32> var_4;
        wp::vec_t<6, wp::float32> var_5;
        wp::vec_t<6, wp::float32>* var_6;
        wp::vec_t<6, wp::float32> var_7;
        wp::vec_t<6, wp::float32> var_8;
        wp::vec_t<6, wp::float32> var_9;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::vec_t<6, wp::float32> adj_3 = {};
        wp::vec_t<6, wp::float32> adj_4 = {};
        wp::vec_t<6, wp::float32> adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        wp::vec_t<6, wp::float32> adj_7 = {};
        wp::vec_t<6, wp::float32> adj_8 = {};
        wp::vec_t<6, wp::float32> adj_9 = {};
        //---------
        // forward
        // def get_body_com_acc_from_body_com_vel_ordered(                                        <L 120>
        // env_id, user_body_id = wp.tid()                                                        <L 143>
        builtin_tid2d(var_0, var_1);
        // backend_body_id = resolve_backend_index(user_body_id, user_to_backend, has_ordering)       <L 144>
        var_2 = resolve_backend_index_0(var_1, var_user_to_backend, var_has_ordering);
        // body_com_vel = body_com_vel_backend[env_id, backend_body_id]                           <L 145>
        var_3 = wp::address(var_body_com_vel_backend, var_0, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // body_com_acc_user[env_id, user_body_id] = (body_com_vel - prev_body_com_vel_backend[env_id, backend_body_id]) / dt       <L 146>
        var_6 = wp::address(var_prev_body_com_vel_backend, var_0, var_2);
        var_8 = wp::load(var_6);
        var_7 = wp::sub(var_4, var_8);
        var_9 = wp::div(var_7, var_dt);
        // wp::array_store(var_body_com_acc_user, var_0, var_1, var_9);
        // prev_body_com_vel_backend[env_id, backend_body_id] = body_com_vel                      <L 147>
        // wp::array_store(var_prev_body_com_vel_backend, var_0, var_2, var_4);
        //---------
        // reverse
        wp::adj_array_store(var_prev_body_com_vel_backend, var_0, var_2, var_4, adj_prev_body_com_vel_backend, adj_0, adj_2, adj_4);
        // adj: prev_body_com_vel_backend[env_id, backend_body_id] = body_com_vel                 <L 147>
        wp::adj_array_store(var_body_com_acc_user, var_0, var_1, var_9, adj_body_com_acc_user, adj_0, adj_1, adj_9);
        wp::adj_div(var_7, var_dt, adj_7, adj_dt, adj_9);
        wp::adj_sub(var_4, var_8, adj_4, adj_6, adj_7);
        wp::adj_address(var_prev_body_com_vel_backend, var_0, var_2, adj_prev_body_com_vel_backend, adj_0, adj_2, adj_6);
        // adj: body_com_acc_user[env_id, user_body_id] = (body_com_vel - prev_body_com_vel_backend[env_id, backend_body_id]) / dt  <L 146>
        wp::adj_copy(var_5, adj_3, adj_4);
        wp::adj_address(var_body_com_vel_backend, var_0, var_2, adj_body_com_vel_backend, adj_0, adj_2, adj_3);
        // adj: body_com_vel = body_com_vel_backend[env_id, backend_body_id]                      <L 145>
        adj_resolve_backend_index_0(var_1, var_user_to_backend, var_has_ordering, adj_1, adj_user_to_backend, adj_has_ordering, adj_2);
        // adj: backend_body_id = resolve_backend_index(user_body_id, user_to_backend, has_ordering)  <L 144>
        // adj: env_id, user_body_id = wp.tid()                                                   <L 143>
        // adj: def get_body_com_acc_from_body_com_vel_ordered(                                   <L 120>
        continue;
    }
}



extern "C" __global__ void write_joint_state_data_index_64a63208_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_pos_data,
    wp::array_t<wp::float32> var_vel_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    wp::array_t<wp::float32> var_joint_pos,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::array_t<wp::float32> var_joint_acc)
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
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        const wp::float32 var_14 = 0.0;
        //---------
        // forward
        // def write_joint_state_data_index(                                                      <L 1>
        // i, j = wp.tid()                                                                        <L 26>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 27>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 28>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // joint_pos[env_id, joint_id] = pos_data[i, j]                                           <L 29>
        var_8 = wp::address(var_pos_data, var_0, var_1);
        var_9 = wp::load(var_8);
        wp::array_store(var_joint_pos, var_3, var_6, var_9);
        // joint_vel[env_id, joint_id] = vel_data[i, j]                                           <L 30>
        var_10 = wp::address(var_vel_data, var_0, var_1);
        var_11 = wp::load(var_10);
        wp::array_store(var_joint_vel, var_3, var_6, var_11);
        // prev_joint_vel[env_id, joint_id] = vel_data[i, j]                                      <L 31>
        var_12 = wp::address(var_vel_data, var_0, var_1);
        var_13 = wp::load(var_12);
        wp::array_store(var_prev_joint_vel, var_3, var_6, var_13);
        // joint_acc[env_id, joint_id] = 0.0                                                      <L 32>
        wp::array_store(var_joint_acc, var_3, var_6, var_14);
    }
}



extern "C" __global__ void write_joint_state_data_index_64a63208_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_pos_data,
    wp::array_t<wp::float32> var_vel_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    wp::array_t<wp::float32> var_joint_pos,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::array_t<wp::float32> var_joint_acc,
    wp::array_t<wp::float32> adj_pos_data,
    wp::array_t<wp::float32> adj_vel_data,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int64> adj_joint_ids,
    wp::array_t<wp::float32> adj_joint_pos,
    wp::array_t<wp::float32> adj_joint_vel,
    wp::array_t<wp::float32> adj_prev_joint_vel,
    wp::array_t<wp::float32> adj_joint_acc)
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
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        const wp::float32 var_14 = 0.0;
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
        //---------
        // forward
        // def write_joint_state_data_index(                                                      <L 1>
        // i, j = wp.tid()                                                                        <L 26>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 27>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 28>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // joint_pos[env_id, joint_id] = pos_data[i, j]                                           <L 29>
        var_8 = wp::address(var_pos_data, var_0, var_1);
        var_9 = wp::load(var_8);
        // wp::array_store(var_joint_pos, var_3, var_6, var_9);
        // joint_vel[env_id, joint_id] = vel_data[i, j]                                           <L 30>
        var_10 = wp::address(var_vel_data, var_0, var_1);
        var_11 = wp::load(var_10);
        // wp::array_store(var_joint_vel, var_3, var_6, var_11);
        // prev_joint_vel[env_id, joint_id] = vel_data[i, j]                                      <L 31>
        var_12 = wp::address(var_vel_data, var_0, var_1);
        var_13 = wp::load(var_12);
        // wp::array_store(var_prev_joint_vel, var_3, var_6, var_13);
        // joint_acc[env_id, joint_id] = 0.0                                                      <L 32>
        // wp::array_store(var_joint_acc, var_3, var_6, var_14);
        //---------
        // reverse
        wp::adj_array_store(var_joint_acc, var_3, var_6, var_14, adj_joint_acc, adj_3, adj_6, adj_14);
        // adj: joint_acc[env_id, joint_id] = 0.0                                                 <L 32>
        wp::adj_array_store(var_prev_joint_vel, var_3, var_6, var_13, adj_prev_joint_vel, adj_3, adj_6, adj_12);
        wp::adj_address(var_vel_data, var_0, var_1, adj_vel_data, adj_0, adj_1, adj_12);
        // adj: prev_joint_vel[env_id, joint_id] = vel_data[i, j]                                 <L 31>
        wp::adj_array_store(var_joint_vel, var_3, var_6, var_11, adj_joint_vel, adj_3, adj_6, adj_10);
        wp::adj_address(var_vel_data, var_0, var_1, adj_vel_data, adj_0, adj_1, adj_10);
        // adj: joint_vel[env_id, joint_id] = vel_data[i, j]                                      <L 30>
        wp::adj_array_store(var_joint_pos, var_3, var_6, var_9, adj_joint_pos, adj_3, adj_6, adj_8);
        wp::adj_address(var_pos_data, var_0, var_1, adj_pos_data, adj_0, adj_1, adj_8);
        // adj: joint_pos[env_id, joint_id] = pos_data[i, j]                                      <L 29>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 28>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 27>
        // adj: i, j = wp.tid()                                                                   <L 26>
        // adj: def write_joint_state_data_index(                                                 <L 1>
        continue;
    }
}



extern "C" __global__ void write_joint_vel_data_mask_6cd85155_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_joint_mask,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::array_t<wp::float32> var_joint_acc)
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
        wp::float32* var_9;
        wp::float32 var_10;
        const wp::float32 var_11 = 0.0;
        //---------
        // forward
        // def write_joint_vel_data_mask(                                                         <L 953>
        // i, j = wp.tid()                                                                        <L 979>
        builtin_tid2d(var_0, var_1);
        // if env_mask[i] and joint_mask[j]:                                                      <L 980>
        var_3 = wp::address(var_env_mask, var_0);
        var_4 = wp::load(var_3);
        var_2 = var_4;
        if (var_2) {
            var_5 = wp::address(var_joint_mask, var_1);
            var_6 = wp::load(var_5);
            var_2 = var_2 && var_6;
        }
        if (var_2) {
            // joint_vel[i, j] = in_data[i, j]                                                    <L 981>
            var_7 = wp::address(var_in_data, var_0, var_1);
            var_8 = wp::load(var_7);
            wp::array_store(var_joint_vel, var_0, var_1, var_8);
            // prev_joint_vel[i, j] = in_data[i, j]                                               <L 982>
            var_9 = wp::address(var_in_data, var_0, var_1);
            var_10 = wp::load(var_9);
            wp::array_store(var_prev_joint_vel, var_0, var_1, var_10);
            // joint_acc[i, j] = 0.0                                                              <L 983>
            wp::array_store(var_joint_acc, var_0, var_1, var_11);
        }
    }
}



extern "C" __global__ void write_joint_vel_data_mask_6cd85155_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_joint_mask,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::array_t<wp::float32> var_joint_acc,
    wp::array_t<wp::float32> adj_in_data,
    wp::array_t<bool> adj_env_mask,
    wp::array_t<bool> adj_joint_mask,
    wp::array_t<wp::float32> adj_joint_vel,
    wp::array_t<wp::float32> adj_prev_joint_vel,
    wp::array_t<wp::float32> adj_joint_acc)
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
        wp::float32* var_9;
        wp::float32 var_10;
        const wp::float32 var_11 = 0.0;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        bool adj_6 = {};
        wp::float32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        //---------
        // forward
        // def write_joint_vel_data_mask(                                                         <L 953>
        // i, j = wp.tid()                                                                        <L 979>
        builtin_tid2d(var_0, var_1);
        // if env_mask[i] and joint_mask[j]:                                                      <L 980>
        var_3 = wp::address(var_env_mask, var_0);
        var_4 = wp::load(var_3);
        var_2 = var_4;
        if (var_2) {
            var_5 = wp::address(var_joint_mask, var_1);
            var_6 = wp::load(var_5);
            var_2 = var_2 && var_6;
        }
        if (var_2) {
            // joint_vel[i, j] = in_data[i, j]                                                    <L 981>
            var_7 = wp::address(var_in_data, var_0, var_1);
            var_8 = wp::load(var_7);
            // wp::array_store(var_joint_vel, var_0, var_1, var_8);
            // prev_joint_vel[i, j] = in_data[i, j]                                               <L 982>
            var_9 = wp::address(var_in_data, var_0, var_1);
            var_10 = wp::load(var_9);
            // wp::array_store(var_prev_joint_vel, var_0, var_1, var_10);
            // joint_acc[i, j] = 0.0                                                              <L 983>
            // wp::array_store(var_joint_acc, var_0, var_1, var_11);
        }
        //---------
        // reverse
        if (var_2) {
            wp::adj_array_store(var_joint_acc, var_0, var_1, var_11, adj_joint_acc, adj_0, adj_1, adj_11);
            // adj: joint_acc[i, j] = 0.0                                                         <L 983>
            wp::adj_array_store(var_prev_joint_vel, var_0, var_1, var_10, adj_prev_joint_vel, adj_0, adj_1, adj_9);
            wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_9);
            // adj: prev_joint_vel[i, j] = in_data[i, j]                                          <L 982>
            wp::adj_array_store(var_joint_vel, var_0, var_1, var_8, adj_joint_vel, adj_0, adj_1, adj_7);
            wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_7);
            // adj: joint_vel[i, j] = in_data[i, j]                                               <L 981>
        }
        if (var_2) {
            wp::adj_address(var_joint_mask, var_1, adj_joint_mask, adj_1, adj_5);
        }
        wp::adj_address(var_env_mask, var_0, adj_env_mask, adj_0, adj_3);
        // adj: if env_mask[i] and joint_mask[j]:                                                 <L 980>
        // adj: i, j = wp.tid()                                                                   <L 979>
        // adj: def write_joint_vel_data_mask(                                                    <L 953>
        continue;
    }
}



extern "C" __global__ void shift_jacobian_com_to_origin_6fa687e1_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::int32 var_link_offset,
    wp::array_t<wp::float32> var_src,
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
        wp::transform_t<wp::float32>* var_4;
        wp::quat_t<wp::float32> var_5;
        wp::transform_t<wp::float32> var_6;
        wp::vec_t<3, wp::float32>* var_7;
        wp::vec_t<3, wp::float32> var_8;
        wp::vec_t<3, wp::float32> var_9;
        const wp::int32 var_10 = 0;
        wp::float32* var_11;
        const wp::int32 var_12 = 1;
        wp::float32* var_13;
        const wp::int32 var_14 = 2;
        wp::float32* var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        const wp::int32 var_20 = 3;
        wp::float32* var_21;
        const wp::int32 var_22 = 4;
        wp::float32* var_23;
        const wp::int32 var_24 = 5;
        wp::float32* var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32> var_31;
        const wp::int32 var_32 = 0;
        wp::float32 var_33;
        const wp::int32 var_34 = 0;
        const wp::int32 var_35 = 1;
        wp::float32 var_36;
        const wp::int32 var_37 = 1;
        const wp::int32 var_38 = 2;
        wp::float32 var_39;
        const wp::int32 var_40 = 2;
        const wp::int32 var_41 = 0;
        wp::float32 var_42;
        const wp::int32 var_43 = 3;
        const wp::int32 var_44 = 1;
        wp::float32 var_45;
        const wp::int32 var_46 = 4;
        const wp::int32 var_47 = 2;
        wp::float32 var_48;
        const wp::int32 var_49 = 5;
        //---------
        // forward
        // def shift_jacobian_com_to_origin(                                                      <L 839>
        // n, b, dof = wp.tid()                                                                   <L 886>
        builtin_tid3d(var_0, var_1, var_2);
        // full_body_idx = b + link_offset                                                        <L 887>
        var_3 = wp::add(var_1, var_link_offset);
        // R = wp.transform_get_rotation(body_link_pose[n, full_body_idx])                        <L 889>
        var_4 = wp::address(var_body_link_pose, var_0, var_3);
        var_6 = wp::load(var_4);
        var_5 = wp::transform_get_rotation(var_6);
        // c_world = wp.quat_rotate(R, body_com_pos_b[n, full_body_idx])                          <L 890>
        var_7 = wp::address(var_body_com_pos_b, var_0, var_3);
        var_9 = wp::load(var_7);
        var_8 = wp::quat_rotate(var_5, var_9);
        // v_com = wp.vec3(src[n, b, 0, dof], src[n, b, 1, dof], src[n, b, 2, dof])               <L 892>
        var_11 = wp::address(var_src, var_0, var_1, var_10, var_2);
        var_13 = wp::address(var_src, var_0, var_1, var_12, var_2);
        var_15 = wp::address(var_src, var_0, var_1, var_14, var_2);
        var_17 = wp::load(var_11);
        var_18 = wp::load(var_13);
        var_19 = wp::load(var_15);
        var_16 = wp::vec_t<3, wp::float32>(var_17, var_18, var_19);
        // omega = wp.vec3(src[n, b, 3, dof], src[n, b, 4, dof], src[n, b, 5, dof])               <L 893>
        var_21 = wp::address(var_src, var_0, var_1, var_20, var_2);
        var_23 = wp::address(var_src, var_0, var_1, var_22, var_2);
        var_25 = wp::address(var_src, var_0, var_1, var_24, var_2);
        var_27 = wp::load(var_21);
        var_28 = wp::load(var_23);
        var_29 = wp::load(var_25);
        var_26 = wp::vec_t<3, wp::float32>(var_27, var_28, var_29);
        // v_origin = v_com - wp.cross(omega, c_world)                                            <L 895>
        var_30 = wp::cross(var_26, var_8);
        var_31 = wp::sub(var_16, var_30);
        // dst[n, b, 0, dof] = v_origin[0]                                                        <L 897>
        var_33 = wp::extract(var_31, var_32);
        wp::array_store(var_dst, var_0, var_1, var_34, var_2, var_33);
        // dst[n, b, 1, dof] = v_origin[1]                                                        <L 898>
        var_36 = wp::extract(var_31, var_35);
        wp::array_store(var_dst, var_0, var_1, var_37, var_2, var_36);
        // dst[n, b, 2, dof] = v_origin[2]                                                        <L 899>
        var_39 = wp::extract(var_31, var_38);
        wp::array_store(var_dst, var_0, var_1, var_40, var_2, var_39);
        // dst[n, b, 3, dof] = omega[0]                                                           <L 900>
        var_42 = wp::extract(var_26, var_41);
        wp::array_store(var_dst, var_0, var_1, var_43, var_2, var_42);
        // dst[n, b, 4, dof] = omega[1]                                                           <L 901>
        var_45 = wp::extract(var_26, var_44);
        wp::array_store(var_dst, var_0, var_1, var_46, var_2, var_45);
        // dst[n, b, 5, dof] = omega[2]                                                           <L 902>
        var_48 = wp::extract(var_26, var_47);
        wp::array_store(var_dst, var_0, var_1, var_49, var_2, var_48);
    }
}



extern "C" __global__ void shift_jacobian_com_to_origin_6fa687e1_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::int32 var_link_offset,
    wp::array_t<wp::float32> var_src,
    wp::array_t<wp::float32> var_dst,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_link_pose,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_body_com_pos_b,
    wp::int32 adj_link_offset,
    wp::array_t<wp::float32> adj_src,
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
        wp::transform_t<wp::float32>* var_4;
        wp::quat_t<wp::float32> var_5;
        wp::transform_t<wp::float32> var_6;
        wp::vec_t<3, wp::float32>* var_7;
        wp::vec_t<3, wp::float32> var_8;
        wp::vec_t<3, wp::float32> var_9;
        const wp::int32 var_10 = 0;
        wp::float32* var_11;
        const wp::int32 var_12 = 1;
        wp::float32* var_13;
        const wp::int32 var_14 = 2;
        wp::float32* var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        const wp::int32 var_20 = 3;
        wp::float32* var_21;
        const wp::int32 var_22 = 4;
        wp::float32* var_23;
        const wp::int32 var_24 = 5;
        wp::float32* var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32> var_31;
        const wp::int32 var_32 = 0;
        wp::float32 var_33;
        const wp::int32 var_34 = 0;
        const wp::int32 var_35 = 1;
        wp::float32 var_36;
        const wp::int32 var_37 = 1;
        const wp::int32 var_38 = 2;
        wp::float32 var_39;
        const wp::int32 var_40 = 2;
        const wp::int32 var_41 = 0;
        wp::float32 var_42;
        const wp::int32 var_43 = 3;
        const wp::int32 var_44 = 1;
        wp::float32 var_45;
        const wp::int32 var_46 = 4;
        const wp::int32 var_47 = 2;
        wp::float32 var_48;
        const wp::int32 var_49 = 5;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::transform_t<wp::float32> adj_4 = {};
        wp::quat_t<wp::float32> adj_5 = {};
        wp::transform_t<wp::float32> adj_6 = {};
        wp::vec_t<3, wp::float32> adj_7 = {};
        wp::vec_t<3, wp::float32> adj_8 = {};
        wp::vec_t<3, wp::float32> adj_9 = {};
        wp::int32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::float32 adj_25 = {};
        wp::vec_t<3, wp::float32> adj_26 = {};
        wp::float32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::float32 adj_29 = {};
        wp::vec_t<3, wp::float32> adj_30 = {};
        wp::vec_t<3, wp::float32> adj_31 = {};
        wp::int32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::float32 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::float32 adj_42 = {};
        wp::int32 adj_43 = {};
        wp::int32 adj_44 = {};
        wp::float32 adj_45 = {};
        wp::int32 adj_46 = {};
        wp::int32 adj_47 = {};
        wp::float32 adj_48 = {};
        wp::int32 adj_49 = {};
        //---------
        // forward
        // def shift_jacobian_com_to_origin(                                                      <L 839>
        // n, b, dof = wp.tid()                                                                   <L 886>
        builtin_tid3d(var_0, var_1, var_2);
        // full_body_idx = b + link_offset                                                        <L 887>
        var_3 = wp::add(var_1, var_link_offset);
        // R = wp.transform_get_rotation(body_link_pose[n, full_body_idx])                        <L 889>
        var_4 = wp::address(var_body_link_pose, var_0, var_3);
        var_6 = wp::load(var_4);
        var_5 = wp::transform_get_rotation(var_6);
        // c_world = wp.quat_rotate(R, body_com_pos_b[n, full_body_idx])                          <L 890>
        var_7 = wp::address(var_body_com_pos_b, var_0, var_3);
        var_9 = wp::load(var_7);
        var_8 = wp::quat_rotate(var_5, var_9);
        // v_com = wp.vec3(src[n, b, 0, dof], src[n, b, 1, dof], src[n, b, 2, dof])               <L 892>
        var_11 = wp::address(var_src, var_0, var_1, var_10, var_2);
        var_13 = wp::address(var_src, var_0, var_1, var_12, var_2);
        var_15 = wp::address(var_src, var_0, var_1, var_14, var_2);
        var_17 = wp::load(var_11);
        var_18 = wp::load(var_13);
        var_19 = wp::load(var_15);
        var_16 = wp::vec_t<3, wp::float32>(var_17, var_18, var_19);
        // omega = wp.vec3(src[n, b, 3, dof], src[n, b, 4, dof], src[n, b, 5, dof])               <L 893>
        var_21 = wp::address(var_src, var_0, var_1, var_20, var_2);
        var_23 = wp::address(var_src, var_0, var_1, var_22, var_2);
        var_25 = wp::address(var_src, var_0, var_1, var_24, var_2);
        var_27 = wp::load(var_21);
        var_28 = wp::load(var_23);
        var_29 = wp::load(var_25);
        var_26 = wp::vec_t<3, wp::float32>(var_27, var_28, var_29);
        // v_origin = v_com - wp.cross(omega, c_world)                                            <L 895>
        var_30 = wp::cross(var_26, var_8);
        var_31 = wp::sub(var_16, var_30);
        // dst[n, b, 0, dof] = v_origin[0]                                                        <L 897>
        var_33 = wp::extract(var_31, var_32);
        // wp::array_store(var_dst, var_0, var_1, var_34, var_2, var_33);
        // dst[n, b, 1, dof] = v_origin[1]                                                        <L 898>
        var_36 = wp::extract(var_31, var_35);
        // wp::array_store(var_dst, var_0, var_1, var_37, var_2, var_36);
        // dst[n, b, 2, dof] = v_origin[2]                                                        <L 899>
        var_39 = wp::extract(var_31, var_38);
        // wp::array_store(var_dst, var_0, var_1, var_40, var_2, var_39);
        // dst[n, b, 3, dof] = omega[0]                                                           <L 900>
        var_42 = wp::extract(var_26, var_41);
        // wp::array_store(var_dst, var_0, var_1, var_43, var_2, var_42);
        // dst[n, b, 4, dof] = omega[1]                                                           <L 901>
        var_45 = wp::extract(var_26, var_44);
        // wp::array_store(var_dst, var_0, var_1, var_46, var_2, var_45);
        // dst[n, b, 5, dof] = omega[2]                                                           <L 902>
        var_48 = wp::extract(var_26, var_47);
        // wp::array_store(var_dst, var_0, var_1, var_49, var_2, var_48);
        //---------
        // reverse
        wp::adj_array_store(var_dst, var_0, var_1, var_49, var_2, var_48, adj_dst, adj_0, adj_1, adj_49, adj_2, adj_48);
        wp::adj_extract(var_26, var_47, adj_26, adj_47, adj_48);
        // adj: dst[n, b, 5, dof] = omega[2]                                                      <L 902>
        wp::adj_array_store(var_dst, var_0, var_1, var_46, var_2, var_45, adj_dst, adj_0, adj_1, adj_46, adj_2, adj_45);
        wp::adj_extract(var_26, var_44, adj_26, adj_44, adj_45);
        // adj: dst[n, b, 4, dof] = omega[1]                                                      <L 901>
        wp::adj_array_store(var_dst, var_0, var_1, var_43, var_2, var_42, adj_dst, adj_0, adj_1, adj_43, adj_2, adj_42);
        wp::adj_extract(var_26, var_41, adj_26, adj_41, adj_42);
        // adj: dst[n, b, 3, dof] = omega[0]                                                      <L 900>
        wp::adj_array_store(var_dst, var_0, var_1, var_40, var_2, var_39, adj_dst, adj_0, adj_1, adj_40, adj_2, adj_39);
        wp::adj_extract(var_31, var_38, adj_31, adj_38, adj_39);
        // adj: dst[n, b, 2, dof] = v_origin[2]                                                   <L 899>
        wp::adj_array_store(var_dst, var_0, var_1, var_37, var_2, var_36, adj_dst, adj_0, adj_1, adj_37, adj_2, adj_36);
        wp::adj_extract(var_31, var_35, adj_31, adj_35, adj_36);
        // adj: dst[n, b, 1, dof] = v_origin[1]                                                   <L 898>
        wp::adj_array_store(var_dst, var_0, var_1, var_34, var_2, var_33, adj_dst, adj_0, adj_1, adj_34, adj_2, adj_33);
        wp::adj_extract(var_31, var_32, adj_31, adj_32, adj_33);
        // adj: dst[n, b, 0, dof] = v_origin[0]                                                   <L 897>
        wp::adj_sub(var_16, var_30, adj_16, adj_30, adj_31);
        wp::adj_cross(var_26, var_8, adj_26, adj_8, adj_30);
        // adj: v_origin = v_com - wp.cross(omega, c_world)                                       <L 895>
        wp::adj_vec_t(var_27, var_28, var_29, adj_21, adj_23, adj_25, adj_26);
        wp::adj_address(var_src, var_0, var_1, var_24, var_2, adj_src, adj_0, adj_1, adj_24, adj_2, adj_25);
        wp::adj_address(var_src, var_0, var_1, var_22, var_2, adj_src, adj_0, adj_1, adj_22, adj_2, adj_23);
        wp::adj_address(var_src, var_0, var_1, var_20, var_2, adj_src, adj_0, adj_1, adj_20, adj_2, adj_21);
        // adj: omega = wp.vec3(src[n, b, 3, dof], src[n, b, 4, dof], src[n, b, 5, dof])          <L 893>
        wp::adj_vec_t(var_17, var_18, var_19, adj_11, adj_13, adj_15, adj_16);
        wp::adj_address(var_src, var_0, var_1, var_14, var_2, adj_src, adj_0, adj_1, adj_14, adj_2, adj_15);
        wp::adj_address(var_src, var_0, var_1, var_12, var_2, adj_src, adj_0, adj_1, adj_12, adj_2, adj_13);
        wp::adj_address(var_src, var_0, var_1, var_10, var_2, adj_src, adj_0, adj_1, adj_10, adj_2, adj_11);
        // adj: v_com = wp.vec3(src[n, b, 0, dof], src[n, b, 1, dof], src[n, b, 2, dof])          <L 892>
        wp::adj_quat_rotate(var_5, var_9, adj_5, adj_7, adj_8);
        wp::adj_address(var_body_com_pos_b, var_0, var_3, adj_body_com_pos_b, adj_0, adj_3, adj_7);
        // adj: c_world = wp.quat_rotate(R, body_com_pos_b[n, full_body_idx])                     <L 890>
        wp::adj_transform_get_rotation(var_6, adj_4, adj_5);
        wp::adj_address(var_body_link_pose, var_0, var_3, adj_body_link_pose, adj_0, adj_3, adj_4);
        // adj: R = wp.transform_get_rotation(body_link_pose[n, full_body_idx])                   <L 889>
        wp::adj_add(var_1, var_link_offset, adj_1, adj_link_offset, adj_3);
        // adj: full_body_idx = b + link_offset                                                   <L 887>
        // adj: n, b, dof = wp.tid()                                                              <L 886>
        // adj: def shift_jacobian_com_to_origin(                                                 <L 839>
        continue;
    }
}



extern "C" __global__ void write_joint_limit_data_to_buffer_index_7bf6bc6c_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_in_data,
    wp::float32 var_soft_limit_factor,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::array_t<wp::float32> var_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_soft_joint_pos_limits,
    wp::array_t<wp::float32> var_default_joint_pos,
    wp::array_t<wp::int32> var_clamped_defaults)
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
        wp::vec_t<2, wp::float32>* var_8;
        const wp::int32 var_9 = 0;
        wp::float32 var_10;
        wp::vec_t<2, wp::float32> var_11;
        wp::vec_t<2, wp::float32>* var_12;
        const wp::int32 var_13 = 1;
        wp::float32 var_14;
        wp::vec_t<2, wp::float32> var_15;
        bool var_16;
        wp::float32* var_17;
        wp::float32* var_18;
        bool var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32* var_22;
        wp::float32* var_23;
        bool var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        const wp::int32 var_27 = 0;
        const wp::int32 var_28 = 1;
        wp::int32 var_29;
        wp::float32* var_30;
        wp::float32* var_31;
        wp::float32* var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32* var_37;
        wp::float32* var_38;
        wp::vec_t<2, wp::float32> var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::vec_t<2, wp::float32> var_42;
        //---------
        // forward
        // def write_joint_limit_data_to_buffer_index(                                            <L 1>
        // i, j = wp.tid()                                                                        <L 35>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 36>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 37>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // joint_pos_limits_lower[env_id, joint_id] = in_data[i, j][0]                            <L 38>
        var_8 = wp::address(var_in_data, var_0, var_1);
        var_11 = wp::load(var_8);
        var_10 = wp::extract(var_11, var_9);
        wp::array_store(var_joint_pos_limits_lower, var_3, var_6, var_10);
        // joint_pos_limits_upper[env_id, joint_id] = in_data[i, j][1]                            <L 39>
        var_12 = wp::address(var_in_data, var_0, var_1);
        var_15 = wp::load(var_12);
        var_14 = wp::extract(var_15, var_13);
        wp::array_store(var_joint_pos_limits_upper, var_3, var_6, var_14);
        // if (default_joint_pos[env_id, joint_id] < joint_pos_limits_lower[env_id, joint_id]) or default_joint_pos[       <L 40>
        var_17 = wp::address(var_default_joint_pos, var_3, var_6);
        var_18 = wp::address(var_joint_pos_limits_lower, var_3, var_6);
        var_20 = wp::load(var_17);
        var_21 = wp::load(var_18);
        var_19 = (var_20 < var_21);
        var_16 = var_19;
        if (!var_16) {
            // env_id, joint_id                                                                   <L 41>
            var_22 = wp::address(var_default_joint_pos, var_3, var_6);
            // ] > joint_pos_limits_upper[env_id, joint_id]:                                      <L 42>
            var_23 = wp::address(var_joint_pos_limits_upper, var_3, var_6);
            var_25 = wp::load(var_22);
            var_26 = wp::load(var_23);
            var_24 = (var_25 > var_26);
            var_16 = var_16 || var_24;
        }
        if (var_16) {
            // wp.atomic_add(clamped_defaults, 0, 1)                                              <L 43>
            var_29 = wp::atomic_add(var_clamped_defaults, var_27, var_28);
            // default_joint_pos[env_id, joint_id] = wp.clamp(                                    <L 44>
            // default_joint_pos[env_id, joint_id],                                               <L 45>
            var_30 = wp::address(var_default_joint_pos, var_3, var_6);
            // joint_pos_limits_lower[env_id, joint_id],                                          <L 46>
            var_31 = wp::address(var_joint_pos_limits_lower, var_3, var_6);
            // joint_pos_limits_upper[env_id, joint_id],                                          <L 47>
            var_32 = wp::address(var_joint_pos_limits_upper, var_3, var_6);
            var_34 = wp::load(var_30);
            var_35 = wp::load(var_31);
            var_36 = wp::load(var_32);
            var_33 = wp::clamp(var_34, var_35, var_36);
            // default_joint_pos[env_id, joint_id] = wp.clamp(                                    <L 44>
            wp::array_store(var_default_joint_pos, var_3, var_6, var_33);
        }
        // soft_joint_pos_limits[env_id, joint_id] = compute_soft_joint_pos_limits_func(          <L 49>
        // wp.vec2f(joint_pos_limits_lower[env_id, joint_id], joint_pos_limits_upper[env_id, joint_id]),       <L 50>
        var_37 = wp::address(var_joint_pos_limits_lower, var_3, var_6);
        var_38 = wp::address(var_joint_pos_limits_upper, var_3, var_6);
        var_40 = wp::load(var_37);
        var_41 = wp::load(var_38);
        var_39 = wp::vec_t<2, wp::float32>(var_40, var_41);
        // soft_limit_factor,                                                                     <L 51>
        var_42 = compute_soft_joint_pos_limits_func_0(var_39, var_soft_limit_factor);
        // soft_joint_pos_limits[env_id, joint_id] = compute_soft_joint_pos_limits_func(          <L 49>
        wp::array_store(var_soft_joint_pos_limits, var_3, var_6, var_42);
    }
}



extern "C" __global__ void write_joint_limit_data_to_buffer_index_7bf6bc6c_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_in_data,
    wp::float32 var_soft_limit_factor,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::array_t<wp::float32> var_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_soft_joint_pos_limits,
    wp::array_t<wp::float32> var_default_joint_pos,
    wp::array_t<wp::int32> var_clamped_defaults,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_in_data,
    wp::float32 adj_soft_limit_factor,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int32> adj_joint_ids,
    wp::array_t<wp::float32> adj_joint_pos_limits_lower,
    wp::array_t<wp::float32> adj_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_soft_joint_pos_limits,
    wp::array_t<wp::float32> adj_default_joint_pos,
    wp::array_t<wp::int32> adj_clamped_defaults)
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
        wp::vec_t<2, wp::float32>* var_8;
        const wp::int32 var_9 = 0;
        wp::float32 var_10;
        wp::vec_t<2, wp::float32> var_11;
        wp::vec_t<2, wp::float32>* var_12;
        const wp::int32 var_13 = 1;
        wp::float32 var_14;
        wp::vec_t<2, wp::float32> var_15;
        bool var_16;
        wp::float32* var_17;
        wp::float32* var_18;
        bool var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32* var_22;
        wp::float32* var_23;
        bool var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        const wp::int32 var_27 = 0;
        const wp::int32 var_28 = 1;
        wp::int32 var_29;
        wp::float32* var_30;
        wp::float32* var_31;
        wp::float32* var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32* var_37;
        wp::float32* var_38;
        wp::vec_t<2, wp::float32> var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::vec_t<2, wp::float32> var_42;
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
        wp::vec_t<2, wp::float32> adj_8 = {};
        wp::int32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::vec_t<2, wp::float32> adj_11 = {};
        wp::vec_t<2, wp::float32> adj_12 = {};
        wp::int32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::vec_t<2, wp::float32> adj_15 = {};
        bool adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        bool adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        bool adj_24 = {};
        wp::float32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::float32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::vec_t<2, wp::float32> adj_39 = {};
        wp::float32 adj_40 = {};
        wp::float32 adj_41 = {};
        wp::vec_t<2, wp::float32> adj_42 = {};
        //---------
        // forward
        // def write_joint_limit_data_to_buffer_index(                                            <L 1>
        // i, j = wp.tid()                                                                        <L 35>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 36>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 37>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // joint_pos_limits_lower[env_id, joint_id] = in_data[i, j][0]                            <L 38>
        var_8 = wp::address(var_in_data, var_0, var_1);
        var_11 = wp::load(var_8);
        var_10 = wp::extract(var_11, var_9);
        // wp::array_store(var_joint_pos_limits_lower, var_3, var_6, var_10);
        // joint_pos_limits_upper[env_id, joint_id] = in_data[i, j][1]                            <L 39>
        var_12 = wp::address(var_in_data, var_0, var_1);
        var_15 = wp::load(var_12);
        var_14 = wp::extract(var_15, var_13);
        // wp::array_store(var_joint_pos_limits_upper, var_3, var_6, var_14);
        // if (default_joint_pos[env_id, joint_id] < joint_pos_limits_lower[env_id, joint_id]) or default_joint_pos[       <L 40>
        var_17 = wp::address(var_default_joint_pos, var_3, var_6);
        var_18 = wp::address(var_joint_pos_limits_lower, var_3, var_6);
        var_20 = wp::load(var_17);
        var_21 = wp::load(var_18);
        var_19 = (var_20 < var_21);
        var_16 = var_19;
        if (!var_16) {
            // env_id, joint_id                                                                   <L 41>
            var_22 = wp::address(var_default_joint_pos, var_3, var_6);
            // ] > joint_pos_limits_upper[env_id, joint_id]:                                      <L 42>
            var_23 = wp::address(var_joint_pos_limits_upper, var_3, var_6);
            var_25 = wp::load(var_22);
            var_26 = wp::load(var_23);
            var_24 = (var_25 > var_26);
            var_16 = var_16 || var_24;
        }
        if (var_16) {
            // wp.atomic_add(clamped_defaults, 0, 1)                                              <L 43>
            // var_29 = wp::atomic_add(var_clamped_defaults, var_27, var_28);
            // default_joint_pos[env_id, joint_id] = wp.clamp(                                    <L 44>
            // default_joint_pos[env_id, joint_id],                                               <L 45>
            var_30 = wp::address(var_default_joint_pos, var_3, var_6);
            // joint_pos_limits_lower[env_id, joint_id],                                          <L 46>
            var_31 = wp::address(var_joint_pos_limits_lower, var_3, var_6);
            // joint_pos_limits_upper[env_id, joint_id],                                          <L 47>
            var_32 = wp::address(var_joint_pos_limits_upper, var_3, var_6);
            var_34 = wp::load(var_30);
            var_35 = wp::load(var_31);
            var_36 = wp::load(var_32);
            var_33 = wp::clamp(var_34, var_35, var_36);
            // default_joint_pos[env_id, joint_id] = wp.clamp(                                    <L 44>
            // wp::array_store(var_default_joint_pos, var_3, var_6, var_33);
        }
        // soft_joint_pos_limits[env_id, joint_id] = compute_soft_joint_pos_limits_func(          <L 49>
        // wp.vec2f(joint_pos_limits_lower[env_id, joint_id], joint_pos_limits_upper[env_id, joint_id]),       <L 50>
        var_37 = wp::address(var_joint_pos_limits_lower, var_3, var_6);
        var_38 = wp::address(var_joint_pos_limits_upper, var_3, var_6);
        var_40 = wp::load(var_37);
        var_41 = wp::load(var_38);
        var_39 = wp::vec_t<2, wp::float32>(var_40, var_41);
        // soft_limit_factor,                                                                     <L 51>
        var_42 = compute_soft_joint_pos_limits_func_0(var_39, var_soft_limit_factor);
        // soft_joint_pos_limits[env_id, joint_id] = compute_soft_joint_pos_limits_func(          <L 49>
        // wp::array_store(var_soft_joint_pos_limits, var_3, var_6, var_42);
        //---------
        // reverse
        wp::adj_array_store(var_soft_joint_pos_limits, var_3, var_6, var_42, adj_soft_joint_pos_limits, adj_3, adj_6, adj_42);
        // adj: soft_joint_pos_limits[env_id, joint_id] = compute_soft_joint_pos_limits_func(     <L 49>
        adj_compute_soft_joint_pos_limits_func_0(var_39, var_soft_limit_factor, adj_39, adj_soft_limit_factor, adj_42);
        // adj: soft_limit_factor,                                                                <L 51>
        wp::adj_vec_t(var_40, var_41, adj_37, adj_38, adj_39);
        wp::adj_address(var_joint_pos_limits_upper, var_3, var_6, adj_joint_pos_limits_upper, adj_3, adj_6, adj_38);
        wp::adj_address(var_joint_pos_limits_lower, var_3, var_6, adj_joint_pos_limits_lower, adj_3, adj_6, adj_37);
        // adj: wp.vec2f(joint_pos_limits_lower[env_id, joint_id], joint_pos_limits_upper[env_id, joint_id]),  <L 50>
        // adj: soft_joint_pos_limits[env_id, joint_id] = compute_soft_joint_pos_limits_func(     <L 49>
        if (var_16) {
            wp::adj_array_store(var_default_joint_pos, var_3, var_6, var_33, adj_default_joint_pos, adj_3, adj_6, adj_33);
            // adj: default_joint_pos[env_id, joint_id] = wp.clamp(                               <L 44>
            wp::adj_clamp(var_34, var_35, var_36, adj_30, adj_31, adj_32, adj_33);
            wp::adj_address(var_joint_pos_limits_upper, var_3, var_6, adj_joint_pos_limits_upper, adj_3, adj_6, adj_32);
            // adj: joint_pos_limits_upper[env_id, joint_id],                                     <L 47>
            wp::adj_address(var_joint_pos_limits_lower, var_3, var_6, adj_joint_pos_limits_lower, adj_3, adj_6, adj_31);
            // adj: joint_pos_limits_lower[env_id, joint_id],                                     <L 46>
            wp::adj_address(var_default_joint_pos, var_3, var_6, adj_default_joint_pos, adj_3, adj_6, adj_30);
            // adj: default_joint_pos[env_id, joint_id],                                          <L 45>
            // adj: default_joint_pos[env_id, joint_id] = wp.clamp(                               <L 44>
            wp::adj_atomic_add(var_clamped_defaults, var_27, var_28, adj_clamped_defaults, adj_27, adj_28, adj_29);
            // adj: wp.atomic_add(clamped_defaults, 0, 1)                                         <L 43>
        }
        if (!var_16) {
            wp::adj_address(var_joint_pos_limits_upper, var_3, var_6, adj_joint_pos_limits_upper, adj_3, adj_6, adj_23);
            // adj: ] > joint_pos_limits_upper[env_id, joint_id]:                                 <L 42>
            wp::adj_address(var_default_joint_pos, var_3, var_6, adj_default_joint_pos, adj_3, adj_6, adj_22);
            // adj: env_id, joint_id                                                              <L 41>
        }
        wp::adj_address(var_joint_pos_limits_lower, var_3, var_6, adj_joint_pos_limits_lower, adj_3, adj_6, adj_18);
        wp::adj_address(var_default_joint_pos, var_3, var_6, adj_default_joint_pos, adj_3, adj_6, adj_17);
        // adj: if (default_joint_pos[env_id, joint_id] < joint_pos_limits_lower[env_id, joint_id]) or default_joint_pos[  <L 40>
        wp::adj_array_store(var_joint_pos_limits_upper, var_3, var_6, var_14, adj_joint_pos_limits_upper, adj_3, adj_6, adj_14);
        wp::adj_extract(var_15, var_13, adj_12, adj_13, adj_14);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_12);
        // adj: joint_pos_limits_upper[env_id, joint_id] = in_data[i, j][1]                       <L 39>
        wp::adj_array_store(var_joint_pos_limits_lower, var_3, var_6, var_10, adj_joint_pos_limits_lower, adj_3, adj_6, adj_10);
        wp::adj_extract(var_11, var_9, adj_8, adj_9, adj_10);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_8);
        // adj: joint_pos_limits_lower[env_id, joint_id] = in_data[i, j][0]                       <L 38>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 37>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 36>
        // adj: i, j = wp.tid()                                                                   <L 35>
        // adj: def write_joint_limit_data_to_buffer_index(                                       <L 1>
        continue;
    }
}



extern "C" __global__ void write_joint_limit_data_to_user_and_backend_index_81b3d0bb_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_in_data,
    wp::float32 var_soft_limit_factor,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_user_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_user_joint_pos_limits,
    wp::array_t<wp::float32> var_backend_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_backend_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_soft_joint_pos_limits,
    wp::array_t<wp::float32> var_default_joint_pos,
    wp::array_t<wp::int32> var_clamped_defaults)
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
        wp::int32 var_8;
        wp::vec_t<2, wp::float32>* var_9;
        wp::vec_t<2, wp::float32> var_10;
        wp::vec_t<2, wp::float32> var_11;
        const wp::int32 var_12 = 0;
        wp::float32 var_13;
        const wp::int32 var_14 = 1;
        wp::float32 var_15;
        bool var_16;
        wp::float32* var_17;
        bool var_18;
        wp::float32 var_19;
        wp::float32* var_20;
        bool var_21;
        wp::float32 var_22;
        const wp::int32 var_23 = 0;
        const wp::int32 var_24 = 1;
        wp::int32 var_25;
        wp::float32* var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::vec_t<2, wp::float32> var_29;
        //---------
        // forward
        // def write_joint_limit_data_to_user_and_backend_index(                                  <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 19>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 20>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // backend_id = resolve_backend_index(user_id, user_to_backend, has_ordering)             <L 21>
        var_8 = resolve_backend_index_0(var_6, var_user_to_backend, var_has_ordering);
        // limits = in_data[i, j]                                                                 <L 22>
        var_9 = wp::address(var_in_data, var_0, var_1);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // lower = limits[0]                                                                      <L 23>
        var_13 = wp::extract(var_10, var_12);
        // upper = limits[1]                                                                      <L 24>
        var_15 = wp::extract(var_10, var_14);
        // user_joint_pos_limits_lower[env_id, user_id] = lower                                   <L 26>
        wp::array_store(var_user_joint_pos_limits_lower, var_3, var_6, var_13);
        // user_joint_pos_limits_upper[env_id, user_id] = upper                                   <L 27>
        wp::array_store(var_user_joint_pos_limits_upper, var_3, var_6, var_15);
        // user_joint_pos_limits[env_id, user_id] = limits                                        <L 28>
        wp::array_store(var_user_joint_pos_limits, var_3, var_6, var_10);
        // backend_joint_pos_limits_lower[env_id, backend_id] = lower                             <L 29>
        wp::array_store(var_backend_joint_pos_limits_lower, var_3, var_8, var_13);
        // backend_joint_pos_limits_upper[env_id, backend_id] = upper                             <L 30>
        wp::array_store(var_backend_joint_pos_limits_upper, var_3, var_8, var_15);
        // if (default_joint_pos[env_id, user_id] < lower) or default_joint_pos[env_id, user_id] > upper:       <L 32>
        var_17 = wp::address(var_default_joint_pos, var_3, var_6);
        var_19 = wp::load(var_17);
        var_18 = (var_19 < var_13);
        var_16 = var_18;
        if (!var_16) {
            var_20 = wp::address(var_default_joint_pos, var_3, var_6);
            var_22 = wp::load(var_20);
            var_21 = (var_22 > var_15);
            var_16 = var_16 || var_21;
        }
        if (var_16) {
            // wp.atomic_add(clamped_defaults, 0, 1)                                              <L 33>
            var_25 = wp::atomic_add(var_clamped_defaults, var_23, var_24);
            // default_joint_pos[env_id, user_id] = wp.clamp(default_joint_pos[env_id, user_id], lower, upper)       <L 34>
            var_26 = wp::address(var_default_joint_pos, var_3, var_6);
            var_28 = wp::load(var_26);
            var_27 = wp::clamp(var_28, var_13, var_15);
            wp::array_store(var_default_joint_pos, var_3, var_6, var_27);
        }
        // soft_joint_pos_limits[env_id, user_id] = compute_soft_joint_pos_limits_func(limits, soft_limit_factor)       <L 36>
        var_29 = compute_soft_joint_pos_limits_func_0(var_10, var_soft_limit_factor);
        wp::array_store(var_soft_joint_pos_limits, var_3, var_6, var_29);
    }
}



extern "C" __global__ void write_joint_limit_data_to_user_and_backend_index_81b3d0bb_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_in_data,
    wp::float32 var_soft_limit_factor,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_user_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_user_joint_pos_limits,
    wp::array_t<wp::float32> var_backend_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_backend_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_soft_joint_pos_limits,
    wp::array_t<wp::float32> var_default_joint_pos,
    wp::array_t<wp::int32> var_clamped_defaults,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_in_data,
    wp::float32 adj_soft_limit_factor,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int32> adj_user_ids,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    wp::array_t<wp::float32> adj_user_joint_pos_limits_lower,
    wp::array_t<wp::float32> adj_user_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_user_joint_pos_limits,
    wp::array_t<wp::float32> adj_backend_joint_pos_limits_lower,
    wp::array_t<wp::float32> adj_backend_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_soft_joint_pos_limits,
    wp::array_t<wp::float32> adj_default_joint_pos,
    wp::array_t<wp::int32> adj_clamped_defaults)
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
        wp::int32 var_8;
        wp::vec_t<2, wp::float32>* var_9;
        wp::vec_t<2, wp::float32> var_10;
        wp::vec_t<2, wp::float32> var_11;
        const wp::int32 var_12 = 0;
        wp::float32 var_13;
        const wp::int32 var_14 = 1;
        wp::float32 var_15;
        bool var_16;
        wp::float32* var_17;
        bool var_18;
        wp::float32 var_19;
        wp::float32* var_20;
        bool var_21;
        wp::float32 var_22;
        const wp::int32 var_23 = 0;
        const wp::int32 var_24 = 1;
        wp::int32 var_25;
        wp::float32* var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::vec_t<2, wp::float32> var_29;
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
        wp::vec_t<2, wp::float32> adj_9 = {};
        wp::vec_t<2, wp::float32> adj_10 = {};
        wp::vec_t<2, wp::float32> adj_11 = {};
        wp::int32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::float32 adj_15 = {};
        bool adj_16 = {};
        wp::float32 adj_17 = {};
        bool adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        bool adj_21 = {};
        wp::float32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::vec_t<2, wp::float32> adj_29 = {};
        //---------
        // forward
        // def write_joint_limit_data_to_user_and_backend_index(                                  <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 19>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 20>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // backend_id = resolve_backend_index(user_id, user_to_backend, has_ordering)             <L 21>
        var_8 = resolve_backend_index_0(var_6, var_user_to_backend, var_has_ordering);
        // limits = in_data[i, j]                                                                 <L 22>
        var_9 = wp::address(var_in_data, var_0, var_1);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // lower = limits[0]                                                                      <L 23>
        var_13 = wp::extract(var_10, var_12);
        // upper = limits[1]                                                                      <L 24>
        var_15 = wp::extract(var_10, var_14);
        // user_joint_pos_limits_lower[env_id, user_id] = lower                                   <L 26>
        // wp::array_store(var_user_joint_pos_limits_lower, var_3, var_6, var_13);
        // user_joint_pos_limits_upper[env_id, user_id] = upper                                   <L 27>
        // wp::array_store(var_user_joint_pos_limits_upper, var_3, var_6, var_15);
        // user_joint_pos_limits[env_id, user_id] = limits                                        <L 28>
        // wp::array_store(var_user_joint_pos_limits, var_3, var_6, var_10);
        // backend_joint_pos_limits_lower[env_id, backend_id] = lower                             <L 29>
        // wp::array_store(var_backend_joint_pos_limits_lower, var_3, var_8, var_13);
        // backend_joint_pos_limits_upper[env_id, backend_id] = upper                             <L 30>
        // wp::array_store(var_backend_joint_pos_limits_upper, var_3, var_8, var_15);
        // if (default_joint_pos[env_id, user_id] < lower) or default_joint_pos[env_id, user_id] > upper:       <L 32>
        var_17 = wp::address(var_default_joint_pos, var_3, var_6);
        var_19 = wp::load(var_17);
        var_18 = (var_19 < var_13);
        var_16 = var_18;
        if (!var_16) {
            var_20 = wp::address(var_default_joint_pos, var_3, var_6);
            var_22 = wp::load(var_20);
            var_21 = (var_22 > var_15);
            var_16 = var_16 || var_21;
        }
        if (var_16) {
            // wp.atomic_add(clamped_defaults, 0, 1)                                              <L 33>
            // var_25 = wp::atomic_add(var_clamped_defaults, var_23, var_24);
            // default_joint_pos[env_id, user_id] = wp.clamp(default_joint_pos[env_id, user_id], lower, upper)       <L 34>
            var_26 = wp::address(var_default_joint_pos, var_3, var_6);
            var_28 = wp::load(var_26);
            var_27 = wp::clamp(var_28, var_13, var_15);
            // wp::array_store(var_default_joint_pos, var_3, var_6, var_27);
        }
        // soft_joint_pos_limits[env_id, user_id] = compute_soft_joint_pos_limits_func(limits, soft_limit_factor)       <L 36>
        var_29 = compute_soft_joint_pos_limits_func_0(var_10, var_soft_limit_factor);
        // wp::array_store(var_soft_joint_pos_limits, var_3, var_6, var_29);
        //---------
        // reverse
        wp::adj_array_store(var_soft_joint_pos_limits, var_3, var_6, var_29, adj_soft_joint_pos_limits, adj_3, adj_6, adj_29);
        adj_compute_soft_joint_pos_limits_func_0(var_10, var_soft_limit_factor, adj_10, adj_soft_limit_factor, adj_29);
        // adj: soft_joint_pos_limits[env_id, user_id] = compute_soft_joint_pos_limits_func(limits, soft_limit_factor)  <L 36>
        if (var_16) {
            wp::adj_array_store(var_default_joint_pos, var_3, var_6, var_27, adj_default_joint_pos, adj_3, adj_6, adj_27);
            wp::adj_clamp(var_28, var_13, var_15, adj_26, adj_13, adj_15, adj_27);
            wp::adj_address(var_default_joint_pos, var_3, var_6, adj_default_joint_pos, adj_3, adj_6, adj_26);
            // adj: default_joint_pos[env_id, user_id] = wp.clamp(default_joint_pos[env_id, user_id], lower, upper)  <L 34>
            wp::adj_atomic_add(var_clamped_defaults, var_23, var_24, adj_clamped_defaults, adj_23, adj_24, adj_25);
            // adj: wp.atomic_add(clamped_defaults, 0, 1)                                         <L 33>
        }
        if (!var_16) {
            wp::adj_address(var_default_joint_pos, var_3, var_6, adj_default_joint_pos, adj_3, adj_6, adj_20);
        }
        wp::adj_address(var_default_joint_pos, var_3, var_6, adj_default_joint_pos, adj_3, adj_6, adj_17);
        // adj: if (default_joint_pos[env_id, user_id] < lower) or default_joint_pos[env_id, user_id] > upper:  <L 32>
        wp::adj_array_store(var_backend_joint_pos_limits_upper, var_3, var_8, var_15, adj_backend_joint_pos_limits_upper, adj_3, adj_8, adj_15);
        // adj: backend_joint_pos_limits_upper[env_id, backend_id] = upper                        <L 30>
        wp::adj_array_store(var_backend_joint_pos_limits_lower, var_3, var_8, var_13, adj_backend_joint_pos_limits_lower, adj_3, adj_8, adj_13);
        // adj: backend_joint_pos_limits_lower[env_id, backend_id] = lower                        <L 29>
        wp::adj_array_store(var_user_joint_pos_limits, var_3, var_6, var_10, adj_user_joint_pos_limits, adj_3, adj_6, adj_10);
        // adj: user_joint_pos_limits[env_id, user_id] = limits                                   <L 28>
        wp::adj_array_store(var_user_joint_pos_limits_upper, var_3, var_6, var_15, adj_user_joint_pos_limits_upper, adj_3, adj_6, adj_15);
        // adj: user_joint_pos_limits_upper[env_id, user_id] = upper                              <L 27>
        wp::adj_array_store(var_user_joint_pos_limits_lower, var_3, var_6, var_13, adj_user_joint_pos_limits_lower, adj_3, adj_6, adj_13);
        // adj: user_joint_pos_limits_lower[env_id, user_id] = lower                              <L 26>
        wp::adj_extract(var_10, var_14, adj_10, adj_14, adj_15);
        // adj: upper = limits[1]                                                                 <L 24>
        wp::adj_extract(var_10, var_12, adj_10, adj_12, adj_13);
        // adj: lower = limits[0]                                                                 <L 23>
        wp::adj_copy(var_11, adj_9, adj_10);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_9);
        // adj: limits = in_data[i, j]                                                            <L 22>
        adj_resolve_backend_index_0(var_6, var_user_to_backend, var_has_ordering, adj_6, adj_user_to_backend, adj_has_ordering, adj_8);
        // adj: backend_id = resolve_backend_index(user_id, user_to_backend, has_ordering)        <L 21>
        wp::adj_address(var_user_ids, var_1, adj_user_ids, adj_1, adj_5);
        // adj: user_id = wp.int32(user_ids[j])                                                   <L 20>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 19>
        // adj: i, j = wp.tid()                                                                   <L 18>
        // adj: def write_joint_limit_data_to_user_and_backend_index(                             <L 1>
        continue;
    }
}



extern "C" __global__ void write_joint_state_data_index_84f4bc38_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_pos_data,
    wp::array_t<wp::float32> var_vel_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::array_t<wp::float32> var_joint_pos,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::array_t<wp::float32> var_joint_acc)
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
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        const wp::float32 var_14 = 0.0;
        //---------
        // forward
        // def write_joint_state_data_index(                                                      <L 1>
        // i, j = wp.tid()                                                                        <L 26>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 27>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 28>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // joint_pos[env_id, joint_id] = pos_data[i, j]                                           <L 29>
        var_8 = wp::address(var_pos_data, var_0, var_1);
        var_9 = wp::load(var_8);
        wp::array_store(var_joint_pos, var_3, var_6, var_9);
        // joint_vel[env_id, joint_id] = vel_data[i, j]                                           <L 30>
        var_10 = wp::address(var_vel_data, var_0, var_1);
        var_11 = wp::load(var_10);
        wp::array_store(var_joint_vel, var_3, var_6, var_11);
        // prev_joint_vel[env_id, joint_id] = vel_data[i, j]                                      <L 31>
        var_12 = wp::address(var_vel_data, var_0, var_1);
        var_13 = wp::load(var_12);
        wp::array_store(var_prev_joint_vel, var_3, var_6, var_13);
        // joint_acc[env_id, joint_id] = 0.0                                                      <L 32>
        wp::array_store(var_joint_acc, var_3, var_6, var_14);
    }
}



extern "C" __global__ void write_joint_state_data_index_84f4bc38_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_pos_data,
    wp::array_t<wp::float32> var_vel_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::array_t<wp::float32> var_joint_pos,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::array_t<wp::float32> var_joint_acc,
    wp::array_t<wp::float32> adj_pos_data,
    wp::array_t<wp::float32> adj_vel_data,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int32> adj_joint_ids,
    wp::array_t<wp::float32> adj_joint_pos,
    wp::array_t<wp::float32> adj_joint_vel,
    wp::array_t<wp::float32> adj_prev_joint_vel,
    wp::array_t<wp::float32> adj_joint_acc)
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
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        const wp::float32 var_14 = 0.0;
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
        //---------
        // forward
        // def write_joint_state_data_index(                                                      <L 1>
        // i, j = wp.tid()                                                                        <L 26>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 27>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 28>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // joint_pos[env_id, joint_id] = pos_data[i, j]                                           <L 29>
        var_8 = wp::address(var_pos_data, var_0, var_1);
        var_9 = wp::load(var_8);
        // wp::array_store(var_joint_pos, var_3, var_6, var_9);
        // joint_vel[env_id, joint_id] = vel_data[i, j]                                           <L 30>
        var_10 = wp::address(var_vel_data, var_0, var_1);
        var_11 = wp::load(var_10);
        // wp::array_store(var_joint_vel, var_3, var_6, var_11);
        // prev_joint_vel[env_id, joint_id] = vel_data[i, j]                                      <L 31>
        var_12 = wp::address(var_vel_data, var_0, var_1);
        var_13 = wp::load(var_12);
        // wp::array_store(var_prev_joint_vel, var_3, var_6, var_13);
        // joint_acc[env_id, joint_id] = 0.0                                                      <L 32>
        // wp::array_store(var_joint_acc, var_3, var_6, var_14);
        //---------
        // reverse
        wp::adj_array_store(var_joint_acc, var_3, var_6, var_14, adj_joint_acc, adj_3, adj_6, adj_14);
        // adj: joint_acc[env_id, joint_id] = 0.0                                                 <L 32>
        wp::adj_array_store(var_prev_joint_vel, var_3, var_6, var_13, adj_prev_joint_vel, adj_3, adj_6, adj_12);
        wp::adj_address(var_vel_data, var_0, var_1, adj_vel_data, adj_0, adj_1, adj_12);
        // adj: prev_joint_vel[env_id, joint_id] = vel_data[i, j]                                 <L 31>
        wp::adj_array_store(var_joint_vel, var_3, var_6, var_11, adj_joint_vel, adj_3, adj_6, adj_10);
        wp::adj_address(var_vel_data, var_0, var_1, adj_vel_data, adj_0, adj_1, adj_10);
        // adj: joint_vel[env_id, joint_id] = vel_data[i, j]                                      <L 30>
        wp::adj_array_store(var_joint_pos, var_3, var_6, var_9, adj_joint_pos, adj_3, adj_6, adj_8);
        wp::adj_address(var_pos_data, var_0, var_1, adj_pos_data, adj_0, adj_1, adj_8);
        // adj: joint_pos[env_id, joint_id] = pos_data[i, j]                                      <L 29>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 28>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 27>
        // adj: i, j = wp.tid()                                                                   <L 26>
        // adj: def write_joint_state_data_index(                                                 <L 1>
        continue;
    }
}



extern "C" __global__ void write_joint_state_data_index_89167d45_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_pos_data,
    wp::array_t<wp::float32> var_vel_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::array_t<wp::float32> var_joint_pos,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::array_t<wp::float32> var_joint_acc)
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
        wp::float32* var_12;
        wp::float32 var_13;
        const wp::float32 var_14 = 0.0;
        //---------
        // forward
        // def write_joint_state_data_index(                                                      <L 1>
        // i, j = wp.tid()                                                                        <L 26>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 27>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 28>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // joint_pos[env_id, joint_id] = pos_data[i, j]                                           <L 29>
        var_8 = wp::address(var_pos_data, var_0, var_1);
        var_9 = wp::load(var_8);
        wp::array_store(var_joint_pos, var_3, var_6, var_9);
        // joint_vel[env_id, joint_id] = vel_data[i, j]                                           <L 30>
        var_10 = wp::address(var_vel_data, var_0, var_1);
        var_11 = wp::load(var_10);
        wp::array_store(var_joint_vel, var_3, var_6, var_11);
        // prev_joint_vel[env_id, joint_id] = vel_data[i, j]                                      <L 31>
        var_12 = wp::address(var_vel_data, var_0, var_1);
        var_13 = wp::load(var_12);
        wp::array_store(var_prev_joint_vel, var_3, var_6, var_13);
        // joint_acc[env_id, joint_id] = 0.0                                                      <L 32>
        wp::array_store(var_joint_acc, var_3, var_6, var_14);
    }
}



extern "C" __global__ void write_joint_state_data_index_89167d45_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_pos_data,
    wp::array_t<wp::float32> var_vel_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::array_t<wp::float32> var_joint_pos,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::array_t<wp::float32> var_joint_acc,
    wp::array_t<wp::float32> adj_pos_data,
    wp::array_t<wp::float32> adj_vel_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_joint_ids,
    wp::array_t<wp::float32> adj_joint_pos,
    wp::array_t<wp::float32> adj_joint_vel,
    wp::array_t<wp::float32> adj_prev_joint_vel,
    wp::array_t<wp::float32> adj_joint_acc)
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
        wp::float32* var_12;
        wp::float32 var_13;
        const wp::float32 var_14 = 0.0;
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
        //---------
        // forward
        // def write_joint_state_data_index(                                                      <L 1>
        // i, j = wp.tid()                                                                        <L 26>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 27>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 28>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // joint_pos[env_id, joint_id] = pos_data[i, j]                                           <L 29>
        var_8 = wp::address(var_pos_data, var_0, var_1);
        var_9 = wp::load(var_8);
        // wp::array_store(var_joint_pos, var_3, var_6, var_9);
        // joint_vel[env_id, joint_id] = vel_data[i, j]                                           <L 30>
        var_10 = wp::address(var_vel_data, var_0, var_1);
        var_11 = wp::load(var_10);
        // wp::array_store(var_joint_vel, var_3, var_6, var_11);
        // prev_joint_vel[env_id, joint_id] = vel_data[i, j]                                      <L 31>
        var_12 = wp::address(var_vel_data, var_0, var_1);
        var_13 = wp::load(var_12);
        // wp::array_store(var_prev_joint_vel, var_3, var_6, var_13);
        // joint_acc[env_id, joint_id] = 0.0                                                      <L 32>
        // wp::array_store(var_joint_acc, var_3, var_6, var_14);
        //---------
        // reverse
        wp::adj_array_store(var_joint_acc, var_3, var_6, var_14, adj_joint_acc, adj_3, adj_6, adj_14);
        // adj: joint_acc[env_id, joint_id] = 0.0                                                 <L 32>
        wp::adj_array_store(var_prev_joint_vel, var_3, var_6, var_13, adj_prev_joint_vel, adj_3, adj_6, adj_12);
        wp::adj_address(var_vel_data, var_0, var_1, adj_vel_data, adj_0, adj_1, adj_12);
        // adj: prev_joint_vel[env_id, joint_id] = vel_data[i, j]                                 <L 31>
        wp::adj_array_store(var_joint_vel, var_3, var_6, var_11, adj_joint_vel, adj_3, adj_6, adj_10);
        wp::adj_address(var_vel_data, var_0, var_1, adj_vel_data, adj_0, adj_1, adj_10);
        // adj: joint_vel[env_id, joint_id] = vel_data[i, j]                                      <L 30>
        wp::adj_array_store(var_joint_pos, var_3, var_6, var_9, adj_joint_pos, adj_3, adj_6, adj_8);
        wp::adj_address(var_pos_data, var_0, var_1, adj_pos_data, adj_0, adj_1, adj_8);
        // adj: joint_pos[env_id, joint_id] = pos_data[i, j]                                      <L 29>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 28>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 27>
        // adj: i, j = wp.tid()                                                                   <L 26>
        // adj: def write_joint_state_data_index(                                                 <L 1>
        continue;
    }
}



extern "C" __global__ void update_default_joint_values_94439699_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_source,
    wp::array_t<wp::int32> var_ids,
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
        wp::float32* var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::float32 var_5;
        //---------
        // forward
        // def update_default_joint_values(                                                       <L 571>
        // i, j = wp.tid()                                                                        <L 588>
        builtin_tid2d(var_0, var_1);
        // target[i, ids[j]] = source[j]                                                          <L 589>
        var_2 = wp::address(var_source, var_1);
        var_3 = wp::address(var_ids, var_1);
        var_4 = wp::load(var_3);
        var_5 = wp::load(var_2);
        wp::array_store(var_target, var_0, var_4, var_5);
    }
}



extern "C" __global__ void update_default_joint_values_94439699_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_source,
    wp::array_t<wp::int32> var_ids,
    wp::array_t<wp::float32> var_target,
    wp::array_t<wp::float32> adj_source,
    wp::array_t<wp::int32> adj_ids,
    wp::array_t<wp::float32> adj_target)
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
        wp::float32* var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::float32 var_5;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::float32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::float32 adj_5 = {};
        //---------
        // forward
        // def update_default_joint_values(                                                       <L 571>
        // i, j = wp.tid()                                                                        <L 588>
        builtin_tid2d(var_0, var_1);
        // target[i, ids[j]] = source[j]                                                          <L 589>
        var_2 = wp::address(var_source, var_1);
        var_3 = wp::address(var_ids, var_1);
        var_4 = wp::load(var_3);
        var_5 = wp::load(var_2);
        // wp::array_store(var_target, var_0, var_4, var_5);
        //---------
        // reverse
        wp::adj_array_store(var_target, var_0, var_4, var_5, adj_target, adj_0, adj_3, adj_2);
        wp::adj_address(var_ids, var_1, adj_ids, adj_1, adj_3);
        wp::adj_address(var_source, var_1, adj_source, adj_1, adj_2);
        // adj: target[i, ids[j]] = source[j]                                                     <L 589>
        // adj: i, j = wp.tid()                                                                   <L 588>
        // adj: def update_default_joint_values(                                                  <L 571>
        continue;
    }
}



extern "C" __global__ void write_joint_friction_data_to_buffer_9f3200a8_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_friction,
    wp::array_t<wp::float32> var_in_dynamic_friction,
    wp::array_t<wp::float32> var_in_viscous_friction,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    bool var_from_mask,
    wp::array_t<wp::float32> var_out_friction,
    wp::array_t<wp::float32> var_out_dynamic_friction,
    wp::array_t<wp::float32> var_out_viscous_friction,
    wp::array_t<wp::float32> var_friction_props)
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
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32* var_16;
        wp::float32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32* var_20;
        const wp::int32 var_21 = 0;
        wp::float32 var_22;
        wp::float32* var_23;
        const wp::int32 var_24 = 1;
        wp::float32 var_25;
        wp::float32* var_26;
        const wp::int32 var_27 = 2;
        wp::float32 var_28;
        //---------
        // forward
        // def write_joint_friction_data_to_buffer(                                               <L 1>
        // i, j = wp.tid()                                                                        <L 43>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 44>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 45>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 47>
        if (var_from_mask) {
            // out_friction[env_id, joint_id] = in_friction[env_id, joint_id]                     <L 48>
            var_8 = wp::address(var_in_friction, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_out_friction, var_3, var_6, var_9);
            // if in_dynamic_friction:                                                            <L 49>
            if (var_in_dynamic_friction) {
                // out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[env_id, joint_id]       <L 50>
                var_10 = wp::address(var_in_dynamic_friction, var_3, var_6);
                var_11 = wp::load(var_10);
                wp::array_store(var_out_dynamic_friction, var_3, var_6, var_11);
            }
            // if in_viscous_friction:                                                            <L 51>
            if (var_in_viscous_friction) {
                // out_viscous_friction[env_id, joint_id] = in_viscous_friction[env_id, joint_id]       <L 52>
                var_12 = wp::address(var_in_viscous_friction, var_3, var_6);
                var_13 = wp::load(var_12);
                wp::array_store(var_out_viscous_friction, var_3, var_6, var_13);
            }
        }
        if (!var_from_mask) {
            // out_friction[env_id, joint_id] = in_friction[i, j]                                 <L 54>
            var_14 = wp::address(var_in_friction, var_0, var_1);
            var_15 = wp::load(var_14);
            wp::array_store(var_out_friction, var_3, var_6, var_15);
            // if in_dynamic_friction:                                                            <L 55>
            if (var_in_dynamic_friction) {
                // out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[i, j]             <L 56>
                var_16 = wp::address(var_in_dynamic_friction, var_0, var_1);
                var_17 = wp::load(var_16);
                wp::array_store(var_out_dynamic_friction, var_3, var_6, var_17);
            }
            // if in_viscous_friction:                                                            <L 57>
            if (var_in_viscous_friction) {
                // out_viscous_friction[env_id, joint_id] = in_viscous_friction[i, j]             <L 58>
                var_18 = wp::address(var_in_viscous_friction, var_0, var_1);
                var_19 = wp::load(var_18);
                wp::array_store(var_out_viscous_friction, var_3, var_6, var_19);
            }
        }
        // friction_props[env_id, joint_id, 0] = out_friction[env_id, joint_id]                   <L 60>
        var_20 = wp::address(var_out_friction, var_3, var_6);
        var_22 = wp::load(var_20);
        wp::array_store(var_friction_props, var_3, var_6, var_21, var_22);
        // if in_dynamic_friction:                                                                <L 61>
        if (var_in_dynamic_friction) {
            // friction_props[env_id, joint_id, 1] = out_dynamic_friction[env_id, joint_id]       <L 62>
            var_23 = wp::address(var_out_dynamic_friction, var_3, var_6);
            var_25 = wp::load(var_23);
            wp::array_store(var_friction_props, var_3, var_6, var_24, var_25);
        }
        // if in_viscous_friction:                                                                <L 63>
        if (var_in_viscous_friction) {
            // friction_props[env_id, joint_id, 2] = out_viscous_friction[env_id, joint_id]       <L 64>
            var_26 = wp::address(var_out_viscous_friction, var_3, var_6);
            var_28 = wp::load(var_26);
            wp::array_store(var_friction_props, var_3, var_6, var_27, var_28);
        }
    }
}



extern "C" __global__ void write_joint_friction_data_to_buffer_9f3200a8_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_friction,
    wp::array_t<wp::float32> var_in_dynamic_friction,
    wp::array_t<wp::float32> var_in_viscous_friction,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    bool var_from_mask,
    wp::array_t<wp::float32> var_out_friction,
    wp::array_t<wp::float32> var_out_dynamic_friction,
    wp::array_t<wp::float32> var_out_viscous_friction,
    wp::array_t<wp::float32> var_friction_props,
    wp::array_t<wp::float32> adj_in_friction,
    wp::array_t<wp::float32> adj_in_dynamic_friction,
    wp::array_t<wp::float32> adj_in_viscous_friction,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int64> adj_joint_ids,
    bool adj_from_mask,
    wp::array_t<wp::float32> adj_out_friction,
    wp::array_t<wp::float32> adj_out_dynamic_friction,
    wp::array_t<wp::float32> adj_out_viscous_friction,
    wp::array_t<wp::float32> adj_friction_props)
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
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32* var_16;
        wp::float32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32* var_20;
        const wp::int32 var_21 = 0;
        wp::float32 var_22;
        wp::float32* var_23;
        const wp::int32 var_24 = 1;
        wp::float32 var_25;
        wp::float32* var_26;
        const wp::int32 var_27 = 2;
        wp::float32 var_28;
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
        wp::int32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::float32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::float32 adj_28 = {};
        //---------
        // forward
        // def write_joint_friction_data_to_buffer(                                               <L 1>
        // i, j = wp.tid()                                                                        <L 43>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 44>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 45>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 47>
        if (var_from_mask) {
            // out_friction[env_id, joint_id] = in_friction[env_id, joint_id]                     <L 48>
            var_8 = wp::address(var_in_friction, var_3, var_6);
            var_9 = wp::load(var_8);
            // wp::array_store(var_out_friction, var_3, var_6, var_9);
            // if in_dynamic_friction:                                                            <L 49>
            if (var_in_dynamic_friction) {
                // out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[env_id, joint_id]       <L 50>
                var_10 = wp::address(var_in_dynamic_friction, var_3, var_6);
                var_11 = wp::load(var_10);
                // wp::array_store(var_out_dynamic_friction, var_3, var_6, var_11);
            }
            // if in_viscous_friction:                                                            <L 51>
            if (var_in_viscous_friction) {
                // out_viscous_friction[env_id, joint_id] = in_viscous_friction[env_id, joint_id]       <L 52>
                var_12 = wp::address(var_in_viscous_friction, var_3, var_6);
                var_13 = wp::load(var_12);
                // wp::array_store(var_out_viscous_friction, var_3, var_6, var_13);
            }
        }
        if (!var_from_mask) {
            // out_friction[env_id, joint_id] = in_friction[i, j]                                 <L 54>
            var_14 = wp::address(var_in_friction, var_0, var_1);
            var_15 = wp::load(var_14);
            // wp::array_store(var_out_friction, var_3, var_6, var_15);
            // if in_dynamic_friction:                                                            <L 55>
            if (var_in_dynamic_friction) {
                // out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[i, j]             <L 56>
                var_16 = wp::address(var_in_dynamic_friction, var_0, var_1);
                var_17 = wp::load(var_16);
                // wp::array_store(var_out_dynamic_friction, var_3, var_6, var_17);
            }
            // if in_viscous_friction:                                                            <L 57>
            if (var_in_viscous_friction) {
                // out_viscous_friction[env_id, joint_id] = in_viscous_friction[i, j]             <L 58>
                var_18 = wp::address(var_in_viscous_friction, var_0, var_1);
                var_19 = wp::load(var_18);
                // wp::array_store(var_out_viscous_friction, var_3, var_6, var_19);
            }
        }
        // friction_props[env_id, joint_id, 0] = out_friction[env_id, joint_id]                   <L 60>
        var_20 = wp::address(var_out_friction, var_3, var_6);
        var_22 = wp::load(var_20);
        // wp::array_store(var_friction_props, var_3, var_6, var_21, var_22);
        // if in_dynamic_friction:                                                                <L 61>
        if (var_in_dynamic_friction) {
            // friction_props[env_id, joint_id, 1] = out_dynamic_friction[env_id, joint_id]       <L 62>
            var_23 = wp::address(var_out_dynamic_friction, var_3, var_6);
            var_25 = wp::load(var_23);
            // wp::array_store(var_friction_props, var_3, var_6, var_24, var_25);
        }
        // if in_viscous_friction:                                                                <L 63>
        if (var_in_viscous_friction) {
            // friction_props[env_id, joint_id, 2] = out_viscous_friction[env_id, joint_id]       <L 64>
            var_26 = wp::address(var_out_viscous_friction, var_3, var_6);
            var_28 = wp::load(var_26);
            // wp::array_store(var_friction_props, var_3, var_6, var_27, var_28);
        }
        //---------
        // reverse
        if (var_in_viscous_friction) {
            wp::adj_array_store(var_friction_props, var_3, var_6, var_27, var_28, adj_friction_props, adj_3, adj_6, adj_27, adj_26);
            wp::adj_address(var_out_viscous_friction, var_3, var_6, adj_out_viscous_friction, adj_3, adj_6, adj_26);
            // adj: friction_props[env_id, joint_id, 2] = out_viscous_friction[env_id, joint_id]  <L 64>
        }
        // adj: if in_viscous_friction:                                                           <L 63>
        if (var_in_dynamic_friction) {
            wp::adj_array_store(var_friction_props, var_3, var_6, var_24, var_25, adj_friction_props, adj_3, adj_6, adj_24, adj_23);
            wp::adj_address(var_out_dynamic_friction, var_3, var_6, adj_out_dynamic_friction, adj_3, adj_6, adj_23);
            // adj: friction_props[env_id, joint_id, 1] = out_dynamic_friction[env_id, joint_id]  <L 62>
        }
        // adj: if in_dynamic_friction:                                                           <L 61>
        wp::adj_array_store(var_friction_props, var_3, var_6, var_21, var_22, adj_friction_props, adj_3, adj_6, adj_21, adj_20);
        wp::adj_address(var_out_friction, var_3, var_6, adj_out_friction, adj_3, adj_6, adj_20);
        // adj: friction_props[env_id, joint_id, 0] = out_friction[env_id, joint_id]              <L 60>
        if (!var_from_mask) {
            if (var_in_viscous_friction) {
                wp::adj_array_store(var_out_viscous_friction, var_3, var_6, var_19, adj_out_viscous_friction, adj_3, adj_6, adj_18);
                wp::adj_address(var_in_viscous_friction, var_0, var_1, adj_in_viscous_friction, adj_0, adj_1, adj_18);
                // adj: out_viscous_friction[env_id, joint_id] = in_viscous_friction[i, j]        <L 58>
            }
            // adj: if in_viscous_friction:                                                       <L 57>
            if (var_in_dynamic_friction) {
                wp::adj_array_store(var_out_dynamic_friction, var_3, var_6, var_17, adj_out_dynamic_friction, adj_3, adj_6, adj_16);
                wp::adj_address(var_in_dynamic_friction, var_0, var_1, adj_in_dynamic_friction, adj_0, adj_1, adj_16);
                // adj: out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[i, j]        <L 56>
            }
            // adj: if in_dynamic_friction:                                                       <L 55>
            wp::adj_array_store(var_out_friction, var_3, var_6, var_15, adj_out_friction, adj_3, adj_6, adj_14);
            wp::adj_address(var_in_friction, var_0, var_1, adj_in_friction, adj_0, adj_1, adj_14);
            // adj: out_friction[env_id, joint_id] = in_friction[i, j]                            <L 54>
        }
        if (var_from_mask) {
            if (var_in_viscous_friction) {
                wp::adj_array_store(var_out_viscous_friction, var_3, var_6, var_13, adj_out_viscous_friction, adj_3, adj_6, adj_12);
                wp::adj_address(var_in_viscous_friction, var_3, var_6, adj_in_viscous_friction, adj_3, adj_6, adj_12);
                // adj: out_viscous_friction[env_id, joint_id] = in_viscous_friction[env_id, joint_id]  <L 52>
            }
            // adj: if in_viscous_friction:                                                       <L 51>
            if (var_in_dynamic_friction) {
                wp::adj_array_store(var_out_dynamic_friction, var_3, var_6, var_11, adj_out_dynamic_friction, adj_3, adj_6, adj_10);
                wp::adj_address(var_in_dynamic_friction, var_3, var_6, adj_in_dynamic_friction, adj_3, adj_6, adj_10);
                // adj: out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[env_id, joint_id]  <L 50>
            }
            // adj: if in_dynamic_friction:                                                       <L 49>
            wp::adj_array_store(var_out_friction, var_3, var_6, var_9, adj_out_friction, adj_3, adj_6, adj_8);
            wp::adj_address(var_in_friction, var_3, var_6, adj_in_friction, adj_3, adj_6, adj_8);
            // adj: out_friction[env_id, joint_id] = in_friction[env_id, joint_id]                <L 48>
        }
        // adj: if from_mask:                                                                     <L 47>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 45>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 44>
        // adj: i, j = wp.tid()                                                                   <L 43>
        // adj: def write_joint_friction_data_to_buffer(                                          <L 1>
        continue;
    }
}



extern "C" __global__ void write_joint_limit_data_to_user_and_backend_index_ab329408_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_in_data,
    wp::float32 var_soft_limit_factor,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_user_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_user_joint_pos_limits,
    wp::array_t<wp::float32> var_backend_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_backend_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_soft_joint_pos_limits,
    wp::array_t<wp::float32> var_default_joint_pos,
    wp::array_t<wp::int32> var_clamped_defaults)
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
        wp::int32 var_8;
        wp::vec_t<2, wp::float32>* var_9;
        wp::vec_t<2, wp::float32> var_10;
        wp::vec_t<2, wp::float32> var_11;
        const wp::int32 var_12 = 0;
        wp::float32 var_13;
        const wp::int32 var_14 = 1;
        wp::float32 var_15;
        bool var_16;
        wp::float32* var_17;
        bool var_18;
        wp::float32 var_19;
        wp::float32* var_20;
        bool var_21;
        wp::float32 var_22;
        const wp::int32 var_23 = 0;
        const wp::int32 var_24 = 1;
        wp::int32 var_25;
        wp::float32* var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::vec_t<2, wp::float32> var_29;
        //---------
        // forward
        // def write_joint_limit_data_to_user_and_backend_index(                                  <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 19>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 20>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // backend_id = resolve_backend_index(user_id, user_to_backend, has_ordering)             <L 21>
        var_8 = resolve_backend_index_0(var_6, var_user_to_backend, var_has_ordering);
        // limits = in_data[i, j]                                                                 <L 22>
        var_9 = wp::address(var_in_data, var_0, var_1);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // lower = limits[0]                                                                      <L 23>
        var_13 = wp::extract(var_10, var_12);
        // upper = limits[1]                                                                      <L 24>
        var_15 = wp::extract(var_10, var_14);
        // user_joint_pos_limits_lower[env_id, user_id] = lower                                   <L 26>
        wp::array_store(var_user_joint_pos_limits_lower, var_3, var_6, var_13);
        // user_joint_pos_limits_upper[env_id, user_id] = upper                                   <L 27>
        wp::array_store(var_user_joint_pos_limits_upper, var_3, var_6, var_15);
        // user_joint_pos_limits[env_id, user_id] = limits                                        <L 28>
        wp::array_store(var_user_joint_pos_limits, var_3, var_6, var_10);
        // backend_joint_pos_limits_lower[env_id, backend_id] = lower                             <L 29>
        wp::array_store(var_backend_joint_pos_limits_lower, var_3, var_8, var_13);
        // backend_joint_pos_limits_upper[env_id, backend_id] = upper                             <L 30>
        wp::array_store(var_backend_joint_pos_limits_upper, var_3, var_8, var_15);
        // if (default_joint_pos[env_id, user_id] < lower) or default_joint_pos[env_id, user_id] > upper:       <L 32>
        var_17 = wp::address(var_default_joint_pos, var_3, var_6);
        var_19 = wp::load(var_17);
        var_18 = (var_19 < var_13);
        var_16 = var_18;
        if (!var_16) {
            var_20 = wp::address(var_default_joint_pos, var_3, var_6);
            var_22 = wp::load(var_20);
            var_21 = (var_22 > var_15);
            var_16 = var_16 || var_21;
        }
        if (var_16) {
            // wp.atomic_add(clamped_defaults, 0, 1)                                              <L 33>
            var_25 = wp::atomic_add(var_clamped_defaults, var_23, var_24);
            // default_joint_pos[env_id, user_id] = wp.clamp(default_joint_pos[env_id, user_id], lower, upper)       <L 34>
            var_26 = wp::address(var_default_joint_pos, var_3, var_6);
            var_28 = wp::load(var_26);
            var_27 = wp::clamp(var_28, var_13, var_15);
            wp::array_store(var_default_joint_pos, var_3, var_6, var_27);
        }
        // soft_joint_pos_limits[env_id, user_id] = compute_soft_joint_pos_limits_func(limits, soft_limit_factor)       <L 36>
        var_29 = compute_soft_joint_pos_limits_func_0(var_10, var_soft_limit_factor);
        wp::array_store(var_soft_joint_pos_limits, var_3, var_6, var_29);
    }
}



extern "C" __global__ void write_joint_limit_data_to_user_and_backend_index_ab329408_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_in_data,
    wp::float32 var_soft_limit_factor,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_user_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_user_joint_pos_limits,
    wp::array_t<wp::float32> var_backend_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_backend_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_soft_joint_pos_limits,
    wp::array_t<wp::float32> var_default_joint_pos,
    wp::array_t<wp::int32> var_clamped_defaults,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_in_data,
    wp::float32 adj_soft_limit_factor,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int64> adj_user_ids,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    wp::array_t<wp::float32> adj_user_joint_pos_limits_lower,
    wp::array_t<wp::float32> adj_user_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_user_joint_pos_limits,
    wp::array_t<wp::float32> adj_backend_joint_pos_limits_lower,
    wp::array_t<wp::float32> adj_backend_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_soft_joint_pos_limits,
    wp::array_t<wp::float32> adj_default_joint_pos,
    wp::array_t<wp::int32> adj_clamped_defaults)
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
        wp::int32 var_8;
        wp::vec_t<2, wp::float32>* var_9;
        wp::vec_t<2, wp::float32> var_10;
        wp::vec_t<2, wp::float32> var_11;
        const wp::int32 var_12 = 0;
        wp::float32 var_13;
        const wp::int32 var_14 = 1;
        wp::float32 var_15;
        bool var_16;
        wp::float32* var_17;
        bool var_18;
        wp::float32 var_19;
        wp::float32* var_20;
        bool var_21;
        wp::float32 var_22;
        const wp::int32 var_23 = 0;
        const wp::int32 var_24 = 1;
        wp::int32 var_25;
        wp::float32* var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::vec_t<2, wp::float32> var_29;
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
        wp::vec_t<2, wp::float32> adj_9 = {};
        wp::vec_t<2, wp::float32> adj_10 = {};
        wp::vec_t<2, wp::float32> adj_11 = {};
        wp::int32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::float32 adj_15 = {};
        bool adj_16 = {};
        wp::float32 adj_17 = {};
        bool adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        bool adj_21 = {};
        wp::float32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::vec_t<2, wp::float32> adj_29 = {};
        //---------
        // forward
        // def write_joint_limit_data_to_user_and_backend_index(                                  <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 19>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 20>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // backend_id = resolve_backend_index(user_id, user_to_backend, has_ordering)             <L 21>
        var_8 = resolve_backend_index_0(var_6, var_user_to_backend, var_has_ordering);
        // limits = in_data[i, j]                                                                 <L 22>
        var_9 = wp::address(var_in_data, var_0, var_1);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // lower = limits[0]                                                                      <L 23>
        var_13 = wp::extract(var_10, var_12);
        // upper = limits[1]                                                                      <L 24>
        var_15 = wp::extract(var_10, var_14);
        // user_joint_pos_limits_lower[env_id, user_id] = lower                                   <L 26>
        // wp::array_store(var_user_joint_pos_limits_lower, var_3, var_6, var_13);
        // user_joint_pos_limits_upper[env_id, user_id] = upper                                   <L 27>
        // wp::array_store(var_user_joint_pos_limits_upper, var_3, var_6, var_15);
        // user_joint_pos_limits[env_id, user_id] = limits                                        <L 28>
        // wp::array_store(var_user_joint_pos_limits, var_3, var_6, var_10);
        // backend_joint_pos_limits_lower[env_id, backend_id] = lower                             <L 29>
        // wp::array_store(var_backend_joint_pos_limits_lower, var_3, var_8, var_13);
        // backend_joint_pos_limits_upper[env_id, backend_id] = upper                             <L 30>
        // wp::array_store(var_backend_joint_pos_limits_upper, var_3, var_8, var_15);
        // if (default_joint_pos[env_id, user_id] < lower) or default_joint_pos[env_id, user_id] > upper:       <L 32>
        var_17 = wp::address(var_default_joint_pos, var_3, var_6);
        var_19 = wp::load(var_17);
        var_18 = (var_19 < var_13);
        var_16 = var_18;
        if (!var_16) {
            var_20 = wp::address(var_default_joint_pos, var_3, var_6);
            var_22 = wp::load(var_20);
            var_21 = (var_22 > var_15);
            var_16 = var_16 || var_21;
        }
        if (var_16) {
            // wp.atomic_add(clamped_defaults, 0, 1)                                              <L 33>
            // var_25 = wp::atomic_add(var_clamped_defaults, var_23, var_24);
            // default_joint_pos[env_id, user_id] = wp.clamp(default_joint_pos[env_id, user_id], lower, upper)       <L 34>
            var_26 = wp::address(var_default_joint_pos, var_3, var_6);
            var_28 = wp::load(var_26);
            var_27 = wp::clamp(var_28, var_13, var_15);
            // wp::array_store(var_default_joint_pos, var_3, var_6, var_27);
        }
        // soft_joint_pos_limits[env_id, user_id] = compute_soft_joint_pos_limits_func(limits, soft_limit_factor)       <L 36>
        var_29 = compute_soft_joint_pos_limits_func_0(var_10, var_soft_limit_factor);
        // wp::array_store(var_soft_joint_pos_limits, var_3, var_6, var_29);
        //---------
        // reverse
        wp::adj_array_store(var_soft_joint_pos_limits, var_3, var_6, var_29, adj_soft_joint_pos_limits, adj_3, adj_6, adj_29);
        adj_compute_soft_joint_pos_limits_func_0(var_10, var_soft_limit_factor, adj_10, adj_soft_limit_factor, adj_29);
        // adj: soft_joint_pos_limits[env_id, user_id] = compute_soft_joint_pos_limits_func(limits, soft_limit_factor)  <L 36>
        if (var_16) {
            wp::adj_array_store(var_default_joint_pos, var_3, var_6, var_27, adj_default_joint_pos, adj_3, adj_6, adj_27);
            wp::adj_clamp(var_28, var_13, var_15, adj_26, adj_13, adj_15, adj_27);
            wp::adj_address(var_default_joint_pos, var_3, var_6, adj_default_joint_pos, adj_3, adj_6, adj_26);
            // adj: default_joint_pos[env_id, user_id] = wp.clamp(default_joint_pos[env_id, user_id], lower, upper)  <L 34>
            wp::adj_atomic_add(var_clamped_defaults, var_23, var_24, adj_clamped_defaults, adj_23, adj_24, adj_25);
            // adj: wp.atomic_add(clamped_defaults, 0, 1)                                         <L 33>
        }
        if (!var_16) {
            wp::adj_address(var_default_joint_pos, var_3, var_6, adj_default_joint_pos, adj_3, adj_6, adj_20);
        }
        wp::adj_address(var_default_joint_pos, var_3, var_6, adj_default_joint_pos, adj_3, adj_6, adj_17);
        // adj: if (default_joint_pos[env_id, user_id] < lower) or default_joint_pos[env_id, user_id] > upper:  <L 32>
        wp::adj_array_store(var_backend_joint_pos_limits_upper, var_3, var_8, var_15, adj_backend_joint_pos_limits_upper, adj_3, adj_8, adj_15);
        // adj: backend_joint_pos_limits_upper[env_id, backend_id] = upper                        <L 30>
        wp::adj_array_store(var_backend_joint_pos_limits_lower, var_3, var_8, var_13, adj_backend_joint_pos_limits_lower, adj_3, adj_8, adj_13);
        // adj: backend_joint_pos_limits_lower[env_id, backend_id] = lower                        <L 29>
        wp::adj_array_store(var_user_joint_pos_limits, var_3, var_6, var_10, adj_user_joint_pos_limits, adj_3, adj_6, adj_10);
        // adj: user_joint_pos_limits[env_id, user_id] = limits                                   <L 28>
        wp::adj_array_store(var_user_joint_pos_limits_upper, var_3, var_6, var_15, adj_user_joint_pos_limits_upper, adj_3, adj_6, adj_15);
        // adj: user_joint_pos_limits_upper[env_id, user_id] = upper                              <L 27>
        wp::adj_array_store(var_user_joint_pos_limits_lower, var_3, var_6, var_13, adj_user_joint_pos_limits_lower, adj_3, adj_6, adj_13);
        // adj: user_joint_pos_limits_lower[env_id, user_id] = lower                              <L 26>
        wp::adj_extract(var_10, var_14, adj_10, adj_14, adj_15);
        // adj: upper = limits[1]                                                                 <L 24>
        wp::adj_extract(var_10, var_12, adj_10, adj_12, adj_13);
        // adj: lower = limits[0]                                                                 <L 23>
        wp::adj_copy(var_11, adj_9, adj_10);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_9);
        // adj: limits = in_data[i, j]                                                            <L 22>
        adj_resolve_backend_index_0(var_6, var_user_to_backend, var_has_ordering, adj_6, adj_user_to_backend, adj_has_ordering, adj_8);
        // adj: backend_id = resolve_backend_index(user_id, user_to_backend, has_ordering)        <L 21>
        wp::adj_address(var_user_ids, var_1, adj_user_ids, adj_1, adj_5);
        // adj: user_id = wp.int32(user_ids[j])                                                   <L 20>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 19>
        // adj: i, j = wp.tid()                                                                   <L 18>
        // adj: def write_joint_limit_data_to_user_and_backend_index(                             <L 1>
        continue;
    }
}



extern "C" __global__ void update_targets_acac1b6f_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_source_joint_positions,
    wp::array_t<wp::float32> var_source_joint_velocities,
    wp::array_t<wp::float32> var_source_joint_efforts,
    wp::array_t<wp::int32> var_joint_indices,
    wp::array_t<wp::float32> var_target_joint_positions,
    wp::array_t<wp::float32> var_target_joint_velocities,
    wp::array_t<wp::float32> var_target_joint_efforts)
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
        wp::float32* var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::float32 var_5;
        wp::float32* var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::float32 var_9;
        wp::float32* var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::float32 var_13;
        //---------
        // forward
        // def update_targets(                                                                    <L 593>
        // i, j = wp.tid()                                                                        <L 624>
        builtin_tid2d(var_0, var_1);
        // if source_joint_positions:                                                             <L 625>
        if (var_source_joint_positions) {
            // target_joint_positions[i, joint_indices[j]] = source_joint_positions[i, j]         <L 626>
            var_2 = wp::address(var_source_joint_positions, var_0, var_1);
            var_3 = wp::address(var_joint_indices, var_1);
            var_4 = wp::load(var_3);
            var_5 = wp::load(var_2);
            wp::array_store(var_target_joint_positions, var_0, var_4, var_5);
        }
        // if source_joint_velocities:                                                            <L 627>
        if (var_source_joint_velocities) {
            // target_joint_velocities[i, joint_indices[j]] = source_joint_velocities[i, j]       <L 628>
            var_6 = wp::address(var_source_joint_velocities, var_0, var_1);
            var_7 = wp::address(var_joint_indices, var_1);
            var_8 = wp::load(var_7);
            var_9 = wp::load(var_6);
            wp::array_store(var_target_joint_velocities, var_0, var_8, var_9);
        }
        // if source_joint_efforts:                                                               <L 629>
        if (var_source_joint_efforts) {
            // target_joint_efforts[i, joint_indices[j]] = source_joint_efforts[i, j]             <L 630>
            var_10 = wp::address(var_source_joint_efforts, var_0, var_1);
            var_11 = wp::address(var_joint_indices, var_1);
            var_12 = wp::load(var_11);
            var_13 = wp::load(var_10);
            wp::array_store(var_target_joint_efforts, var_0, var_12, var_13);
        }
    }
}



extern "C" __global__ void update_targets_acac1b6f_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_source_joint_positions,
    wp::array_t<wp::float32> var_source_joint_velocities,
    wp::array_t<wp::float32> var_source_joint_efforts,
    wp::array_t<wp::int32> var_joint_indices,
    wp::array_t<wp::float32> var_target_joint_positions,
    wp::array_t<wp::float32> var_target_joint_velocities,
    wp::array_t<wp::float32> var_target_joint_efforts,
    wp::array_t<wp::float32> adj_source_joint_positions,
    wp::array_t<wp::float32> adj_source_joint_velocities,
    wp::array_t<wp::float32> adj_source_joint_efforts,
    wp::array_t<wp::int32> adj_joint_indices,
    wp::array_t<wp::float32> adj_target_joint_positions,
    wp::array_t<wp::float32> adj_target_joint_velocities,
    wp::array_t<wp::float32> adj_target_joint_efforts)
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
        wp::float32* var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::float32 var_5;
        wp::float32* var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::float32 var_9;
        wp::float32* var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::float32 var_13;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::float32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::float32 adj_13 = {};
        //---------
        // forward
        // def update_targets(                                                                    <L 593>
        // i, j = wp.tid()                                                                        <L 624>
        builtin_tid2d(var_0, var_1);
        // if source_joint_positions:                                                             <L 625>
        if (var_source_joint_positions) {
            // target_joint_positions[i, joint_indices[j]] = source_joint_positions[i, j]         <L 626>
            var_2 = wp::address(var_source_joint_positions, var_0, var_1);
            var_3 = wp::address(var_joint_indices, var_1);
            var_4 = wp::load(var_3);
            var_5 = wp::load(var_2);
            // wp::array_store(var_target_joint_positions, var_0, var_4, var_5);
        }
        // if source_joint_velocities:                                                            <L 627>
        if (var_source_joint_velocities) {
            // target_joint_velocities[i, joint_indices[j]] = source_joint_velocities[i, j]       <L 628>
            var_6 = wp::address(var_source_joint_velocities, var_0, var_1);
            var_7 = wp::address(var_joint_indices, var_1);
            var_8 = wp::load(var_7);
            var_9 = wp::load(var_6);
            // wp::array_store(var_target_joint_velocities, var_0, var_8, var_9);
        }
        // if source_joint_efforts:                                                               <L 629>
        if (var_source_joint_efforts) {
            // target_joint_efforts[i, joint_indices[j]] = source_joint_efforts[i, j]             <L 630>
            var_10 = wp::address(var_source_joint_efforts, var_0, var_1);
            var_11 = wp::address(var_joint_indices, var_1);
            var_12 = wp::load(var_11);
            var_13 = wp::load(var_10);
            // wp::array_store(var_target_joint_efforts, var_0, var_12, var_13);
        }
        //---------
        // reverse
        if (var_source_joint_efforts) {
            wp::adj_array_store(var_target_joint_efforts, var_0, var_12, var_13, adj_target_joint_efforts, adj_0, adj_11, adj_10);
            wp::adj_address(var_joint_indices, var_1, adj_joint_indices, adj_1, adj_11);
            wp::adj_address(var_source_joint_efforts, var_0, var_1, adj_source_joint_efforts, adj_0, adj_1, adj_10);
            // adj: target_joint_efforts[i, joint_indices[j]] = source_joint_efforts[i, j]        <L 630>
        }
        // adj: if source_joint_efforts:                                                          <L 629>
        if (var_source_joint_velocities) {
            wp::adj_array_store(var_target_joint_velocities, var_0, var_8, var_9, adj_target_joint_velocities, adj_0, adj_7, adj_6);
            wp::adj_address(var_joint_indices, var_1, adj_joint_indices, adj_1, adj_7);
            wp::adj_address(var_source_joint_velocities, var_0, var_1, adj_source_joint_velocities, adj_0, adj_1, adj_6);
            // adj: target_joint_velocities[i, joint_indices[j]] = source_joint_velocities[i, j]  <L 628>
        }
        // adj: if source_joint_velocities:                                                       <L 627>
        if (var_source_joint_positions) {
            wp::adj_array_store(var_target_joint_positions, var_0, var_4, var_5, adj_target_joint_positions, adj_0, adj_3, adj_2);
            wp::adj_address(var_joint_indices, var_1, adj_joint_indices, adj_1, adj_3);
            wp::adj_address(var_source_joint_positions, var_0, var_1, adj_source_joint_positions, adj_0, adj_1, adj_2);
            // adj: target_joint_positions[i, joint_indices[j]] = source_joint_positions[i, j]    <L 626>
        }
        // adj: if source_joint_positions:                                                        <L 625>
        // adj: i, j = wp.tid()                                                                   <L 624>
        // adj: def update_targets(                                                               <L 593>
        continue;
    }
}



extern "C" __global__ void float_data_to_buffer_with_indices_af435e43_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    wp::array_t<wp::float32> var_out_data)
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
        //---------
        // forward
        // def float_data_to_buffer_with_indices(                                                 <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 19>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 20>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // out_data[env_id, joint_id] = in_data                                                   <L 21>
        wp::array_store(var_out_data, var_3, var_6, var_in_data);
    }
}



extern "C" __global__ void float_data_to_buffer_with_indices_af435e43_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    wp::array_t<wp::float32> var_out_data,
    wp::float32 adj_in_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int64> adj_joint_ids,
    wp::array_t<wp::float32> adj_out_data)
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
        //---------
        // forward
        // def float_data_to_buffer_with_indices(                                                 <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 19>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 20>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // out_data[env_id, joint_id] = in_data                                                   <L 21>
        // wp::array_store(var_out_data, var_3, var_6, var_in_data);
        //---------
        // reverse
        wp::adj_array_store(var_out_data, var_3, var_6, var_in_data, adj_out_data, adj_3, adj_6, adj_in_data);
        // adj: out_data[env_id, joint_id] = in_data                                              <L 21>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 20>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 19>
        // adj: i, j = wp.tid()                                                                   <L 18>
        // adj: def float_data_to_buffer_with_indices(                                            <L 1>
        continue;
    }
}



extern "C" __global__ void write_joint_friction_data_to_buffer_b0650efb_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_friction,
    wp::array_t<wp::float32> var_in_dynamic_friction,
    wp::array_t<wp::float32> var_in_viscous_friction,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    bool var_from_mask,
    wp::array_t<wp::float32> var_out_friction,
    wp::array_t<wp::float32> var_out_dynamic_friction,
    wp::array_t<wp::float32> var_out_viscous_friction,
    wp::array_t<wp::float32> var_friction_props)
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
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32* var_16;
        wp::float32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32* var_20;
        const wp::int32 var_21 = 0;
        wp::float32 var_22;
        wp::float32* var_23;
        const wp::int32 var_24 = 1;
        wp::float32 var_25;
        wp::float32* var_26;
        const wp::int32 var_27 = 2;
        wp::float32 var_28;
        //---------
        // forward
        // def write_joint_friction_data_to_buffer(                                               <L 1>
        // i, j = wp.tid()                                                                        <L 43>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 44>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 45>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 47>
        if (var_from_mask) {
            // out_friction[env_id, joint_id] = in_friction[env_id, joint_id]                     <L 48>
            var_8 = wp::address(var_in_friction, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_out_friction, var_3, var_6, var_9);
            // if in_dynamic_friction:                                                            <L 49>
            if (var_in_dynamic_friction) {
                // out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[env_id, joint_id]       <L 50>
                var_10 = wp::address(var_in_dynamic_friction, var_3, var_6);
                var_11 = wp::load(var_10);
                wp::array_store(var_out_dynamic_friction, var_3, var_6, var_11);
            }
            // if in_viscous_friction:                                                            <L 51>
            if (var_in_viscous_friction) {
                // out_viscous_friction[env_id, joint_id] = in_viscous_friction[env_id, joint_id]       <L 52>
                var_12 = wp::address(var_in_viscous_friction, var_3, var_6);
                var_13 = wp::load(var_12);
                wp::array_store(var_out_viscous_friction, var_3, var_6, var_13);
            }
        }
        if (!var_from_mask) {
            // out_friction[env_id, joint_id] = in_friction[i, j]                                 <L 54>
            var_14 = wp::address(var_in_friction, var_0, var_1);
            var_15 = wp::load(var_14);
            wp::array_store(var_out_friction, var_3, var_6, var_15);
            // if in_dynamic_friction:                                                            <L 55>
            if (var_in_dynamic_friction) {
                // out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[i, j]             <L 56>
                var_16 = wp::address(var_in_dynamic_friction, var_0, var_1);
                var_17 = wp::load(var_16);
                wp::array_store(var_out_dynamic_friction, var_3, var_6, var_17);
            }
            // if in_viscous_friction:                                                            <L 57>
            if (var_in_viscous_friction) {
                // out_viscous_friction[env_id, joint_id] = in_viscous_friction[i, j]             <L 58>
                var_18 = wp::address(var_in_viscous_friction, var_0, var_1);
                var_19 = wp::load(var_18);
                wp::array_store(var_out_viscous_friction, var_3, var_6, var_19);
            }
        }
        // friction_props[env_id, joint_id, 0] = out_friction[env_id, joint_id]                   <L 60>
        var_20 = wp::address(var_out_friction, var_3, var_6);
        var_22 = wp::load(var_20);
        wp::array_store(var_friction_props, var_3, var_6, var_21, var_22);
        // if in_dynamic_friction:                                                                <L 61>
        if (var_in_dynamic_friction) {
            // friction_props[env_id, joint_id, 1] = out_dynamic_friction[env_id, joint_id]       <L 62>
            var_23 = wp::address(var_out_dynamic_friction, var_3, var_6);
            var_25 = wp::load(var_23);
            wp::array_store(var_friction_props, var_3, var_6, var_24, var_25);
        }
        // if in_viscous_friction:                                                                <L 63>
        if (var_in_viscous_friction) {
            // friction_props[env_id, joint_id, 2] = out_viscous_friction[env_id, joint_id]       <L 64>
            var_26 = wp::address(var_out_viscous_friction, var_3, var_6);
            var_28 = wp::load(var_26);
            wp::array_store(var_friction_props, var_3, var_6, var_27, var_28);
        }
    }
}



extern "C" __global__ void write_joint_friction_data_to_buffer_b0650efb_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_friction,
    wp::array_t<wp::float32> var_in_dynamic_friction,
    wp::array_t<wp::float32> var_in_viscous_friction,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    bool var_from_mask,
    wp::array_t<wp::float32> var_out_friction,
    wp::array_t<wp::float32> var_out_dynamic_friction,
    wp::array_t<wp::float32> var_out_viscous_friction,
    wp::array_t<wp::float32> var_friction_props,
    wp::array_t<wp::float32> adj_in_friction,
    wp::array_t<wp::float32> adj_in_dynamic_friction,
    wp::array_t<wp::float32> adj_in_viscous_friction,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int64> adj_joint_ids,
    bool adj_from_mask,
    wp::array_t<wp::float32> adj_out_friction,
    wp::array_t<wp::float32> adj_out_dynamic_friction,
    wp::array_t<wp::float32> adj_out_viscous_friction,
    wp::array_t<wp::float32> adj_friction_props)
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
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32* var_16;
        wp::float32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32* var_20;
        const wp::int32 var_21 = 0;
        wp::float32 var_22;
        wp::float32* var_23;
        const wp::int32 var_24 = 1;
        wp::float32 var_25;
        wp::float32* var_26;
        const wp::int32 var_27 = 2;
        wp::float32 var_28;
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
        wp::int32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::float32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::float32 adj_28 = {};
        //---------
        // forward
        // def write_joint_friction_data_to_buffer(                                               <L 1>
        // i, j = wp.tid()                                                                        <L 43>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 44>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 45>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 47>
        if (var_from_mask) {
            // out_friction[env_id, joint_id] = in_friction[env_id, joint_id]                     <L 48>
            var_8 = wp::address(var_in_friction, var_3, var_6);
            var_9 = wp::load(var_8);
            // wp::array_store(var_out_friction, var_3, var_6, var_9);
            // if in_dynamic_friction:                                                            <L 49>
            if (var_in_dynamic_friction) {
                // out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[env_id, joint_id]       <L 50>
                var_10 = wp::address(var_in_dynamic_friction, var_3, var_6);
                var_11 = wp::load(var_10);
                // wp::array_store(var_out_dynamic_friction, var_3, var_6, var_11);
            }
            // if in_viscous_friction:                                                            <L 51>
            if (var_in_viscous_friction) {
                // out_viscous_friction[env_id, joint_id] = in_viscous_friction[env_id, joint_id]       <L 52>
                var_12 = wp::address(var_in_viscous_friction, var_3, var_6);
                var_13 = wp::load(var_12);
                // wp::array_store(var_out_viscous_friction, var_3, var_6, var_13);
            }
        }
        if (!var_from_mask) {
            // out_friction[env_id, joint_id] = in_friction[i, j]                                 <L 54>
            var_14 = wp::address(var_in_friction, var_0, var_1);
            var_15 = wp::load(var_14);
            // wp::array_store(var_out_friction, var_3, var_6, var_15);
            // if in_dynamic_friction:                                                            <L 55>
            if (var_in_dynamic_friction) {
                // out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[i, j]             <L 56>
                var_16 = wp::address(var_in_dynamic_friction, var_0, var_1);
                var_17 = wp::load(var_16);
                // wp::array_store(var_out_dynamic_friction, var_3, var_6, var_17);
            }
            // if in_viscous_friction:                                                            <L 57>
            if (var_in_viscous_friction) {
                // out_viscous_friction[env_id, joint_id] = in_viscous_friction[i, j]             <L 58>
                var_18 = wp::address(var_in_viscous_friction, var_0, var_1);
                var_19 = wp::load(var_18);
                // wp::array_store(var_out_viscous_friction, var_3, var_6, var_19);
            }
        }
        // friction_props[env_id, joint_id, 0] = out_friction[env_id, joint_id]                   <L 60>
        var_20 = wp::address(var_out_friction, var_3, var_6);
        var_22 = wp::load(var_20);
        // wp::array_store(var_friction_props, var_3, var_6, var_21, var_22);
        // if in_dynamic_friction:                                                                <L 61>
        if (var_in_dynamic_friction) {
            // friction_props[env_id, joint_id, 1] = out_dynamic_friction[env_id, joint_id]       <L 62>
            var_23 = wp::address(var_out_dynamic_friction, var_3, var_6);
            var_25 = wp::load(var_23);
            // wp::array_store(var_friction_props, var_3, var_6, var_24, var_25);
        }
        // if in_viscous_friction:                                                                <L 63>
        if (var_in_viscous_friction) {
            // friction_props[env_id, joint_id, 2] = out_viscous_friction[env_id, joint_id]       <L 64>
            var_26 = wp::address(var_out_viscous_friction, var_3, var_6);
            var_28 = wp::load(var_26);
            // wp::array_store(var_friction_props, var_3, var_6, var_27, var_28);
        }
        //---------
        // reverse
        if (var_in_viscous_friction) {
            wp::adj_array_store(var_friction_props, var_3, var_6, var_27, var_28, adj_friction_props, adj_3, adj_6, adj_27, adj_26);
            wp::adj_address(var_out_viscous_friction, var_3, var_6, adj_out_viscous_friction, adj_3, adj_6, adj_26);
            // adj: friction_props[env_id, joint_id, 2] = out_viscous_friction[env_id, joint_id]  <L 64>
        }
        // adj: if in_viscous_friction:                                                           <L 63>
        if (var_in_dynamic_friction) {
            wp::adj_array_store(var_friction_props, var_3, var_6, var_24, var_25, adj_friction_props, adj_3, adj_6, adj_24, adj_23);
            wp::adj_address(var_out_dynamic_friction, var_3, var_6, adj_out_dynamic_friction, adj_3, adj_6, adj_23);
            // adj: friction_props[env_id, joint_id, 1] = out_dynamic_friction[env_id, joint_id]  <L 62>
        }
        // adj: if in_dynamic_friction:                                                           <L 61>
        wp::adj_array_store(var_friction_props, var_3, var_6, var_21, var_22, adj_friction_props, adj_3, adj_6, adj_21, adj_20);
        wp::adj_address(var_out_friction, var_3, var_6, adj_out_friction, adj_3, adj_6, adj_20);
        // adj: friction_props[env_id, joint_id, 0] = out_friction[env_id, joint_id]              <L 60>
        if (!var_from_mask) {
            if (var_in_viscous_friction) {
                wp::adj_array_store(var_out_viscous_friction, var_3, var_6, var_19, adj_out_viscous_friction, adj_3, adj_6, adj_18);
                wp::adj_address(var_in_viscous_friction, var_0, var_1, adj_in_viscous_friction, adj_0, adj_1, adj_18);
                // adj: out_viscous_friction[env_id, joint_id] = in_viscous_friction[i, j]        <L 58>
            }
            // adj: if in_viscous_friction:                                                       <L 57>
            if (var_in_dynamic_friction) {
                wp::adj_array_store(var_out_dynamic_friction, var_3, var_6, var_17, adj_out_dynamic_friction, adj_3, adj_6, adj_16);
                wp::adj_address(var_in_dynamic_friction, var_0, var_1, adj_in_dynamic_friction, adj_0, adj_1, adj_16);
                // adj: out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[i, j]        <L 56>
            }
            // adj: if in_dynamic_friction:                                                       <L 55>
            wp::adj_array_store(var_out_friction, var_3, var_6, var_15, adj_out_friction, adj_3, adj_6, adj_14);
            wp::adj_address(var_in_friction, var_0, var_1, adj_in_friction, adj_0, adj_1, adj_14);
            // adj: out_friction[env_id, joint_id] = in_friction[i, j]                            <L 54>
        }
        if (var_from_mask) {
            if (var_in_viscous_friction) {
                wp::adj_array_store(var_out_viscous_friction, var_3, var_6, var_13, adj_out_viscous_friction, adj_3, adj_6, adj_12);
                wp::adj_address(var_in_viscous_friction, var_3, var_6, adj_in_viscous_friction, adj_3, adj_6, adj_12);
                // adj: out_viscous_friction[env_id, joint_id] = in_viscous_friction[env_id, joint_id]  <L 52>
            }
            // adj: if in_viscous_friction:                                                       <L 51>
            if (var_in_dynamic_friction) {
                wp::adj_array_store(var_out_dynamic_friction, var_3, var_6, var_11, adj_out_dynamic_friction, adj_3, adj_6, adj_10);
                wp::adj_address(var_in_dynamic_friction, var_3, var_6, adj_in_dynamic_friction, adj_3, adj_6, adj_10);
                // adj: out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[env_id, joint_id]  <L 50>
            }
            // adj: if in_dynamic_friction:                                                       <L 49>
            wp::adj_array_store(var_out_friction, var_3, var_6, var_9, adj_out_friction, adj_3, adj_6, adj_8);
            wp::adj_address(var_in_friction, var_3, var_6, adj_in_friction, adj_3, adj_6, adj_8);
            // adj: out_friction[env_id, joint_id] = in_friction[env_id, joint_id]                <L 48>
        }
        // adj: if from_mask:                                                                     <L 47>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 45>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 44>
        // adj: i, j = wp.tid()                                                                   <L 43>
        // adj: def write_joint_friction_data_to_buffer(                                          <L 1>
        continue;
    }
}



extern "C" __global__ void write_joint_vel_data_index_b646a6ab_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::array_t<wp::float32> var_joint_acc)
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
        wp::float32* var_10;
        wp::float32 var_11;
        const wp::float32 var_12 = 0.0;
        //---------
        // forward
        // def write_joint_vel_data_index(                                                        <L 1>
        // i, j = wp.tid()                                                                        <L 27>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 28>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 29>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // joint_vel[env_id, joint_id] = in_data[i, j]                                            <L 30>
        var_8 = wp::address(var_in_data, var_0, var_1);
        var_9 = wp::load(var_8);
        wp::array_store(var_joint_vel, var_3, var_6, var_9);
        // prev_joint_vel[env_id, joint_id] = in_data[i, j]                                       <L 31>
        var_10 = wp::address(var_in_data, var_0, var_1);
        var_11 = wp::load(var_10);
        wp::array_store(var_prev_joint_vel, var_3, var_6, var_11);
        // joint_acc[env_id, joint_id] = 0.0                                                      <L 32>
        wp::array_store(var_joint_acc, var_3, var_6, var_12);
    }
}



extern "C" __global__ void write_joint_vel_data_index_b646a6ab_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::array_t<wp::float32> var_joint_acc,
    wp::array_t<wp::float32> adj_in_data,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int64> adj_joint_ids,
    wp::array_t<wp::float32> adj_joint_vel,
    wp::array_t<wp::float32> adj_prev_joint_vel,
    wp::array_t<wp::float32> adj_joint_acc)
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
        wp::float32* var_10;
        wp::float32 var_11;
        const wp::float32 var_12 = 0.0;
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
        //---------
        // forward
        // def write_joint_vel_data_index(                                                        <L 1>
        // i, j = wp.tid()                                                                        <L 27>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 28>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 29>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // joint_vel[env_id, joint_id] = in_data[i, j]                                            <L 30>
        var_8 = wp::address(var_in_data, var_0, var_1);
        var_9 = wp::load(var_8);
        // wp::array_store(var_joint_vel, var_3, var_6, var_9);
        // prev_joint_vel[env_id, joint_id] = in_data[i, j]                                       <L 31>
        var_10 = wp::address(var_in_data, var_0, var_1);
        var_11 = wp::load(var_10);
        // wp::array_store(var_prev_joint_vel, var_3, var_6, var_11);
        // joint_acc[env_id, joint_id] = 0.0                                                      <L 32>
        // wp::array_store(var_joint_acc, var_3, var_6, var_12);
        //---------
        // reverse
        wp::adj_array_store(var_joint_acc, var_3, var_6, var_12, adj_joint_acc, adj_3, adj_6, adj_12);
        // adj: joint_acc[env_id, joint_id] = 0.0                                                 <L 32>
        wp::adj_array_store(var_prev_joint_vel, var_3, var_6, var_11, adj_prev_joint_vel, adj_3, adj_6, adj_10);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_10);
        // adj: prev_joint_vel[env_id, joint_id] = in_data[i, j]                                  <L 31>
        wp::adj_array_store(var_joint_vel, var_3, var_6, var_9, adj_joint_vel, adj_3, adj_6, adj_8);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_8);
        // adj: joint_vel[env_id, joint_id] = in_data[i, j]                                       <L 30>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 29>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 28>
        // adj: i, j = wp.tid()                                                                   <L 27>
        // adj: def write_joint_vel_data_index(                                                   <L 1>
        continue;
    }
}



extern "C" __global__ void extract_friction_properties_bf847ed9_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_friction_props,
    wp::array_t<wp::float32> var_out_friction,
    wp::array_t<wp::float32> var_out_dynamic_friction,
    wp::array_t<wp::float32> var_out_viscous_friction)
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
        const wp::int32 var_2 = 0;
        wp::float32* var_3;
        wp::float32 var_4;
        const wp::int32 var_5 = 1;
        wp::float32* var_6;
        wp::float32 var_7;
        const wp::int32 var_8 = 2;
        wp::float32* var_9;
        wp::float32 var_10;
        //---------
        // forward
        // def extract_friction_properties(                                                       <L 634>
        // i, j = wp.tid()                                                                        <L 656>
        builtin_tid2d(var_0, var_1);
        // out_friction[i, j] = friction_props[i, j, 0]                                           <L 657>
        var_3 = wp::address(var_friction_props, var_0, var_1, var_2);
        var_4 = wp::load(var_3);
        wp::array_store(var_out_friction, var_0, var_1, var_4);
        // out_dynamic_friction[i, j] = friction_props[i, j, 1]                                   <L 658>
        var_6 = wp::address(var_friction_props, var_0, var_1, var_5);
        var_7 = wp::load(var_6);
        wp::array_store(var_out_dynamic_friction, var_0, var_1, var_7);
        // out_viscous_friction[i, j] = friction_props[i, j, 2]                                   <L 659>
        var_9 = wp::address(var_friction_props, var_0, var_1, var_8);
        var_10 = wp::load(var_9);
        wp::array_store(var_out_viscous_friction, var_0, var_1, var_10);
    }
}



extern "C" __global__ void extract_friction_properties_bf847ed9_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_friction_props,
    wp::array_t<wp::float32> var_out_friction,
    wp::array_t<wp::float32> var_out_dynamic_friction,
    wp::array_t<wp::float32> var_out_viscous_friction,
    wp::array_t<wp::float32> adj_friction_props,
    wp::array_t<wp::float32> adj_out_friction,
    wp::array_t<wp::float32> adj_out_dynamic_friction,
    wp::array_t<wp::float32> adj_out_viscous_friction)
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
        const wp::int32 var_2 = 0;
        wp::float32* var_3;
        wp::float32 var_4;
        const wp::int32 var_5 = 1;
        wp::float32* var_6;
        wp::float32 var_7;
        const wp::int32 var_8 = 2;
        wp::float32* var_9;
        wp::float32 var_10;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::float32 adj_3 = {};
        wp::float32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        //---------
        // forward
        // def extract_friction_properties(                                                       <L 634>
        // i, j = wp.tid()                                                                        <L 656>
        builtin_tid2d(var_0, var_1);
        // out_friction[i, j] = friction_props[i, j, 0]                                           <L 657>
        var_3 = wp::address(var_friction_props, var_0, var_1, var_2);
        var_4 = wp::load(var_3);
        // wp::array_store(var_out_friction, var_0, var_1, var_4);
        // out_dynamic_friction[i, j] = friction_props[i, j, 1]                                   <L 658>
        var_6 = wp::address(var_friction_props, var_0, var_1, var_5);
        var_7 = wp::load(var_6);
        // wp::array_store(var_out_dynamic_friction, var_0, var_1, var_7);
        // out_viscous_friction[i, j] = friction_props[i, j, 2]                                   <L 659>
        var_9 = wp::address(var_friction_props, var_0, var_1, var_8);
        var_10 = wp::load(var_9);
        // wp::array_store(var_out_viscous_friction, var_0, var_1, var_10);
        //---------
        // reverse
        wp::adj_array_store(var_out_viscous_friction, var_0, var_1, var_10, adj_out_viscous_friction, adj_0, adj_1, adj_9);
        wp::adj_address(var_friction_props, var_0, var_1, var_8, adj_friction_props, adj_0, adj_1, adj_8, adj_9);
        // adj: out_viscous_friction[i, j] = friction_props[i, j, 2]                              <L 659>
        wp::adj_array_store(var_out_dynamic_friction, var_0, var_1, var_7, adj_out_dynamic_friction, adj_0, adj_1, adj_6);
        wp::adj_address(var_friction_props, var_0, var_1, var_5, adj_friction_props, adj_0, adj_1, adj_5, adj_6);
        // adj: out_dynamic_friction[i, j] = friction_props[i, j, 1]                              <L 658>
        wp::adj_array_store(var_out_friction, var_0, var_1, var_4, adj_out_friction, adj_0, adj_1, adj_3);
        wp::adj_address(var_friction_props, var_0, var_1, var_2, adj_friction_props, adj_0, adj_1, adj_2, adj_3);
        // adj: out_friction[i, j] = friction_props[i, j, 0]                                      <L 657>
        // adj: i, j = wp.tid()                                                                   <L 656>
        // adj: def extract_friction_properties(                                                  <L 634>
        continue;
    }
}



extern "C" __global__ void write_joint_state_data_index_c5990926_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_pos_data,
    wp::array_t<wp::float32> var_vel_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    wp::array_t<wp::float32> var_joint_pos,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::array_t<wp::float32> var_joint_acc)
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
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        const wp::float32 var_14 = 0.0;
        //---------
        // forward
        // def write_joint_state_data_index(                                                      <L 1>
        // i, j = wp.tid()                                                                        <L 26>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 27>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 28>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // joint_pos[env_id, joint_id] = pos_data[i, j]                                           <L 29>
        var_8 = wp::address(var_pos_data, var_0, var_1);
        var_9 = wp::load(var_8);
        wp::array_store(var_joint_pos, var_3, var_6, var_9);
        // joint_vel[env_id, joint_id] = vel_data[i, j]                                           <L 30>
        var_10 = wp::address(var_vel_data, var_0, var_1);
        var_11 = wp::load(var_10);
        wp::array_store(var_joint_vel, var_3, var_6, var_11);
        // prev_joint_vel[env_id, joint_id] = vel_data[i, j]                                      <L 31>
        var_12 = wp::address(var_vel_data, var_0, var_1);
        var_13 = wp::load(var_12);
        wp::array_store(var_prev_joint_vel, var_3, var_6, var_13);
        // joint_acc[env_id, joint_id] = 0.0                                                      <L 32>
        wp::array_store(var_joint_acc, var_3, var_6, var_14);
    }
}



extern "C" __global__ void write_joint_state_data_index_c5990926_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_pos_data,
    wp::array_t<wp::float32> var_vel_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    wp::array_t<wp::float32> var_joint_pos,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::array_t<wp::float32> var_joint_acc,
    wp::array_t<wp::float32> adj_pos_data,
    wp::array_t<wp::float32> adj_vel_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int64> adj_joint_ids,
    wp::array_t<wp::float32> adj_joint_pos,
    wp::array_t<wp::float32> adj_joint_vel,
    wp::array_t<wp::float32> adj_prev_joint_vel,
    wp::array_t<wp::float32> adj_joint_acc)
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
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        const wp::float32 var_14 = 0.0;
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
        //---------
        // forward
        // def write_joint_state_data_index(                                                      <L 1>
        // i, j = wp.tid()                                                                        <L 26>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 27>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 28>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // joint_pos[env_id, joint_id] = pos_data[i, j]                                           <L 29>
        var_8 = wp::address(var_pos_data, var_0, var_1);
        var_9 = wp::load(var_8);
        // wp::array_store(var_joint_pos, var_3, var_6, var_9);
        // joint_vel[env_id, joint_id] = vel_data[i, j]                                           <L 30>
        var_10 = wp::address(var_vel_data, var_0, var_1);
        var_11 = wp::load(var_10);
        // wp::array_store(var_joint_vel, var_3, var_6, var_11);
        // prev_joint_vel[env_id, joint_id] = vel_data[i, j]                                      <L 31>
        var_12 = wp::address(var_vel_data, var_0, var_1);
        var_13 = wp::load(var_12);
        // wp::array_store(var_prev_joint_vel, var_3, var_6, var_13);
        // joint_acc[env_id, joint_id] = 0.0                                                      <L 32>
        // wp::array_store(var_joint_acc, var_3, var_6, var_14);
        //---------
        // reverse
        wp::adj_array_store(var_joint_acc, var_3, var_6, var_14, adj_joint_acc, adj_3, adj_6, adj_14);
        // adj: joint_acc[env_id, joint_id] = 0.0                                                 <L 32>
        wp::adj_array_store(var_prev_joint_vel, var_3, var_6, var_13, adj_prev_joint_vel, adj_3, adj_6, adj_12);
        wp::adj_address(var_vel_data, var_0, var_1, adj_vel_data, adj_0, adj_1, adj_12);
        // adj: prev_joint_vel[env_id, joint_id] = vel_data[i, j]                                 <L 31>
        wp::adj_array_store(var_joint_vel, var_3, var_6, var_11, adj_joint_vel, adj_3, adj_6, adj_10);
        wp::adj_address(var_vel_data, var_0, var_1, adj_vel_data, adj_0, adj_1, adj_10);
        // adj: joint_vel[env_id, joint_id] = vel_data[i, j]                                      <L 30>
        wp::adj_array_store(var_joint_pos, var_3, var_6, var_9, adj_joint_pos, adj_3, adj_6, adj_8);
        wp::adj_address(var_pos_data, var_0, var_1, adj_pos_data, adj_0, adj_1, adj_8);
        // adj: joint_pos[env_id, joint_id] = pos_data[i, j]                                      <L 29>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 28>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 27>
        // adj: i, j = wp.tid()                                                                   <L 26>
        // adj: def write_joint_state_data_index(                                                 <L 1>
        continue;
    }
}



extern "C" __global__ void update_soft_joint_pos_limits_ca01ff7f_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_joint_pos_limits,
    wp::float32 var_soft_limit_factor,
    wp::array_t<wp::vec_t<2, wp::float32>> var_soft_joint_pos_limits)
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
        wp::vec_t<2, wp::float32>* var_2;
        wp::vec_t<2, wp::float32> var_3;
        wp::vec_t<2, wp::float32> var_4;
        //---------
        // forward
        // def update_soft_joint_pos_limits(                                                      <L 546>
        // i, j = wp.tid()                                                                        <L 566>
        builtin_tid2d(var_0, var_1);
        // soft_joint_pos_limits[i, j] = compute_soft_joint_pos_limits_func(joint_pos_limits[i, j], soft_limit_factor)       <L 567>
        var_2 = wp::address(var_joint_pos_limits, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = compute_soft_joint_pos_limits_func_0(var_4, var_soft_limit_factor);
        wp::array_store(var_soft_joint_pos_limits, var_0, var_1, var_3);
    }
}



extern "C" __global__ void update_soft_joint_pos_limits_ca01ff7f_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_joint_pos_limits,
    wp::float32 var_soft_limit_factor,
    wp::array_t<wp::vec_t<2, wp::float32>> var_soft_joint_pos_limits,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_joint_pos_limits,
    wp::float32 adj_soft_limit_factor,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_soft_joint_pos_limits)
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
        wp::vec_t<2, wp::float32>* var_2;
        wp::vec_t<2, wp::float32> var_3;
        wp::vec_t<2, wp::float32> var_4;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::vec_t<2, wp::float32> adj_2 = {};
        wp::vec_t<2, wp::float32> adj_3 = {};
        wp::vec_t<2, wp::float32> adj_4 = {};
        //---------
        // forward
        // def update_soft_joint_pos_limits(                                                      <L 546>
        // i, j = wp.tid()                                                                        <L 566>
        builtin_tid2d(var_0, var_1);
        // soft_joint_pos_limits[i, j] = compute_soft_joint_pos_limits_func(joint_pos_limits[i, j], soft_limit_factor)       <L 567>
        var_2 = wp::address(var_joint_pos_limits, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = compute_soft_joint_pos_limits_func_0(var_4, var_soft_limit_factor);
        // wp::array_store(var_soft_joint_pos_limits, var_0, var_1, var_3);
        //---------
        // reverse
        wp::adj_array_store(var_soft_joint_pos_limits, var_0, var_1, var_3, adj_soft_joint_pos_limits, adj_0, adj_1, adj_3);
        adj_compute_soft_joint_pos_limits_func_0(var_4, var_soft_limit_factor, adj_2, adj_soft_limit_factor, adj_3);
        wp::adj_address(var_joint_pos_limits, var_0, var_1, adj_joint_pos_limits, adj_0, adj_1, adj_2);
        // adj: soft_joint_pos_limits[i, j] = compute_soft_joint_pos_limits_func(joint_pos_limits[i, j], soft_limit_factor)  <L 567>
        // adj: i, j = wp.tid()                                                                   <L 566>
        // adj: def update_soft_joint_pos_limits(                                                 <L 546>
        continue;
    }
}



extern "C" __global__ void write_joint_friction_data_to_buffer_d2461c2d_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_friction,
    wp::array_t<wp::float32> var_in_dynamic_friction,
    wp::array_t<wp::float32> var_in_viscous_friction,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    bool var_from_mask,
    wp::array_t<wp::float32> var_out_friction,
    wp::array_t<wp::float32> var_out_dynamic_friction,
    wp::array_t<wp::float32> var_out_viscous_friction,
    wp::array_t<wp::float32> var_friction_props)
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
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32* var_16;
        wp::float32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32* var_20;
        const wp::int32 var_21 = 0;
        wp::float32 var_22;
        wp::float32* var_23;
        const wp::int32 var_24 = 1;
        wp::float32 var_25;
        wp::float32* var_26;
        const wp::int32 var_27 = 2;
        wp::float32 var_28;
        //---------
        // forward
        // def write_joint_friction_data_to_buffer(                                               <L 1>
        // i, j = wp.tid()                                                                        <L 43>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 44>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 45>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 47>
        if (var_from_mask) {
            // out_friction[env_id, joint_id] = in_friction[env_id, joint_id]                     <L 48>
            var_8 = wp::address(var_in_friction, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_out_friction, var_3, var_6, var_9);
            // if in_dynamic_friction:                                                            <L 49>
            if (var_in_dynamic_friction) {
                // out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[env_id, joint_id]       <L 50>
                var_10 = wp::address(var_in_dynamic_friction, var_3, var_6);
                var_11 = wp::load(var_10);
                wp::array_store(var_out_dynamic_friction, var_3, var_6, var_11);
            }
            // if in_viscous_friction:                                                            <L 51>
            if (var_in_viscous_friction) {
                // out_viscous_friction[env_id, joint_id] = in_viscous_friction[env_id, joint_id]       <L 52>
                var_12 = wp::address(var_in_viscous_friction, var_3, var_6);
                var_13 = wp::load(var_12);
                wp::array_store(var_out_viscous_friction, var_3, var_6, var_13);
            }
        }
        if (!var_from_mask) {
            // out_friction[env_id, joint_id] = in_friction[i, j]                                 <L 54>
            var_14 = wp::address(var_in_friction, var_0, var_1);
            var_15 = wp::load(var_14);
            wp::array_store(var_out_friction, var_3, var_6, var_15);
            // if in_dynamic_friction:                                                            <L 55>
            if (var_in_dynamic_friction) {
                // out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[i, j]             <L 56>
                var_16 = wp::address(var_in_dynamic_friction, var_0, var_1);
                var_17 = wp::load(var_16);
                wp::array_store(var_out_dynamic_friction, var_3, var_6, var_17);
            }
            // if in_viscous_friction:                                                            <L 57>
            if (var_in_viscous_friction) {
                // out_viscous_friction[env_id, joint_id] = in_viscous_friction[i, j]             <L 58>
                var_18 = wp::address(var_in_viscous_friction, var_0, var_1);
                var_19 = wp::load(var_18);
                wp::array_store(var_out_viscous_friction, var_3, var_6, var_19);
            }
        }
        // friction_props[env_id, joint_id, 0] = out_friction[env_id, joint_id]                   <L 60>
        var_20 = wp::address(var_out_friction, var_3, var_6);
        var_22 = wp::load(var_20);
        wp::array_store(var_friction_props, var_3, var_6, var_21, var_22);
        // if in_dynamic_friction:                                                                <L 61>
        if (var_in_dynamic_friction) {
            // friction_props[env_id, joint_id, 1] = out_dynamic_friction[env_id, joint_id]       <L 62>
            var_23 = wp::address(var_out_dynamic_friction, var_3, var_6);
            var_25 = wp::load(var_23);
            wp::array_store(var_friction_props, var_3, var_6, var_24, var_25);
        }
        // if in_viscous_friction:                                                                <L 63>
        if (var_in_viscous_friction) {
            // friction_props[env_id, joint_id, 2] = out_viscous_friction[env_id, joint_id]       <L 64>
            var_26 = wp::address(var_out_viscous_friction, var_3, var_6);
            var_28 = wp::load(var_26);
            wp::array_store(var_friction_props, var_3, var_6, var_27, var_28);
        }
    }
}



extern "C" __global__ void write_joint_friction_data_to_buffer_d2461c2d_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_friction,
    wp::array_t<wp::float32> var_in_dynamic_friction,
    wp::array_t<wp::float32> var_in_viscous_friction,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    bool var_from_mask,
    wp::array_t<wp::float32> var_out_friction,
    wp::array_t<wp::float32> var_out_dynamic_friction,
    wp::array_t<wp::float32> var_out_viscous_friction,
    wp::array_t<wp::float32> var_friction_props,
    wp::array_t<wp::float32> adj_in_friction,
    wp::array_t<wp::float32> adj_in_dynamic_friction,
    wp::array_t<wp::float32> adj_in_viscous_friction,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int32> adj_joint_ids,
    bool adj_from_mask,
    wp::array_t<wp::float32> adj_out_friction,
    wp::array_t<wp::float32> adj_out_dynamic_friction,
    wp::array_t<wp::float32> adj_out_viscous_friction,
    wp::array_t<wp::float32> adj_friction_props)
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
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32* var_16;
        wp::float32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32* var_20;
        const wp::int32 var_21 = 0;
        wp::float32 var_22;
        wp::float32* var_23;
        const wp::int32 var_24 = 1;
        wp::float32 var_25;
        wp::float32* var_26;
        const wp::int32 var_27 = 2;
        wp::float32 var_28;
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
        wp::int32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::float32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::float32 adj_28 = {};
        //---------
        // forward
        // def write_joint_friction_data_to_buffer(                                               <L 1>
        // i, j = wp.tid()                                                                        <L 43>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 44>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 45>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 47>
        if (var_from_mask) {
            // out_friction[env_id, joint_id] = in_friction[env_id, joint_id]                     <L 48>
            var_8 = wp::address(var_in_friction, var_3, var_6);
            var_9 = wp::load(var_8);
            // wp::array_store(var_out_friction, var_3, var_6, var_9);
            // if in_dynamic_friction:                                                            <L 49>
            if (var_in_dynamic_friction) {
                // out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[env_id, joint_id]       <L 50>
                var_10 = wp::address(var_in_dynamic_friction, var_3, var_6);
                var_11 = wp::load(var_10);
                // wp::array_store(var_out_dynamic_friction, var_3, var_6, var_11);
            }
            // if in_viscous_friction:                                                            <L 51>
            if (var_in_viscous_friction) {
                // out_viscous_friction[env_id, joint_id] = in_viscous_friction[env_id, joint_id]       <L 52>
                var_12 = wp::address(var_in_viscous_friction, var_3, var_6);
                var_13 = wp::load(var_12);
                // wp::array_store(var_out_viscous_friction, var_3, var_6, var_13);
            }
        }
        if (!var_from_mask) {
            // out_friction[env_id, joint_id] = in_friction[i, j]                                 <L 54>
            var_14 = wp::address(var_in_friction, var_0, var_1);
            var_15 = wp::load(var_14);
            // wp::array_store(var_out_friction, var_3, var_6, var_15);
            // if in_dynamic_friction:                                                            <L 55>
            if (var_in_dynamic_friction) {
                // out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[i, j]             <L 56>
                var_16 = wp::address(var_in_dynamic_friction, var_0, var_1);
                var_17 = wp::load(var_16);
                // wp::array_store(var_out_dynamic_friction, var_3, var_6, var_17);
            }
            // if in_viscous_friction:                                                            <L 57>
            if (var_in_viscous_friction) {
                // out_viscous_friction[env_id, joint_id] = in_viscous_friction[i, j]             <L 58>
                var_18 = wp::address(var_in_viscous_friction, var_0, var_1);
                var_19 = wp::load(var_18);
                // wp::array_store(var_out_viscous_friction, var_3, var_6, var_19);
            }
        }
        // friction_props[env_id, joint_id, 0] = out_friction[env_id, joint_id]                   <L 60>
        var_20 = wp::address(var_out_friction, var_3, var_6);
        var_22 = wp::load(var_20);
        // wp::array_store(var_friction_props, var_3, var_6, var_21, var_22);
        // if in_dynamic_friction:                                                                <L 61>
        if (var_in_dynamic_friction) {
            // friction_props[env_id, joint_id, 1] = out_dynamic_friction[env_id, joint_id]       <L 62>
            var_23 = wp::address(var_out_dynamic_friction, var_3, var_6);
            var_25 = wp::load(var_23);
            // wp::array_store(var_friction_props, var_3, var_6, var_24, var_25);
        }
        // if in_viscous_friction:                                                                <L 63>
        if (var_in_viscous_friction) {
            // friction_props[env_id, joint_id, 2] = out_viscous_friction[env_id, joint_id]       <L 64>
            var_26 = wp::address(var_out_viscous_friction, var_3, var_6);
            var_28 = wp::load(var_26);
            // wp::array_store(var_friction_props, var_3, var_6, var_27, var_28);
        }
        //---------
        // reverse
        if (var_in_viscous_friction) {
            wp::adj_array_store(var_friction_props, var_3, var_6, var_27, var_28, adj_friction_props, adj_3, adj_6, adj_27, adj_26);
            wp::adj_address(var_out_viscous_friction, var_3, var_6, adj_out_viscous_friction, adj_3, adj_6, adj_26);
            // adj: friction_props[env_id, joint_id, 2] = out_viscous_friction[env_id, joint_id]  <L 64>
        }
        // adj: if in_viscous_friction:                                                           <L 63>
        if (var_in_dynamic_friction) {
            wp::adj_array_store(var_friction_props, var_3, var_6, var_24, var_25, adj_friction_props, adj_3, adj_6, adj_24, adj_23);
            wp::adj_address(var_out_dynamic_friction, var_3, var_6, adj_out_dynamic_friction, adj_3, adj_6, adj_23);
            // adj: friction_props[env_id, joint_id, 1] = out_dynamic_friction[env_id, joint_id]  <L 62>
        }
        // adj: if in_dynamic_friction:                                                           <L 61>
        wp::adj_array_store(var_friction_props, var_3, var_6, var_21, var_22, adj_friction_props, adj_3, adj_6, adj_21, adj_20);
        wp::adj_address(var_out_friction, var_3, var_6, adj_out_friction, adj_3, adj_6, adj_20);
        // adj: friction_props[env_id, joint_id, 0] = out_friction[env_id, joint_id]              <L 60>
        if (!var_from_mask) {
            if (var_in_viscous_friction) {
                wp::adj_array_store(var_out_viscous_friction, var_3, var_6, var_19, adj_out_viscous_friction, adj_3, adj_6, adj_18);
                wp::adj_address(var_in_viscous_friction, var_0, var_1, adj_in_viscous_friction, adj_0, adj_1, adj_18);
                // adj: out_viscous_friction[env_id, joint_id] = in_viscous_friction[i, j]        <L 58>
            }
            // adj: if in_viscous_friction:                                                       <L 57>
            if (var_in_dynamic_friction) {
                wp::adj_array_store(var_out_dynamic_friction, var_3, var_6, var_17, adj_out_dynamic_friction, adj_3, adj_6, adj_16);
                wp::adj_address(var_in_dynamic_friction, var_0, var_1, adj_in_dynamic_friction, adj_0, adj_1, adj_16);
                // adj: out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[i, j]        <L 56>
            }
            // adj: if in_dynamic_friction:                                                       <L 55>
            wp::adj_array_store(var_out_friction, var_3, var_6, var_15, adj_out_friction, adj_3, adj_6, adj_14);
            wp::adj_address(var_in_friction, var_0, var_1, adj_in_friction, adj_0, adj_1, adj_14);
            // adj: out_friction[env_id, joint_id] = in_friction[i, j]                            <L 54>
        }
        if (var_from_mask) {
            if (var_in_viscous_friction) {
                wp::adj_array_store(var_out_viscous_friction, var_3, var_6, var_13, adj_out_viscous_friction, adj_3, adj_6, adj_12);
                wp::adj_address(var_in_viscous_friction, var_3, var_6, adj_in_viscous_friction, adj_3, adj_6, adj_12);
                // adj: out_viscous_friction[env_id, joint_id] = in_viscous_friction[env_id, joint_id]  <L 52>
            }
            // adj: if in_viscous_friction:                                                       <L 51>
            if (var_in_dynamic_friction) {
                wp::adj_array_store(var_out_dynamic_friction, var_3, var_6, var_11, adj_out_dynamic_friction, adj_3, adj_6, adj_10);
                wp::adj_address(var_in_dynamic_friction, var_3, var_6, adj_in_dynamic_friction, adj_3, adj_6, adj_10);
                // adj: out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[env_id, joint_id]  <L 50>
            }
            // adj: if in_dynamic_friction:                                                       <L 49>
            wp::adj_array_store(var_out_friction, var_3, var_6, var_9, adj_out_friction, adj_3, adj_6, adj_8);
            wp::adj_address(var_in_friction, var_3, var_6, adj_in_friction, adj_3, adj_6, adj_8);
            // adj: out_friction[env_id, joint_id] = in_friction[env_id, joint_id]                <L 48>
        }
        // adj: if from_mask:                                                                     <L 47>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 45>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 44>
        // adj: i, j = wp.tid()                                                                   <L 43>
        // adj: def write_joint_friction_data_to_buffer(                                          <L 1>
        continue;
    }
}



extern "C" __global__ void write_joint_limit_data_to_user_and_backend_index_e0976898_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_in_data,
    wp::float32 var_soft_limit_factor,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_user_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_user_joint_pos_limits,
    wp::array_t<wp::float32> var_backend_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_backend_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_soft_joint_pos_limits,
    wp::array_t<wp::float32> var_default_joint_pos,
    wp::array_t<wp::int32> var_clamped_defaults)
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
        wp::int32 var_8;
        wp::vec_t<2, wp::float32>* var_9;
        wp::vec_t<2, wp::float32> var_10;
        wp::vec_t<2, wp::float32> var_11;
        const wp::int32 var_12 = 0;
        wp::float32 var_13;
        const wp::int32 var_14 = 1;
        wp::float32 var_15;
        bool var_16;
        wp::float32* var_17;
        bool var_18;
        wp::float32 var_19;
        wp::float32* var_20;
        bool var_21;
        wp::float32 var_22;
        const wp::int32 var_23 = 0;
        const wp::int32 var_24 = 1;
        wp::int32 var_25;
        wp::float32* var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::vec_t<2, wp::float32> var_29;
        //---------
        // forward
        // def write_joint_limit_data_to_user_and_backend_index(                                  <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 19>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 20>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // backend_id = resolve_backend_index(user_id, user_to_backend, has_ordering)             <L 21>
        var_8 = resolve_backend_index_0(var_6, var_user_to_backend, var_has_ordering);
        // limits = in_data[i, j]                                                                 <L 22>
        var_9 = wp::address(var_in_data, var_0, var_1);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // lower = limits[0]                                                                      <L 23>
        var_13 = wp::extract(var_10, var_12);
        // upper = limits[1]                                                                      <L 24>
        var_15 = wp::extract(var_10, var_14);
        // user_joint_pos_limits_lower[env_id, user_id] = lower                                   <L 26>
        wp::array_store(var_user_joint_pos_limits_lower, var_3, var_6, var_13);
        // user_joint_pos_limits_upper[env_id, user_id] = upper                                   <L 27>
        wp::array_store(var_user_joint_pos_limits_upper, var_3, var_6, var_15);
        // user_joint_pos_limits[env_id, user_id] = limits                                        <L 28>
        wp::array_store(var_user_joint_pos_limits, var_3, var_6, var_10);
        // backend_joint_pos_limits_lower[env_id, backend_id] = lower                             <L 29>
        wp::array_store(var_backend_joint_pos_limits_lower, var_3, var_8, var_13);
        // backend_joint_pos_limits_upper[env_id, backend_id] = upper                             <L 30>
        wp::array_store(var_backend_joint_pos_limits_upper, var_3, var_8, var_15);
        // if (default_joint_pos[env_id, user_id] < lower) or default_joint_pos[env_id, user_id] > upper:       <L 32>
        var_17 = wp::address(var_default_joint_pos, var_3, var_6);
        var_19 = wp::load(var_17);
        var_18 = (var_19 < var_13);
        var_16 = var_18;
        if (!var_16) {
            var_20 = wp::address(var_default_joint_pos, var_3, var_6);
            var_22 = wp::load(var_20);
            var_21 = (var_22 > var_15);
            var_16 = var_16 || var_21;
        }
        if (var_16) {
            // wp.atomic_add(clamped_defaults, 0, 1)                                              <L 33>
            var_25 = wp::atomic_add(var_clamped_defaults, var_23, var_24);
            // default_joint_pos[env_id, user_id] = wp.clamp(default_joint_pos[env_id, user_id], lower, upper)       <L 34>
            var_26 = wp::address(var_default_joint_pos, var_3, var_6);
            var_28 = wp::load(var_26);
            var_27 = wp::clamp(var_28, var_13, var_15);
            wp::array_store(var_default_joint_pos, var_3, var_6, var_27);
        }
        // soft_joint_pos_limits[env_id, user_id] = compute_soft_joint_pos_limits_func(limits, soft_limit_factor)       <L 36>
        var_29 = compute_soft_joint_pos_limits_func_0(var_10, var_soft_limit_factor);
        wp::array_store(var_soft_joint_pos_limits, var_3, var_6, var_29);
    }
}



extern "C" __global__ void write_joint_limit_data_to_user_and_backend_index_e0976898_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_in_data,
    wp::float32 var_soft_limit_factor,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_user_ids,
    wp::array_t<wp::int32> var_user_to_backend,
    bool var_has_ordering,
    wp::array_t<wp::float32> var_user_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_user_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_user_joint_pos_limits,
    wp::array_t<wp::float32> var_backend_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_backend_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_soft_joint_pos_limits,
    wp::array_t<wp::float32> var_default_joint_pos,
    wp::array_t<wp::int32> var_clamped_defaults,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_in_data,
    wp::float32 adj_soft_limit_factor,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int64> adj_user_ids,
    wp::array_t<wp::int32> adj_user_to_backend,
    bool adj_has_ordering,
    wp::array_t<wp::float32> adj_user_joint_pos_limits_lower,
    wp::array_t<wp::float32> adj_user_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_user_joint_pos_limits,
    wp::array_t<wp::float32> adj_backend_joint_pos_limits_lower,
    wp::array_t<wp::float32> adj_backend_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_soft_joint_pos_limits,
    wp::array_t<wp::float32> adj_default_joint_pos,
    wp::array_t<wp::int32> adj_clamped_defaults)
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
        wp::int32 var_8;
        wp::vec_t<2, wp::float32>* var_9;
        wp::vec_t<2, wp::float32> var_10;
        wp::vec_t<2, wp::float32> var_11;
        const wp::int32 var_12 = 0;
        wp::float32 var_13;
        const wp::int32 var_14 = 1;
        wp::float32 var_15;
        bool var_16;
        wp::float32* var_17;
        bool var_18;
        wp::float32 var_19;
        wp::float32* var_20;
        bool var_21;
        wp::float32 var_22;
        const wp::int32 var_23 = 0;
        const wp::int32 var_24 = 1;
        wp::int32 var_25;
        wp::float32* var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::vec_t<2, wp::float32> var_29;
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
        wp::vec_t<2, wp::float32> adj_9 = {};
        wp::vec_t<2, wp::float32> adj_10 = {};
        wp::vec_t<2, wp::float32> adj_11 = {};
        wp::int32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::float32 adj_15 = {};
        bool adj_16 = {};
        wp::float32 adj_17 = {};
        bool adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        bool adj_21 = {};
        wp::float32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::vec_t<2, wp::float32> adj_29 = {};
        //---------
        // forward
        // def write_joint_limit_data_to_user_and_backend_index(                                  <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 19>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_id = wp.int32(user_ids[j])                                                        <L 20>
        var_5 = wp::address(var_user_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // backend_id = resolve_backend_index(user_id, user_to_backend, has_ordering)             <L 21>
        var_8 = resolve_backend_index_0(var_6, var_user_to_backend, var_has_ordering);
        // limits = in_data[i, j]                                                                 <L 22>
        var_9 = wp::address(var_in_data, var_0, var_1);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // lower = limits[0]                                                                      <L 23>
        var_13 = wp::extract(var_10, var_12);
        // upper = limits[1]                                                                      <L 24>
        var_15 = wp::extract(var_10, var_14);
        // user_joint_pos_limits_lower[env_id, user_id] = lower                                   <L 26>
        // wp::array_store(var_user_joint_pos_limits_lower, var_3, var_6, var_13);
        // user_joint_pos_limits_upper[env_id, user_id] = upper                                   <L 27>
        // wp::array_store(var_user_joint_pos_limits_upper, var_3, var_6, var_15);
        // user_joint_pos_limits[env_id, user_id] = limits                                        <L 28>
        // wp::array_store(var_user_joint_pos_limits, var_3, var_6, var_10);
        // backend_joint_pos_limits_lower[env_id, backend_id] = lower                             <L 29>
        // wp::array_store(var_backend_joint_pos_limits_lower, var_3, var_8, var_13);
        // backend_joint_pos_limits_upper[env_id, backend_id] = upper                             <L 30>
        // wp::array_store(var_backend_joint_pos_limits_upper, var_3, var_8, var_15);
        // if (default_joint_pos[env_id, user_id] < lower) or default_joint_pos[env_id, user_id] > upper:       <L 32>
        var_17 = wp::address(var_default_joint_pos, var_3, var_6);
        var_19 = wp::load(var_17);
        var_18 = (var_19 < var_13);
        var_16 = var_18;
        if (!var_16) {
            var_20 = wp::address(var_default_joint_pos, var_3, var_6);
            var_22 = wp::load(var_20);
            var_21 = (var_22 > var_15);
            var_16 = var_16 || var_21;
        }
        if (var_16) {
            // wp.atomic_add(clamped_defaults, 0, 1)                                              <L 33>
            // var_25 = wp::atomic_add(var_clamped_defaults, var_23, var_24);
            // default_joint_pos[env_id, user_id] = wp.clamp(default_joint_pos[env_id, user_id], lower, upper)       <L 34>
            var_26 = wp::address(var_default_joint_pos, var_3, var_6);
            var_28 = wp::load(var_26);
            var_27 = wp::clamp(var_28, var_13, var_15);
            // wp::array_store(var_default_joint_pos, var_3, var_6, var_27);
        }
        // soft_joint_pos_limits[env_id, user_id] = compute_soft_joint_pos_limits_func(limits, soft_limit_factor)       <L 36>
        var_29 = compute_soft_joint_pos_limits_func_0(var_10, var_soft_limit_factor);
        // wp::array_store(var_soft_joint_pos_limits, var_3, var_6, var_29);
        //---------
        // reverse
        wp::adj_array_store(var_soft_joint_pos_limits, var_3, var_6, var_29, adj_soft_joint_pos_limits, adj_3, adj_6, adj_29);
        adj_compute_soft_joint_pos_limits_func_0(var_10, var_soft_limit_factor, adj_10, adj_soft_limit_factor, adj_29);
        // adj: soft_joint_pos_limits[env_id, user_id] = compute_soft_joint_pos_limits_func(limits, soft_limit_factor)  <L 36>
        if (var_16) {
            wp::adj_array_store(var_default_joint_pos, var_3, var_6, var_27, adj_default_joint_pos, adj_3, adj_6, adj_27);
            wp::adj_clamp(var_28, var_13, var_15, adj_26, adj_13, adj_15, adj_27);
            wp::adj_address(var_default_joint_pos, var_3, var_6, adj_default_joint_pos, adj_3, adj_6, adj_26);
            // adj: default_joint_pos[env_id, user_id] = wp.clamp(default_joint_pos[env_id, user_id], lower, upper)  <L 34>
            wp::adj_atomic_add(var_clamped_defaults, var_23, var_24, adj_clamped_defaults, adj_23, adj_24, adj_25);
            // adj: wp.atomic_add(clamped_defaults, 0, 1)                                         <L 33>
        }
        if (!var_16) {
            wp::adj_address(var_default_joint_pos, var_3, var_6, adj_default_joint_pos, adj_3, adj_6, adj_20);
        }
        wp::adj_address(var_default_joint_pos, var_3, var_6, adj_default_joint_pos, adj_3, adj_6, adj_17);
        // adj: if (default_joint_pos[env_id, user_id] < lower) or default_joint_pos[env_id, user_id] > upper:  <L 32>
        wp::adj_array_store(var_backend_joint_pos_limits_upper, var_3, var_8, var_15, adj_backend_joint_pos_limits_upper, adj_3, adj_8, adj_15);
        // adj: backend_joint_pos_limits_upper[env_id, backend_id] = upper                        <L 30>
        wp::adj_array_store(var_backend_joint_pos_limits_lower, var_3, var_8, var_13, adj_backend_joint_pos_limits_lower, adj_3, adj_8, adj_13);
        // adj: backend_joint_pos_limits_lower[env_id, backend_id] = lower                        <L 29>
        wp::adj_array_store(var_user_joint_pos_limits, var_3, var_6, var_10, adj_user_joint_pos_limits, adj_3, adj_6, adj_10);
        // adj: user_joint_pos_limits[env_id, user_id] = limits                                   <L 28>
        wp::adj_array_store(var_user_joint_pos_limits_upper, var_3, var_6, var_15, adj_user_joint_pos_limits_upper, adj_3, adj_6, adj_15);
        // adj: user_joint_pos_limits_upper[env_id, user_id] = upper                              <L 27>
        wp::adj_array_store(var_user_joint_pos_limits_lower, var_3, var_6, var_13, adj_user_joint_pos_limits_lower, adj_3, adj_6, adj_13);
        // adj: user_joint_pos_limits_lower[env_id, user_id] = lower                              <L 26>
        wp::adj_extract(var_10, var_14, adj_10, adj_14, adj_15);
        // adj: upper = limits[1]                                                                 <L 24>
        wp::adj_extract(var_10, var_12, adj_10, adj_12, adj_13);
        // adj: lower = limits[0]                                                                 <L 23>
        wp::adj_copy(var_11, adj_9, adj_10);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_9);
        // adj: limits = in_data[i, j]                                                            <L 22>
        adj_resolve_backend_index_0(var_6, var_user_to_backend, var_has_ordering, adj_6, adj_user_to_backend, adj_has_ordering, adj_8);
        // adj: backend_id = resolve_backend_index(user_id, user_to_backend, has_ordering)        <L 21>
        wp::adj_address(var_user_ids, var_1, adj_user_ids, adj_1, adj_5);
        // adj: user_id = wp.int32(user_ids[j])                                                   <L 20>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 19>
        // adj: i, j = wp.tid()                                                                   <L 18>
        // adj: def write_joint_limit_data_to_user_and_backend_index(                             <L 1>
        continue;
    }
}



extern "C" __global__ void write_joint_limit_data_to_buffer_index_e662f843_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_in_data,
    wp::float32 var_soft_limit_factor,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    wp::array_t<wp::float32> var_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_soft_joint_pos_limits,
    wp::array_t<wp::float32> var_default_joint_pos,
    wp::array_t<wp::int32> var_clamped_defaults)
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
        wp::vec_t<2, wp::float32>* var_8;
        const wp::int32 var_9 = 0;
        wp::float32 var_10;
        wp::vec_t<2, wp::float32> var_11;
        wp::vec_t<2, wp::float32>* var_12;
        const wp::int32 var_13 = 1;
        wp::float32 var_14;
        wp::vec_t<2, wp::float32> var_15;
        bool var_16;
        wp::float32* var_17;
        wp::float32* var_18;
        bool var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32* var_22;
        wp::float32* var_23;
        bool var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        const wp::int32 var_27 = 0;
        const wp::int32 var_28 = 1;
        wp::int32 var_29;
        wp::float32* var_30;
        wp::float32* var_31;
        wp::float32* var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32* var_37;
        wp::float32* var_38;
        wp::vec_t<2, wp::float32> var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::vec_t<2, wp::float32> var_42;
        //---------
        // forward
        // def write_joint_limit_data_to_buffer_index(                                            <L 1>
        // i, j = wp.tid()                                                                        <L 35>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 36>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 37>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // joint_pos_limits_lower[env_id, joint_id] = in_data[i, j][0]                            <L 38>
        var_8 = wp::address(var_in_data, var_0, var_1);
        var_11 = wp::load(var_8);
        var_10 = wp::extract(var_11, var_9);
        wp::array_store(var_joint_pos_limits_lower, var_3, var_6, var_10);
        // joint_pos_limits_upper[env_id, joint_id] = in_data[i, j][1]                            <L 39>
        var_12 = wp::address(var_in_data, var_0, var_1);
        var_15 = wp::load(var_12);
        var_14 = wp::extract(var_15, var_13);
        wp::array_store(var_joint_pos_limits_upper, var_3, var_6, var_14);
        // if (default_joint_pos[env_id, joint_id] < joint_pos_limits_lower[env_id, joint_id]) or default_joint_pos[       <L 40>
        var_17 = wp::address(var_default_joint_pos, var_3, var_6);
        var_18 = wp::address(var_joint_pos_limits_lower, var_3, var_6);
        var_20 = wp::load(var_17);
        var_21 = wp::load(var_18);
        var_19 = (var_20 < var_21);
        var_16 = var_19;
        if (!var_16) {
            // env_id, joint_id                                                                   <L 41>
            var_22 = wp::address(var_default_joint_pos, var_3, var_6);
            // ] > joint_pos_limits_upper[env_id, joint_id]:                                      <L 42>
            var_23 = wp::address(var_joint_pos_limits_upper, var_3, var_6);
            var_25 = wp::load(var_22);
            var_26 = wp::load(var_23);
            var_24 = (var_25 > var_26);
            var_16 = var_16 || var_24;
        }
        if (var_16) {
            // wp.atomic_add(clamped_defaults, 0, 1)                                              <L 43>
            var_29 = wp::atomic_add(var_clamped_defaults, var_27, var_28);
            // default_joint_pos[env_id, joint_id] = wp.clamp(                                    <L 44>
            // default_joint_pos[env_id, joint_id],                                               <L 45>
            var_30 = wp::address(var_default_joint_pos, var_3, var_6);
            // joint_pos_limits_lower[env_id, joint_id],                                          <L 46>
            var_31 = wp::address(var_joint_pos_limits_lower, var_3, var_6);
            // joint_pos_limits_upper[env_id, joint_id],                                          <L 47>
            var_32 = wp::address(var_joint_pos_limits_upper, var_3, var_6);
            var_34 = wp::load(var_30);
            var_35 = wp::load(var_31);
            var_36 = wp::load(var_32);
            var_33 = wp::clamp(var_34, var_35, var_36);
            // default_joint_pos[env_id, joint_id] = wp.clamp(                                    <L 44>
            wp::array_store(var_default_joint_pos, var_3, var_6, var_33);
        }
        // soft_joint_pos_limits[env_id, joint_id] = compute_soft_joint_pos_limits_func(          <L 49>
        // wp.vec2f(joint_pos_limits_lower[env_id, joint_id], joint_pos_limits_upper[env_id, joint_id]),       <L 50>
        var_37 = wp::address(var_joint_pos_limits_lower, var_3, var_6);
        var_38 = wp::address(var_joint_pos_limits_upper, var_3, var_6);
        var_40 = wp::load(var_37);
        var_41 = wp::load(var_38);
        var_39 = wp::vec_t<2, wp::float32>(var_40, var_41);
        // soft_limit_factor,                                                                     <L 51>
        var_42 = compute_soft_joint_pos_limits_func_0(var_39, var_soft_limit_factor);
        // soft_joint_pos_limits[env_id, joint_id] = compute_soft_joint_pos_limits_func(          <L 49>
        wp::array_store(var_soft_joint_pos_limits, var_3, var_6, var_42);
    }
}



extern "C" __global__ void write_joint_limit_data_to_buffer_index_e662f843_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_in_data,
    wp::float32 var_soft_limit_factor,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    wp::array_t<wp::float32> var_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_soft_joint_pos_limits,
    wp::array_t<wp::float32> var_default_joint_pos,
    wp::array_t<wp::int32> var_clamped_defaults,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_in_data,
    wp::float32 adj_soft_limit_factor,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int64> adj_joint_ids,
    wp::array_t<wp::float32> adj_joint_pos_limits_lower,
    wp::array_t<wp::float32> adj_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_soft_joint_pos_limits,
    wp::array_t<wp::float32> adj_default_joint_pos,
    wp::array_t<wp::int32> adj_clamped_defaults)
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
        wp::vec_t<2, wp::float32>* var_8;
        const wp::int32 var_9 = 0;
        wp::float32 var_10;
        wp::vec_t<2, wp::float32> var_11;
        wp::vec_t<2, wp::float32>* var_12;
        const wp::int32 var_13 = 1;
        wp::float32 var_14;
        wp::vec_t<2, wp::float32> var_15;
        bool var_16;
        wp::float32* var_17;
        wp::float32* var_18;
        bool var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32* var_22;
        wp::float32* var_23;
        bool var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        const wp::int32 var_27 = 0;
        const wp::int32 var_28 = 1;
        wp::int32 var_29;
        wp::float32* var_30;
        wp::float32* var_31;
        wp::float32* var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32* var_37;
        wp::float32* var_38;
        wp::vec_t<2, wp::float32> var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::vec_t<2, wp::float32> var_42;
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
        wp::vec_t<2, wp::float32> adj_8 = {};
        wp::int32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::vec_t<2, wp::float32> adj_11 = {};
        wp::vec_t<2, wp::float32> adj_12 = {};
        wp::int32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::vec_t<2, wp::float32> adj_15 = {};
        bool adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        bool adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        bool adj_24 = {};
        wp::float32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::float32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::vec_t<2, wp::float32> adj_39 = {};
        wp::float32 adj_40 = {};
        wp::float32 adj_41 = {};
        wp::vec_t<2, wp::float32> adj_42 = {};
        //---------
        // forward
        // def write_joint_limit_data_to_buffer_index(                                            <L 1>
        // i, j = wp.tid()                                                                        <L 35>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 36>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 37>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // joint_pos_limits_lower[env_id, joint_id] = in_data[i, j][0]                            <L 38>
        var_8 = wp::address(var_in_data, var_0, var_1);
        var_11 = wp::load(var_8);
        var_10 = wp::extract(var_11, var_9);
        // wp::array_store(var_joint_pos_limits_lower, var_3, var_6, var_10);
        // joint_pos_limits_upper[env_id, joint_id] = in_data[i, j][1]                            <L 39>
        var_12 = wp::address(var_in_data, var_0, var_1);
        var_15 = wp::load(var_12);
        var_14 = wp::extract(var_15, var_13);
        // wp::array_store(var_joint_pos_limits_upper, var_3, var_6, var_14);
        // if (default_joint_pos[env_id, joint_id] < joint_pos_limits_lower[env_id, joint_id]) or default_joint_pos[       <L 40>
        var_17 = wp::address(var_default_joint_pos, var_3, var_6);
        var_18 = wp::address(var_joint_pos_limits_lower, var_3, var_6);
        var_20 = wp::load(var_17);
        var_21 = wp::load(var_18);
        var_19 = (var_20 < var_21);
        var_16 = var_19;
        if (!var_16) {
            // env_id, joint_id                                                                   <L 41>
            var_22 = wp::address(var_default_joint_pos, var_3, var_6);
            // ] > joint_pos_limits_upper[env_id, joint_id]:                                      <L 42>
            var_23 = wp::address(var_joint_pos_limits_upper, var_3, var_6);
            var_25 = wp::load(var_22);
            var_26 = wp::load(var_23);
            var_24 = (var_25 > var_26);
            var_16 = var_16 || var_24;
        }
        if (var_16) {
            // wp.atomic_add(clamped_defaults, 0, 1)                                              <L 43>
            // var_29 = wp::atomic_add(var_clamped_defaults, var_27, var_28);
            // default_joint_pos[env_id, joint_id] = wp.clamp(                                    <L 44>
            // default_joint_pos[env_id, joint_id],                                               <L 45>
            var_30 = wp::address(var_default_joint_pos, var_3, var_6);
            // joint_pos_limits_lower[env_id, joint_id],                                          <L 46>
            var_31 = wp::address(var_joint_pos_limits_lower, var_3, var_6);
            // joint_pos_limits_upper[env_id, joint_id],                                          <L 47>
            var_32 = wp::address(var_joint_pos_limits_upper, var_3, var_6);
            var_34 = wp::load(var_30);
            var_35 = wp::load(var_31);
            var_36 = wp::load(var_32);
            var_33 = wp::clamp(var_34, var_35, var_36);
            // default_joint_pos[env_id, joint_id] = wp.clamp(                                    <L 44>
            // wp::array_store(var_default_joint_pos, var_3, var_6, var_33);
        }
        // soft_joint_pos_limits[env_id, joint_id] = compute_soft_joint_pos_limits_func(          <L 49>
        // wp.vec2f(joint_pos_limits_lower[env_id, joint_id], joint_pos_limits_upper[env_id, joint_id]),       <L 50>
        var_37 = wp::address(var_joint_pos_limits_lower, var_3, var_6);
        var_38 = wp::address(var_joint_pos_limits_upper, var_3, var_6);
        var_40 = wp::load(var_37);
        var_41 = wp::load(var_38);
        var_39 = wp::vec_t<2, wp::float32>(var_40, var_41);
        // soft_limit_factor,                                                                     <L 51>
        var_42 = compute_soft_joint_pos_limits_func_0(var_39, var_soft_limit_factor);
        // soft_joint_pos_limits[env_id, joint_id] = compute_soft_joint_pos_limits_func(          <L 49>
        // wp::array_store(var_soft_joint_pos_limits, var_3, var_6, var_42);
        //---------
        // reverse
        wp::adj_array_store(var_soft_joint_pos_limits, var_3, var_6, var_42, adj_soft_joint_pos_limits, adj_3, adj_6, adj_42);
        // adj: soft_joint_pos_limits[env_id, joint_id] = compute_soft_joint_pos_limits_func(     <L 49>
        adj_compute_soft_joint_pos_limits_func_0(var_39, var_soft_limit_factor, adj_39, adj_soft_limit_factor, adj_42);
        // adj: soft_limit_factor,                                                                <L 51>
        wp::adj_vec_t(var_40, var_41, adj_37, adj_38, adj_39);
        wp::adj_address(var_joint_pos_limits_upper, var_3, var_6, adj_joint_pos_limits_upper, adj_3, adj_6, adj_38);
        wp::adj_address(var_joint_pos_limits_lower, var_3, var_6, adj_joint_pos_limits_lower, adj_3, adj_6, adj_37);
        // adj: wp.vec2f(joint_pos_limits_lower[env_id, joint_id], joint_pos_limits_upper[env_id, joint_id]),  <L 50>
        // adj: soft_joint_pos_limits[env_id, joint_id] = compute_soft_joint_pos_limits_func(     <L 49>
        if (var_16) {
            wp::adj_array_store(var_default_joint_pos, var_3, var_6, var_33, adj_default_joint_pos, adj_3, adj_6, adj_33);
            // adj: default_joint_pos[env_id, joint_id] = wp.clamp(                               <L 44>
            wp::adj_clamp(var_34, var_35, var_36, adj_30, adj_31, adj_32, adj_33);
            wp::adj_address(var_joint_pos_limits_upper, var_3, var_6, adj_joint_pos_limits_upper, adj_3, adj_6, adj_32);
            // adj: joint_pos_limits_upper[env_id, joint_id],                                     <L 47>
            wp::adj_address(var_joint_pos_limits_lower, var_3, var_6, adj_joint_pos_limits_lower, adj_3, adj_6, adj_31);
            // adj: joint_pos_limits_lower[env_id, joint_id],                                     <L 46>
            wp::adj_address(var_default_joint_pos, var_3, var_6, adj_default_joint_pos, adj_3, adj_6, adj_30);
            // adj: default_joint_pos[env_id, joint_id],                                          <L 45>
            // adj: default_joint_pos[env_id, joint_id] = wp.clamp(                               <L 44>
            wp::adj_atomic_add(var_clamped_defaults, var_27, var_28, adj_clamped_defaults, adj_27, adj_28, adj_29);
            // adj: wp.atomic_add(clamped_defaults, 0, 1)                                         <L 43>
        }
        if (!var_16) {
            wp::adj_address(var_joint_pos_limits_upper, var_3, var_6, adj_joint_pos_limits_upper, adj_3, adj_6, adj_23);
            // adj: ] > joint_pos_limits_upper[env_id, joint_id]:                                 <L 42>
            wp::adj_address(var_default_joint_pos, var_3, var_6, adj_default_joint_pos, adj_3, adj_6, adj_22);
            // adj: env_id, joint_id                                                              <L 41>
        }
        wp::adj_address(var_joint_pos_limits_lower, var_3, var_6, adj_joint_pos_limits_lower, adj_3, adj_6, adj_18);
        wp::adj_address(var_default_joint_pos, var_3, var_6, adj_default_joint_pos, adj_3, adj_6, adj_17);
        // adj: if (default_joint_pos[env_id, joint_id] < joint_pos_limits_lower[env_id, joint_id]) or default_joint_pos[  <L 40>
        wp::adj_array_store(var_joint_pos_limits_upper, var_3, var_6, var_14, adj_joint_pos_limits_upper, adj_3, adj_6, adj_14);
        wp::adj_extract(var_15, var_13, adj_12, adj_13, adj_14);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_12);
        // adj: joint_pos_limits_upper[env_id, joint_id] = in_data[i, j][1]                       <L 39>
        wp::adj_array_store(var_joint_pos_limits_lower, var_3, var_6, var_10, adj_joint_pos_limits_lower, adj_3, adj_6, adj_10);
        wp::adj_extract(var_11, var_9, adj_8, adj_9, adj_10);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_8);
        // adj: joint_pos_limits_lower[env_id, joint_id] = in_data[i, j][0]                       <L 38>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 37>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 36>
        // adj: i, j = wp.tid()                                                                   <L 35>
        // adj: def write_joint_limit_data_to_buffer_index(                                       <L 1>
        continue;
    }
}



extern "C" __global__ void scatter_fixed_tendon_position_targets_e8e56e12_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_position_target,
    wp::array_t<wp::int32> var_actuator_columns,
    wp::array_t<wp::int32> var_target_tendon_ids,
    wp::array_t<wp::float32> var_commands)
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
        wp::float32* var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::float32 var_8;
        //---------
        // forward
        // def scatter_fixed_tendon_position_targets(                                             <L 1074>
        // env_id, entry = wp.tid()                                                               <L 1087>
        builtin_tid2d(var_0, var_1);
        // commands[env_id, 0, actuator_columns[entry]] = position_target[env_id, target_tendon_ids[entry]]       <L 1088>
        var_2 = wp::address(var_target_tendon_ids, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::address(var_position_target, var_0, var_4);
        var_6 = wp::address(var_actuator_columns, var_1);
        var_7 = wp::load(var_6);
        var_8 = wp::load(var_3);
        wp::array_store(var_commands, var_0, var_5, var_7, var_8);
    }
}



extern "C" __global__ void scatter_fixed_tendon_position_targets_e8e56e12_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_position_target,
    wp::array_t<wp::int32> var_actuator_columns,
    wp::array_t<wp::int32> var_target_tendon_ids,
    wp::array_t<wp::float32> var_commands,
    wp::array_t<wp::float32> adj_position_target,
    wp::array_t<wp::int32> adj_actuator_columns,
    wp::array_t<wp::int32> adj_target_tendon_ids,
    wp::array_t<wp::float32> adj_commands)
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
        wp::float32* var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::float32 var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::float32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::float32 adj_8 = {};
        //---------
        // forward
        // def scatter_fixed_tendon_position_targets(                                             <L 1074>
        // env_id, entry = wp.tid()                                                               <L 1087>
        builtin_tid2d(var_0, var_1);
        // commands[env_id, 0, actuator_columns[entry]] = position_target[env_id, target_tendon_ids[entry]]       <L 1088>
        var_2 = wp::address(var_target_tendon_ids, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::address(var_position_target, var_0, var_4);
        var_6 = wp::address(var_actuator_columns, var_1);
        var_7 = wp::load(var_6);
        var_8 = wp::load(var_3);
        // wp::array_store(var_commands, var_0, var_5, var_7, var_8);
        //---------
        // reverse
        wp::adj_array_store(var_commands, var_0, var_5, var_7, var_8, adj_commands, adj_0, adj_5, adj_6, adj_3);
        wp::adj_address(var_actuator_columns, var_1, adj_actuator_columns, adj_1, adj_6);
        wp::adj_address(var_position_target, var_0, var_4, adj_position_target, adj_0, adj_2, adj_3);
        wp::adj_address(var_target_tendon_ids, var_1, adj_target_tendon_ids, adj_1, adj_2);
        // adj: commands[env_id, 0, actuator_columns[entry]] = position_target[env_id, target_tendon_ids[entry]]  <L 1088>
        // adj: env_id, entry = wp.tid()                                                          <L 1087>
        // adj: def scatter_fixed_tendon_position_targets(                                        <L 1074>
        continue;
    }
}



extern "C" __global__ void float_data_to_buffer_with_mask_ef5c317c_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_in_data,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_joint_mask,
    wp::array_t<wp::float32> var_out_data)
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
        //---------
        // forward
        // def float_data_to_buffer_with_mask(                                                    <L 523>
        // i, j = wp.tid()                                                                        <L 540>
        builtin_tid2d(var_0, var_1);
        // if env_mask[i] and joint_mask[j]:                                                      <L 541>
        var_3 = wp::address(var_env_mask, var_0);
        var_4 = wp::load(var_3);
        var_2 = var_4;
        if (var_2) {
            var_5 = wp::address(var_joint_mask, var_1);
            var_6 = wp::load(var_5);
            var_2 = var_2 && var_6;
        }
        if (var_2) {
            // out_data[i, j] = in_data                                                           <L 542>
            wp::array_store(var_out_data, var_0, var_1, var_in_data);
        }
    }
}



extern "C" __global__ void float_data_to_buffer_with_mask_ef5c317c_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_in_data,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_joint_mask,
    wp::array_t<wp::float32> var_out_data,
    wp::float32 adj_in_data,
    wp::array_t<bool> adj_env_mask,
    wp::array_t<bool> adj_joint_mask,
    wp::array_t<wp::float32> adj_out_data)
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
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        bool adj_6 = {};
        //---------
        // forward
        // def float_data_to_buffer_with_mask(                                                    <L 523>
        // i, j = wp.tid()                                                                        <L 540>
        builtin_tid2d(var_0, var_1);
        // if env_mask[i] and joint_mask[j]:                                                      <L 541>
        var_3 = wp::address(var_env_mask, var_0);
        var_4 = wp::load(var_3);
        var_2 = var_4;
        if (var_2) {
            var_5 = wp::address(var_joint_mask, var_1);
            var_6 = wp::load(var_5);
            var_2 = var_2 && var_6;
        }
        if (var_2) {
            // out_data[i, j] = in_data                                                           <L 542>
            // wp::array_store(var_out_data, var_0, var_1, var_in_data);
        }
        //---------
        // reverse
        if (var_2) {
            wp::adj_array_store(var_out_data, var_0, var_1, var_in_data, adj_out_data, adj_0, adj_1, adj_in_data);
            // adj: out_data[i, j] = in_data                                                      <L 542>
        }
        if (var_2) {
            wp::adj_address(var_joint_mask, var_1, adj_joint_mask, adj_1, adj_5);
        }
        wp::adj_address(var_env_mask, var_0, adj_env_mask, adj_0, adj_3);
        // adj: if env_mask[i] and joint_mask[j]:                                                 <L 541>
        // adj: i, j = wp.tid()                                                                   <L 540>
        // adj: def float_data_to_buffer_with_mask(                                               <L 523>
        continue;
    }
}



extern "C" __global__ void write_joint_vel_data_index_f1969352_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::array_t<wp::float32> var_joint_acc)
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
        const wp::float32 var_12 = 0.0;
        //---------
        // forward
        // def write_joint_vel_data_index(                                                        <L 1>
        // i, j = wp.tid()                                                                        <L 27>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 28>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 29>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // joint_vel[env_id, joint_id] = in_data[i, j]                                            <L 30>
        var_8 = wp::address(var_in_data, var_0, var_1);
        var_9 = wp::load(var_8);
        wp::array_store(var_joint_vel, var_3, var_6, var_9);
        // prev_joint_vel[env_id, joint_id] = in_data[i, j]                                       <L 31>
        var_10 = wp::address(var_in_data, var_0, var_1);
        var_11 = wp::load(var_10);
        wp::array_store(var_prev_joint_vel, var_3, var_6, var_11);
        // joint_acc[env_id, joint_id] = 0.0                                                      <L 32>
        wp::array_store(var_joint_acc, var_3, var_6, var_12);
    }
}



extern "C" __global__ void write_joint_vel_data_index_f1969352_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::array_t<wp::float32> var_joint_acc,
    wp::array_t<wp::float32> adj_in_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_joint_ids,
    wp::array_t<wp::float32> adj_joint_vel,
    wp::array_t<wp::float32> adj_prev_joint_vel,
    wp::array_t<wp::float32> adj_joint_acc)
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
        const wp::float32 var_12 = 0.0;
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
        //---------
        // forward
        // def write_joint_vel_data_index(                                                        <L 1>
        // i, j = wp.tid()                                                                        <L 27>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 28>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 29>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // joint_vel[env_id, joint_id] = in_data[i, j]                                            <L 30>
        var_8 = wp::address(var_in_data, var_0, var_1);
        var_9 = wp::load(var_8);
        // wp::array_store(var_joint_vel, var_3, var_6, var_9);
        // prev_joint_vel[env_id, joint_id] = in_data[i, j]                                       <L 31>
        var_10 = wp::address(var_in_data, var_0, var_1);
        var_11 = wp::load(var_10);
        // wp::array_store(var_prev_joint_vel, var_3, var_6, var_11);
        // joint_acc[env_id, joint_id] = 0.0                                                      <L 32>
        // wp::array_store(var_joint_acc, var_3, var_6, var_12);
        //---------
        // reverse
        wp::adj_array_store(var_joint_acc, var_3, var_6, var_12, adj_joint_acc, adj_3, adj_6, adj_12);
        // adj: joint_acc[env_id, joint_id] = 0.0                                                 <L 32>
        wp::adj_array_store(var_prev_joint_vel, var_3, var_6, var_11, adj_prev_joint_vel, adj_3, adj_6, adj_10);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_10);
        // adj: prev_joint_vel[env_id, joint_id] = in_data[i, j]                                  <L 31>
        wp::adj_array_store(var_joint_vel, var_3, var_6, var_9, adj_joint_vel, adj_3, adj_6, adj_8);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_8);
        // adj: joint_vel[env_id, joint_id] = in_data[i, j]                                       <L 30>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 29>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 28>
        // adj: i, j = wp.tid()                                                                   <L 27>
        // adj: def write_joint_vel_data_index(                                                   <L 1>
        continue;
    }
}



extern "C" __global__ void write_joint_limit_data_to_buffer_index_fbf1bd4c_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_in_data,
    wp::float32 var_soft_limit_factor,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::array_t<wp::float32> var_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_soft_joint_pos_limits,
    wp::array_t<wp::float32> var_default_joint_pos,
    wp::array_t<wp::int32> var_clamped_defaults)
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
        wp::vec_t<2, wp::float32>* var_8;
        const wp::int32 var_9 = 0;
        wp::float32 var_10;
        wp::vec_t<2, wp::float32> var_11;
        wp::vec_t<2, wp::float32>* var_12;
        const wp::int32 var_13 = 1;
        wp::float32 var_14;
        wp::vec_t<2, wp::float32> var_15;
        bool var_16;
        wp::float32* var_17;
        wp::float32* var_18;
        bool var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32* var_22;
        wp::float32* var_23;
        bool var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        const wp::int32 var_27 = 0;
        const wp::int32 var_28 = 1;
        wp::int32 var_29;
        wp::float32* var_30;
        wp::float32* var_31;
        wp::float32* var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32* var_37;
        wp::float32* var_38;
        wp::vec_t<2, wp::float32> var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::vec_t<2, wp::float32> var_42;
        //---------
        // forward
        // def write_joint_limit_data_to_buffer_index(                                            <L 1>
        // i, j = wp.tid()                                                                        <L 35>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 36>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 37>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // joint_pos_limits_lower[env_id, joint_id] = in_data[i, j][0]                            <L 38>
        var_8 = wp::address(var_in_data, var_0, var_1);
        var_11 = wp::load(var_8);
        var_10 = wp::extract(var_11, var_9);
        wp::array_store(var_joint_pos_limits_lower, var_3, var_6, var_10);
        // joint_pos_limits_upper[env_id, joint_id] = in_data[i, j][1]                            <L 39>
        var_12 = wp::address(var_in_data, var_0, var_1);
        var_15 = wp::load(var_12);
        var_14 = wp::extract(var_15, var_13);
        wp::array_store(var_joint_pos_limits_upper, var_3, var_6, var_14);
        // if (default_joint_pos[env_id, joint_id] < joint_pos_limits_lower[env_id, joint_id]) or default_joint_pos[       <L 40>
        var_17 = wp::address(var_default_joint_pos, var_3, var_6);
        var_18 = wp::address(var_joint_pos_limits_lower, var_3, var_6);
        var_20 = wp::load(var_17);
        var_21 = wp::load(var_18);
        var_19 = (var_20 < var_21);
        var_16 = var_19;
        if (!var_16) {
            // env_id, joint_id                                                                   <L 41>
            var_22 = wp::address(var_default_joint_pos, var_3, var_6);
            // ] > joint_pos_limits_upper[env_id, joint_id]:                                      <L 42>
            var_23 = wp::address(var_joint_pos_limits_upper, var_3, var_6);
            var_25 = wp::load(var_22);
            var_26 = wp::load(var_23);
            var_24 = (var_25 > var_26);
            var_16 = var_16 || var_24;
        }
        if (var_16) {
            // wp.atomic_add(clamped_defaults, 0, 1)                                              <L 43>
            var_29 = wp::atomic_add(var_clamped_defaults, var_27, var_28);
            // default_joint_pos[env_id, joint_id] = wp.clamp(                                    <L 44>
            // default_joint_pos[env_id, joint_id],                                               <L 45>
            var_30 = wp::address(var_default_joint_pos, var_3, var_6);
            // joint_pos_limits_lower[env_id, joint_id],                                          <L 46>
            var_31 = wp::address(var_joint_pos_limits_lower, var_3, var_6);
            // joint_pos_limits_upper[env_id, joint_id],                                          <L 47>
            var_32 = wp::address(var_joint_pos_limits_upper, var_3, var_6);
            var_34 = wp::load(var_30);
            var_35 = wp::load(var_31);
            var_36 = wp::load(var_32);
            var_33 = wp::clamp(var_34, var_35, var_36);
            // default_joint_pos[env_id, joint_id] = wp.clamp(                                    <L 44>
            wp::array_store(var_default_joint_pos, var_3, var_6, var_33);
        }
        // soft_joint_pos_limits[env_id, joint_id] = compute_soft_joint_pos_limits_func(          <L 49>
        // wp.vec2f(joint_pos_limits_lower[env_id, joint_id], joint_pos_limits_upper[env_id, joint_id]),       <L 50>
        var_37 = wp::address(var_joint_pos_limits_lower, var_3, var_6);
        var_38 = wp::address(var_joint_pos_limits_upper, var_3, var_6);
        var_40 = wp::load(var_37);
        var_41 = wp::load(var_38);
        var_39 = wp::vec_t<2, wp::float32>(var_40, var_41);
        // soft_limit_factor,                                                                     <L 51>
        var_42 = compute_soft_joint_pos_limits_func_0(var_39, var_soft_limit_factor);
        // soft_joint_pos_limits[env_id, joint_id] = compute_soft_joint_pos_limits_func(          <L 49>
        wp::array_store(var_soft_joint_pos_limits, var_3, var_6, var_42);
    }
}



extern "C" __global__ void write_joint_limit_data_to_buffer_index_fbf1bd4c_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_in_data,
    wp::float32 var_soft_limit_factor,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::array_t<wp::float32> var_joint_pos_limits_lower,
    wp::array_t<wp::float32> var_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> var_soft_joint_pos_limits,
    wp::array_t<wp::float32> var_default_joint_pos,
    wp::array_t<wp::int32> var_clamped_defaults,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_in_data,
    wp::float32 adj_soft_limit_factor,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_joint_ids,
    wp::array_t<wp::float32> adj_joint_pos_limits_lower,
    wp::array_t<wp::float32> adj_joint_pos_limits_upper,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_soft_joint_pos_limits,
    wp::array_t<wp::float32> adj_default_joint_pos,
    wp::array_t<wp::int32> adj_clamped_defaults)
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
        wp::vec_t<2, wp::float32>* var_8;
        const wp::int32 var_9 = 0;
        wp::float32 var_10;
        wp::vec_t<2, wp::float32> var_11;
        wp::vec_t<2, wp::float32>* var_12;
        const wp::int32 var_13 = 1;
        wp::float32 var_14;
        wp::vec_t<2, wp::float32> var_15;
        bool var_16;
        wp::float32* var_17;
        wp::float32* var_18;
        bool var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32* var_22;
        wp::float32* var_23;
        bool var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        const wp::int32 var_27 = 0;
        const wp::int32 var_28 = 1;
        wp::int32 var_29;
        wp::float32* var_30;
        wp::float32* var_31;
        wp::float32* var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32* var_37;
        wp::float32* var_38;
        wp::vec_t<2, wp::float32> var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::vec_t<2, wp::float32> var_42;
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
        wp::vec_t<2, wp::float32> adj_8 = {};
        wp::int32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::vec_t<2, wp::float32> adj_11 = {};
        wp::vec_t<2, wp::float32> adj_12 = {};
        wp::int32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::vec_t<2, wp::float32> adj_15 = {};
        bool adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        bool adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        bool adj_24 = {};
        wp::float32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::float32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::vec_t<2, wp::float32> adj_39 = {};
        wp::float32 adj_40 = {};
        wp::float32 adj_41 = {};
        wp::vec_t<2, wp::float32> adj_42 = {};
        //---------
        // forward
        // def write_joint_limit_data_to_buffer_index(                                            <L 1>
        // i, j = wp.tid()                                                                        <L 35>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 36>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 37>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // joint_pos_limits_lower[env_id, joint_id] = in_data[i, j][0]                            <L 38>
        var_8 = wp::address(var_in_data, var_0, var_1);
        var_11 = wp::load(var_8);
        var_10 = wp::extract(var_11, var_9);
        // wp::array_store(var_joint_pos_limits_lower, var_3, var_6, var_10);
        // joint_pos_limits_upper[env_id, joint_id] = in_data[i, j][1]                            <L 39>
        var_12 = wp::address(var_in_data, var_0, var_1);
        var_15 = wp::load(var_12);
        var_14 = wp::extract(var_15, var_13);
        // wp::array_store(var_joint_pos_limits_upper, var_3, var_6, var_14);
        // if (default_joint_pos[env_id, joint_id] < joint_pos_limits_lower[env_id, joint_id]) or default_joint_pos[       <L 40>
        var_17 = wp::address(var_default_joint_pos, var_3, var_6);
        var_18 = wp::address(var_joint_pos_limits_lower, var_3, var_6);
        var_20 = wp::load(var_17);
        var_21 = wp::load(var_18);
        var_19 = (var_20 < var_21);
        var_16 = var_19;
        if (!var_16) {
            // env_id, joint_id                                                                   <L 41>
            var_22 = wp::address(var_default_joint_pos, var_3, var_6);
            // ] > joint_pos_limits_upper[env_id, joint_id]:                                      <L 42>
            var_23 = wp::address(var_joint_pos_limits_upper, var_3, var_6);
            var_25 = wp::load(var_22);
            var_26 = wp::load(var_23);
            var_24 = (var_25 > var_26);
            var_16 = var_16 || var_24;
        }
        if (var_16) {
            // wp.atomic_add(clamped_defaults, 0, 1)                                              <L 43>
            // var_29 = wp::atomic_add(var_clamped_defaults, var_27, var_28);
            // default_joint_pos[env_id, joint_id] = wp.clamp(                                    <L 44>
            // default_joint_pos[env_id, joint_id],                                               <L 45>
            var_30 = wp::address(var_default_joint_pos, var_3, var_6);
            // joint_pos_limits_lower[env_id, joint_id],                                          <L 46>
            var_31 = wp::address(var_joint_pos_limits_lower, var_3, var_6);
            // joint_pos_limits_upper[env_id, joint_id],                                          <L 47>
            var_32 = wp::address(var_joint_pos_limits_upper, var_3, var_6);
            var_34 = wp::load(var_30);
            var_35 = wp::load(var_31);
            var_36 = wp::load(var_32);
            var_33 = wp::clamp(var_34, var_35, var_36);
            // default_joint_pos[env_id, joint_id] = wp.clamp(                                    <L 44>
            // wp::array_store(var_default_joint_pos, var_3, var_6, var_33);
        }
        // soft_joint_pos_limits[env_id, joint_id] = compute_soft_joint_pos_limits_func(          <L 49>
        // wp.vec2f(joint_pos_limits_lower[env_id, joint_id], joint_pos_limits_upper[env_id, joint_id]),       <L 50>
        var_37 = wp::address(var_joint_pos_limits_lower, var_3, var_6);
        var_38 = wp::address(var_joint_pos_limits_upper, var_3, var_6);
        var_40 = wp::load(var_37);
        var_41 = wp::load(var_38);
        var_39 = wp::vec_t<2, wp::float32>(var_40, var_41);
        // soft_limit_factor,                                                                     <L 51>
        var_42 = compute_soft_joint_pos_limits_func_0(var_39, var_soft_limit_factor);
        // soft_joint_pos_limits[env_id, joint_id] = compute_soft_joint_pos_limits_func(          <L 49>
        // wp::array_store(var_soft_joint_pos_limits, var_3, var_6, var_42);
        //---------
        // reverse
        wp::adj_array_store(var_soft_joint_pos_limits, var_3, var_6, var_42, adj_soft_joint_pos_limits, adj_3, adj_6, adj_42);
        // adj: soft_joint_pos_limits[env_id, joint_id] = compute_soft_joint_pos_limits_func(     <L 49>
        adj_compute_soft_joint_pos_limits_func_0(var_39, var_soft_limit_factor, adj_39, adj_soft_limit_factor, adj_42);
        // adj: soft_limit_factor,                                                                <L 51>
        wp::adj_vec_t(var_40, var_41, adj_37, adj_38, adj_39);
        wp::adj_address(var_joint_pos_limits_upper, var_3, var_6, adj_joint_pos_limits_upper, adj_3, adj_6, adj_38);
        wp::adj_address(var_joint_pos_limits_lower, var_3, var_6, adj_joint_pos_limits_lower, adj_3, adj_6, adj_37);
        // adj: wp.vec2f(joint_pos_limits_lower[env_id, joint_id], joint_pos_limits_upper[env_id, joint_id]),  <L 50>
        // adj: soft_joint_pos_limits[env_id, joint_id] = compute_soft_joint_pos_limits_func(     <L 49>
        if (var_16) {
            wp::adj_array_store(var_default_joint_pos, var_3, var_6, var_33, adj_default_joint_pos, adj_3, adj_6, adj_33);
            // adj: default_joint_pos[env_id, joint_id] = wp.clamp(                               <L 44>
            wp::adj_clamp(var_34, var_35, var_36, adj_30, adj_31, adj_32, adj_33);
            wp::adj_address(var_joint_pos_limits_upper, var_3, var_6, adj_joint_pos_limits_upper, adj_3, adj_6, adj_32);
            // adj: joint_pos_limits_upper[env_id, joint_id],                                     <L 47>
            wp::adj_address(var_joint_pos_limits_lower, var_3, var_6, adj_joint_pos_limits_lower, adj_3, adj_6, adj_31);
            // adj: joint_pos_limits_lower[env_id, joint_id],                                     <L 46>
            wp::adj_address(var_default_joint_pos, var_3, var_6, adj_default_joint_pos, adj_3, adj_6, adj_30);
            // adj: default_joint_pos[env_id, joint_id],                                          <L 45>
            // adj: default_joint_pos[env_id, joint_id] = wp.clamp(                               <L 44>
            wp::adj_atomic_add(var_clamped_defaults, var_27, var_28, adj_clamped_defaults, adj_27, adj_28, adj_29);
            // adj: wp.atomic_add(clamped_defaults, 0, 1)                                         <L 43>
        }
        if (!var_16) {
            wp::adj_address(var_joint_pos_limits_upper, var_3, var_6, adj_joint_pos_limits_upper, adj_3, adj_6, adj_23);
            // adj: ] > joint_pos_limits_upper[env_id, joint_id]:                                 <L 42>
            wp::adj_address(var_default_joint_pos, var_3, var_6, adj_default_joint_pos, adj_3, adj_6, adj_22);
            // adj: env_id, joint_id                                                              <L 41>
        }
        wp::adj_address(var_joint_pos_limits_lower, var_3, var_6, adj_joint_pos_limits_lower, adj_3, adj_6, adj_18);
        wp::adj_address(var_default_joint_pos, var_3, var_6, adj_default_joint_pos, adj_3, adj_6, adj_17);
        // adj: if (default_joint_pos[env_id, joint_id] < joint_pos_limits_lower[env_id, joint_id]) or default_joint_pos[  <L 40>
        wp::adj_array_store(var_joint_pos_limits_upper, var_3, var_6, var_14, adj_joint_pos_limits_upper, adj_3, adj_6, adj_14);
        wp::adj_extract(var_15, var_13, adj_12, adj_13, adj_14);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_12);
        // adj: joint_pos_limits_upper[env_id, joint_id] = in_data[i, j][1]                       <L 39>
        wp::adj_array_store(var_joint_pos_limits_lower, var_3, var_6, var_10, adj_joint_pos_limits_lower, adj_3, adj_6, adj_10);
        wp::adj_extract(var_11, var_9, adj_8, adj_9, adj_10);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_8);
        // adj: joint_pos_limits_lower[env_id, joint_id] = in_data[i, j][0]                       <L 38>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 37>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 36>
        // adj: i, j = wp.tid()                                                                   <L 35>
        // adj: def write_joint_limit_data_to_buffer_index(                                       <L 1>
        continue;
    }
}

