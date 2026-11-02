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



extern "C" __global__ void copy_from_newton_kernel_53c6608c_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_env_mask,
    wp::int32 var_num_sensors,
    wp::array_t<wp::vec_t<3, wp::float32>> var_newton_total_force,
    wp::array_t<wp::vec_t<3, wp::float32>> var_newton_total_force_friction,
    wp::array_t<wp::vec_t<3, wp::float32>> var_newton_force_matrix,
    wp::array_t<wp::vec_t<3, wp::float32>> var_newton_force_matrix_friction,
    wp::array_t<wp::vec_t<3, wp::float32>> var_newton_position_matrix,
    wp::array_t<wp::float32> var_timestamp,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_force,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_normal_force,
    wp::array_t<wp::vec_t<3, wp::float32>> var_force_matrix,
    wp::array_t<wp::vec_t<3, wp::float32>> var_normal_force_matrix,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_friction_force,
    wp::array_t<wp::vec_t<3, wp::float32>> var_friction_force_matrix,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_w)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
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
        bool var_5;
        wp::float32* var_6;
        const wp::float32 var_7 = 0.0;
        bool var_8;
        wp::float32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        const wp::int32 var_12 = 0;
        bool var_13;
        wp::vec_t<3, wp::float32>* var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32>* var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        bool var_30;
        wp::vec_t<3, wp::float32>* var_31;
        wp::float32 var_32;
        wp::vec_t<3, wp::float32> var_33;
        const wp::float32 var_34 = 0.0;
        bool var_35;
        wp::vec_t<3, wp::float32>* var_36;
        wp::vec_t<3, wp::float32> var_37;
        const wp::float32 var_38 = NAN;
        const wp::float32 var_39 = NAN;
        wp::vec_t<3, wp::float32> var_40;
        //---------
        // forward
        // def copy_from_newton_kernel(                                                           <L 13>
        // env, sensor, f_idx = wp.tid()                                                          <L 37>
        builtin_tid3d(var_0, var_1, var_2);
        // if env_mask:                                                                           <L 39>
        if (var_env_mask) {
            // if not env_mask[env]:                                                              <L 40>
            var_3 = wp::address(var_env_mask, var_0);
            var_5 = wp::load(var_3);
            var_4 = wp::unot(var_5);
            if (var_4) {
                // return                                                                         <L 41>
                continue;
            }
        }
        // if timestamp[env] == 0.0:                                                              <L 45>
        var_6 = wp::address(var_timestamp, var_0);
        var_9 = wp::load(var_6);
        var_8 = (var_9 == var_7);
        if (var_8) {
            // return                                                                             <L 46>
            continue;
        }
        // src_idx = env * num_sensors + sensor                                                   <L 49>
        var_10 = wp::mul(var_0, var_num_sensors);
        var_11 = wp::add(var_10, var_1);
        // if f_idx == 0:                                                                         <L 50>
        var_13 = (var_2 == var_12);
        if (var_13) {
            // total = newton_total_force[src_idx]                                                <L 51>
            var_14 = wp::address(var_newton_total_force, var_11);
            var_16 = wp::load(var_14);
            var_15 = wp::copy(var_16);
            // friction = newton_total_force_friction[src_idx]                                    <L 52>
            var_17 = wp::address(var_newton_total_force_friction, var_11);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // net_force[env, sensor] = total                                                     <L 53>
            wp::array_store(var_net_force, var_0, var_1, var_15);
            // net_normal_force[env, sensor] = total - friction                                   <L 54>
            var_20 = wp::sub(var_15, var_18);
            wp::array_store(var_net_normal_force, var_0, var_1, var_20);
            // if net_friction_force:                                                             <L 55>
            if (var_net_friction_force) {
                // net_friction_force[env, sensor] = friction                                     <L 56>
                wp::array_store(var_net_friction_force, var_0, var_1, var_18);
            }
        }
        // if force_matrix:                                                                       <L 59>
        if (var_force_matrix) {
            // total = newton_force_matrix[src_idx, f_idx]                                        <L 60>
            var_21 = wp::address(var_newton_force_matrix, var_11, var_2);
            var_23 = wp::load(var_21);
            var_22 = wp::copy(var_23);
            // friction = newton_force_matrix_friction[src_idx, f_idx]                            <L 61>
            var_24 = wp::address(var_newton_force_matrix_friction, var_11, var_2);
            var_26 = wp::load(var_24);
            var_25 = wp::copy(var_26);
            // force_matrix[env, sensor, f_idx] = total                                           <L 62>
            wp::array_store(var_force_matrix, var_0, var_1, var_2, var_22);
            // normal_force_matrix[env, sensor, f_idx] = total - friction                         <L 63>
            var_27 = wp::sub(var_22, var_25);
            wp::array_store(var_normal_force_matrix, var_0, var_1, var_2, var_27);
            // if friction_force_matrix:                                                          <L 64>
            if (var_friction_force_matrix) {
                // friction_force_matrix[env, sensor, f_idx] = friction                           <L 65>
                wp::array_store(var_friction_force_matrix, var_0, var_1, var_2, var_25);
            }
        }
        var_28 = wp::where(var_force_matrix, var_22, var_15);
        var_29 = wp::where(var_force_matrix, var_25, var_18);
        // if contact_pos_w and normal_force_matrix:                                              <L 69>
        var_30 = var_contact_pos_w;
        if (var_30) {
            var_30 = var_30 && var_normal_force_matrix;
        }
        if (var_30) {
            // if wp.length_sq(normal_force_matrix[env, sensor, f_idx]) > 0.0:                    <L 70>
            var_31 = wp::address(var_normal_force_matrix, var_0, var_1, var_2);
            var_33 = wp::load(var_31);
            var_32 = wp::length_sq(var_33);
            var_35 = (var_32 > var_34);
            if (var_35) {
                // contact_pos_w[env, sensor, f_idx] = newton_position_matrix[src_idx, f_idx]       <L 71>
                var_36 = wp::address(var_newton_position_matrix, var_11, var_2);
                var_37 = wp::load(var_36);
                wp::array_store(var_contact_pos_w, var_0, var_1, var_2, var_37);
            }
            if (!var_35) {
                // contact_pos_w[env, sensor, f_idx] = wp.vec3f(wp.nan)                           <L 73>
                var_40 = wp::vec_t<3, wp::float32>(var_39);
                wp::array_store(var_contact_pos_w, var_0, var_1, var_2, var_40);
            }
        }
    }
}



extern "C" __global__ void copy_from_newton_kernel_53c6608c_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_env_mask,
    wp::int32 var_num_sensors,
    wp::array_t<wp::vec_t<3, wp::float32>> var_newton_total_force,
    wp::array_t<wp::vec_t<3, wp::float32>> var_newton_total_force_friction,
    wp::array_t<wp::vec_t<3, wp::float32>> var_newton_force_matrix,
    wp::array_t<wp::vec_t<3, wp::float32>> var_newton_force_matrix_friction,
    wp::array_t<wp::vec_t<3, wp::float32>> var_newton_position_matrix,
    wp::array_t<wp::float32> var_timestamp,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_force,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_normal_force,
    wp::array_t<wp::vec_t<3, wp::float32>> var_force_matrix,
    wp::array_t<wp::vec_t<3, wp::float32>> var_normal_force_matrix,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_friction_force,
    wp::array_t<wp::vec_t<3, wp::float32>> var_friction_force_matrix,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_w,
    wp::array_t<bool> adj_env_mask,
    wp::int32 adj_num_sensors,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_newton_total_force,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_newton_total_force_friction,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_newton_force_matrix,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_newton_force_matrix_friction,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_newton_position_matrix,
    wp::array_t<wp::float32> adj_timestamp,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_net_force,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_net_normal_force,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_force_matrix,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_normal_force_matrix,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_net_friction_force,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_friction_force_matrix,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_contact_pos_w)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
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
        bool var_5;
        wp::float32* var_6;
        const wp::float32 var_7 = 0.0;
        bool var_8;
        wp::float32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        const wp::int32 var_12 = 0;
        bool var_13;
        wp::vec_t<3, wp::float32>* var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32>* var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        bool var_30;
        wp::vec_t<3, wp::float32>* var_31;
        wp::float32 var_32;
        wp::vec_t<3, wp::float32> var_33;
        const wp::float32 var_34 = 0.0;
        bool var_35;
        wp::vec_t<3, wp::float32>* var_36;
        wp::vec_t<3, wp::float32> var_37;
        const wp::float32 var_38 = NAN;
        const wp::float32 var_39 = NAN;
        wp::vec_t<3, wp::float32> var_40;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        bool adj_8 = {};
        wp::float32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        bool adj_13 = {};
        wp::vec_t<3, wp::float32> adj_14 = {};
        wp::vec_t<3, wp::float32> adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::vec_t<3, wp::float32> adj_17 = {};
        wp::vec_t<3, wp::float32> adj_18 = {};
        wp::vec_t<3, wp::float32> adj_19 = {};
        wp::vec_t<3, wp::float32> adj_20 = {};
        wp::vec_t<3, wp::float32> adj_21 = {};
        wp::vec_t<3, wp::float32> adj_22 = {};
        wp::vec_t<3, wp::float32> adj_23 = {};
        wp::vec_t<3, wp::float32> adj_24 = {};
        wp::vec_t<3, wp::float32> adj_25 = {};
        wp::vec_t<3, wp::float32> adj_26 = {};
        wp::vec_t<3, wp::float32> adj_27 = {};
        wp::vec_t<3, wp::float32> adj_28 = {};
        wp::vec_t<3, wp::float32> adj_29 = {};
        bool adj_30 = {};
        wp::vec_t<3, wp::float32> adj_31 = {};
        wp::float32 adj_32 = {};
        wp::vec_t<3, wp::float32> adj_33 = {};
        wp::float32 adj_34 = {};
        bool adj_35 = {};
        wp::vec_t<3, wp::float32> adj_36 = {};
        wp::vec_t<3, wp::float32> adj_37 = {};
        wp::float32 adj_38 = {};
        wp::float32 adj_39 = {};
        wp::vec_t<3, wp::float32> adj_40 = {};
        //---------
        // forward
        // def copy_from_newton_kernel(                                                           <L 13>
        // env, sensor, f_idx = wp.tid()                                                          <L 37>
        builtin_tid3d(var_0, var_1, var_2);
        // if env_mask:                                                                           <L 39>
        if (var_env_mask) {
            // if not env_mask[env]:                                                              <L 40>
            var_3 = wp::address(var_env_mask, var_0);
            var_5 = wp::load(var_3);
            var_4 = wp::unot(var_5);
            if (var_4) {
                // return                                                                         <L 41>
                goto label0;
            }
        }
        // if timestamp[env] == 0.0:                                                              <L 45>
        var_6 = wp::address(var_timestamp, var_0);
        var_9 = wp::load(var_6);
        var_8 = (var_9 == var_7);
        if (var_8) {
            // return                                                                             <L 46>
            goto label1;
        }
        // src_idx = env * num_sensors + sensor                                                   <L 49>
        var_10 = wp::mul(var_0, var_num_sensors);
        var_11 = wp::add(var_10, var_1);
        // if f_idx == 0:                                                                         <L 50>
        var_13 = (var_2 == var_12);
        if (var_13) {
            // total = newton_total_force[src_idx]                                                <L 51>
            var_14 = wp::address(var_newton_total_force, var_11);
            var_16 = wp::load(var_14);
            var_15 = wp::copy(var_16);
            // friction = newton_total_force_friction[src_idx]                                    <L 52>
            var_17 = wp::address(var_newton_total_force_friction, var_11);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // net_force[env, sensor] = total                                                     <L 53>
            // wp::array_store(var_net_force, var_0, var_1, var_15);
            // net_normal_force[env, sensor] = total - friction                                   <L 54>
            var_20 = wp::sub(var_15, var_18);
            // wp::array_store(var_net_normal_force, var_0, var_1, var_20);
            // if net_friction_force:                                                             <L 55>
            if (var_net_friction_force) {
                // net_friction_force[env, sensor] = friction                                     <L 56>
                // wp::array_store(var_net_friction_force, var_0, var_1, var_18);
            }
        }
        // if force_matrix:                                                                       <L 59>
        if (var_force_matrix) {
            // total = newton_force_matrix[src_idx, f_idx]                                        <L 60>
            var_21 = wp::address(var_newton_force_matrix, var_11, var_2);
            var_23 = wp::load(var_21);
            var_22 = wp::copy(var_23);
            // friction = newton_force_matrix_friction[src_idx, f_idx]                            <L 61>
            var_24 = wp::address(var_newton_force_matrix_friction, var_11, var_2);
            var_26 = wp::load(var_24);
            var_25 = wp::copy(var_26);
            // force_matrix[env, sensor, f_idx] = total                                           <L 62>
            // wp::array_store(var_force_matrix, var_0, var_1, var_2, var_22);
            // normal_force_matrix[env, sensor, f_idx] = total - friction                         <L 63>
            var_27 = wp::sub(var_22, var_25);
            // wp::array_store(var_normal_force_matrix, var_0, var_1, var_2, var_27);
            // if friction_force_matrix:                                                          <L 64>
            if (var_friction_force_matrix) {
                // friction_force_matrix[env, sensor, f_idx] = friction                           <L 65>
                // wp::array_store(var_friction_force_matrix, var_0, var_1, var_2, var_25);
            }
        }
        var_28 = wp::where(var_force_matrix, var_22, var_15);
        var_29 = wp::where(var_force_matrix, var_25, var_18);
        // if contact_pos_w and normal_force_matrix:                                              <L 69>
        var_30 = var_contact_pos_w;
        if (var_30) {
            var_30 = var_30 && var_normal_force_matrix;
        }
        if (var_30) {
            // if wp.length_sq(normal_force_matrix[env, sensor, f_idx]) > 0.0:                    <L 70>
            var_31 = wp::address(var_normal_force_matrix, var_0, var_1, var_2);
            var_33 = wp::load(var_31);
            var_32 = wp::length_sq(var_33);
            var_35 = (var_32 > var_34);
            if (var_35) {
                // contact_pos_w[env, sensor, f_idx] = newton_position_matrix[src_idx, f_idx]       <L 71>
                var_36 = wp::address(var_newton_position_matrix, var_11, var_2);
                var_37 = wp::load(var_36);
                // wp::array_store(var_contact_pos_w, var_0, var_1, var_2, var_37);
            }
            if (!var_35) {
                // contact_pos_w[env, sensor, f_idx] = wp.vec3f(wp.nan)                           <L 73>
                var_40 = wp::vec_t<3, wp::float32>(var_39);
                // wp::array_store(var_contact_pos_w, var_0, var_1, var_2, var_40);
            }
        }
        //---------
        // reverse
        if (var_30) {
            if (!var_35) {
                wp::adj_array_store(var_contact_pos_w, var_0, var_1, var_2, var_40, adj_contact_pos_w, adj_0, adj_1, adj_2, adj_40);
                wp::adj_vec_t(var_39, adj_39, adj_40);
                // adj: contact_pos_w[env, sensor, f_idx] = wp.vec3f(wp.nan)                      <L 73>
            }
            if (var_35) {
                wp::adj_array_store(var_contact_pos_w, var_0, var_1, var_2, var_37, adj_contact_pos_w, adj_0, adj_1, adj_2, adj_36);
                wp::adj_address(var_newton_position_matrix, var_11, var_2, adj_newton_position_matrix, adj_11, adj_2, adj_36);
                // adj: contact_pos_w[env, sensor, f_idx] = newton_position_matrix[src_idx, f_idx]  <L 71>
            }
            wp::adj_length_sq(var_33, adj_31, adj_32);
            wp::adj_address(var_normal_force_matrix, var_0, var_1, var_2, adj_normal_force_matrix, adj_0, adj_1, adj_2, adj_31);
            // adj: if wp.length_sq(normal_force_matrix[env, sensor, f_idx]) > 0.0:               <L 70>
        }
        if (var_30) {
        }
        // adj: if contact_pos_w and normal_force_matrix:                                         <L 69>
        wp::adj_where(var_force_matrix, var_25, var_18, adj_force_matrix, adj_25, adj_18, adj_29);
        wp::adj_where(var_force_matrix, var_22, var_15, adj_force_matrix, adj_22, adj_15, adj_28);
        if (var_force_matrix) {
            if (var_friction_force_matrix) {
                wp::adj_array_store(var_friction_force_matrix, var_0, var_1, var_2, var_25, adj_friction_force_matrix, adj_0, adj_1, adj_2, adj_25);
                // adj: friction_force_matrix[env, sensor, f_idx] = friction                      <L 65>
            }
            // adj: if friction_force_matrix:                                                     <L 64>
            wp::adj_array_store(var_normal_force_matrix, var_0, var_1, var_2, var_27, adj_normal_force_matrix, adj_0, adj_1, adj_2, adj_27);
            wp::adj_sub(var_22, var_25, adj_22, adj_25, adj_27);
            // adj: normal_force_matrix[env, sensor, f_idx] = total - friction                    <L 63>
            wp::adj_array_store(var_force_matrix, var_0, var_1, var_2, var_22, adj_force_matrix, adj_0, adj_1, adj_2, adj_22);
            // adj: force_matrix[env, sensor, f_idx] = total                                      <L 62>
            wp::adj_copy(var_26, adj_24, adj_25);
            wp::adj_address(var_newton_force_matrix_friction, var_11, var_2, adj_newton_force_matrix_friction, adj_11, adj_2, adj_24);
            // adj: friction = newton_force_matrix_friction[src_idx, f_idx]                       <L 61>
            wp::adj_copy(var_23, adj_21, adj_22);
            wp::adj_address(var_newton_force_matrix, var_11, var_2, adj_newton_force_matrix, adj_11, adj_2, adj_21);
            // adj: total = newton_force_matrix[src_idx, f_idx]                                   <L 60>
        }
        // adj: if force_matrix:                                                                  <L 59>
        if (var_13) {
            if (var_net_friction_force) {
                wp::adj_array_store(var_net_friction_force, var_0, var_1, var_18, adj_net_friction_force, adj_0, adj_1, adj_18);
                // adj: net_friction_force[env, sensor] = friction                                <L 56>
            }
            // adj: if net_friction_force:                                                        <L 55>
            wp::adj_array_store(var_net_normal_force, var_0, var_1, var_20, adj_net_normal_force, adj_0, adj_1, adj_20);
            wp::adj_sub(var_15, var_18, adj_15, adj_18, adj_20);
            // adj: net_normal_force[env, sensor] = total - friction                              <L 54>
            wp::adj_array_store(var_net_force, var_0, var_1, var_15, adj_net_force, adj_0, adj_1, adj_15);
            // adj: net_force[env, sensor] = total                                                <L 53>
            wp::adj_copy(var_19, adj_17, adj_18);
            wp::adj_address(var_newton_total_force_friction, var_11, adj_newton_total_force_friction, adj_11, adj_17);
            // adj: friction = newton_total_force_friction[src_idx]                               <L 52>
            wp::adj_copy(var_16, adj_14, adj_15);
            wp::adj_address(var_newton_total_force, var_11, adj_newton_total_force, adj_11, adj_14);
            // adj: total = newton_total_force[src_idx]                                           <L 51>
        }
        // adj: if f_idx == 0:                                                                    <L 50>
        wp::adj_add(var_10, var_1, adj_10, adj_1, adj_11);
        wp::adj_mul(var_0, var_num_sensors, adj_0, adj_num_sensors, adj_10);
        // adj: src_idx = env * num_sensors + sensor                                              <L 49>
        if (var_8) {
            label1:;
            // adj: return                                                                        <L 46>
        }
        wp::adj_address(var_timestamp, var_0, adj_timestamp, adj_0, adj_6);
        // adj: if timestamp[env] == 0.0:                                                         <L 45>
        if (var_env_mask) {
            if (var_4) {
                label0:;
                // adj: return                                                                    <L 41>
            }
            wp::adj_address(var_env_mask, var_0, adj_env_mask, adj_0, adj_3);
            // adj: if not env_mask[env]:                                                         <L 40>
        }
        // adj: if env_mask:                                                                      <L 39>
        // adj: env, sensor, f_idx = wp.tid()                                                     <L 37>
        // adj: def copy_from_newton_kernel(                                                      <L 13>
        continue;
    }
}



extern "C" __global__ void update_contact_sensor_kernel_7e4412e0_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_history_length,
    wp::int32 var_num_filter_objects,
    wp::float32 var_contact_force_threshold,
    wp::array_t<bool> var_env_mask,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> var_force_matrix,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_normal_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> var_normal_force_matrix,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_friction_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> var_friction_force_matrix,
    wp::array_t<wp::float32> var_timestamp,
    wp::array_t<wp::float32> var_timestamp_last_update,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_forces_history,
    wp::array_t<wp::vec_t<3, wp::float32>> var_force_matrix_history,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_normal_forces_history,
    wp::array_t<wp::vec_t<3, wp::float32>> var_normal_force_matrix_history,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_friction_forces_history,
    wp::array_t<wp::vec_t<3, wp::float32>> var_friction_force_matrix_history,
    wp::array_t<wp::float32> var_current_air_time,
    wp::array_t<wp::float32> var_current_contact_time,
    wp::array_t<wp::float32> var_last_air_time,
    wp::array_t<wp::float32> var_last_contact_time)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
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
        bool var_4;
        const wp::int32 var_5 = 1;
        wp::int32 var_6;
        const wp::int32 var_7 = 0;
        const wp::int32 var_8 = -1;
        wp::range_t var_9;
        wp::int32 var_10;
        const wp::int32 var_11 = 1;
        wp::int32 var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32>* var_15;
        const wp::int32 var_16 = 0;
        wp::vec_t<3, wp::float32> var_17;
        wp::range_t var_18;
        wp::int32 var_19;
        const wp::int32 var_20 = 1;
        wp::int32 var_21;
        const wp::int32 var_22 = 0;
        const wp::int32 var_23 = -1;
        wp::range_t var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 1;
        wp::int32 var_27;
        wp::vec_t<3, wp::float32>* var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32>* var_30;
        const wp::int32 var_31 = 0;
        wp::vec_t<3, wp::float32> var_32;
        const wp::int32 var_33 = 1;
        wp::int32 var_34;
        const wp::int32 var_35 = 0;
        const wp::int32 var_36 = -1;
        wp::range_t var_37;
        wp::int32 var_38;
        const wp::int32 var_39 = 1;
        wp::int32 var_40;
        wp::vec_t<3, wp::float32>* var_41;
        wp::vec_t<3, wp::float32> var_42;
        wp::vec_t<3, wp::float32>* var_43;
        const wp::int32 var_44 = 0;
        wp::vec_t<3, wp::float32> var_45;
        wp::int32 var_46;
        wp::range_t var_47;
        wp::int32 var_48;
        const wp::int32 var_49 = 1;
        wp::int32 var_50;
        const wp::int32 var_51 = 0;
        const wp::int32 var_52 = -1;
        wp::range_t var_53;
        wp::int32 var_54;
        const wp::int32 var_55 = 1;
        wp::int32 var_56;
        wp::vec_t<3, wp::float32>* var_57;
        wp::vec_t<3, wp::float32> var_58;
        wp::vec_t<3, wp::float32>* var_59;
        const wp::int32 var_60 = 0;
        wp::vec_t<3, wp::float32> var_61;
        wp::int32 var_62;
        const wp::int32 var_63 = 1;
        wp::int32 var_64;
        const wp::int32 var_65 = 0;
        const wp::int32 var_66 = -1;
        wp::range_t var_67;
        wp::int32 var_68;
        const wp::int32 var_69 = 1;
        wp::int32 var_70;
        wp::vec_t<3, wp::float32>* var_71;
        wp::vec_t<3, wp::float32> var_72;
        wp::vec_t<3, wp::float32>* var_73;
        const wp::int32 var_74 = 0;
        wp::vec_t<3, wp::float32> var_75;
        wp::int32 var_76;
        wp::range_t var_77;
        wp::int32 var_78;
        const wp::int32 var_79 = 1;
        wp::int32 var_80;
        const wp::int32 var_81 = 0;
        const wp::int32 var_82 = -1;
        wp::range_t var_83;
        wp::int32 var_84;
        const wp::int32 var_85 = 1;
        wp::int32 var_86;
        wp::vec_t<3, wp::float32>* var_87;
        wp::vec_t<3, wp::float32> var_88;
        wp::vec_t<3, wp::float32>* var_89;
        const wp::int32 var_90 = 0;
        wp::vec_t<3, wp::float32> var_91;
        wp::int32 var_92;
        wp::float32* var_93;
        wp::float32* var_94;
        wp::float32 var_95;
        wp::float32 var_96;
        wp::float32 var_97;
        wp::vec_t<3, wp::float32>* var_98;
        wp::float32 var_99;
        wp::vec_t<3, wp::float32> var_100;
        wp::float32 var_101;
        bool var_102;
        wp::float32* var_103;
        wp::float32 var_104;
        wp::float32 var_105;
        wp::float32* var_106;
        wp::float32 var_107;
        wp::float32 var_108;
        bool var_109;
        const wp::float32 var_110 = 0.0;
        bool var_111;
        bool var_112;
        bool var_113;
        const wp::float32 var_114 = 0.0;
        bool var_115;
        wp::float32 var_116;
        wp::float32 var_117;
        wp::float32 var_118;
        const wp::float32 var_119 = 0.0;
        wp::float32 var_120;
        const wp::float32 var_121 = 0.0;
        wp::float32 var_122;
        wp::float32 var_123;
        //---------
        // forward
        // def update_contact_sensor_kernel(                                                      <L 162>
        // env, sensor = wp.tid()                                                                 <L 193>
        builtin_tid2d(var_0, var_1);
        // if env_mask:                                                                           <L 195>
        if (var_env_mask) {
            // if not env_mask[env]:                                                              <L 196>
            var_2 = wp::address(var_env_mask, var_0);
            var_4 = wp::load(var_2);
            var_3 = wp::unot(var_4);
            if (var_3) {
                // return                                                                         <L 197>
                continue;
            }
        }
        // if net_forces_history:                                                                 <L 200>
        if (var_net_forces_history) {
            // for i in range(history_length - 1, 0, -1):                                         <L 201>
            var_6 = wp::sub(var_history_length, var_5);
            var_9 = wp::range(var_6, var_7, var_8);
            start_for_1:;
                if (iter_cmp(var_9) == 0) goto end_for_1;
                var_10 = wp::iter_next(var_9);
                // net_forces_history[env, i, sensor] = net_forces_history[env, i - 1, sensor]       <L 202>
                var_12 = wp::sub(var_10, var_11);
                var_13 = wp::address(var_net_forces_history, var_0, var_12, var_1);
                var_14 = wp::load(var_13);
                wp::array_store(var_net_forces_history, var_0, var_10, var_1, var_14);
                goto start_for_1;
            end_for_1:;
            // net_forces_history[env, 0, sensor] = net_forces[env, sensor]                       <L 203>
            var_15 = wp::address(var_net_forces, var_0, var_1);
            var_17 = wp::load(var_15);
            wp::array_store(var_net_forces_history, var_0, var_16, var_1, var_17);
        }
        // if force_matrix_history:                                                               <L 205>
        if (var_force_matrix_history) {
            // for f in range(num_filter_objects):                                                <L 206>
            var_18 = wp::range(var_num_filter_objects);
            start_for_3:;
                if (iter_cmp(var_18) == 0) goto end_for_3;
                var_19 = wp::iter_next(var_18);
                // for i in range(history_length - 1, 0, -1):                                     <L 207>
                var_21 = wp::sub(var_history_length, var_20);
                var_24 = wp::range(var_21, var_22, var_23);
                start_for_5:;
                    if (iter_cmp(var_24) == 0) goto end_for_5;
                    var_25 = wp::iter_next(var_24);
                    // force_matrix_history[env, i, sensor, f] = force_matrix_history[env, i - 1, sensor, f]       <L 208>
                    var_27 = wp::sub(var_25, var_26);
                    var_28 = wp::address(var_force_matrix_history, var_0, var_27, var_1, var_19);
                    var_29 = wp::load(var_28);
                    wp::array_store(var_force_matrix_history, var_0, var_25, var_1, var_19, var_29);
                    goto start_for_5;
                end_for_5:;
                // force_matrix_history[env, 0, sensor, f] = force_matrix[env, sensor, f]         <L 209>
                var_30 = wp::address(var_force_matrix, var_0, var_1, var_19);
                var_32 = wp::load(var_30);
                wp::array_store(var_force_matrix_history, var_0, var_31, var_1, var_19, var_32);
                wp::assign(var_10, var_25);
                goto start_for_3;
            end_for_3:;
        }
        // if net_normal_forces_history:                                                          <L 212>
        if (var_net_normal_forces_history) {
            // for i in range(history_length - 1, 0, -1):                                         <L 213>
            var_34 = wp::sub(var_history_length, var_33);
            var_37 = wp::range(var_34, var_35, var_36);
            start_for_7:;
                if (iter_cmp(var_37) == 0) goto end_for_7;
                var_38 = wp::iter_next(var_37);
                // net_normal_forces_history[env, i, sensor] = net_normal_forces_history[env, i - 1, sensor]       <L 214>
                var_40 = wp::sub(var_38, var_39);
                var_41 = wp::address(var_net_normal_forces_history, var_0, var_40, var_1);
                var_42 = wp::load(var_41);
                wp::array_store(var_net_normal_forces_history, var_0, var_38, var_1, var_42);
                goto start_for_7;
            end_for_7:;
            // net_normal_forces_history[env, 0, sensor] = net_normal_forces[env, sensor]         <L 215>
            var_43 = wp::address(var_net_normal_forces, var_0, var_1);
            var_45 = wp::load(var_43);
            wp::array_store(var_net_normal_forces_history, var_0, var_44, var_1, var_45);
        }
        var_46 = wp::where(var_net_normal_forces_history, var_38, var_10);
        // if normal_force_matrix_history:                                                        <L 217>
        if (var_normal_force_matrix_history) {
            // for f in range(num_filter_objects):                                                <L 218>
            var_47 = wp::range(var_num_filter_objects);
            start_for_9:;
                if (iter_cmp(var_47) == 0) goto end_for_9;
                var_48 = wp::iter_next(var_47);
                // for i in range(history_length - 1, 0, -1):                                     <L 219>
                var_50 = wp::sub(var_history_length, var_49);
                var_53 = wp::range(var_50, var_51, var_52);
                start_for_11:;
                    if (iter_cmp(var_53) == 0) goto end_for_11;
                    var_54 = wp::iter_next(var_53);
                    // normal_force_matrix_history[env, i, sensor, f] = normal_force_matrix_history[env, i - 1, sensor, f]       <L 220>
                    var_56 = wp::sub(var_54, var_55);
                    var_57 = wp::address(var_normal_force_matrix_history, var_0, var_56, var_1, var_48);
                    var_58 = wp::load(var_57);
                    wp::array_store(var_normal_force_matrix_history, var_0, var_54, var_1, var_48, var_58);
                    goto start_for_11;
                end_for_11:;
                // normal_force_matrix_history[env, 0, sensor, f] = normal_force_matrix[env, sensor, f]       <L 221>
                var_59 = wp::address(var_normal_force_matrix, var_0, var_1, var_48);
                var_61 = wp::load(var_59);
                wp::array_store(var_normal_force_matrix_history, var_0, var_60, var_1, var_48, var_61);
                wp::assign(var_46, var_54);
                goto start_for_9;
            end_for_9:;
        }
        var_62 = wp::where(var_normal_force_matrix_history, var_48, var_19);
        // if net_friction_forces_history:                                                        <L 224>
        if (var_net_friction_forces_history) {
            // for i in range(history_length - 1, 0, -1):                                         <L 225>
            var_64 = wp::sub(var_history_length, var_63);
            var_67 = wp::range(var_64, var_65, var_66);
            start_for_13:;
                if (iter_cmp(var_67) == 0) goto end_for_13;
                var_68 = wp::iter_next(var_67);
                // net_friction_forces_history[env, i, sensor] = net_friction_forces_history[env, i - 1, sensor]       <L 226>
                var_70 = wp::sub(var_68, var_69);
                var_71 = wp::address(var_net_friction_forces_history, var_0, var_70, var_1);
                var_72 = wp::load(var_71);
                wp::array_store(var_net_friction_forces_history, var_0, var_68, var_1, var_72);
                goto start_for_13;
            end_for_13:;
            // net_friction_forces_history[env, 0, sensor] = net_friction_forces[env, sensor]       <L 227>
            var_73 = wp::address(var_net_friction_forces, var_0, var_1);
            var_75 = wp::load(var_73);
            wp::array_store(var_net_friction_forces_history, var_0, var_74, var_1, var_75);
        }
        var_76 = wp::where(var_net_friction_forces_history, var_68, var_46);
        // if friction_force_matrix_history:                                                      <L 229>
        if (var_friction_force_matrix_history) {
            // for f in range(num_filter_objects):                                                <L 230>
            var_77 = wp::range(var_num_filter_objects);
            start_for_15:;
                if (iter_cmp(var_77) == 0) goto end_for_15;
                var_78 = wp::iter_next(var_77);
                // for i in range(history_length - 1, 0, -1):                                     <L 231>
                var_80 = wp::sub(var_history_length, var_79);
                var_83 = wp::range(var_80, var_81, var_82);
                start_for_17:;
                    if (iter_cmp(var_83) == 0) goto end_for_17;
                    var_84 = wp::iter_next(var_83);
                    // friction_force_matrix_history[env, i, sensor, f] = friction_force_matrix_history[env, i - 1, sensor, f]       <L 232>
                    var_86 = wp::sub(var_84, var_85);
                    var_87 = wp::address(var_friction_force_matrix_history, var_0, var_86, var_1, var_78);
                    var_88 = wp::load(var_87);
                    wp::array_store(var_friction_force_matrix_history, var_0, var_84, var_1, var_78, var_88);
                    goto start_for_17;
                end_for_17:;
                // friction_force_matrix_history[env, 0, sensor, f] = friction_force_matrix[env, sensor, f]       <L 233>
                var_89 = wp::address(var_friction_force_matrix, var_0, var_1, var_78);
                var_91 = wp::load(var_89);
                wp::array_store(var_friction_force_matrix_history, var_0, var_90, var_1, var_78, var_91);
                wp::assign(var_76, var_84);
                goto start_for_15;
            end_for_15:;
        }
        var_92 = wp::where(var_friction_force_matrix_history, var_78, var_62);
        // if current_air_time:                                                                   <L 236>
        if (var_current_air_time) {
            // elapsed_time = timestamp[env] - timestamp_last_update[env]                         <L 237>
            var_93 = wp::address(var_timestamp, var_0);
            var_94 = wp::address(var_timestamp_last_update, var_0);
            var_96 = wp::load(var_93);
            var_97 = wp::load(var_94);
            var_95 = wp::sub(var_96, var_97);
            // in_contact = wp.length_sq(net_normal_forces[env, sensor]) > contact_force_threshold * contact_force_threshold       <L 238>
            var_98 = wp::address(var_net_normal_forces, var_0, var_1);
            var_100 = wp::load(var_98);
            var_99 = wp::length_sq(var_100);
            var_101 = wp::mul(var_contact_force_threshold, var_contact_force_threshold);
            var_102 = (var_99 > var_101);
            // cat = current_air_time[env, sensor]                                                <L 240>
            var_103 = wp::address(var_current_air_time, var_0, var_1);
            var_105 = wp::load(var_103);
            var_104 = wp::copy(var_105);
            // cct = current_contact_time[env, sensor]                                            <L 241>
            var_106 = wp::address(var_current_contact_time, var_0, var_1);
            var_108 = wp::load(var_106);
            var_107 = wp::copy(var_108);
            // is_first_contact = in_contact and (cat > 0.0)                                      <L 242>
            var_109 = var_102;
            if (var_109) {
                var_111 = (var_104 > var_110);
                var_109 = var_109 && var_111;
            }
            // is_first_detached = not in_contact and (cct > 0.0)                                 <L 243>
            var_113 = wp::unot(var_102);
            var_112 = var_113;
            if (var_112) {
                var_115 = (var_107 > var_114);
                var_112 = var_112 && var_115;
            }
            // if is_first_contact:                                                               <L 245>
            if (var_109) {
                // last_air_time[env, sensor] = cat + elapsed_time                                <L 246>
                var_116 = wp::add(var_104, var_95);
                wp::array_store(var_last_air_time, var_0, var_1, var_116);
            }
            if (!var_109) {
                // elif is_first_detached:                                                        <L 247>
                if (var_112) {
                    // last_contact_time[env, sensor] = cct + elapsed_time                        <L 248>
                    var_117 = wp::add(var_107, var_95);
                    wp::array_store(var_last_contact_time, var_0, var_1, var_117);
                }
            }
            // current_contact_time[env, sensor] = wp.where(in_contact, cct + elapsed_time, 0.0)       <L 250>
            var_118 = wp::add(var_107, var_95);
            var_120 = wp::where(var_102, var_118, var_119);
            wp::array_store(var_current_contact_time, var_0, var_1, var_120);
            // current_air_time[env, sensor] = wp.where(in_contact, 0.0, cat + elapsed_time)       <L 251>
            var_122 = wp::add(var_104, var_95);
            var_123 = wp::where(var_102, var_121, var_122);
            wp::array_store(var_current_air_time, var_0, var_1, var_123);
        }
    }
}



extern "C" __global__ void update_contact_sensor_kernel_7e4412e0_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_history_length,
    wp::int32 var_num_filter_objects,
    wp::float32 var_contact_force_threshold,
    wp::array_t<bool> var_env_mask,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> var_force_matrix,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_normal_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> var_normal_force_matrix,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_friction_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> var_friction_force_matrix,
    wp::array_t<wp::float32> var_timestamp,
    wp::array_t<wp::float32> var_timestamp_last_update,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_forces_history,
    wp::array_t<wp::vec_t<3, wp::float32>> var_force_matrix_history,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_normal_forces_history,
    wp::array_t<wp::vec_t<3, wp::float32>> var_normal_force_matrix_history,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_friction_forces_history,
    wp::array_t<wp::vec_t<3, wp::float32>> var_friction_force_matrix_history,
    wp::array_t<wp::float32> var_current_air_time,
    wp::array_t<wp::float32> var_current_contact_time,
    wp::array_t<wp::float32> var_last_air_time,
    wp::array_t<wp::float32> var_last_contact_time,
    wp::int32 adj_history_length,
    wp::int32 adj_num_filter_objects,
    wp::float32 adj_contact_force_threshold,
    wp::array_t<bool> adj_env_mask,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_net_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_force_matrix,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_net_normal_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_normal_force_matrix,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_net_friction_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_friction_force_matrix,
    wp::array_t<wp::float32> adj_timestamp,
    wp::array_t<wp::float32> adj_timestamp_last_update,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_net_forces_history,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_force_matrix_history,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_net_normal_forces_history,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_normal_force_matrix_history,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_net_friction_forces_history,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_friction_force_matrix_history,
    wp::array_t<wp::float32> adj_current_air_time,
    wp::array_t<wp::float32> adj_current_contact_time,
    wp::array_t<wp::float32> adj_last_air_time,
    wp::array_t<wp::float32> adj_last_contact_time)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
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
        bool var_4;
        const wp::int32 var_5 = 1;
        wp::int32 var_6;
        const wp::int32 var_7 = 0;
        const wp::int32 var_8 = -1;
        wp::range_t var_9;
        wp::int32 var_10;
        const wp::int32 var_11 = 1;
        wp::int32 var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32>* var_15;
        const wp::int32 var_16 = 0;
        wp::vec_t<3, wp::float32> var_17;
        wp::range_t var_18;
        wp::int32 var_19;
        const wp::int32 var_20 = 1;
        wp::int32 var_21;
        const wp::int32 var_22 = 0;
        const wp::int32 var_23 = -1;
        wp::range_t var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 1;
        wp::int32 var_27;
        wp::vec_t<3, wp::float32>* var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32>* var_30;
        const wp::int32 var_31 = 0;
        wp::vec_t<3, wp::float32> var_32;
        const wp::int32 var_33 = 1;
        wp::int32 var_34;
        const wp::int32 var_35 = 0;
        const wp::int32 var_36 = -1;
        wp::range_t var_37;
        wp::int32 var_38;
        const wp::int32 var_39 = 1;
        wp::int32 var_40;
        wp::vec_t<3, wp::float32>* var_41;
        wp::vec_t<3, wp::float32> var_42;
        wp::vec_t<3, wp::float32>* var_43;
        const wp::int32 var_44 = 0;
        wp::vec_t<3, wp::float32> var_45;
        wp::int32 var_46;
        wp::range_t var_47;
        wp::int32 var_48;
        const wp::int32 var_49 = 1;
        wp::int32 var_50;
        const wp::int32 var_51 = 0;
        const wp::int32 var_52 = -1;
        wp::range_t var_53;
        wp::int32 var_54;
        const wp::int32 var_55 = 1;
        wp::int32 var_56;
        wp::vec_t<3, wp::float32>* var_57;
        wp::vec_t<3, wp::float32> var_58;
        wp::vec_t<3, wp::float32>* var_59;
        const wp::int32 var_60 = 0;
        wp::vec_t<3, wp::float32> var_61;
        wp::int32 var_62;
        const wp::int32 var_63 = 1;
        wp::int32 var_64;
        const wp::int32 var_65 = 0;
        const wp::int32 var_66 = -1;
        wp::range_t var_67;
        wp::int32 var_68;
        const wp::int32 var_69 = 1;
        wp::int32 var_70;
        wp::vec_t<3, wp::float32>* var_71;
        wp::vec_t<3, wp::float32> var_72;
        wp::vec_t<3, wp::float32>* var_73;
        const wp::int32 var_74 = 0;
        wp::vec_t<3, wp::float32> var_75;
        wp::int32 var_76;
        wp::range_t var_77;
        wp::int32 var_78;
        const wp::int32 var_79 = 1;
        wp::int32 var_80;
        const wp::int32 var_81 = 0;
        const wp::int32 var_82 = -1;
        wp::range_t var_83;
        wp::int32 var_84;
        const wp::int32 var_85 = 1;
        wp::int32 var_86;
        wp::vec_t<3, wp::float32>* var_87;
        wp::vec_t<3, wp::float32> var_88;
        wp::vec_t<3, wp::float32>* var_89;
        const wp::int32 var_90 = 0;
        wp::vec_t<3, wp::float32> var_91;
        wp::int32 var_92;
        wp::float32* var_93;
        wp::float32* var_94;
        wp::float32 var_95;
        wp::float32 var_96;
        wp::float32 var_97;
        wp::vec_t<3, wp::float32>* var_98;
        wp::float32 var_99;
        wp::vec_t<3, wp::float32> var_100;
        wp::float32 var_101;
        bool var_102;
        wp::float32* var_103;
        wp::float32 var_104;
        wp::float32 var_105;
        wp::float32* var_106;
        wp::float32 var_107;
        wp::float32 var_108;
        bool var_109;
        const wp::float32 var_110 = 0.0;
        bool var_111;
        bool var_112;
        bool var_113;
        const wp::float32 var_114 = 0.0;
        bool var_115;
        wp::float32 var_116;
        wp::float32 var_117;
        wp::float32 var_118;
        const wp::float32 var_119 = 0.0;
        wp::float32 var_120;
        const wp::float32 var_121 = 0.0;
        wp::float32 var_122;
        wp::float32 var_123;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::range_t adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::vec_t<3, wp::float32> adj_13 = {};
        wp::vec_t<3, wp::float32> adj_14 = {};
        wp::vec_t<3, wp::float32> adj_15 = {};
        wp::int32 adj_16 = {};
        wp::vec_t<3, wp::float32> adj_17 = {};
        wp::range_t adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::range_t adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::vec_t<3, wp::float32> adj_28 = {};
        wp::vec_t<3, wp::float32> adj_29 = {};
        wp::vec_t<3, wp::float32> adj_30 = {};
        wp::int32 adj_31 = {};
        wp::vec_t<3, wp::float32> adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::range_t adj_37 = {};
        wp::int32 adj_38 = {};
        wp::int32 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::vec_t<3, wp::float32> adj_41 = {};
        wp::vec_t<3, wp::float32> adj_42 = {};
        wp::vec_t<3, wp::float32> adj_43 = {};
        wp::int32 adj_44 = {};
        wp::vec_t<3, wp::float32> adj_45 = {};
        wp::int32 adj_46 = {};
        wp::range_t adj_47 = {};
        wp::int32 adj_48 = {};
        wp::int32 adj_49 = {};
        wp::int32 adj_50 = {};
        wp::int32 adj_51 = {};
        wp::int32 adj_52 = {};
        wp::range_t adj_53 = {};
        wp::int32 adj_54 = {};
        wp::int32 adj_55 = {};
        wp::int32 adj_56 = {};
        wp::vec_t<3, wp::float32> adj_57 = {};
        wp::vec_t<3, wp::float32> adj_58 = {};
        wp::vec_t<3, wp::float32> adj_59 = {};
        wp::int32 adj_60 = {};
        wp::vec_t<3, wp::float32> adj_61 = {};
        wp::int32 adj_62 = {};
        wp::int32 adj_63 = {};
        wp::int32 adj_64 = {};
        wp::int32 adj_65 = {};
        wp::int32 adj_66 = {};
        wp::range_t adj_67 = {};
        wp::int32 adj_68 = {};
        wp::int32 adj_69 = {};
        wp::int32 adj_70 = {};
        wp::vec_t<3, wp::float32> adj_71 = {};
        wp::vec_t<3, wp::float32> adj_72 = {};
        wp::vec_t<3, wp::float32> adj_73 = {};
        wp::int32 adj_74 = {};
        wp::vec_t<3, wp::float32> adj_75 = {};
        wp::int32 adj_76 = {};
        wp::range_t adj_77 = {};
        wp::int32 adj_78 = {};
        wp::int32 adj_79 = {};
        wp::int32 adj_80 = {};
        wp::int32 adj_81 = {};
        wp::int32 adj_82 = {};
        wp::range_t adj_83 = {};
        wp::int32 adj_84 = {};
        wp::int32 adj_85 = {};
        wp::int32 adj_86 = {};
        wp::vec_t<3, wp::float32> adj_87 = {};
        wp::vec_t<3, wp::float32> adj_88 = {};
        wp::vec_t<3, wp::float32> adj_89 = {};
        wp::int32 adj_90 = {};
        wp::vec_t<3, wp::float32> adj_91 = {};
        wp::int32 adj_92 = {};
        wp::float32 adj_93 = {};
        wp::float32 adj_94 = {};
        wp::float32 adj_95 = {};
        wp::float32 adj_96 = {};
        wp::float32 adj_97 = {};
        wp::vec_t<3, wp::float32> adj_98 = {};
        wp::float32 adj_99 = {};
        wp::vec_t<3, wp::float32> adj_100 = {};
        wp::float32 adj_101 = {};
        bool adj_102 = {};
        wp::float32 adj_103 = {};
        wp::float32 adj_104 = {};
        wp::float32 adj_105 = {};
        wp::float32 adj_106 = {};
        wp::float32 adj_107 = {};
        wp::float32 adj_108 = {};
        bool adj_109 = {};
        wp::float32 adj_110 = {};
        bool adj_111 = {};
        bool adj_112 = {};
        bool adj_113 = {};
        wp::float32 adj_114 = {};
        bool adj_115 = {};
        wp::float32 adj_116 = {};
        wp::float32 adj_117 = {};
        wp::float32 adj_118 = {};
        wp::float32 adj_119 = {};
        wp::float32 adj_120 = {};
        wp::float32 adj_121 = {};
        wp::float32 adj_122 = {};
        wp::float32 adj_123 = {};
        //---------
        // forward
        // def update_contact_sensor_kernel(                                                      <L 162>
        // env, sensor = wp.tid()                                                                 <L 193>
        builtin_tid2d(var_0, var_1);
        // if env_mask:                                                                           <L 195>
        if (var_env_mask) {
            // if not env_mask[env]:                                                              <L 196>
            var_2 = wp::address(var_env_mask, var_0);
            var_4 = wp::load(var_2);
            var_3 = wp::unot(var_4);
            if (var_3) {
                // return                                                                         <L 197>
                goto label0;
            }
        }
        // if net_forces_history:                                                                 <L 200>
        if (var_net_forces_history) {
            // for i in range(history_length - 1, 0, -1):                                         <L 201>
            var_6 = wp::sub(var_history_length, var_5);
            var_9 = wp::range(var_6, var_7, var_8);
            // net_forces_history[env, 0, sensor] = net_forces[env, sensor]                       <L 203>
            var_15 = wp::address(var_net_forces, var_0, var_1);
            var_17 = wp::load(var_15);
            // wp::array_store(var_net_forces_history, var_0, var_16, var_1, var_17);
        }
        // if force_matrix_history:                                                               <L 205>
        if (var_force_matrix_history) {
            // for f in range(num_filter_objects):                                                <L 206>
            var_18 = wp::range(var_num_filter_objects);
        }
        // if net_normal_forces_history:                                                          <L 212>
        if (var_net_normal_forces_history) {
            // for i in range(history_length - 1, 0, -1):                                         <L 213>
            var_34 = wp::sub(var_history_length, var_33);
            var_37 = wp::range(var_34, var_35, var_36);
            // net_normal_forces_history[env, 0, sensor] = net_normal_forces[env, sensor]         <L 215>
            var_43 = wp::address(var_net_normal_forces, var_0, var_1);
            var_45 = wp::load(var_43);
            // wp::array_store(var_net_normal_forces_history, var_0, var_44, var_1, var_45);
        }
        var_46 = wp::where(var_net_normal_forces_history, var_38, var_10);
        // if normal_force_matrix_history:                                                        <L 217>
        if (var_normal_force_matrix_history) {
            // for f in range(num_filter_objects):                                                <L 218>
            var_47 = wp::range(var_num_filter_objects);
        }
        var_62 = wp::where(var_normal_force_matrix_history, var_48, var_19);
        // if net_friction_forces_history:                                                        <L 224>
        if (var_net_friction_forces_history) {
            // for i in range(history_length - 1, 0, -1):                                         <L 225>
            var_64 = wp::sub(var_history_length, var_63);
            var_67 = wp::range(var_64, var_65, var_66);
            // net_friction_forces_history[env, 0, sensor] = net_friction_forces[env, sensor]       <L 227>
            var_73 = wp::address(var_net_friction_forces, var_0, var_1);
            var_75 = wp::load(var_73);
            // wp::array_store(var_net_friction_forces_history, var_0, var_74, var_1, var_75);
        }
        var_76 = wp::where(var_net_friction_forces_history, var_68, var_46);
        // if friction_force_matrix_history:                                                      <L 229>
        if (var_friction_force_matrix_history) {
            // for f in range(num_filter_objects):                                                <L 230>
            var_77 = wp::range(var_num_filter_objects);
        }
        var_92 = wp::where(var_friction_force_matrix_history, var_78, var_62);
        // if current_air_time:                                                                   <L 236>
        if (var_current_air_time) {
            // elapsed_time = timestamp[env] - timestamp_last_update[env]                         <L 237>
            var_93 = wp::address(var_timestamp, var_0);
            var_94 = wp::address(var_timestamp_last_update, var_0);
            var_96 = wp::load(var_93);
            var_97 = wp::load(var_94);
            var_95 = wp::sub(var_96, var_97);
            // in_contact = wp.length_sq(net_normal_forces[env, sensor]) > contact_force_threshold * contact_force_threshold       <L 238>
            var_98 = wp::address(var_net_normal_forces, var_0, var_1);
            var_100 = wp::load(var_98);
            var_99 = wp::length_sq(var_100);
            var_101 = wp::mul(var_contact_force_threshold, var_contact_force_threshold);
            var_102 = (var_99 > var_101);
            // cat = current_air_time[env, sensor]                                                <L 240>
            var_103 = wp::address(var_current_air_time, var_0, var_1);
            var_105 = wp::load(var_103);
            var_104 = wp::copy(var_105);
            // cct = current_contact_time[env, sensor]                                            <L 241>
            var_106 = wp::address(var_current_contact_time, var_0, var_1);
            var_108 = wp::load(var_106);
            var_107 = wp::copy(var_108);
            // is_first_contact = in_contact and (cat > 0.0)                                      <L 242>
            var_109 = var_102;
            if (var_109) {
                var_111 = (var_104 > var_110);
                var_109 = var_109 && var_111;
            }
            // is_first_detached = not in_contact and (cct > 0.0)                                 <L 243>
            var_113 = wp::unot(var_102);
            var_112 = var_113;
            if (var_112) {
                var_115 = (var_107 > var_114);
                var_112 = var_112 && var_115;
            }
            // if is_first_contact:                                                               <L 245>
            if (var_109) {
                // last_air_time[env, sensor] = cat + elapsed_time                                <L 246>
                var_116 = wp::add(var_104, var_95);
                // wp::array_store(var_last_air_time, var_0, var_1, var_116);
            }
            if (!var_109) {
                // elif is_first_detached:                                                        <L 247>
                if (var_112) {
                    // last_contact_time[env, sensor] = cct + elapsed_time                        <L 248>
                    var_117 = wp::add(var_107, var_95);
                    // wp::array_store(var_last_contact_time, var_0, var_1, var_117);
                }
            }
            // current_contact_time[env, sensor] = wp.where(in_contact, cct + elapsed_time, 0.0)       <L 250>
            var_118 = wp::add(var_107, var_95);
            var_120 = wp::where(var_102, var_118, var_119);
            // wp::array_store(var_current_contact_time, var_0, var_1, var_120);
            // current_air_time[env, sensor] = wp.where(in_contact, 0.0, cat + elapsed_time)       <L 251>
            var_122 = wp::add(var_104, var_95);
            var_123 = wp::where(var_102, var_121, var_122);
            // wp::array_store(var_current_air_time, var_0, var_1, var_123);
        }
        //---------
        // reverse
        if (var_current_air_time) {
            wp::adj_array_store(var_current_air_time, var_0, var_1, var_123, adj_current_air_time, adj_0, adj_1, adj_123);
            wp::adj_where(var_102, var_121, var_122, adj_102, adj_121, adj_122, adj_123);
            wp::adj_add(var_104, var_95, adj_104, adj_95, adj_122);
            // adj: current_air_time[env, sensor] = wp.where(in_contact, 0.0, cat + elapsed_time)  <L 251>
            wp::adj_array_store(var_current_contact_time, var_0, var_1, var_120, adj_current_contact_time, adj_0, adj_1, adj_120);
            wp::adj_where(var_102, var_118, var_119, adj_102, adj_118, adj_119, adj_120);
            wp::adj_add(var_107, var_95, adj_107, adj_95, adj_118);
            // adj: current_contact_time[env, sensor] = wp.where(in_contact, cct + elapsed_time, 0.0)  <L 250>
            if (!var_109) {
                if (var_112) {
                    wp::adj_array_store(var_last_contact_time, var_0, var_1, var_117, adj_last_contact_time, adj_0, adj_1, adj_117);
                    wp::adj_add(var_107, var_95, adj_107, adj_95, adj_117);
                    // adj: last_contact_time[env, sensor] = cct + elapsed_time                   <L 248>
                }
                // adj: elif is_first_detached:                                                   <L 247>
            }
            if (var_109) {
                wp::adj_array_store(var_last_air_time, var_0, var_1, var_116, adj_last_air_time, adj_0, adj_1, adj_116);
                wp::adj_add(var_104, var_95, adj_104, adj_95, adj_116);
                // adj: last_air_time[env, sensor] = cat + elapsed_time                           <L 246>
            }
            // adj: if is_first_contact:                                                          <L 245>
            if (var_112) {
            }
            // adj: is_first_detached = not in_contact and (cct > 0.0)                            <L 243>
            if (var_109) {
            }
            // adj: is_first_contact = in_contact and (cat > 0.0)                                 <L 242>
            wp::adj_copy(var_108, adj_106, adj_107);
            wp::adj_address(var_current_contact_time, var_0, var_1, adj_current_contact_time, adj_0, adj_1, adj_106);
            // adj: cct = current_contact_time[env, sensor]                                       <L 241>
            wp::adj_copy(var_105, adj_103, adj_104);
            wp::adj_address(var_current_air_time, var_0, var_1, adj_current_air_time, adj_0, adj_1, adj_103);
            // adj: cat = current_air_time[env, sensor]                                           <L 240>
            wp::adj_mul(var_contact_force_threshold, var_contact_force_threshold, adj_contact_force_threshold, adj_contact_force_threshold, adj_101);
            wp::adj_length_sq(var_100, adj_98, adj_99);
            wp::adj_address(var_net_normal_forces, var_0, var_1, adj_net_normal_forces, adj_0, adj_1, adj_98);
            // adj: in_contact = wp.length_sq(net_normal_forces[env, sensor]) > contact_force_threshold * contact_force_threshold  <L 238>
            wp::adj_sub(var_96, var_97, adj_93, adj_94, adj_95);
            wp::adj_address(var_timestamp_last_update, var_0, adj_timestamp_last_update, adj_0, adj_94);
            wp::adj_address(var_timestamp, var_0, adj_timestamp, adj_0, adj_93);
            // adj: elapsed_time = timestamp[env] - timestamp_last_update[env]                    <L 237>
        }
        // adj: if current_air_time:                                                              <L 236>
        wp::adj_where(var_friction_force_matrix_history, var_78, var_62, adj_friction_force_matrix_history, adj_78, adj_62, adj_92);
        if (var_friction_force_matrix_history) {
            var_77 = wp::iter_reverse(var_77);
            start_for_15:;
                if (iter_cmp(var_77) == 0) goto end_for_15;
                var_78 = wp::iter_next(var_77);
            	adj_79 = {};
            	adj_80 = {};
            	adj_81 = {};
            	adj_82 = {};
            	adj_83 = {};
            	adj_89 = {};
            	adj_90 = {};
            	adj_91 = {};
                // for i in range(history_length - 1, 0, -1):                                     <L 231>
                var_80 = wp::sub(var_history_length, var_79);
                var_83 = wp::range(var_80, var_81, var_82);
                // friction_force_matrix_history[env, 0, sensor, f] = friction_force_matrix[env, sensor, f]       <L 233>
                var_89 = wp::address(var_friction_force_matrix, var_0, var_1, var_78);
                var_91 = wp::load(var_89);
                // wp::array_store(var_friction_force_matrix_history, var_0, var_90, var_1, var_78, var_91);
                wp::assign(var_76, var_84);
                wp::adj_assign(var_76, var_84, adj_76, adj_84);
                wp::adj_array_store(var_friction_force_matrix_history, var_0, var_90, var_1, var_78, var_91, adj_friction_force_matrix_history, adj_0, adj_90, adj_1, adj_78, adj_89);
                wp::adj_address(var_friction_force_matrix, var_0, var_1, var_78, adj_friction_force_matrix, adj_0, adj_1, adj_78, adj_89);
                // adj: friction_force_matrix_history[env, 0, sensor, f] = friction_force_matrix[env, sensor, f]  <L 233>
                var_83 = wp::iter_reverse(var_83);
                start_for_17:;
                    if (iter_cmp(var_83) == 0) goto end_for_17;
                    var_84 = wp::iter_next(var_83);
                	adj_85 = {};
                	adj_86 = {};
                	adj_87 = {};
                	adj_88 = {};
                    // friction_force_matrix_history[env, i, sensor, f] = friction_force_matrix_history[env, i - 1, sensor, f]       <L 232>
                    var_86 = wp::sub(var_84, var_85);
                    var_87 = wp::address(var_friction_force_matrix_history, var_0, var_86, var_1, var_78);
                    var_88 = wp::load(var_87);
                    // wp::array_store(var_friction_force_matrix_history, var_0, var_84, var_1, var_78, var_88);
                    wp::adj_array_store(var_friction_force_matrix_history, var_0, var_84, var_1, var_78, var_88, adj_friction_force_matrix_history, adj_0, adj_84, adj_1, adj_78, adj_87);
                    wp::adj_address(var_friction_force_matrix_history, var_0, var_86, var_1, var_78, adj_friction_force_matrix_history, adj_0, adj_86, adj_1, adj_78, adj_87);
                    wp::adj_sub(var_84, var_85, adj_84, adj_85, adj_86);
                    // adj: friction_force_matrix_history[env, i, sensor, f] = friction_force_matrix_history[env, i - 1, sensor, f]  <L 232>
                	goto start_for_17;
                end_for_17:;
                wp::adj_sub(var_history_length, var_79, adj_history_length, adj_79, adj_80);
                // adj: for i in range(history_length - 1, 0, -1):                                <L 231>
            	goto start_for_15;
            end_for_15:;
            // adj: for f in range(num_filter_objects):                                           <L 230>
        }
        // adj: if friction_force_matrix_history:                                                 <L 229>
        wp::adj_where(var_net_friction_forces_history, var_68, var_46, adj_net_friction_forces_history, adj_68, adj_46, adj_76);
        if (var_net_friction_forces_history) {
            wp::adj_array_store(var_net_friction_forces_history, var_0, var_74, var_1, var_75, adj_net_friction_forces_history, adj_0, adj_74, adj_1, adj_73);
            wp::adj_address(var_net_friction_forces, var_0, var_1, adj_net_friction_forces, adj_0, adj_1, adj_73);
            // adj: net_friction_forces_history[env, 0, sensor] = net_friction_forces[env, sensor]  <L 227>
            var_67 = wp::iter_reverse(var_67);
            start_for_13:;
                if (iter_cmp(var_67) == 0) goto end_for_13;
                var_68 = wp::iter_next(var_67);
            	adj_69 = {};
            	adj_70 = {};
            	adj_71 = {};
            	adj_72 = {};
                // net_friction_forces_history[env, i, sensor] = net_friction_forces_history[env, i - 1, sensor]       <L 226>
                var_70 = wp::sub(var_68, var_69);
                var_71 = wp::address(var_net_friction_forces_history, var_0, var_70, var_1);
                var_72 = wp::load(var_71);
                // wp::array_store(var_net_friction_forces_history, var_0, var_68, var_1, var_72);
                wp::adj_array_store(var_net_friction_forces_history, var_0, var_68, var_1, var_72, adj_net_friction_forces_history, adj_0, adj_68, adj_1, adj_71);
                wp::adj_address(var_net_friction_forces_history, var_0, var_70, var_1, adj_net_friction_forces_history, adj_0, adj_70, adj_1, adj_71);
                wp::adj_sub(var_68, var_69, adj_68, adj_69, adj_70);
                // adj: net_friction_forces_history[env, i, sensor] = net_friction_forces_history[env, i - 1, sensor]  <L 226>
            	goto start_for_13;
            end_for_13:;
            wp::adj_sub(var_history_length, var_63, adj_history_length, adj_63, adj_64);
            // adj: for i in range(history_length - 1, 0, -1):                                    <L 225>
        }
        // adj: if net_friction_forces_history:                                                   <L 224>
        wp::adj_where(var_normal_force_matrix_history, var_48, var_19, adj_normal_force_matrix_history, adj_48, adj_19, adj_62);
        if (var_normal_force_matrix_history) {
            var_47 = wp::iter_reverse(var_47);
            start_for_9:;
                if (iter_cmp(var_47) == 0) goto end_for_9;
                var_48 = wp::iter_next(var_47);
            	adj_49 = {};
            	adj_50 = {};
            	adj_51 = {};
            	adj_52 = {};
            	adj_53 = {};
            	adj_59 = {};
            	adj_60 = {};
            	adj_61 = {};
                // for i in range(history_length - 1, 0, -1):                                     <L 219>
                var_50 = wp::sub(var_history_length, var_49);
                var_53 = wp::range(var_50, var_51, var_52);
                // normal_force_matrix_history[env, 0, sensor, f] = normal_force_matrix[env, sensor, f]       <L 221>
                var_59 = wp::address(var_normal_force_matrix, var_0, var_1, var_48);
                var_61 = wp::load(var_59);
                // wp::array_store(var_normal_force_matrix_history, var_0, var_60, var_1, var_48, var_61);
                wp::assign(var_46, var_54);
                wp::adj_assign(var_46, var_54, adj_46, adj_54);
                wp::adj_array_store(var_normal_force_matrix_history, var_0, var_60, var_1, var_48, var_61, adj_normal_force_matrix_history, adj_0, adj_60, adj_1, adj_48, adj_59);
                wp::adj_address(var_normal_force_matrix, var_0, var_1, var_48, adj_normal_force_matrix, adj_0, adj_1, adj_48, adj_59);
                // adj: normal_force_matrix_history[env, 0, sensor, f] = normal_force_matrix[env, sensor, f]  <L 221>
                var_53 = wp::iter_reverse(var_53);
                start_for_11:;
                    if (iter_cmp(var_53) == 0) goto end_for_11;
                    var_54 = wp::iter_next(var_53);
                	adj_55 = {};
                	adj_56 = {};
                	adj_57 = {};
                	adj_58 = {};
                    // normal_force_matrix_history[env, i, sensor, f] = normal_force_matrix_history[env, i - 1, sensor, f]       <L 220>
                    var_56 = wp::sub(var_54, var_55);
                    var_57 = wp::address(var_normal_force_matrix_history, var_0, var_56, var_1, var_48);
                    var_58 = wp::load(var_57);
                    // wp::array_store(var_normal_force_matrix_history, var_0, var_54, var_1, var_48, var_58);
                    wp::adj_array_store(var_normal_force_matrix_history, var_0, var_54, var_1, var_48, var_58, adj_normal_force_matrix_history, adj_0, adj_54, adj_1, adj_48, adj_57);
                    wp::adj_address(var_normal_force_matrix_history, var_0, var_56, var_1, var_48, adj_normal_force_matrix_history, adj_0, adj_56, adj_1, adj_48, adj_57);
                    wp::adj_sub(var_54, var_55, adj_54, adj_55, adj_56);
                    // adj: normal_force_matrix_history[env, i, sensor, f] = normal_force_matrix_history[env, i - 1, sensor, f]  <L 220>
                	goto start_for_11;
                end_for_11:;
                wp::adj_sub(var_history_length, var_49, adj_history_length, adj_49, adj_50);
                // adj: for i in range(history_length - 1, 0, -1):                                <L 219>
            	goto start_for_9;
            end_for_9:;
            // adj: for f in range(num_filter_objects):                                           <L 218>
        }
        // adj: if normal_force_matrix_history:                                                   <L 217>
        wp::adj_where(var_net_normal_forces_history, var_38, var_10, adj_net_normal_forces_history, adj_38, adj_10, adj_46);
        if (var_net_normal_forces_history) {
            wp::adj_array_store(var_net_normal_forces_history, var_0, var_44, var_1, var_45, adj_net_normal_forces_history, adj_0, adj_44, adj_1, adj_43);
            wp::adj_address(var_net_normal_forces, var_0, var_1, adj_net_normal_forces, adj_0, adj_1, adj_43);
            // adj: net_normal_forces_history[env, 0, sensor] = net_normal_forces[env, sensor]    <L 215>
            var_37 = wp::iter_reverse(var_37);
            start_for_7:;
                if (iter_cmp(var_37) == 0) goto end_for_7;
                var_38 = wp::iter_next(var_37);
            	adj_39 = {};
            	adj_40 = {};
            	adj_41 = {};
            	adj_42 = {};
                // net_normal_forces_history[env, i, sensor] = net_normal_forces_history[env, i - 1, sensor]       <L 214>
                var_40 = wp::sub(var_38, var_39);
                var_41 = wp::address(var_net_normal_forces_history, var_0, var_40, var_1);
                var_42 = wp::load(var_41);
                // wp::array_store(var_net_normal_forces_history, var_0, var_38, var_1, var_42);
                wp::adj_array_store(var_net_normal_forces_history, var_0, var_38, var_1, var_42, adj_net_normal_forces_history, adj_0, adj_38, adj_1, adj_41);
                wp::adj_address(var_net_normal_forces_history, var_0, var_40, var_1, adj_net_normal_forces_history, adj_0, adj_40, adj_1, adj_41);
                wp::adj_sub(var_38, var_39, adj_38, adj_39, adj_40);
                // adj: net_normal_forces_history[env, i, sensor] = net_normal_forces_history[env, i - 1, sensor]  <L 214>
            	goto start_for_7;
            end_for_7:;
            wp::adj_sub(var_history_length, var_33, adj_history_length, adj_33, adj_34);
            // adj: for i in range(history_length - 1, 0, -1):                                    <L 213>
        }
        // adj: if net_normal_forces_history:                                                     <L 212>
        if (var_force_matrix_history) {
            var_18 = wp::iter_reverse(var_18);
            start_for_3:;
                if (iter_cmp(var_18) == 0) goto end_for_3;
                var_19 = wp::iter_next(var_18);
            	adj_20 = {};
            	adj_21 = {};
            	adj_22 = {};
            	adj_23 = {};
            	adj_24 = {};
            	adj_30 = {};
            	adj_31 = {};
            	adj_32 = {};
                // for i in range(history_length - 1, 0, -1):                                     <L 207>
                var_21 = wp::sub(var_history_length, var_20);
                var_24 = wp::range(var_21, var_22, var_23);
                // force_matrix_history[env, 0, sensor, f] = force_matrix[env, sensor, f]         <L 209>
                var_30 = wp::address(var_force_matrix, var_0, var_1, var_19);
                var_32 = wp::load(var_30);
                // wp::array_store(var_force_matrix_history, var_0, var_31, var_1, var_19, var_32);
                wp::assign(var_10, var_25);
                wp::adj_assign(var_10, var_25, adj_10, adj_25);
                wp::adj_array_store(var_force_matrix_history, var_0, var_31, var_1, var_19, var_32, adj_force_matrix_history, adj_0, adj_31, adj_1, adj_19, adj_30);
                wp::adj_address(var_force_matrix, var_0, var_1, var_19, adj_force_matrix, adj_0, adj_1, adj_19, adj_30);
                // adj: force_matrix_history[env, 0, sensor, f] = force_matrix[env, sensor, f]    <L 209>
                var_24 = wp::iter_reverse(var_24);
                start_for_5:;
                    if (iter_cmp(var_24) == 0) goto end_for_5;
                    var_25 = wp::iter_next(var_24);
                	adj_26 = {};
                	adj_27 = {};
                	adj_28 = {};
                	adj_29 = {};
                    // force_matrix_history[env, i, sensor, f] = force_matrix_history[env, i - 1, sensor, f]       <L 208>
                    var_27 = wp::sub(var_25, var_26);
                    var_28 = wp::address(var_force_matrix_history, var_0, var_27, var_1, var_19);
                    var_29 = wp::load(var_28);
                    // wp::array_store(var_force_matrix_history, var_0, var_25, var_1, var_19, var_29);
                    wp::adj_array_store(var_force_matrix_history, var_0, var_25, var_1, var_19, var_29, adj_force_matrix_history, adj_0, adj_25, adj_1, adj_19, adj_28);
                    wp::adj_address(var_force_matrix_history, var_0, var_27, var_1, var_19, adj_force_matrix_history, adj_0, adj_27, adj_1, adj_19, adj_28);
                    wp::adj_sub(var_25, var_26, adj_25, adj_26, adj_27);
                    // adj: force_matrix_history[env, i, sensor, f] = force_matrix_history[env, i - 1, sensor, f]  <L 208>
                	goto start_for_5;
                end_for_5:;
                wp::adj_sub(var_history_length, var_20, adj_history_length, adj_20, adj_21);
                // adj: for i in range(history_length - 1, 0, -1):                                <L 207>
            	goto start_for_3;
            end_for_3:;
            // adj: for f in range(num_filter_objects):                                           <L 206>
        }
        // adj: if force_matrix_history:                                                          <L 205>
        if (var_net_forces_history) {
            wp::adj_array_store(var_net_forces_history, var_0, var_16, var_1, var_17, adj_net_forces_history, adj_0, adj_16, adj_1, adj_15);
            wp::adj_address(var_net_forces, var_0, var_1, adj_net_forces, adj_0, adj_1, adj_15);
            // adj: net_forces_history[env, 0, sensor] = net_forces[env, sensor]                  <L 203>
            var_9 = wp::iter_reverse(var_9);
            start_for_1:;
                if (iter_cmp(var_9) == 0) goto end_for_1;
                var_10 = wp::iter_next(var_9);
            	adj_11 = {};
            	adj_12 = {};
            	adj_13 = {};
            	adj_14 = {};
                // net_forces_history[env, i, sensor] = net_forces_history[env, i - 1, sensor]       <L 202>
                var_12 = wp::sub(var_10, var_11);
                var_13 = wp::address(var_net_forces_history, var_0, var_12, var_1);
                var_14 = wp::load(var_13);
                // wp::array_store(var_net_forces_history, var_0, var_10, var_1, var_14);
                wp::adj_array_store(var_net_forces_history, var_0, var_10, var_1, var_14, adj_net_forces_history, adj_0, adj_10, adj_1, adj_13);
                wp::adj_address(var_net_forces_history, var_0, var_12, var_1, adj_net_forces_history, adj_0, adj_12, adj_1, adj_13);
                wp::adj_sub(var_10, var_11, adj_10, adj_11, adj_12);
                // adj: net_forces_history[env, i, sensor] = net_forces_history[env, i - 1, sensor]  <L 202>
            	goto start_for_1;
            end_for_1:;
            wp::adj_sub(var_history_length, var_5, adj_history_length, adj_5, adj_6);
            // adj: for i in range(history_length - 1, 0, -1):                                    <L 201>
        }
        // adj: if net_forces_history:                                                            <L 200>
        if (var_env_mask) {
            if (var_3) {
                label0:;
                // adj: return                                                                    <L 197>
            }
            wp::adj_address(var_env_mask, var_0, adj_env_mask, adj_0, adj_2);
            // adj: if not env_mask[env]:                                                         <L 196>
        }
        // adj: if env_mask:                                                                      <L 195>
        // adj: env, sensor = wp.tid()                                                            <L 193>
        // adj: def update_contact_sensor_kernel(                                                 <L 162>
        continue;
    }
}



extern "C" __global__ void reset_contact_sensor_kernel_923c105d_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_history_length,
    wp::int32 var_num_filter_objects,
    wp::array_t<bool> var_env_mask,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_forces_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_forces_w_history,
    wp::array_t<wp::vec_t<3, wp::float32>> var_force_matrix_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_force_matrix_w_history,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_normal_forces_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_normal_forces_w_history,
    wp::array_t<wp::vec_t<3, wp::float32>> var_normal_force_matrix_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_normal_force_matrix_w_history,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_friction_forces_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_friction_forces_w_history,
    wp::array_t<wp::vec_t<3, wp::float32>> var_friction_force_matrix_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_friction_force_matrix_w_history,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_w,
    wp::array_t<wp::float32> var_current_air_time,
    wp::array_t<wp::float32> var_last_air_time,
    wp::array_t<wp::float32> var_current_contact_time,
    wp::array_t<wp::float32> var_last_contact_time)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
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
        bool var_4;
        const wp::float32 var_5 = 0.0;
        wp::vec_t<3, wp::float32> var_6;
        const wp::float32 var_7 = 0.0;
        wp::vec_t<3, wp::float32> var_8;
        wp::range_t var_9;
        wp::int32 var_10;
        const wp::float32 var_11 = 0.0;
        wp::vec_t<3, wp::float32> var_12;
        wp::range_t var_13;
        wp::int32 var_14;
        const wp::float32 var_15 = 0.0;
        wp::vec_t<3, wp::float32> var_16;
        wp::int32 var_17;
        const wp::float32 var_18 = 0.0;
        wp::vec_t<3, wp::float32> var_19;
        wp::range_t var_20;
        wp::int32 var_21;
        const wp::float32 var_22 = 0.0;
        wp::vec_t<3, wp::float32> var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::range_t var_26;
        wp::int32 var_27;
        const wp::float32 var_28 = 0.0;
        wp::vec_t<3, wp::float32> var_29;
        wp::range_t var_30;
        wp::int32 var_31;
        const wp::float32 var_32 = 0.0;
        wp::vec_t<3, wp::float32> var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        const wp::float32 var_36 = 0.0;
        wp::vec_t<3, wp::float32> var_37;
        wp::range_t var_38;
        wp::int32 var_39;
        const wp::float32 var_40 = 0.0;
        wp::vec_t<3, wp::float32> var_41;
        wp::int32 var_42;
        const wp::float32 var_43 = 0.0;
        wp::vec_t<3, wp::float32> var_44;
        wp::range_t var_45;
        wp::int32 var_46;
        const wp::float32 var_47 = 0.0;
        wp::vec_t<3, wp::float32> var_48;
        wp::int32 var_49;
        wp::int32 var_50;
        wp::range_t var_51;
        wp::int32 var_52;
        const wp::float32 var_53 = NAN;
        const wp::float32 var_54 = NAN;
        wp::vec_t<3, wp::float32> var_55;
        wp::int32 var_56;
        const wp::float32 var_57 = 0.0;
        const wp::float32 var_58 = 0.0;
        const wp::float32 var_59 = 0.0;
        const wp::float32 var_60 = 0.0;
        //---------
        // forward
        // def reset_contact_sensor_kernel(                                                       <L 77>
        // env, sensor = wp.tid()                                                                 <L 106>
        builtin_tid2d(var_0, var_1);
        // if env_mask:                                                                           <L 108>
        if (var_env_mask) {
            // if not env_mask[env]:                                                              <L 109>
            var_2 = wp::address(var_env_mask, var_0);
            var_4 = wp::load(var_2);
            var_3 = wp::unot(var_4);
            if (var_3) {
                // return                                                                         <L 110>
                continue;
            }
        }
        // net_forces_w[env, sensor] = wp.vec3f(0.0)                                              <L 113>
        var_6 = wp::vec_t<3, wp::float32>(var_5);
        wp::array_store(var_net_forces_w, var_0, var_1, var_6);
        // net_normal_forces_w[env, sensor] = wp.vec3f(0.0)                                       <L 114>
        var_8 = wp::vec_t<3, wp::float32>(var_7);
        wp::array_store(var_net_normal_forces_w, var_0, var_1, var_8);
        // if net_forces_w_history:                                                               <L 117>
        if (var_net_forces_w_history) {
            // for i in range(history_length):                                                    <L 118>
            var_9 = wp::range(var_history_length);
            start_for_1:;
                if (iter_cmp(var_9) == 0) goto end_for_1;
                var_10 = wp::iter_next(var_9);
                // net_forces_w_history[env, i, sensor] = wp.vec3f(0.0)                           <L 119>
                var_12 = wp::vec_t<3, wp::float32>(var_11);
                wp::array_store(var_net_forces_w_history, var_0, var_10, var_1, var_12);
                goto start_for_1;
            end_for_1:;
        }
        // if net_normal_forces_w_history:                                                        <L 120>
        if (var_net_normal_forces_w_history) {
            // for i in range(history_length):                                                    <L 121>
            var_13 = wp::range(var_history_length);
            start_for_3:;
                if (iter_cmp(var_13) == 0) goto end_for_3;
                var_14 = wp::iter_next(var_13);
                // net_normal_forces_w_history[env, i, sensor] = wp.vec3f(0.0)                    <L 122>
                var_16 = wp::vec_t<3, wp::float32>(var_15);
                wp::array_store(var_net_normal_forces_w_history, var_0, var_14, var_1, var_16);
                goto start_for_3;
            end_for_3:;
        }
        var_17 = wp::where(var_net_normal_forces_w_history, var_14, var_10);
        // if net_friction_forces_w:                                                              <L 124>
        if (var_net_friction_forces_w) {
            // net_friction_forces_w[env, sensor] = wp.vec3f(0.0)                                 <L 125>
            var_19 = wp::vec_t<3, wp::float32>(var_18);
            wp::array_store(var_net_friction_forces_w, var_0, var_1, var_19);
            // if net_friction_forces_w_history:                                                  <L 126>
            if (var_net_friction_forces_w_history) {
                // for i in range(history_length):                                                <L 127>
                var_20 = wp::range(var_history_length);
                start_for_5:;
                    if (iter_cmp(var_20) == 0) goto end_for_5;
                    var_21 = wp::iter_next(var_20);
                    // net_friction_forces_w_history[env, i, sensor] = wp.vec3f(0.0)              <L 128>
                    var_23 = wp::vec_t<3, wp::float32>(var_22);
                    wp::array_store(var_net_friction_forces_w_history, var_0, var_21, var_1, var_23);
                    goto start_for_5;
                end_for_5:;
            }
            var_24 = wp::where(var_net_friction_forces_w_history, var_21, var_17);
        }
        var_25 = wp::where(var_net_friction_forces_w, var_24, var_17);
        // if normal_force_matrix_w:                                                              <L 131>
        if (var_normal_force_matrix_w) {
            // for f in range(num_filter_objects):                                                <L 132>
            var_26 = wp::range(var_num_filter_objects);
            start_for_7:;
                if (iter_cmp(var_26) == 0) goto end_for_7;
                var_27 = wp::iter_next(var_26);
                // if force_matrix_w:                                                             <L 133>
                if (var_force_matrix_w) {
                    // force_matrix_w[env, sensor, f] = wp.vec3f(0.0)                             <L 134>
                    var_29 = wp::vec_t<3, wp::float32>(var_28);
                    wp::array_store(var_force_matrix_w, var_0, var_1, var_27, var_29);
                    // if force_matrix_w_history:                                                 <L 135>
                    if (var_force_matrix_w_history) {
                        // for i in range(history_length):                                        <L 136>
                        var_30 = wp::range(var_history_length);
                        start_for_9:;
                            if (iter_cmp(var_30) == 0) goto end_for_9;
                            var_31 = wp::iter_next(var_30);
                            // force_matrix_w_history[env, i, sensor, f] = wp.vec3f(0.0)          <L 137>
                            var_33 = wp::vec_t<3, wp::float32>(var_32);
                            wp::array_store(var_force_matrix_w_history, var_0, var_31, var_1, var_27, var_33);
                            goto start_for_9;
                        end_for_9:;
                    }
                    var_34 = wp::where(var_force_matrix_w_history, var_31, var_25);
                }
                var_35 = wp::where(var_force_matrix_w, var_34, var_25);
                // normal_force_matrix_w[env, sensor, f] = wp.vec3f(0.0)                          <L 138>
                var_37 = wp::vec_t<3, wp::float32>(var_36);
                wp::array_store(var_normal_force_matrix_w, var_0, var_1, var_27, var_37);
                // if normal_force_matrix_w_history:                                              <L 139>
                if (var_normal_force_matrix_w_history) {
                    // for i in range(history_length):                                            <L 140>
                    var_38 = wp::range(var_history_length);
                    start_for_11:;
                        if (iter_cmp(var_38) == 0) goto end_for_11;
                        var_39 = wp::iter_next(var_38);
                        // normal_force_matrix_w_history[env, i, sensor, f] = wp.vec3f(0.0)       <L 141>
                        var_41 = wp::vec_t<3, wp::float32>(var_40);
                        wp::array_store(var_normal_force_matrix_w_history, var_0, var_39, var_1, var_27, var_41);
                        goto start_for_11;
                    end_for_11:;
                }
                var_42 = wp::where(var_normal_force_matrix_w_history, var_39, var_35);
                // if friction_force_matrix_w:                                                    <L 142>
                if (var_friction_force_matrix_w) {
                    // friction_force_matrix_w[env, sensor, f] = wp.vec3f(0.0)                    <L 143>
                    var_44 = wp::vec_t<3, wp::float32>(var_43);
                    wp::array_store(var_friction_force_matrix_w, var_0, var_1, var_27, var_44);
                    // if friction_force_matrix_w_history:                                        <L 144>
                    if (var_friction_force_matrix_w_history) {
                        // for i in range(history_length):                                        <L 145>
                        var_45 = wp::range(var_history_length);
                        start_for_13:;
                            if (iter_cmp(var_45) == 0) goto end_for_13;
                            var_46 = wp::iter_next(var_45);
                            // friction_force_matrix_w_history[env, i, sensor, f] = wp.vec3f(0.0)       <L 146>
                            var_48 = wp::vec_t<3, wp::float32>(var_47);
                            wp::array_store(var_friction_force_matrix_w_history, var_0, var_46, var_1, var_27, var_48);
                            goto start_for_13;
                        end_for_13:;
                    }
                    var_49 = wp::where(var_friction_force_matrix_w_history, var_46, var_42);
                }
                var_50 = wp::where(var_friction_force_matrix_w, var_49, var_42);
                wp::assign(var_25, var_50);
                goto start_for_7;
            end_for_7:;
        }
        // if contact_pos_w:                                                                      <L 149>
        if (var_contact_pos_w) {
            // for f in range(num_filter_objects):                                                <L 150>
            var_51 = wp::range(var_num_filter_objects);
            start_for_15:;
                if (iter_cmp(var_51) == 0) goto end_for_15;
                var_52 = wp::iter_next(var_51);
                // contact_pos_w[env, sensor, f] = wp.vec3f(wp.nan)                               <L 151>
                var_55 = wp::vec_t<3, wp::float32>(var_54);
                wp::array_store(var_contact_pos_w, var_0, var_1, var_52, var_55);
                goto start_for_15;
            end_for_15:;
        }
        var_56 = wp::where(var_contact_pos_w, var_52, var_27);
        // if current_air_time:                                                                   <L 154>
        if (var_current_air_time) {
            // current_air_time[env, sensor] = 0.0                                                <L 155>
            wp::array_store(var_current_air_time, var_0, var_1, var_57);
            // last_air_time[env, sensor] = 0.0                                                   <L 156>
            wp::array_store(var_last_air_time, var_0, var_1, var_58);
            // current_contact_time[env, sensor] = 0.0                                            <L 157>
            wp::array_store(var_current_contact_time, var_0, var_1, var_59);
            // last_contact_time[env, sensor] = 0.0                                               <L 158>
            wp::array_store(var_last_contact_time, var_0, var_1, var_60);
        }
    }
}



extern "C" __global__ void reset_contact_sensor_kernel_923c105d_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_history_length,
    wp::int32 var_num_filter_objects,
    wp::array_t<bool> var_env_mask,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_forces_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_forces_w_history,
    wp::array_t<wp::vec_t<3, wp::float32>> var_force_matrix_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_force_matrix_w_history,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_normal_forces_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_normal_forces_w_history,
    wp::array_t<wp::vec_t<3, wp::float32>> var_normal_force_matrix_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_normal_force_matrix_w_history,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_friction_forces_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_friction_forces_w_history,
    wp::array_t<wp::vec_t<3, wp::float32>> var_friction_force_matrix_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_friction_force_matrix_w_history,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_w,
    wp::array_t<wp::float32> var_current_air_time,
    wp::array_t<wp::float32> var_last_air_time,
    wp::array_t<wp::float32> var_current_contact_time,
    wp::array_t<wp::float32> var_last_contact_time,
    wp::int32 adj_history_length,
    wp::int32 adj_num_filter_objects,
    wp::array_t<bool> adj_env_mask,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_net_forces_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_net_forces_w_history,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_force_matrix_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_force_matrix_w_history,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_net_normal_forces_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_net_normal_forces_w_history,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_normal_force_matrix_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_normal_force_matrix_w_history,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_net_friction_forces_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_net_friction_forces_w_history,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_friction_force_matrix_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_friction_force_matrix_w_history,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_contact_pos_w,
    wp::array_t<wp::float32> adj_current_air_time,
    wp::array_t<wp::float32> adj_last_air_time,
    wp::array_t<wp::float32> adj_current_contact_time,
    wp::array_t<wp::float32> adj_last_contact_time)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
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
        bool var_4;
        const wp::float32 var_5 = 0.0;
        wp::vec_t<3, wp::float32> var_6;
        const wp::float32 var_7 = 0.0;
        wp::vec_t<3, wp::float32> var_8;
        wp::range_t var_9;
        wp::int32 var_10;
        const wp::float32 var_11 = 0.0;
        wp::vec_t<3, wp::float32> var_12;
        wp::range_t var_13;
        wp::int32 var_14;
        const wp::float32 var_15 = 0.0;
        wp::vec_t<3, wp::float32> var_16;
        wp::int32 var_17;
        const wp::float32 var_18 = 0.0;
        wp::vec_t<3, wp::float32> var_19;
        wp::range_t var_20;
        wp::int32 var_21;
        const wp::float32 var_22 = 0.0;
        wp::vec_t<3, wp::float32> var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::range_t var_26;
        wp::int32 var_27;
        const wp::float32 var_28 = 0.0;
        wp::vec_t<3, wp::float32> var_29;
        wp::range_t var_30;
        wp::int32 var_31;
        const wp::float32 var_32 = 0.0;
        wp::vec_t<3, wp::float32> var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        const wp::float32 var_36 = 0.0;
        wp::vec_t<3, wp::float32> var_37;
        wp::range_t var_38;
        wp::int32 var_39;
        const wp::float32 var_40 = 0.0;
        wp::vec_t<3, wp::float32> var_41;
        wp::int32 var_42;
        const wp::float32 var_43 = 0.0;
        wp::vec_t<3, wp::float32> var_44;
        wp::range_t var_45;
        wp::int32 var_46;
        const wp::float32 var_47 = 0.0;
        wp::vec_t<3, wp::float32> var_48;
        wp::int32 var_49;
        wp::int32 var_50;
        wp::range_t var_51;
        wp::int32 var_52;
        const wp::float32 var_53 = NAN;
        const wp::float32 var_54 = NAN;
        wp::vec_t<3, wp::float32> var_55;
        wp::int32 var_56;
        const wp::float32 var_57 = 0.0;
        const wp::float32 var_58 = 0.0;
        const wp::float32 var_59 = 0.0;
        const wp::float32 var_60 = 0.0;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::float32 adj_5 = {};
        wp::vec_t<3, wp::float32> adj_6 = {};
        wp::float32 adj_7 = {};
        wp::vec_t<3, wp::float32> adj_8 = {};
        wp::range_t adj_9 = {};
        wp::int32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::vec_t<3, wp::float32> adj_12 = {};
        wp::range_t adj_13 = {};
        wp::int32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::int32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::vec_t<3, wp::float32> adj_19 = {};
        wp::range_t adj_20 = {};
        wp::int32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::vec_t<3, wp::float32> adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::range_t adj_26 = {};
        wp::int32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::vec_t<3, wp::float32> adj_29 = {};
        wp::range_t adj_30 = {};
        wp::int32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::vec_t<3, wp::float32> adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::vec_t<3, wp::float32> adj_37 = {};
        wp::range_t adj_38 = {};
        wp::int32 adj_39 = {};
        wp::float32 adj_40 = {};
        wp::vec_t<3, wp::float32> adj_41 = {};
        wp::int32 adj_42 = {};
        wp::float32 adj_43 = {};
        wp::vec_t<3, wp::float32> adj_44 = {};
        wp::range_t adj_45 = {};
        wp::int32 adj_46 = {};
        wp::float32 adj_47 = {};
        wp::vec_t<3, wp::float32> adj_48 = {};
        wp::int32 adj_49 = {};
        wp::int32 adj_50 = {};
        wp::range_t adj_51 = {};
        wp::int32 adj_52 = {};
        wp::float32 adj_53 = {};
        wp::float32 adj_54 = {};
        wp::vec_t<3, wp::float32> adj_55 = {};
        wp::int32 adj_56 = {};
        wp::float32 adj_57 = {};
        wp::float32 adj_58 = {};
        wp::float32 adj_59 = {};
        wp::float32 adj_60 = {};
        //---------
        // forward
        // def reset_contact_sensor_kernel(                                                       <L 77>
        // env, sensor = wp.tid()                                                                 <L 106>
        builtin_tid2d(var_0, var_1);
        // if env_mask:                                                                           <L 108>
        if (var_env_mask) {
            // if not env_mask[env]:                                                              <L 109>
            var_2 = wp::address(var_env_mask, var_0);
            var_4 = wp::load(var_2);
            var_3 = wp::unot(var_4);
            if (var_3) {
                // return                                                                         <L 110>
                goto label0;
            }
        }
        // net_forces_w[env, sensor] = wp.vec3f(0.0)                                              <L 113>
        var_6 = wp::vec_t<3, wp::float32>(var_5);
        // wp::array_store(var_net_forces_w, var_0, var_1, var_6);
        // net_normal_forces_w[env, sensor] = wp.vec3f(0.0)                                       <L 114>
        var_8 = wp::vec_t<3, wp::float32>(var_7);
        // wp::array_store(var_net_normal_forces_w, var_0, var_1, var_8);
        // if net_forces_w_history:                                                               <L 117>
        if (var_net_forces_w_history) {
            // for i in range(history_length):                                                    <L 118>
            var_9 = wp::range(var_history_length);
        }
        // if net_normal_forces_w_history:                                                        <L 120>
        if (var_net_normal_forces_w_history) {
            // for i in range(history_length):                                                    <L 121>
            var_13 = wp::range(var_history_length);
        }
        var_17 = wp::where(var_net_normal_forces_w_history, var_14, var_10);
        // if net_friction_forces_w:                                                              <L 124>
        if (var_net_friction_forces_w) {
            // net_friction_forces_w[env, sensor] = wp.vec3f(0.0)                                 <L 125>
            var_19 = wp::vec_t<3, wp::float32>(var_18);
            // wp::array_store(var_net_friction_forces_w, var_0, var_1, var_19);
            // if net_friction_forces_w_history:                                                  <L 126>
            if (var_net_friction_forces_w_history) {
                // for i in range(history_length):                                                <L 127>
                var_20 = wp::range(var_history_length);
            }
            var_24 = wp::where(var_net_friction_forces_w_history, var_21, var_17);
        }
        var_25 = wp::where(var_net_friction_forces_w, var_24, var_17);
        // if normal_force_matrix_w:                                                              <L 131>
        if (var_normal_force_matrix_w) {
            // for f in range(num_filter_objects):                                                <L 132>
            var_26 = wp::range(var_num_filter_objects);
        }
        // if contact_pos_w:                                                                      <L 149>
        if (var_contact_pos_w) {
            // for f in range(num_filter_objects):                                                <L 150>
            var_51 = wp::range(var_num_filter_objects);
        }
        var_56 = wp::where(var_contact_pos_w, var_52, var_27);
        // if current_air_time:                                                                   <L 154>
        if (var_current_air_time) {
            // current_air_time[env, sensor] = 0.0                                                <L 155>
            // wp::array_store(var_current_air_time, var_0, var_1, var_57);
            // last_air_time[env, sensor] = 0.0                                                   <L 156>
            // wp::array_store(var_last_air_time, var_0, var_1, var_58);
            // current_contact_time[env, sensor] = 0.0                                            <L 157>
            // wp::array_store(var_current_contact_time, var_0, var_1, var_59);
            // last_contact_time[env, sensor] = 0.0                                               <L 158>
            // wp::array_store(var_last_contact_time, var_0, var_1, var_60);
        }
        //---------
        // reverse
        if (var_current_air_time) {
            wp::adj_array_store(var_last_contact_time, var_0, var_1, var_60, adj_last_contact_time, adj_0, adj_1, adj_60);
            // adj: last_contact_time[env, sensor] = 0.0                                          <L 158>
            wp::adj_array_store(var_current_contact_time, var_0, var_1, var_59, adj_current_contact_time, adj_0, adj_1, adj_59);
            // adj: current_contact_time[env, sensor] = 0.0                                       <L 157>
            wp::adj_array_store(var_last_air_time, var_0, var_1, var_58, adj_last_air_time, adj_0, adj_1, adj_58);
            // adj: last_air_time[env, sensor] = 0.0                                              <L 156>
            wp::adj_array_store(var_current_air_time, var_0, var_1, var_57, adj_current_air_time, adj_0, adj_1, adj_57);
            // adj: current_air_time[env, sensor] = 0.0                                           <L 155>
        }
        // adj: if current_air_time:                                                              <L 154>
        wp::adj_where(var_contact_pos_w, var_52, var_27, adj_contact_pos_w, adj_52, adj_27, adj_56);
        if (var_contact_pos_w) {
            var_51 = wp::iter_reverse(var_51);
            start_for_15:;
                if (iter_cmp(var_51) == 0) goto end_for_15;
                var_52 = wp::iter_next(var_51);
            	adj_53 = {};
            	adj_54 = {};
            	adj_55 = {};
                // contact_pos_w[env, sensor, f] = wp.vec3f(wp.nan)                               <L 151>
                var_55 = wp::vec_t<3, wp::float32>(var_54);
                // wp::array_store(var_contact_pos_w, var_0, var_1, var_52, var_55);
                wp::adj_array_store(var_contact_pos_w, var_0, var_1, var_52, var_55, adj_contact_pos_w, adj_0, adj_1, adj_52, adj_55);
                wp::adj_vec_t(var_54, adj_54, adj_55);
                // adj: contact_pos_w[env, sensor, f] = wp.vec3f(wp.nan)                          <L 151>
            	goto start_for_15;
            end_for_15:;
            // adj: for f in range(num_filter_objects):                                           <L 150>
        }
        // adj: if contact_pos_w:                                                                 <L 149>
        if (var_normal_force_matrix_w) {
            var_26 = wp::iter_reverse(var_26);
            start_for_7:;
                if (iter_cmp(var_26) == 0) goto end_for_7;
                var_27 = wp::iter_next(var_26);
            	adj_28 = {};
            	adj_29 = {};
            	adj_30 = {};
            	adj_34 = {};
            	adj_35 = {};
            	adj_36 = {};
            	adj_37 = {};
            	adj_38 = {};
            	adj_42 = {};
            	adj_43 = {};
            	adj_44 = {};
            	adj_45 = {};
            	adj_49 = {};
            	adj_50 = {};
                // if force_matrix_w:                                                             <L 133>
                if (var_force_matrix_w) {
                    // force_matrix_w[env, sensor, f] = wp.vec3f(0.0)                             <L 134>
                    var_29 = wp::vec_t<3, wp::float32>(var_28);
                    // wp::array_store(var_force_matrix_w, var_0, var_1, var_27, var_29);
                    // if force_matrix_w_history:                                                 <L 135>
                    if (var_force_matrix_w_history) {
                        // for i in range(history_length):                                        <L 136>
                        var_30 = wp::range(var_history_length);
                    }
                    var_34 = wp::where(var_force_matrix_w_history, var_31, var_25);
                }
                var_35 = wp::where(var_force_matrix_w, var_34, var_25);
                // normal_force_matrix_w[env, sensor, f] = wp.vec3f(0.0)                          <L 138>
                var_37 = wp::vec_t<3, wp::float32>(var_36);
                // wp::array_store(var_normal_force_matrix_w, var_0, var_1, var_27, var_37);
                // if normal_force_matrix_w_history:                                              <L 139>
                if (var_normal_force_matrix_w_history) {
                    // for i in range(history_length):                                            <L 140>
                    var_38 = wp::range(var_history_length);
                }
                var_42 = wp::where(var_normal_force_matrix_w_history, var_39, var_35);
                // if friction_force_matrix_w:                                                    <L 142>
                if (var_friction_force_matrix_w) {
                    // friction_force_matrix_w[env, sensor, f] = wp.vec3f(0.0)                    <L 143>
                    var_44 = wp::vec_t<3, wp::float32>(var_43);
                    // wp::array_store(var_friction_force_matrix_w, var_0, var_1, var_27, var_44);
                    // if friction_force_matrix_w_history:                                        <L 144>
                    if (var_friction_force_matrix_w_history) {
                        // for i in range(history_length):                                        <L 145>
                        var_45 = wp::range(var_history_length);
                    }
                    var_49 = wp::where(var_friction_force_matrix_w_history, var_46, var_42);
                }
                var_50 = wp::where(var_friction_force_matrix_w, var_49, var_42);
                wp::assign(var_25, var_50);
                wp::adj_assign(var_25, var_50, adj_25, adj_50);
                wp::adj_where(var_friction_force_matrix_w, var_49, var_42, adj_friction_force_matrix_w, adj_49, adj_42, adj_50);
                if (var_friction_force_matrix_w) {
                    wp::adj_where(var_friction_force_matrix_w_history, var_46, var_42, adj_friction_force_matrix_w_history, adj_46, adj_42, adj_49);
                    if (var_friction_force_matrix_w_history) {
                        var_45 = wp::iter_reverse(var_45);
                        start_for_13:;
                            if (iter_cmp(var_45) == 0) goto end_for_13;
                            var_46 = wp::iter_next(var_45);
                        	adj_47 = {};
                        	adj_48 = {};
                            // friction_force_matrix_w_history[env, i, sensor, f] = wp.vec3f(0.0)       <L 146>
                            var_48 = wp::vec_t<3, wp::float32>(var_47);
                            // wp::array_store(var_friction_force_matrix_w_history, var_0, var_46, var_1, var_27, var_48);
                            wp::adj_array_store(var_friction_force_matrix_w_history, var_0, var_46, var_1, var_27, var_48, adj_friction_force_matrix_w_history, adj_0, adj_46, adj_1, adj_27, adj_48);
                            wp::adj_vec_t(var_47, adj_47, adj_48);
                            // adj: friction_force_matrix_w_history[env, i, sensor, f] = wp.vec3f(0.0)  <L 146>
                        	goto start_for_13;
                        end_for_13:;
                        // adj: for i in range(history_length):                                   <L 145>
                    }
                    // adj: if friction_force_matrix_w_history:                                   <L 144>
                    wp::adj_array_store(var_friction_force_matrix_w, var_0, var_1, var_27, var_44, adj_friction_force_matrix_w, adj_0, adj_1, adj_27, adj_44);
                    wp::adj_vec_t(var_43, adj_43, adj_44);
                    // adj: friction_force_matrix_w[env, sensor, f] = wp.vec3f(0.0)               <L 143>
                }
                // adj: if friction_force_matrix_w:                                               <L 142>
                wp::adj_where(var_normal_force_matrix_w_history, var_39, var_35, adj_normal_force_matrix_w_history, adj_39, adj_35, adj_42);
                if (var_normal_force_matrix_w_history) {
                    var_38 = wp::iter_reverse(var_38);
                    start_for_11:;
                        if (iter_cmp(var_38) == 0) goto end_for_11;
                        var_39 = wp::iter_next(var_38);
                    	adj_40 = {};
                    	adj_41 = {};
                        // normal_force_matrix_w_history[env, i, sensor, f] = wp.vec3f(0.0)       <L 141>
                        var_41 = wp::vec_t<3, wp::float32>(var_40);
                        // wp::array_store(var_normal_force_matrix_w_history, var_0, var_39, var_1, var_27, var_41);
                        wp::adj_array_store(var_normal_force_matrix_w_history, var_0, var_39, var_1, var_27, var_41, adj_normal_force_matrix_w_history, adj_0, adj_39, adj_1, adj_27, adj_41);
                        wp::adj_vec_t(var_40, adj_40, adj_41);
                        // adj: normal_force_matrix_w_history[env, i, sensor, f] = wp.vec3f(0.0)  <L 141>
                    	goto start_for_11;
                    end_for_11:;
                    // adj: for i in range(history_length):                                       <L 140>
                }
                // adj: if normal_force_matrix_w_history:                                         <L 139>
                wp::adj_array_store(var_normal_force_matrix_w, var_0, var_1, var_27, var_37, adj_normal_force_matrix_w, adj_0, adj_1, adj_27, adj_37);
                wp::adj_vec_t(var_36, adj_36, adj_37);
                // adj: normal_force_matrix_w[env, sensor, f] = wp.vec3f(0.0)                     <L 138>
                wp::adj_where(var_force_matrix_w, var_34, var_25, adj_force_matrix_w, adj_34, adj_25, adj_35);
                if (var_force_matrix_w) {
                    wp::adj_where(var_force_matrix_w_history, var_31, var_25, adj_force_matrix_w_history, adj_31, adj_25, adj_34);
                    if (var_force_matrix_w_history) {
                        var_30 = wp::iter_reverse(var_30);
                        start_for_9:;
                            if (iter_cmp(var_30) == 0) goto end_for_9;
                            var_31 = wp::iter_next(var_30);
                        	adj_32 = {};
                        	adj_33 = {};
                            // force_matrix_w_history[env, i, sensor, f] = wp.vec3f(0.0)          <L 137>
                            var_33 = wp::vec_t<3, wp::float32>(var_32);
                            // wp::array_store(var_force_matrix_w_history, var_0, var_31, var_1, var_27, var_33);
                            wp::adj_array_store(var_force_matrix_w_history, var_0, var_31, var_1, var_27, var_33, adj_force_matrix_w_history, adj_0, adj_31, adj_1, adj_27, adj_33);
                            wp::adj_vec_t(var_32, adj_32, adj_33);
                            // adj: force_matrix_w_history[env, i, sensor, f] = wp.vec3f(0.0)     <L 137>
                        	goto start_for_9;
                        end_for_9:;
                        // adj: for i in range(history_length):                                   <L 136>
                    }
                    // adj: if force_matrix_w_history:                                            <L 135>
                    wp::adj_array_store(var_force_matrix_w, var_0, var_1, var_27, var_29, adj_force_matrix_w, adj_0, adj_1, adj_27, adj_29);
                    wp::adj_vec_t(var_28, adj_28, adj_29);
                    // adj: force_matrix_w[env, sensor, f] = wp.vec3f(0.0)                        <L 134>
                }
                // adj: if force_matrix_w:                                                        <L 133>
            	goto start_for_7;
            end_for_7:;
            // adj: for f in range(num_filter_objects):                                           <L 132>
        }
        // adj: if normal_force_matrix_w:                                                         <L 131>
        wp::adj_where(var_net_friction_forces_w, var_24, var_17, adj_net_friction_forces_w, adj_24, adj_17, adj_25);
        if (var_net_friction_forces_w) {
            wp::adj_where(var_net_friction_forces_w_history, var_21, var_17, adj_net_friction_forces_w_history, adj_21, adj_17, adj_24);
            if (var_net_friction_forces_w_history) {
                var_20 = wp::iter_reverse(var_20);
                start_for_5:;
                    if (iter_cmp(var_20) == 0) goto end_for_5;
                    var_21 = wp::iter_next(var_20);
                	adj_22 = {};
                	adj_23 = {};
                    // net_friction_forces_w_history[env, i, sensor] = wp.vec3f(0.0)              <L 128>
                    var_23 = wp::vec_t<3, wp::float32>(var_22);
                    // wp::array_store(var_net_friction_forces_w_history, var_0, var_21, var_1, var_23);
                    wp::adj_array_store(var_net_friction_forces_w_history, var_0, var_21, var_1, var_23, adj_net_friction_forces_w_history, adj_0, adj_21, adj_1, adj_23);
                    wp::adj_vec_t(var_22, adj_22, adj_23);
                    // adj: net_friction_forces_w_history[env, i, sensor] = wp.vec3f(0.0)         <L 128>
                	goto start_for_5;
                end_for_5:;
                // adj: for i in range(history_length):                                           <L 127>
            }
            // adj: if net_friction_forces_w_history:                                             <L 126>
            wp::adj_array_store(var_net_friction_forces_w, var_0, var_1, var_19, adj_net_friction_forces_w, adj_0, adj_1, adj_19);
            wp::adj_vec_t(var_18, adj_18, adj_19);
            // adj: net_friction_forces_w[env, sensor] = wp.vec3f(0.0)                            <L 125>
        }
        // adj: if net_friction_forces_w:                                                         <L 124>
        wp::adj_where(var_net_normal_forces_w_history, var_14, var_10, adj_net_normal_forces_w_history, adj_14, adj_10, adj_17);
        if (var_net_normal_forces_w_history) {
            var_13 = wp::iter_reverse(var_13);
            start_for_3:;
                if (iter_cmp(var_13) == 0) goto end_for_3;
                var_14 = wp::iter_next(var_13);
            	adj_15 = {};
            	adj_16 = {};
                // net_normal_forces_w_history[env, i, sensor] = wp.vec3f(0.0)                    <L 122>
                var_16 = wp::vec_t<3, wp::float32>(var_15);
                // wp::array_store(var_net_normal_forces_w_history, var_0, var_14, var_1, var_16);
                wp::adj_array_store(var_net_normal_forces_w_history, var_0, var_14, var_1, var_16, adj_net_normal_forces_w_history, adj_0, adj_14, adj_1, adj_16);
                wp::adj_vec_t(var_15, adj_15, adj_16);
                // adj: net_normal_forces_w_history[env, i, sensor] = wp.vec3f(0.0)               <L 122>
            	goto start_for_3;
            end_for_3:;
            // adj: for i in range(history_length):                                               <L 121>
        }
        // adj: if net_normal_forces_w_history:                                                   <L 120>
        if (var_net_forces_w_history) {
            var_9 = wp::iter_reverse(var_9);
            start_for_1:;
                if (iter_cmp(var_9) == 0) goto end_for_1;
                var_10 = wp::iter_next(var_9);
            	adj_11 = {};
            	adj_12 = {};
                // net_forces_w_history[env, i, sensor] = wp.vec3f(0.0)                           <L 119>
                var_12 = wp::vec_t<3, wp::float32>(var_11);
                // wp::array_store(var_net_forces_w_history, var_0, var_10, var_1, var_12);
                wp::adj_array_store(var_net_forces_w_history, var_0, var_10, var_1, var_12, adj_net_forces_w_history, adj_0, adj_10, adj_1, adj_12);
                wp::adj_vec_t(var_11, adj_11, adj_12);
                // adj: net_forces_w_history[env, i, sensor] = wp.vec3f(0.0)                      <L 119>
            	goto start_for_1;
            end_for_1:;
            // adj: for i in range(history_length):                                               <L 118>
        }
        // adj: if net_forces_w_history:                                                          <L 117>
        wp::adj_array_store(var_net_normal_forces_w, var_0, var_1, var_8, adj_net_normal_forces_w, adj_0, adj_1, adj_8);
        wp::adj_vec_t(var_7, adj_7, adj_8);
        // adj: net_normal_forces_w[env, sensor] = wp.vec3f(0.0)                                  <L 114>
        wp::adj_array_store(var_net_forces_w, var_0, var_1, var_6, adj_net_forces_w, adj_0, adj_1, adj_6);
        wp::adj_vec_t(var_5, adj_5, adj_6);
        // adj: net_forces_w[env, sensor] = wp.vec3f(0.0)                                         <L 113>
        if (var_env_mask) {
            if (var_3) {
                label0:;
                // adj: return                                                                    <L 110>
            }
            wp::adj_address(var_env_mask, var_0, adj_env_mask, adj_0, adj_2);
            // adj: if not env_mask[env]:                                                         <L 109>
        }
        // adj: if env_mask:                                                                      <L 108>
        // adj: env, sensor = wp.tid()                                                            <L 106>
        // adj: def reset_contact_sensor_kernel(                                                  <L 77>
        continue;
    }
}



extern "C" __global__ void compute_first_transition_kernel_bea0eb5c_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_threshold,
    wp::array_t<wp::float32> var_time,
    wp::array_t<wp::float32> var_result)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
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
        wp::float32 var_3;
        wp::float32 var_4;
        bool var_5;
        const wp::float32 var_6 = 0.0;
        bool var_7;
        bool var_8;
        const wp::float32 var_9 = 1.0;
        const wp::float32 var_10 = 0.0;
        //---------
        // forward
        // def compute_first_transition_kernel(                                                   <L 255>
        // env, sensor = wp.tid()                                                                 <L 269>
        builtin_tid2d(var_0, var_1);
        // t = time[env, sensor]                                                                  <L 270>
        var_2 = wp::address(var_time, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if t > 0.0 and t < threshold:                                                          <L 271>
        var_7 = (var_3 > var_6);
        var_5 = var_7;
        if (var_5) {
            var_8 = (var_3 < var_threshold);
            var_5 = var_5 && var_8;
        }
        if (var_5) {
            // result[env, sensor] = 1.0                                                          <L 272>
            wp::array_store(var_result, var_0, var_1, var_9);
        }
        if (!var_5) {
            // result[env, sensor] = 0.0                                                          <L 274>
            wp::array_store(var_result, var_0, var_1, var_10);
        }
    }
}



extern "C" __global__ void compute_first_transition_kernel_bea0eb5c_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_threshold,
    wp::array_t<wp::float32> var_time,
    wp::array_t<wp::float32> var_result,
    wp::float32 adj_threshold,
    wp::array_t<wp::float32> adj_time,
    wp::array_t<wp::float32> adj_result)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
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
        wp::float32 var_3;
        wp::float32 var_4;
        bool var_5;
        const wp::float32 var_6 = 0.0;
        bool var_7;
        bool var_8;
        const wp::float32 var_9 = 1.0;
        const wp::float32 var_10 = 0.0;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::float32 adj_2 = {};
        wp::float32 adj_3 = {};
        wp::float32 adj_4 = {};
        bool adj_5 = {};
        wp::float32 adj_6 = {};
        bool adj_7 = {};
        bool adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        //---------
        // forward
        // def compute_first_transition_kernel(                                                   <L 255>
        // env, sensor = wp.tid()                                                                 <L 269>
        builtin_tid2d(var_0, var_1);
        // t = time[env, sensor]                                                                  <L 270>
        var_2 = wp::address(var_time, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if t > 0.0 and t < threshold:                                                          <L 271>
        var_7 = (var_3 > var_6);
        var_5 = var_7;
        if (var_5) {
            var_8 = (var_3 < var_threshold);
            var_5 = var_5 && var_8;
        }
        if (var_5) {
            // result[env, sensor] = 1.0                                                          <L 272>
            // wp::array_store(var_result, var_0, var_1, var_9);
        }
        if (!var_5) {
            // result[env, sensor] = 0.0                                                          <L 274>
            // wp::array_store(var_result, var_0, var_1, var_10);
        }
        //---------
        // reverse
        if (!var_5) {
            wp::adj_array_store(var_result, var_0, var_1, var_10, adj_result, adj_0, adj_1, adj_10);
            // adj: result[env, sensor] = 0.0                                                     <L 274>
        }
        if (var_5) {
            wp::adj_array_store(var_result, var_0, var_1, var_9, adj_result, adj_0, adj_1, adj_9);
            // adj: result[env, sensor] = 1.0                                                     <L 272>
        }
        if (var_5) {
        }
        // adj: if t > 0.0 and t < threshold:                                                     <L 271>
        wp::adj_copy(var_4, adj_2, adj_3);
        wp::adj_address(var_time, var_0, var_1, adj_time, adj_0, adj_1, adj_2);
        // adj: t = time[env, sensor]                                                             <L 270>
        // adj: env, sensor = wp.tid()                                                            <L 269>
        // adj: def compute_first_transition_kernel(                                              <L 255>
        continue;
    }
}

