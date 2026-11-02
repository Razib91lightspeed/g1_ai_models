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



extern "C" __global__ void _clamp_dc_motor_kernel_454ff6bd_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float32> var_current_vel,
    wp::array_t<wp::uint32> var_state_indices,
    wp::array_t<wp::float32> var_saturation_effort,
    wp::array_t<wp::float32> var_velocity_limit,
    wp::array_t<wp::float32> var_max_motor_effort,
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
        wp::uint32* var_1;
        wp::uint32 var_2;
        wp::uint32 var_3;
        wp::float32* var_4;
        wp::float32 var_5;
        wp::float32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        const wp::float32 var_14 = 0.0;
        bool var_15;
        wp::float32 var_16;
        bool var_17;
        const wp::float32 var_18 = 1.0;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32* var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        const wp::float32 var_27 = 1.0;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        const wp::float32 var_32 = -1.0;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32* var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        //---------
        // forward
        // def _clamp_dc_motor_kernel(                                                            <L 64>
        // i = wp.tid()                                                                           <L 78>
        var_0 = builtin_tid1d();
        // state_idx = state_indices[i]                                                           <L 79>
        var_1 = wp::address(var_state_indices, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // sat = saturation_effort[i]                                                             <L 80>
        var_4 = wp::address(var_saturation_effort, var_0);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // vel_lim = velocity_limit[i]                                                            <L 81>
        var_7 = wp::address(var_velocity_limit, var_0);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // max_e = max_motor_effort[i]                                                            <L 82>
        var_10 = wp::address(var_max_motor_effort, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // corner = vel_lim                                                                       <L 85>
        var_13 = wp::copy(var_8);
        // if sat > 0.0:                                                                          <L 86>
        var_15 = (var_5 > var_14);
        if (var_15) {
            // ratio = max_e / sat                                                                <L 87>
            var_16 = wp::div(var_11, var_5);
            // if ratio == ratio:                                                                 <L 88>
            var_17 = (var_16 == var_16);
            if (var_17) {
                // corner = vel_lim * (1.0 + ratio)                                               <L 89>
                var_19 = wp::add(var_18, var_16);
                var_20 = wp::mul(var_8, var_19);
            }
            var_21 = wp::where(var_17, var_20, var_13);
        }
        var_22 = wp::where(var_15, var_21, var_13);
        // vel = wp.clamp(current_vel[state_idx], -corner, corner)                                <L 90>
        var_23 = wp::address(var_current_vel, var_2);
        var_24 = wp::neg(var_22);
        var_26 = wp::load(var_23);
        var_25 = wp::clamp(var_26, var_24, var_22);
        // effort_max = wp.min(sat * (1.0 - vel / vel_lim), max_e)                                <L 92>
        var_28 = wp::div(var_25, var_8);
        var_29 = wp::sub(var_27, var_28);
        var_30 = wp::mul(var_5, var_29);
        var_31 = wp::min(var_30, var_11);
        // effort_min = wp.max(sat * (-1.0 - vel / vel_lim), -max_e)                              <L 93>
        var_33 = wp::div(var_25, var_8);
        var_34 = wp::sub(var_32, var_33);
        var_35 = wp::mul(var_5, var_34);
        var_36 = wp::neg(var_11);
        var_37 = wp::max(var_35, var_36);
        // dst[i] = wp.clamp(src[i], effort_min, effort_max)                                      <L 94>
        var_38 = wp::address(var_src, var_0);
        var_40 = wp::load(var_38);
        var_39 = wp::clamp(var_40, var_37, var_31);
        wp::array_store(var_dst, var_0, var_39);
    }
}



extern "C" __global__ void _clamp_dc_motor_kernel_454ff6bd_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float32> var_current_vel,
    wp::array_t<wp::uint32> var_state_indices,
    wp::array_t<wp::float32> var_saturation_effort,
    wp::array_t<wp::float32> var_velocity_limit,
    wp::array_t<wp::float32> var_max_motor_effort,
    wp::array_t<wp::float32> var_src,
    wp::array_t<wp::float32> var_dst,
    wp::array_t<wp::float32> adj_current_vel,
    wp::array_t<wp::uint32> adj_state_indices,
    wp::array_t<wp::float32> adj_saturation_effort,
    wp::array_t<wp::float32> adj_velocity_limit,
    wp::array_t<wp::float32> adj_max_motor_effort,
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
        wp::uint32* var_1;
        wp::uint32 var_2;
        wp::uint32 var_3;
        wp::float32* var_4;
        wp::float32 var_5;
        wp::float32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        const wp::float32 var_14 = 0.0;
        bool var_15;
        wp::float32 var_16;
        bool var_17;
        const wp::float32 var_18 = 1.0;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32* var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        const wp::float32 var_27 = 1.0;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        const wp::float32 var_32 = -1.0;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32* var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::uint32 adj_1 = {};
        wp::uint32 adj_2 = {};
        wp::uint32 adj_3 = {};
        wp::float32 adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        bool adj_15 = {};
        wp::float32 adj_16 = {};
        bool adj_17 = {};
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
        //---------
        // forward
        // def _clamp_dc_motor_kernel(                                                            <L 64>
        // i = wp.tid()                                                                           <L 78>
        var_0 = builtin_tid1d();
        // state_idx = state_indices[i]                                                           <L 79>
        var_1 = wp::address(var_state_indices, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // sat = saturation_effort[i]                                                             <L 80>
        var_4 = wp::address(var_saturation_effort, var_0);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // vel_lim = velocity_limit[i]                                                            <L 81>
        var_7 = wp::address(var_velocity_limit, var_0);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // max_e = max_motor_effort[i]                                                            <L 82>
        var_10 = wp::address(var_max_motor_effort, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // corner = vel_lim                                                                       <L 85>
        var_13 = wp::copy(var_8);
        // if sat > 0.0:                                                                          <L 86>
        var_15 = (var_5 > var_14);
        if (var_15) {
            // ratio = max_e / sat                                                                <L 87>
            var_16 = wp::div(var_11, var_5);
            // if ratio == ratio:                                                                 <L 88>
            var_17 = (var_16 == var_16);
            if (var_17) {
                // corner = vel_lim * (1.0 + ratio)                                               <L 89>
                var_19 = wp::add(var_18, var_16);
                var_20 = wp::mul(var_8, var_19);
            }
            var_21 = wp::where(var_17, var_20, var_13);
        }
        var_22 = wp::where(var_15, var_21, var_13);
        // vel = wp.clamp(current_vel[state_idx], -corner, corner)                                <L 90>
        var_23 = wp::address(var_current_vel, var_2);
        var_24 = wp::neg(var_22);
        var_26 = wp::load(var_23);
        var_25 = wp::clamp(var_26, var_24, var_22);
        // effort_max = wp.min(sat * (1.0 - vel / vel_lim), max_e)                                <L 92>
        var_28 = wp::div(var_25, var_8);
        var_29 = wp::sub(var_27, var_28);
        var_30 = wp::mul(var_5, var_29);
        var_31 = wp::min(var_30, var_11);
        // effort_min = wp.max(sat * (-1.0 - vel / vel_lim), -max_e)                              <L 93>
        var_33 = wp::div(var_25, var_8);
        var_34 = wp::sub(var_32, var_33);
        var_35 = wp::mul(var_5, var_34);
        var_36 = wp::neg(var_11);
        var_37 = wp::max(var_35, var_36);
        // dst[i] = wp.clamp(src[i], effort_min, effort_max)                                      <L 94>
        var_38 = wp::address(var_src, var_0);
        var_40 = wp::load(var_38);
        var_39 = wp::clamp(var_40, var_37, var_31);
        // wp::array_store(var_dst, var_0, var_39);
        //---------
        // reverse
        wp::adj_array_store(var_dst, var_0, var_39, adj_dst, adj_0, adj_39);
        wp::adj_clamp(var_40, var_37, var_31, adj_38, adj_37, adj_31, adj_39);
        wp::adj_address(var_src, var_0, adj_src, adj_0, adj_38);
        // adj: dst[i] = wp.clamp(src[i], effort_min, effort_max)                                 <L 94>
        wp::adj_max(var_35, var_36, adj_35, adj_36, adj_37);
        wp::adj_neg(var_11, adj_11, adj_36);
        wp::adj_mul(var_5, var_34, adj_5, adj_34, adj_35);
        wp::adj_sub(var_32, var_33, adj_32, adj_33, adj_34);
        wp::adj_div(var_25, var_8, var_33, adj_25, adj_8, adj_33);
        // adj: effort_min = wp.max(sat * (-1.0 - vel / vel_lim), -max_e)                         <L 93>
        wp::adj_min(var_30, var_11, adj_30, adj_11, adj_31);
        wp::adj_mul(var_5, var_29, adj_5, adj_29, adj_30);
        wp::adj_sub(var_27, var_28, adj_27, adj_28, adj_29);
        wp::adj_div(var_25, var_8, var_28, adj_25, adj_8, adj_28);
        // adj: effort_max = wp.min(sat * (1.0 - vel / vel_lim), max_e)                           <L 92>
        wp::adj_clamp(var_26, var_24, var_22, adj_23, adj_24, adj_22, adj_25);
        wp::adj_neg(var_22, adj_22, adj_24);
        wp::adj_address(var_current_vel, var_2, adj_current_vel, adj_2, adj_23);
        // adj: vel = wp.clamp(current_vel[state_idx], -corner, corner)                           <L 90>
        wp::adj_where(var_15, var_21, var_13, adj_15, adj_21, adj_13, adj_22);
        if (var_15) {
            wp::adj_where(var_17, var_20, var_13, adj_17, adj_20, adj_13, adj_21);
            if (var_17) {
                wp::adj_mul(var_8, var_19, adj_8, adj_19, adj_20);
                wp::adj_add(var_18, var_16, adj_18, adj_16, adj_19);
                // adj: corner = vel_lim * (1.0 + ratio)                                          <L 89>
            }
            // adj: if ratio == ratio:                                                            <L 88>
            wp::adj_div(var_11, var_5, var_16, adj_11, adj_5, adj_16);
            // adj: ratio = max_e / sat                                                           <L 87>
        }
        // adj: if sat > 0.0:                                                                     <L 86>
        wp::adj_copy(var_8, adj_8, adj_13);
        // adj: corner = vel_lim                                                                  <L 85>
        wp::adj_copy(var_12, adj_10, adj_11);
        wp::adj_address(var_max_motor_effort, var_0, adj_max_motor_effort, adj_0, adj_10);
        // adj: max_e = max_motor_effort[i]                                                       <L 82>
        wp::adj_copy(var_9, adj_7, adj_8);
        wp::adj_address(var_velocity_limit, var_0, adj_velocity_limit, adj_0, adj_7);
        // adj: vel_lim = velocity_limit[i]                                                       <L 81>
        wp::adj_copy(var_6, adj_4, adj_5);
        wp::adj_address(var_saturation_effort, var_0, adj_saturation_effort, adj_0, adj_4);
        // adj: sat = saturation_effort[i]                                                        <L 80>
        wp::adj_copy(var_3, adj_1, adj_2);
        wp::adj_address(var_state_indices, var_0, adj_state_indices, adj_0, adj_1);
        // adj: state_idx = state_indices[i]                                                      <L 79>
        // adj: i = wp.tid()                                                                      <L 78>
        // adj: def _clamp_dc_motor_kernel(                                                       <L 64>
        continue;
    }
}



extern "C" __global__ void _corner_velocity_kernel_dbab1f81_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float32> var_saturation_effort,
    wp::array_t<wp::float32> var_velocity_limit,
    wp::array_t<wp::float32> var_max_motor_effort,
    wp::array_t<wp::float32> var_corner_velocity)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::float32* var_1;
        wp::float32 var_2;
        wp::float32 var_3;
        wp::float32* var_4;
        wp::float32 var_5;
        wp::float32 var_6;
        const wp::float32 var_7 = 0.0;
        bool var_8;
        wp::float32* var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        bool var_12;
        wp::float32* var_13;
        const wp::float32 var_14 = 1.0;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        //---------
        // forward
        // def _corner_velocity_kernel(                                                           <L 16>
        // i = wp.tid()                                                                           <L 23>
        var_0 = builtin_tid1d();
        // sat = saturation_effort[i]                                                             <L 24>
        var_1 = wp::address(var_saturation_effort, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // corner = velocity_limit[i]                                                             <L 25>
        var_4 = wp::address(var_velocity_limit, var_0);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // if sat > 0.0:                                                                          <L 26>
        var_8 = (var_2 > var_7);
        if (var_8) {
            // ratio = max_motor_effort[i] / sat                                                  <L 27>
            var_9 = wp::address(var_max_motor_effort, var_0);
            var_11 = wp::load(var_9);
            var_10 = wp::div(var_11, var_2);
            // if ratio == ratio:                                                                 <L 28>
            var_12 = (var_10 == var_10);
            if (var_12) {
                // corner = velocity_limit[i] * (1.0 + ratio)                                     <L 29>
                var_13 = wp::address(var_velocity_limit, var_0);
                var_15 = wp::add(var_14, var_10);
                var_17 = wp::load(var_13);
                var_16 = wp::mul(var_17, var_15);
            }
            var_18 = wp::where(var_12, var_16, var_5);
        }
        var_19 = wp::where(var_8, var_18, var_5);
        // corner_velocity[i] = corner                                                            <L 30>
        wp::array_store(var_corner_velocity, var_0, var_19);
    }
}

