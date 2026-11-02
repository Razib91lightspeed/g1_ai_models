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


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/math.py:0
static CUDA_CALLABLE wp::float32 safe_div_2(
    wp::float32 var_x,
    wp::float32 var_y)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    bool var_1;
    const wp::float32 var_2 = 1e-15;
    const wp::float32 var_3 = 1e-15;
    wp::float32 var_4;
    wp::float32 var_5;
    //---------
    // forward
    // def safe_div(x: Any, y: Any) -> Any:                                                   <L 1>
    // return x / wp.where(y != 0.0, y, types.MJ_MINVAL)                                      <L 2>
    var_1 = (var_y != var_0);
    var_4 = wp::where(var_1, var_y, var_3);
    var_5 = wp::div(var_x, var_4);
    return var_5;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:2443
static CUDA_CALLABLE wp::float32 _elliptic_hessian_entry_from_projections_0(
    wp::float32 var_dm,
    wp::float32 var_mu_over_t,
    wp::float32 var_mu_n_over_ttt,
    wp::float32 var_tangent_diag,
    wp::float32 var_z01,
    wp::float32 var_z02,
    wp::float32 var_projection1,
    wp::float32 var_projection2,
    wp::float32 var_tangent_dot)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    //---------
    // forward
    // def _elliptic_hessian_entry_from_projections(                                          <L 2444>
    // return dm * (                                                                          <L 2457>
    // z01 * z02                                                                              <L 2458>
    var_0 = wp::mul(var_z01, var_z02);
    // - mu_over_t * (z01 * projection2 + z02 * projection1)                                  <L 2459>
    var_1 = wp::mul(var_z01, var_projection2);
    var_2 = wp::mul(var_z02, var_projection1);
    var_3 = wp::add(var_1, var_2);
    var_4 = wp::mul(var_mu_over_t, var_3);
    var_5 = wp::sub(var_0, var_4);
    // + mu_n_over_ttt * projection1 * projection2                                            <L 2460>
    var_6 = wp::mul(var_mu_n_over_ttt, var_projection1);
    var_7 = wp::mul(var_6, var_projection2);
    var_8 = wp::add(var_5, var_7);
    // + tangent_diag * tangent_dot                                                           <L 2461>
    var_9 = wp::mul(var_tangent_diag, var_tangent_dot);
    var_10 = wp::add(var_8, var_9);
    var_11 = wp::mul(var_dm, var_10);
    return var_11;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/math.py:0
static CUDA_CALLABLE void adj_safe_div_2(
    wp::float32 var_x,
    wp::float32 var_y,
    wp::float32 & adj_x,
    wp::float32 & adj_y,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:2443
static CUDA_CALLABLE void adj__elliptic_hessian_entry_from_projections_0(
    wp::float32 var_dm,
    wp::float32 var_mu_over_t,
    wp::float32 var_mu_n_over_ttt,
    wp::float32 var_tangent_diag,
    wp::float32 var_z01,
    wp::float32 var_z02,
    wp::float32 var_projection1,
    wp::float32 var_projection2,
    wp::float32 var_tangent_dot,
    wp::float32 & adj_dm,
    wp::float32 & adj_mu_over_t,
    wp::float32 & adj_mu_n_over_ttt,
    wp::float32 & adj_tangent_diag,
    wp::float32 & adj_z01,
    wp::float32 & adj_z02,
    wp::float32 & adj_projection1,
    wp::float32 & adj_projection2,
    wp::float32 & adj_tangent_dot,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void _gather_J_sparse_0dd96708_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::int32> var_dof_cdof_in,
    wp::array_t<wp::int32> var_J_rownnz_in,
    wp::array_t<wp::int32> var_J_rowadr_in,
    wp::array_t<wp::int32> var_J_colind_in,
    wp::array_t<wp::float32> var_J_in,
    wp::array_t<wp::float32> var_J_c_out)
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
        wp::int32* var_2;
        bool var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::range_t var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 0;
        wp::int32* var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 0;
        bool var_20;
        const wp::int32 var_21 = 0;
        wp::float32* var_22;
        wp::float32 var_23;
        //---------
        // forward
        // def _gather_J_sparse(                                                                  <L 3978>
        // worldid, efcid = wp.tid()                                                              <L 3990>
        builtin_tid2d(var_0, var_1);
        // if efcid >= nefc_in[worldid]:                                                          <L 3991>
        var_2 = wp::address(var_nefc_in, var_0);
        var_4 = wp::load(var_2);
        var_3 = (var_1 >= var_4);
        if (var_3) {
            // return                                                                             <L 3992>
            return;
        }
        // rowadr = J_rowadr_in[worldid, efcid]                                                   <L 3993>
        var_5 = wp::address(var_J_rowadr_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // for k in range(J_rownnz_in[worldid, efcid]):                                           <L 3994>
        var_8 = wp::address(var_J_rownnz_in, var_0, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::range(var_10);
        start_for_1:;
            if (iter_cmp(var_9) == 0) goto end_for_1;
            var_11 = wp::iter_next(var_9);
            // adr = rowadr + k                                                                   <L 3995>
            var_12 = wp::add(var_6, var_11);
            // cj = dof_cdof_in[worldid, J_colind_in[worldid, 0, adr]]                            <L 3996>
            var_14 = wp::address(var_J_colind_in, var_0, var_13, var_12);
            var_16 = wp::load(var_14);
            var_15 = wp::address(var_dof_cdof_in, var_0, var_16);
            var_18 = wp::load(var_15);
            var_17 = wp::copy(var_18);
            // if cj >= 0:                                                                        <L 3997>
            var_20 = (var_17 >= var_19);
            if (var_20) {
                // J_c_out[worldid, efcid, cj] = J_in[worldid, 0, adr]                            <L 3998>
                var_22 = wp::address(var_J_in, var_0, var_21, var_12);
                var_23 = wp::load(var_22);
                wp::array_store(var_J_c_out, var_0, var_1, var_17, var_23);
            }
            goto start_for_1;
        end_for_1:;
}



extern "C" __global__ void _zero_change_counters_1e273adc_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_quad_changed_count_out,
    wp::array_t<wp::int32> var_state_changed_count_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 0;
        const wp::int32 var_2 = 0;
        //---------
        // forward
        // def _zero_change_counters(                                                             <L 3515>
        // worldid = wp.tid()                                                                     <L 3520>
        var_0 = builtin_tid1d();
        // quad_changed_count_out[worldid] = 0                                                    <L 3521>
        wp::array_store(var_quad_changed_count_out, var_0, var_1);
        // state_changed_count_out[worldid] = 0                                                   <L 3522>
        wp::array_store(var_state_changed_count_out, var_0, var_2);
    }
}



extern "C" __global__ void _solve_beta_accumulate_tiled_2c9d0ce3_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_ctx_grad_in,
    wp::array_t<wp::float32> var_ctx_Mgrad_in,
    wp::array_t<wp::float32> var_ctx_prev_grad_in,
    wp::array_t<wp::float32> var_ctx_prev_Mgrad_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_beta_num_out,
    wp::array_t<wp::float32> var_ctx_beta_den_out)
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
        bool* var_2;
        bool var_3;
        bool var_4;
        const wp::float32 var_5 = 0.0;
        wp::float32 var_6;
        const wp::float32 var_7 = 0.0;
        wp::float32 var_8;
        wp::int32 var_9;
        wp::range_t var_10;
        wp::int32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::float32* var_15;
        wp::float32* var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        const bool var_26 = true;
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>> var_27 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>>{};
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_28 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        const bool var_29 = true;
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>> var_30 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>>{};
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_31 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        const wp::int32 var_32 = 0;
        bool var_33;
        const wp::int32 var_34 = 0;
        wp::float32 var_35;
        const wp::int32 var_36 = 0;
        wp::float32 var_37;
        //---------
        // forward
        // def _solve_beta_accumulate_tiled(                                                      <L 3295>
        // worldid, tid = wp.tid()                                                                <L 3308>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 3310>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 3311>
            return;
        }
        var_4 = wp::load(var_2);
        // local_num = float(0.0)                                                                 <L 3313>
        var_6 = wp::float(var_5);
        // local_den = float(0.0)                                                                 <L 3314>
        var_8 = wp::float(var_7);
        // BLOCK_DIM = wp.block_dim()                                                             <L 3315>
        var_9 = builtin_block_dim();
        // for dofid in range(tid, nv, BLOCK_DIM):                                                <L 3317>
        var_10 = wp::range(var_1, var_nv, var_9);
        start_for_1:;
            if (iter_cmp(var_10) == 0) goto end_for_1;
            var_11 = wp::iter_next(var_10);
            // prev_Mgrad = ctx_prev_Mgrad_in[worldid, dofid]                                     <L 3318>
            var_12 = wp::address(var_ctx_prev_Mgrad_in, var_0, var_11);
            var_14 = wp::load(var_12);
            var_13 = wp::copy(var_14);
            // num = ctx_grad_in[worldid, dofid] * (ctx_Mgrad_in[worldid, dofid] - prev_Mgrad)       <L 3319>
            var_15 = wp::address(var_ctx_grad_in, var_0, var_11);
            var_16 = wp::address(var_ctx_Mgrad_in, var_0, var_11);
            var_18 = wp::load(var_16);
            var_17 = wp::sub(var_18, var_13);
            var_20 = wp::load(var_15);
            var_19 = wp::mul(var_20, var_17);
            // den = ctx_prev_grad_in[worldid, dofid] * prev_Mgrad                                <L 3320>
            var_21 = wp::address(var_ctx_prev_grad_in, var_0, var_11);
            var_23 = wp::load(var_21);
            var_22 = wp::mul(var_23, var_13);
            // local_num += num                                                                   <L 3321>
            var_24 = wp::add(var_6, var_19);
            // local_den += den                                                                   <L 3322>
            var_25 = wp::add(var_8, var_22);
            wp::assign(var_6, var_24);
            wp::assign(var_8, var_25);
            goto start_for_1;
        end_for_1:;
        // num_tile = wp.tile(local_num, preserve_type=True)                                      <L 3324>
        var_27 = wp::tile<wp::float32>(var_6);
        // num_sum = wp.tile_reduce(wp.add, num_tile)                                             <L 3325>
        var_28 = wp::tile_reduce(wp::add, var_27);
        // den_tile = wp.tile(local_den, preserve_type=True)                                      <L 3327>
        var_30 = wp::tile<wp::float32>(var_8);
        // den_sum = wp.tile_reduce(wp.add, den_tile)                                             <L 3328>
        var_31 = wp::tile_reduce(wp::add, var_30);
        // if tid == 0:                                                                           <L 3330>
        var_33 = (var_1 == var_32);
        if (var_33) {
            // ctx_beta_num_out[worldid] = num_sum[0]                                             <L 3331>
            var_35 = wp::tile_extract(var_28, var_34);
            wp::array_store(var_ctx_beta_num_out, var_0, var_35);
            // ctx_beta_den_out[worldid] = den_sum[0]                                             <L 3332>
            var_37 = wp::tile_extract(var_31, var_36);
            wp::array_store(var_ctx_beta_den_out, var_0, var_37);
        }
}



extern "C" __global__ void _solve_init_efc_332d2fcc_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_solver_niter_out,
    wp::array_t<wp::float32> var_ctx_search_dot_out,
    wp::array_t<bool> var_ctx_done_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 0;
        const bool var_2 = false;
        const wp::float32 var_3 = 0.0;
        //---------
        // forward
        // def _solve_init_efc(                                                                   <L 1595>
        // worldid = wp.tid()                                                                     <L 1602>
        var_0 = builtin_tid1d();
        // solver_niter_out[worldid] = 0                                                          <L 1603>
        wp::array_store(var_solver_niter_out, var_0, var_1);
        // ctx_done_out[worldid] = False                                                          <L 1604>
        wp::array_store(var_ctx_done_out, var_0, var_2);
        // ctx_search_dot_out[worldid] = 0.0                                                      <L 1605>
        wp::array_store(var_ctx_search_dot_out, var_0, var_3);
    }
}



extern "C" __global__ void _scatter_solution_3c25285e_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_dof_cdof_in,
    wp::array_t<wp::float32> var_x_in,
    wp::array_t<wp::float32> var_vec_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        const wp::int32 var_7 = 0;
        wp::float32* var_8;
        wp::float32 var_9;
        const wp::float32 var_10 = 0.0;
        //---------
        // forward
        // def _scatter_solution(                                                                 <L 3882>
        // worldid, i = wp.tid()                                                                  <L 3890>
        builtin_tid2d(var_0, var_1);
        // ci = dof_cdof_in[worldid, i]                                                           <L 3891>
        var_2 = wp::address(var_dof_cdof_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if ci >= 0:                                                                            <L 3892>
        var_6 = (var_3 >= var_5);
        if (var_6) {
            // vec_out[worldid, i] = x_in[worldid, ci, 0]                                         <L 3893>
            var_8 = wp::address(var_x_in, var_0, var_3, var_7);
            var_9 = wp::load(var_8);
            wp::array_store(var_vec_out, var_0, var_1, var_9);
        }
        if (!var_6) {
            // vec_out[worldid, i] = 0.0  # frozen inactive DOF                                   <L 3895>
            wp::array_store(var_vec_out, var_0, var_1, var_10);
        }
}



extern "C" __global__ void _init_compact_inertia_48546221_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_ncdof_in,
    wp::array_t<wp::float32> var_M_c_out)
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
        wp::int32 var_2;
        const wp::float32 var_3 = 0.0;
        bool var_4;
        bool var_5;
        wp::int32* var_6;
        bool var_7;
        wp::int32 var_8;
        const wp::float32 var_9 = 1.0;
        wp::float32 var_10;
        //---------
        // forward
        // def _init_compact_inertia(                                                             <L 3757>
        // worldid, i, j = wp.tid()                                                               <L 3763>
        builtin_tid3d(var_0, var_1, var_2);
        // val = 0.0                                                                              <L 3764>
        // if i == j and i >= ncdof_in[worldid]:                                                  <L 3765>
        var_5 = (var_1 == var_2);
        var_4 = var_5;
        if (var_4) {
            var_6 = wp::address(var_ncdof_in, var_0);
            var_8 = wp::load(var_6);
            var_7 = (var_1 >= var_8);
            var_4 = var_4 && var_7;
        }
        if (var_4) {
            // val = 1.0                                                                          <L 3766>
        }
        var_10 = wp::where(var_4, var_9, var_3);
        // M_c_out[worldid, i, j] = val                                                           <L 3767>
        wp::array_store(var_M_c_out, var_0, var_1, var_2, var_10);
}



extern "C" __global__ void _gather_rhs_compact_48b89a4c_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_cdof_dof_in,
    wp::array_t<wp::float32> var_vec_in,
    wp::array_t<wp::float32> var_rhs_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::float32* var_7;
        const wp::int32 var_8 = 0;
        wp::float32 var_9;
        const wp::float32 var_10 = 0.0;
        const wp::int32 var_11 = 0;
        //---------
        // forward
        // def _gather_rhs_compact(                                                               <L 3865>
        // worldid, ci = wp.tid()                                                                 <L 3873>
        builtin_tid2d(var_0, var_1);
        // dof = cdof_dof_in[worldid, ci]                                                         <L 3874>
        var_2 = wp::address(var_cdof_dof_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if dof >= 0:                                                                           <L 3875>
        var_6 = (var_3 >= var_5);
        if (var_6) {
            // rhs_out[worldid, ci, 0] = vec_in[worldid, dof]                                     <L 3876>
            var_7 = wp::address(var_vec_in, var_0, var_3);
            var_9 = wp::load(var_7);
            wp::array_store(var_rhs_out, var_0, var_1, var_8, var_9);
        }
        if (!var_6) {
            // rhs_out[worldid, ci, 0] = 0.0                                                      <L 3878>
            wp::array_store(var_rhs_out, var_0, var_1, var_11, var_10);
        }
}



extern "C" __global__ void _qfrc_constraint_from_grad_4e7e88b1_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_qfrc_smooth_in,
    wp::array_t<wp::float32> var_efc_Ma_in,
    wp::array_t<wp::float32> var_ctx_grad_in,
    wp::array_t<wp::float32> var_ctx_grad_scale_in,
    wp::array_t<wp::float32> var_qfrc_constraint_out)
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
        wp::float32* var_2;
        wp::float32* var_3;
        wp::float32 var_4;
        wp::float32 var_5;
        wp::float32 var_6;
        wp::float32* var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        //---------
        // forward
        // def _qfrc_constraint_from_grad(                                                        <L 1896>
        // worldid, dofid = wp.tid()                                                              <L 1906>
        builtin_tid2d(var_0, var_1);
        // grad = ctx_grad_scale_in[worldid] * ctx_grad_in[worldid, dofid]                        <L 1908>
        var_2 = wp::address(var_ctx_grad_scale_in, var_0);
        var_3 = wp::address(var_ctx_grad_in, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = wp::mul(var_5, var_6);
        // qfrc_constraint_out[worldid, dofid] = efc_Ma_in[worldid, dofid] - qfrc_smooth_in[worldid, dofid] - grad       <L 1909>
        var_7 = wp::address(var_efc_Ma_in, var_0, var_1);
        var_8 = wp::address(var_qfrc_smooth_in, var_0, var_1);
        var_10 = wp::load(var_7);
        var_11 = wp::load(var_8);
        var_9 = wp::sub(var_10, var_11);
        var_12 = wp::sub(var_9, var_4);
        wp::array_store(var_qfrc_constraint_out, var_0, var_1, var_12);
}



extern "C" __global__ void _scatter_dof_vecs_54d8b54e_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_dof_cdof_in,
    wp::array_t<wp::float32> var_qacc_c_in,
    wp::array_t<wp::float32> var_qfrc_constraint_c_in,
    wp::array_t<wp::float32> var_qacc_out,
    wp::array_t<wp::float32> var_qfrc_constraint_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32* var_9;
        wp::float32 var_10;
        const wp::float32 var_11 = 0.0;
        const wp::float32 var_12 = 0.0;
        //---------
        // forward
        // def _scatter_dof_vecs(                                                                 <L 3957>
        // worldid, i = wp.tid()                                                                  <L 3967>
        builtin_tid2d(var_0, var_1);
        // ci = dof_cdof_in[worldid, i]                                                           <L 3968>
        var_2 = wp::address(var_dof_cdof_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if ci >= 0:                                                                            <L 3969>
        var_6 = (var_3 >= var_5);
        if (var_6) {
            // qacc_out[worldid, i] = qacc_c_in[worldid, ci]                                      <L 3970>
            var_7 = wp::address(var_qacc_c_in, var_0, var_3);
            var_8 = wp::load(var_7);
            wp::array_store(var_qacc_out, var_0, var_1, var_8);
            // qfrc_constraint_out[worldid, i] = qfrc_constraint_c_in[worldid, ci]                <L 3971>
            var_9 = wp::address(var_qfrc_constraint_c_in, var_0, var_3);
            var_10 = wp::load(var_9);
            wp::array_store(var_qfrc_constraint_out, var_0, var_1, var_10);
        }
        if (!var_6) {
            // qacc_out[worldid, i] = 0.0                                                         <L 3973>
            wp::array_store(var_qacc_out, var_0, var_1, var_11);
            // qfrc_constraint_out[worldid, i] = 0.0                                              <L 3974>
            wp::array_store(var_qfrc_constraint_out, var_0, var_1, var_12);
        }
}



extern "C" __global__ void _solve_init_search_cg_tiled_5992b048_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_ctx_grad_in,
    wp::array_t<wp::float32> var_ctx_Mgrad_in,
    wp::array_t<wp::float32> var_ctx_search_out,
    wp::array_t<wp::float32> var_ctx_search_dot_out,
    wp::array_t<wp::float32> var_ctx_prev_grad_out,
    wp::array_t<wp::float32> var_ctx_prev_Mgrad_out)
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
        const wp::float32 var_2 = 0.0;
        wp::float32 var_3;
        wp::int32 var_4;
        wp::range_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        const wp::float32 var_10 = -1.0;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        const bool var_16 = true;
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>> var_17 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>>{};
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_18 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        const wp::int32 var_19 = 0;
        bool var_20;
        const wp::int32 var_21 = 0;
        wp::float32 var_22;
        //---------
        // forward
        // def _solve_init_search_cg_tiled(                                                       <L 1665>
        // worldid, tid = wp.tid()                                                                <L 1677>
        builtin_tid2d(var_0, var_1);
        // local_search_dot = float(0.0)                                                          <L 1679>
        var_3 = wp::float(var_2);
        // BLOCK_DIM = wp.block_dim()                                                             <L 1680>
        var_4 = builtin_block_dim();
        // for dofid in range(tid, nv, BLOCK_DIM):                                                <L 1682>
        var_5 = wp::range(var_1, var_nv, var_4);
        start_for_0:;
            if (iter_cmp(var_5) == 0) goto end_for_0;
            var_6 = wp::iter_next(var_5);
            // mgrad = ctx_Mgrad_in[worldid, dofid]                                               <L 1683>
            var_7 = wp::address(var_ctx_Mgrad_in, var_0, var_6);
            var_9 = wp::load(var_7);
            var_8 = wp::copy(var_9);
            // search = -1.0 * mgrad                                                              <L 1684>
            var_11 = wp::mul(var_10, var_8);
            // ctx_search_out[worldid, dofid] = search                                            <L 1685>
            wp::array_store(var_ctx_search_out, var_0, var_6, var_11);
            // local_search_dot += search * search                                                <L 1686>
            var_12 = wp::mul(var_11, var_11);
            var_13 = wp::add(var_3, var_12);
            // ctx_prev_grad_out[worldid, dofid] = ctx_grad_in[worldid, dofid]                    <L 1688>
            var_14 = wp::address(var_ctx_grad_in, var_0, var_6);
            var_15 = wp::load(var_14);
            wp::array_store(var_ctx_prev_grad_out, var_0, var_6, var_15);
            // ctx_prev_Mgrad_out[worldid, dofid] = mgrad                                         <L 1689>
            wp::array_store(var_ctx_prev_Mgrad_out, var_0, var_6, var_8);
            wp::assign(var_3, var_13);
            goto start_for_0;
        end_for_0:;
        // search_dot_tile = wp.tile(local_search_dot, preserve_type=True)                        <L 1691>
        var_17 = wp::tile<wp::float32>(var_3);
        // search_dot_sum = wp.tile_reduce(wp.add, search_dot_tile)                               <L 1692>
        var_18 = wp::tile_reduce(wp::add, var_17);
        // if tid == 0:                                                                           <L 1694>
        var_20 = (var_1 == var_19);
        if (var_20) {
            // ctx_search_dot_out[worldid] = search_dot_sum[0]                                    <L 1695>
            var_22 = wp::tile_extract(var_18, var_21);
            wp::array_store(var_ctx_search_dot_out, var_0, var_22);
        }
}



extern "C" __global__ void _zero_qfrc_constraint_sparse_6b3cbe94_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_state_changed_count_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_qfrc_constraint_out)
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
        bool* var_2;
        bool var_3;
        bool var_4;
        wp::int32* var_5;
        const wp::int32 var_6 = 0;
        bool var_7;
        wp::int32 var_8;
        const wp::float32 var_9 = 0.0;
        //---------
        // forward
        // def _zero_qfrc_constraint_sparse(                                                      <L 1826>
        // worldid, dofid = wp.tid()                                                              <L 1834>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 1836>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 1837>
            return;
        }
        var_4 = wp::load(var_2);
        // if state_changed_count_in[worldid] == 0:                                               <L 1839>
        var_5 = wp::address(var_state_changed_count_in, var_0);
        var_8 = wp::load(var_5);
        var_7 = (var_8 == var_6);
        if (var_7) {
            // return                                                                             <L 1840>
            return;
        }
        // qfrc_constraint_out[worldid, dofid] = 0.0                                              <L 1842>
        wp::array_store(var_qfrc_constraint_out, var_0, var_1, var_9);
}



extern "C" __global__ void _gather_J_dense_7f22c6f3_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::int32> var_dof_cdof_in,
    wp::array_t<wp::float32> var_J_in,
    wp::array_t<wp::float32> var_J_c_out)
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
        wp::int32* var_2;
        bool var_3;
        wp::int32 var_4;
        wp::shape_t* var_5;
        const wp::int32 var_6 = 1;
        wp::int32 var_7;
        wp::shape_t var_8;
        wp::range_t var_9;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        const wp::int32 var_14 = 0;
        bool var_15;
        wp::float32* var_16;
        wp::float32 var_17;
        //---------
        // forward
        // def _gather_J_dense(                                                                   <L 4002>
        // worldid, efcid = wp.tid()                                                              <L 4011>
        builtin_tid2d(var_0, var_1);
        // if efcid >= nefc_in[worldid]:                                                          <L 4012>
        var_2 = wp::address(var_nefc_in, var_0);
        var_4 = wp::load(var_2);
        var_3 = (var_1 >= var_4);
        if (var_3) {
            // return                                                                             <L 4013>
            return;
        }
        // nv = dof_cdof_in.shape[1]                                                              <L 4014>
        var_5 = &(var_dof_cdof_in.shape);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        // for j in range(nv):                                                                    <L 4015>
        var_9 = wp::range(var_7);
        start_for_1:;
            if (iter_cmp(var_9) == 0) goto end_for_1;
            var_10 = wp::iter_next(var_9);
            // cj = dof_cdof_in[worldid, j]                                                       <L 4016>
            var_11 = wp::address(var_dof_cdof_in, var_0, var_10);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
            // if cj >= 0:                                                                        <L 4017>
            var_15 = (var_12 >= var_14);
            if (var_15) {
                // J_c_out[worldid, efcid, cj] = J_in[worldid, efcid, j]                          <L 4018>
                var_16 = wp::address(var_J_in, var_0, var_1, var_10);
                var_17 = wp::load(var_16);
                wp::array_store(var_J_c_out, var_0, var_1, var_12, var_17);
            }
            goto start_for_1;
        end_for_1:;
}



extern "C" __global__ void _gather_M_sparse_8250c474_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_M_rownnz,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::int32> var_M_colind,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<wp::int32> var_dof_cdof_in,
    wp::array_t<wp::float32> var_M_c_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::range_t var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32* var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        const wp::int32 var_20 = 0;
        bool var_21;
        wp::float32* var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        //---------
        // forward
        // def _gather_M_sparse(                                                                  <L 3839>
        // worldid, i = wp.tid()                                                                  <L 3850>
        builtin_tid2d(var_0, var_1);
        // ci = dof_cdof_in[worldid, i]                                                           <L 3851>
        var_2 = wp::address(var_dof_cdof_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if ci < 0:                                                                             <L 3852>
        var_6 = (var_3 < var_5);
        if (var_6) {
            // return                                                                             <L 3853>
            return;
        }
        // rowadr = M_rowadr[i]                                                                   <L 3854>
        var_7 = wp::address(var_M_rowadr, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // for k in range(M_rownnz[i]):                                                           <L 3855>
        var_10 = wp::address(var_M_rownnz, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::range(var_12);
        start_for_1:;
            if (iter_cmp(var_11) == 0) goto end_for_1;
            var_13 = wp::iter_next(var_11);
            // adr = rowadr + k                                                                   <L 3856>
            var_14 = wp::add(var_8, var_13);
            // cj = dof_cdof_in[worldid, M_colind[adr]]                                           <L 3857>
            var_15 = wp::address(var_M_colind, var_14);
            var_17 = wp::load(var_15);
            var_16 = wp::address(var_dof_cdof_in, var_0, var_17);
            var_19 = wp::load(var_16);
            var_18 = wp::copy(var_19);
            // if cj >= 0:                                                                        <L 3858>
            var_21 = (var_18 >= var_20);
            if (var_21) {
                // val = M_in[worldid, adr]                                                       <L 3859>
                var_22 = wp::address(var_M_in, var_0, var_14);
                var_24 = wp::load(var_22);
                var_23 = wp::copy(var_24);
                // M_c_out[worldid, ci, cj] = val                                                 <L 3860>
                wp::array_store(var_M_c_out, var_0, var_3, var_18, var_23);
                // M_c_out[worldid, cj, ci] = val                                                 <L 3861>
                wp::array_store(var_M_c_out, var_0, var_18, var_3, var_23);
            }
            goto start_for_1;
        end_for_1:;
}



extern "C" __global__ void _solve_beta_zero_8deac91b_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float32> var_ctx_beta_num_out,
    wp::array_t<wp::float32> var_ctx_beta_den_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::float32 var_1 = 0.0;
        const wp::float32 var_2 = 0.0;
        //---------
        // forward
        // def _solve_beta_zero(                                                                  <L 3284>
        // worldid = wp.tid()                                                                     <L 3289>
        var_0 = builtin_tid1d();
        // ctx_beta_num_out[worldid] = 0.0                                                        <L 3290>
        wp::array_store(var_ctx_beta_num_out, var_0, var_1);
        // ctx_beta_den_out[worldid] = 0.0                                                        <L 3291>
        wp::array_store(var_ctx_beta_den_out, var_0, var_2);
}



extern "C" __global__ void _solve_beta_accumulate_95bec0bb_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_ctx_grad_in,
    wp::array_t<wp::float32> var_ctx_Mgrad_in,
    wp::array_t<wp::float32> var_ctx_prev_grad_in,
    wp::array_t<wp::float32> var_ctx_prev_Mgrad_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_beta_num_out,
    wp::array_t<wp::float32> var_ctx_beta_den_out)
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
        bool* var_2;
        bool var_3;
        bool var_4;
        wp::float32* var_5;
        wp::float32 var_6;
        wp::float32 var_7;
        wp::float32* var_8;
        wp::float32* var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        //---------
        // forward
        // def _solve_beta_accumulate(                                                            <L 3336>
        // worldid, dofid = wp.tid()                                                              <L 3347>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 3349>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 3350>
            return;
        }
        var_4 = wp::load(var_2);
        // prev_Mgrad = ctx_prev_Mgrad_in[worldid, dofid]                                         <L 3352>
        var_5 = wp::address(var_ctx_prev_Mgrad_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // num = ctx_grad_in[worldid, dofid] * (ctx_Mgrad_in[worldid, dofid] - prev_Mgrad)        <L 3353>
        var_8 = wp::address(var_ctx_grad_in, var_0, var_1);
        var_9 = wp::address(var_ctx_Mgrad_in, var_0, var_1);
        var_11 = wp::load(var_9);
        var_10 = wp::sub(var_11, var_6);
        var_13 = wp::load(var_8);
        var_12 = wp::mul(var_13, var_10);
        // den = ctx_prev_grad_in[worldid, dofid] * prev_Mgrad                                    <L 3354>
        var_14 = wp::address(var_ctx_prev_grad_in, var_0, var_1);
        var_16 = wp::load(var_14);
        var_15 = wp::mul(var_16, var_6);
        // wp.atomic_add(ctx_beta_num_out, worldid, num)                                          <L 3355>
        var_17 = wp::atomic_add(var_ctx_beta_num_out, var_0, var_12);
        // wp.atomic_add(ctx_beta_den_out, worldid, den)                                          <L 3356>
        var_18 = wp::atomic_add(var_ctx_beta_den_out, var_0, var_15);
}



extern "C" __global__ void _solve_search_update_cg_tiled_9a891d26_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_ctx_grad_in,
    wp::array_t<wp::float32> var_ctx_Mgrad_in,
    wp::array_t<wp::float32> var_ctx_search_in,
    wp::array_t<wp::float32> var_ctx_beta_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_search_out,
    wp::array_t<wp::float32> var_ctx_search_dot_out,
    wp::array_t<wp::float32> var_ctx_prev_grad_out,
    wp::array_t<wp::float32> var_ctx_prev_Mgrad_out)
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
        bool* var_2;
        bool var_3;
        bool var_4;
        const wp::float32 var_5 = 0.0;
        wp::float32 var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::range_t var_11;
        wp::int32 var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        const wp::float32 var_16 = -1.0;
        wp::float32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::float32* var_24;
        wp::float32 var_25;
        const bool var_26 = true;
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>> var_27 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>>{};
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_28 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        const wp::int32 var_29 = 0;
        bool var_30;
        const wp::int32 var_31 = 0;
        wp::float32 var_32;
        //---------
        // forward
        // def _solve_search_update_cg_tiled(                                                     <L 3360>
        // worldid, tid = wp.tid()                                                                <L 3375>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 3377>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 3378>
            return;
        }
        var_4 = wp::load(var_2);
        // local_search_dot = float(0.0)                                                          <L 3380>
        var_6 = wp::float(var_5);
        // BLOCK_DIM = wp.block_dim()                                                             <L 3381>
        var_7 = builtin_block_dim();
        // beta = ctx_beta_in[worldid]                                                            <L 3382>
        var_8 = wp::address(var_ctx_beta_in, var_0);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // for dofid in range(tid, nv, BLOCK_DIM):                                                <L 3384>
        var_11 = wp::range(var_1, var_nv, var_7);
        start_for_1:;
            if (iter_cmp(var_11) == 0) goto end_for_1;
            var_12 = wp::iter_next(var_11);
            // mgrad = ctx_Mgrad_in[worldid, dofid]                                               <L 3385>
            var_13 = wp::address(var_ctx_Mgrad_in, var_0, var_12);
            var_15 = wp::load(var_13);
            var_14 = wp::copy(var_15);
            // search = -1.0 * mgrad + beta * ctx_search_in[worldid, dofid]                       <L 3386>
            var_17 = wp::mul(var_16, var_14);
            var_18 = wp::address(var_ctx_search_in, var_0, var_12);
            var_20 = wp::load(var_18);
            var_19 = wp::mul(var_9, var_20);
            var_21 = wp::add(var_17, var_19);
            // ctx_search_out[worldid, dofid] = search                                            <L 3388>
            wp::array_store(var_ctx_search_out, var_0, var_12, var_21);
            // local_search_dot += search * search                                                <L 3389>
            var_22 = wp::mul(var_21, var_21);
            var_23 = wp::add(var_6, var_22);
            // ctx_prev_grad_out[worldid, dofid] = ctx_grad_in[worldid, dofid]                    <L 3391>
            var_24 = wp::address(var_ctx_grad_in, var_0, var_12);
            var_25 = wp::load(var_24);
            wp::array_store(var_ctx_prev_grad_out, var_0, var_12, var_25);
            // ctx_prev_Mgrad_out[worldid, dofid] = mgrad                                         <L 3392>
            wp::array_store(var_ctx_prev_Mgrad_out, var_0, var_12, var_14);
            wp::assign(var_6, var_23);
            goto start_for_1;
        end_for_1:;
        // search_dot_tile = wp.tile(local_search_dot, preserve_type=True)                        <L 3394>
        var_27 = wp::tile<wp::float32>(var_6);
        // search_dot_sum = wp.tile_reduce(wp.add, search_dot_tile)                               <L 3395>
        var_28 = wp::tile_reduce(wp::add, var_27);
        // if tid == 0:                                                                           <L 3397>
        var_30 = (var_1 == var_29);
        if (var_30) {
            // ctx_search_dot_out[worldid] = search_dot_sum[0]                                    <L 3398>
            var_32 = wp::tile_extract(var_28, var_31);
            wp::array_store(var_ctx_search_dot_out, var_0, var_32);
        }
}



extern "C" __global__ void _update_gradient_h_incremental_9c0c45bc_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::int32> var_efc_state_in,
    wp::array_t<wp::int32> var_quad_changed_ids_in,
    wp::array_t<wp::int32> var_quad_changed_count_in,
    wp::array_t<wp::float32> var_ctx_h_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        const wp::int32 var_7 = 1;
        const wp::int32 var_8 = 8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::int32 var_13;
        const wp::int32 var_14 = 1;
        wp::int32 var_15;
        const wp::int32 var_16 = 2;
        wp::int32 var_17;
        const wp::int32 var_18 = 1;
        wp::int32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 2;
        wp::int32 var_22;
        wp::int32 var_23;
        const wp::float32 var_24 = 0.0;
        wp::float32 var_25;
        wp::range_t var_26;
        wp::int32 var_27;
        wp::int32* var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        wp::float32* var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        const wp::float32 var_34 = 0.0;
        bool var_35;
        wp::float32* var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        const wp::float32 var_39 = 0.0;
        bool var_40;
        wp::float32* var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::int32* var_44;
        const wp::int32 var_45 = 1;
        bool var_46;
        wp::int32 var_47;
        wp::float32 var_48;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        wp::float32 var_54;
        const wp::float32 var_55 = 0.0;
        bool var_56;
        wp::float32 var_57;
        //---------
        // forward
        // def _update_gradient_h_incremental(                                                    <L 1951>
        // worldid, elementid = wp.tid()                                                          <L 1967>
        builtin_tid2d(var_0, var_1);
        // n_changes = quad_changed_count_in[worldid]                                             <L 1969>
        var_2 = wp::address(var_quad_changed_count_in, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if n_changes == 0:                                                                     <L 1970>
        var_6 = (var_3 == var_5);
        if (var_6) {
            // return                                                                             <L 1971>
            return;
        }
        // col = (int(wp.sqrt(float(1 + 8 * elementid))) - 1) // 2                                <L 1974>
        var_9 = wp::mul(var_8, var_1);
        var_10 = wp::add(var_7, var_9);
        var_11 = wp::float(var_10);
        var_12 = wp::sqrt(var_11);
        var_13 = wp::int(var_12);
        var_15 = wp::sub(var_13, var_14);
        var_17 = wp::floordiv(var_15, var_16);
        // row = elementid - (col * (col + 1)) // 2                                               <L 1975>
        var_19 = wp::add(var_17, var_18);
        var_20 = wp::mul(var_17, var_19);
        var_22 = wp::floordiv(var_20, var_21);
        var_23 = wp::sub(var_1, var_22);
        // delta = float(0.0)                                                                     <L 1977>
        var_25 = wp::float(var_24);
        // for change_idx in range(n_changes):                                                    <L 1978>
        var_26 = wp::range(var_3);
        start_for_1:;
            if (iter_cmp(var_26) == 0) goto end_for_1;
            var_27 = wp::iter_next(var_26);
            // efcid = quad_changed_ids_in[worldid, change_idx]                                   <L 1979>
            var_28 = wp::address(var_quad_changed_ids_in, var_0, var_27);
            var_30 = wp::load(var_28);
            var_29 = wp::copy(var_30);
            // Jrow = efc_J_in[worldid, efcid, row]                                               <L 1980>
            var_31 = wp::address(var_efc_J_in, var_0, var_29, var_23);
            var_33 = wp::load(var_31);
            var_32 = wp::copy(var_33);
            // if Jrow == 0.0:                                                                    <L 1981>
            var_35 = (var_32 == var_34);
            if (var_35) {
                // continue                                                                       <L 1982>
                goto start_for_1;
            }
            // Jcol = efc_J_in[worldid, efcid, col]                                               <L 1983>
            var_36 = wp::address(var_efc_J_in, var_0, var_29, var_17);
            var_38 = wp::load(var_36);
            var_37 = wp::copy(var_38);
            // if Jcol == 0.0:                                                                    <L 1984>
            var_40 = (var_37 == var_39);
            if (var_40) {
                // continue                                                                       <L 1985>
                goto start_for_1;
            }
            // D = efc_D_in[worldid, efcid]                                                       <L 1987>
            var_41 = wp::address(var_efc_D_in, var_0, var_29);
            var_43 = wp::load(var_41);
            var_42 = wp::copy(var_43);
            // if efc_state_in[worldid, efcid] == types.ConstraintState.QUADRATIC.value:          <L 1988>
            var_44 = wp::address(var_efc_state_in, var_0, var_29);
            var_47 = wp::load(var_44);
            var_46 = (var_47 == var_45);
            if (var_46) {
                // delta += D * Jrow * Jcol                                                       <L 1989>
                var_48 = wp::mul(var_42, var_32);
                var_49 = wp::mul(var_48, var_37);
                var_50 = wp::add(var_25, var_49);
            }
            if (!var_46) {
                // delta -= D * Jrow * Jcol                                                       <L 1991>
                var_51 = wp::mul(var_42, var_32);
                var_52 = wp::mul(var_51, var_37);
                var_53 = wp::sub(var_25, var_52);
            }
            var_54 = wp::where(var_46, var_50, var_53);
            wp::assign(var_25, var_54);
            goto start_for_1;
        end_for_1:;
        // if delta != 0.0:                                                                       <L 1993>
        var_56 = (var_25 != var_55);
        if (var_56) {
            // ctx_h_out[worldid, row, col] += delta                                              <L 1994>
            var_57 = wp::atomic_add(var_ctx_h_out, var_0, var_23, var_17, var_25);
        }
}



extern "C" __global__ void _gather_dof_vecs_compact_a4bcec76_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_qacc_warmstart_in,
    wp::array_t<wp::float32> var_qfrc_smooth_in,
    wp::array_t<wp::float32> var_qacc_smooth_in,
    wp::array_t<wp::int32> var_cdof_dof_in,
    wp::array_t<wp::float32> var_qfrc_smooth_c_out,
    wp::array_t<wp::float32> var_qacc_smooth_c_out,
    wp::array_t<wp::float32> var_qacc_warmstart_c_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32* var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        const wp::float32 var_13 = 0.0;
        const wp::float32 var_14 = 0.0;
        const wp::float32 var_15 = 0.0;
        //---------
        // forward
        // def _gather_dof_vecs_compact(                                                          <L 3933>
        // worldid, ci = wp.tid()                                                                 <L 3944>
        builtin_tid2d(var_0, var_1);
        // dof = cdof_dof_in[worldid, ci]                                                         <L 3945>
        var_2 = wp::address(var_cdof_dof_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if dof >= 0:                                                                           <L 3946>
        var_6 = (var_3 >= var_5);
        if (var_6) {
            // qfrc_smooth_c_out[worldid, ci] = qfrc_smooth_in[worldid, dof]                      <L 3947>
            var_7 = wp::address(var_qfrc_smooth_in, var_0, var_3);
            var_8 = wp::load(var_7);
            wp::array_store(var_qfrc_smooth_c_out, var_0, var_1, var_8);
            // qacc_smooth_c_out[worldid, ci] = qacc_smooth_in[worldid, dof]                      <L 3948>
            var_9 = wp::address(var_qacc_smooth_in, var_0, var_3);
            var_10 = wp::load(var_9);
            wp::array_store(var_qacc_smooth_c_out, var_0, var_1, var_10);
            // qacc_warmstart_c_out[worldid, ci] = qacc_warmstart_in[worldid, dof]                <L 3949>
            var_11 = wp::address(var_qacc_warmstart_in, var_0, var_3);
            var_12 = wp::load(var_11);
            wp::array_store(var_qacc_warmstart_c_out, var_0, var_1, var_12);
        }
        if (!var_6) {
            // qfrc_smooth_c_out[worldid, ci] = 0.0                                               <L 3951>
            wp::array_store(var_qfrc_smooth_c_out, var_0, var_1, var_13);
            // qacc_smooth_c_out[worldid, ci] = 0.0                                               <L 3952>
            wp::array_store(var_qacc_smooth_c_out, var_0, var_1, var_14);
            // qacc_warmstart_c_out[worldid, ci] = 0.0                                            <L 3953>
            wp::array_store(var_qacc_warmstart_c_out, var_0, var_1, var_15);
        }
}



extern "C" __global__ void _update_gradient_grad_tiled_c0425fb5_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_qfrc_smooth_in,
    wp::array_t<wp::float32> var_qfrc_constraint_in,
    wp::array_t<wp::float32> var_efc_Ma_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_grad_out,
    wp::array_t<wp::float32> var_ctx_grad_dot_out)
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
        bool* var_2;
        bool var_3;
        bool var_4;
        const wp::float32 var_5 = 0.0;
        wp::float32 var_6;
        wp::int32 var_7;
        wp::range_t var_8;
        wp::int32 var_9;
        wp::float32* var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::float32* var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        const bool var_20 = true;
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>> var_21 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>>{};
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_22 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        const wp::int32 var_23 = 0;
        bool var_24;
        const wp::int32 var_25 = 0;
        wp::float32 var_26;
        //---------
        // forward
        // def _update_gradient_grad_tiled(                                                       <L 2203>
        // worldid, tid = wp.tid()                                                                <L 2216>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 2218>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 2219>
            return;
        }
        var_4 = wp::load(var_2);
        // local_grad_dot = float(0.0)                                                            <L 2221>
        var_6 = wp::float(var_5);
        // BLOCK_DIM = wp.block_dim()                                                             <L 2222>
        var_7 = builtin_block_dim();
        // for dofid in range(tid, nv, BLOCK_DIM):                                                <L 2224>
        var_8 = wp::range(var_1, var_nv, var_7);
        start_for_1:;
            if (iter_cmp(var_8) == 0) goto end_for_1;
            var_9 = wp::iter_next(var_8);
            // grad = efc_Ma_in[worldid, dofid] - qfrc_smooth_in[worldid, dofid] - qfrc_constraint_in[worldid, dofid]       <L 2225>
            var_10 = wp::address(var_efc_Ma_in, var_0, var_9);
            var_11 = wp::address(var_qfrc_smooth_in, var_0, var_9);
            var_13 = wp::load(var_10);
            var_14 = wp::load(var_11);
            var_12 = wp::sub(var_13, var_14);
            var_15 = wp::address(var_qfrc_constraint_in, var_0, var_9);
            var_17 = wp::load(var_15);
            var_16 = wp::sub(var_12, var_17);
            // ctx_grad_out[worldid, dofid] = grad                                                <L 2226>
            wp::array_store(var_ctx_grad_out, var_0, var_9, var_16);
            // local_grad_dot += grad * grad                                                      <L 2227>
            var_18 = wp::mul(var_16, var_16);
            var_19 = wp::add(var_6, var_18);
            wp::assign(var_6, var_19);
            goto start_for_1;
        end_for_1:;
        // grad_dot_tile = wp.tile(local_grad_dot, preserve_type=True)                            <L 2229>
        var_21 = wp::tile<wp::float32>(var_6);
        // grad_dot_sum = wp.tile_reduce(wp.add, grad_dot_tile)                                   <L 2230>
        var_22 = wp::tile_reduce(wp::add, var_21);
        // if tid == 0:                                                                           <L 2232>
        var_24 = (var_1 == var_23);
        if (var_24) {
            // ctx_grad_dot_out[worldid] = grad_dot_sum[0]                                        <L 2233>
            var_26 = wp::tile_extract(var_22, var_25);
            wp::array_store(var_ctx_grad_dot_out, var_0, var_26);
        }
}



extern "C" __global__ void _padding_h_df4cdf7e_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nv,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_h_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
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
        wp::int32 var_5;
        const wp::float32 var_6 = 1.0;
        //---------
        // forward
        // def _padding_h(nv: int, ctx_done_in: wp.array[bool], ctx_h_out: wp.array3d[float]):       <L 2725>
        // worldid, elementid = wp.tid()                                                          <L 2726>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 2728>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 2729>
            continue;
        }
        var_4 = wp::load(var_2);
        // dofid = nv + elementid                                                                 <L 2731>
        var_5 = wp::add(var_nv, var_1);
        // ctx_h_out[worldid, dofid, dofid] = 1.0                                                 <L 2732>
        wp::array_store(var_ctx_h_out, var_0, var_5, var_5, var_6);
    }
}



extern "C" __global__ void _linesearch_zero_jv_ef61d94f_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<bool> var_skip_in,
    wp::array_t<wp::float32> var_ctx_jv_out)
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
        wp::int32* var_2;
        bool var_3;
        wp::int32 var_4;
        bool* var_5;
        bool var_6;
        bool var_7;
        const wp::float32 var_8 = 0.0;
        //---------
        // forward
        // def _linesearch_zero_jv(                                                               <L 1417>
        // worldid, efcid = wp.tid()                                                              <L 1425>
        builtin_tid2d(var_0, var_1);
        // if efcid >= nefc_in[worldid]:                                                          <L 1427>
        var_2 = wp::address(var_nefc_in, var_0);
        var_4 = wp::load(var_2);
        var_3 = (var_1 >= var_4);
        if (var_3) {
            // return                                                                             <L 1428>
            return;
        }
        // if skip_in[worldid]:                                                                   <L 1430>
        var_5 = wp::address(var_skip_in, var_0);
        var_6 = wp::load(var_5);
        if (var_6) {
            // return                                                                             <L 1431>
            return;
        }
        var_7 = wp::load(var_5);
        // ctx_jv_out[worldid, efcid] = 0.0                                                       <L 1433>
        wp::array_store(var_ctx_jv_out, var_0, var_1, var_8);
}



extern "C" __global__ void _update_gradient_JTCJ_dense_f52c2ac2_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_impratio_invsqrt,
    wp::array_t<wp::int32> var_dof_tri_row,
    wp::array_t<wp::int32> var_dof_tri_col,
    wp::array_t<wp::float32> var_contact_dist_in,
    wp::array_t<wp::float32> var_contact_includemargin_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_contact_worldid_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::int32> var_efc_state_in,
    wp::int32 var_naconmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::float32> var_ctx_Jaref_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::int32 var_nblocks_perblock,
    wp::int32 var_dim_block,
    wp::array_t<wp::float32> var_ctx_h_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::range_t var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        const wp::int32 var_12 = 0;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        bool var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        bool* var_20;
        bool var_21;
        bool var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 1;
        bool var_27;
        wp::float32* var_28;
        wp::float32* var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        const wp::float32 var_33 = 0.0;
        bool var_34;
        const wp::int32 var_35 = 0;
        wp::int32* var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        const wp::int32 var_39 = 0;
        bool var_40;
        wp::int32* var_41;
        const wp::int32 var_42 = 4;
        bool var_43;
        wp::int32 var_44;
        wp::vec_t<5, wp::float32>* var_45;
        wp::vec_t<5, wp::float32> var_46;
        wp::vec_t<5, wp::float32> var_47;
        const wp::int32 var_48 = 0;
        wp::float32 var_49;
        wp::shape_t* var_50;
        const wp::int32 var_51 = 0;
        wp::int32 var_52;
        wp::shape_t var_53;
        wp::int32 var_54;
        wp::float32* var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        wp::float32 var_58;
        wp::float32* var_59;
        const wp::float32 var_60 = 1.0;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::float32 var_64;
        const wp::float32 var_65 = 0.0;
        bool var_66;
        wp::float32* var_67;
        wp::float32 var_68;
        wp::float32 var_69;
        wp::float32* var_70;
        wp::float32 var_71;
        wp::float32 var_72;
        wp::float32* var_73;
        wp::float32 var_74;
        wp::float32 var_75;
        const wp::float32 var_76 = 0.0;
        wp::float32 var_77;
        const wp::float32 var_78 = 0.0;
        wp::float32 var_79;
        const wp::float32 var_80 = 0.0;
        wp::float32 var_81;
        const wp::float32 var_82 = 0.0;
        wp::float32 var_83;
        const wp::int32 var_84 = 1;
        wp::range_t var_85;
        wp::int32 var_86;
        wp::int32* var_87;
        wp::int32 var_88;
        wp::int32 var_89;
        const wp::int32 var_90 = 0;
        bool var_91;
        const wp::int32 var_92 = 1;
        wp::int32 var_93;
        wp::float32 var_94;
        wp::float32* var_95;
        wp::float32 var_96;
        wp::float32 var_97;
        wp::float32* var_98;
        wp::float32 var_99;
        wp::float32 var_100;
        wp::float32* var_101;
        wp::float32 var_102;
        wp::float32 var_103;
        wp::float32 var_104;
        wp::float32 var_105;
        wp::float32 var_106;
        wp::float32 var_107;
        wp::float32 var_108;
        wp::float32 var_109;
        wp::float32 var_110;
        wp::float32 var_111;
        wp::float32 var_112;
        wp::float32 var_113;
        wp::float32 var_114;
        wp::float32 var_115;
        wp::float32 var_116;
        const wp::float32 var_117 = 1e-15;
        const wp::float32 var_118 = 1e-15;
        wp::float32 var_119;
        wp::float32 var_120;
        wp::float32 var_121;
        const wp::float32 var_122 = 1e-15;
        const wp::float32 var_123 = 1e-15;
        wp::float32 var_124;
        wp::float32 var_125;
        wp::float32 var_126;
        wp::float32 var_127;
        wp::float32 var_128;
        wp::float32 var_129;
        wp::float32 var_130;
        wp::float32 var_131;
        //---------
        // forward
        // def _update_gradient_JTCJ_dense(                                                       <L 2466>
        // conid_start, elementid = wp.tid()                                                      <L 2491>
        builtin_tid2d(var_0, var_1);
        // dof1id = dof_tri_row[elementid]                                                        <L 2493>
        var_2 = wp::address(var_dof_tri_row, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // dof2id = dof_tri_col[elementid]                                                        <L 2494>
        var_5 = wp::address(var_dof_tri_col, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // for i in range(nblocks_perblock):                                                      <L 2496>
        var_8 = wp::range(var_nblocks_perblock);
        start_for_0:;
            if (iter_cmp(var_8) == 0) goto end_for_0;
            var_9 = wp::iter_next(var_8);
            // conid = conid_start + i * dim_block                                                <L 2497>
            var_10 = wp::mul(var_9, var_dim_block);
            var_11 = wp::add(var_0, var_10);
            // if conid >= min(nacon_in[0], naconmax_in):                                         <L 2499>
            var_13 = wp::address(var_nacon_in, var_12);
            var_15 = wp::load(var_13);
            var_14 = wp::min(var_15, var_naconmax_in);
            var_16 = (var_11 >= var_14);
            if (var_16) {
                // return                                                                         <L 2500>
                return;
            }
            // worldid = contact_worldid_in[conid]                                                <L 2502>
            var_17 = wp::address(var_contact_worldid_in, var_11);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // if ctx_done_in[worldid]:                                                           <L 2503>
            var_20 = wp::address(var_ctx_done_in, var_18);
            var_21 = wp::load(var_20);
            if (var_21) {
                // continue                                                                       <L 2504>
                goto start_for_0;
            }
            var_22 = wp::load(var_20);
            // condim = contact_dim_in[conid]                                                     <L 2506>
            var_23 = wp::address(var_contact_dim_in, var_11);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // if condim == 1:                                                                    <L 2508>
            var_27 = (var_24 == var_26);
            if (var_27) {
                // continue                                                                       <L 2509>
                goto start_for_0;
            }
            // if contact_dist_in[conid] - contact_includemargin_in[conid] >= 0.0:                <L 2512>
            var_28 = wp::address(var_contact_dist_in, var_11);
            var_29 = wp::address(var_contact_includemargin_in, var_11);
            var_31 = wp::load(var_28);
            var_32 = wp::load(var_29);
            var_30 = wp::sub(var_31, var_32);
            var_34 = (var_30 >= var_33);
            if (var_34) {
                // continue                                                                       <L 2513>
                goto start_for_0;
            }
            // efcid0 = contact_efc_address_in[conid, 0]                                          <L 2515>
            var_36 = wp::address(var_contact_efc_address_in, var_11, var_35);
            var_38 = wp::load(var_36);
            var_37 = wp::copy(var_38);
            // if efcid0 < 0:                                                                     <L 2516>
            var_40 = (var_37 < var_39);
            if (var_40) {
                // continue                                                                       <L 2517>
                goto start_for_0;
            }
            // if efc_state_in[worldid, efcid0] != types.ConstraintState.CONE:                    <L 2518>
            var_41 = wp::address(var_efc_state_in, var_18, var_37);
            var_44 = wp::load(var_41);
            var_43 = (var_44 != var_42);
            if (var_43) {
                // continue                                                                       <L 2519>
                goto start_for_0;
            }
            // fri = contact_friction_in[conid]                                                   <L 2521>
            var_45 = wp::address(var_contact_friction_in, var_11);
            var_47 = wp::load(var_45);
            var_46 = wp::copy(var_47);
            // mu = fri[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]        <L 2522>
            var_49 = wp::extract(var_46, var_48);
            var_50 = &(var_opt_impratio_invsqrt.shape);
            var_53 = wp::load(var_50);
            var_52 = wp::extract(var_53, var_51);
            var_54 = wp::mod(var_18, var_52);
            var_55 = wp::address(var_opt_impratio_invsqrt, var_54);
            var_57 = wp::load(var_55);
            var_56 = wp::mul(var_49, var_57);
            // mu2 = mu * mu                                                                      <L 2524>
            var_58 = wp::mul(var_56, var_56);
            // dm = math.safe_div(efc_D_in[worldid, efcid0], mu2 * (1.0 + mu2))                   <L 2525>
            var_59 = wp::address(var_efc_D_in, var_18, var_37);
            var_61 = wp::add(var_60, var_58);
            var_62 = wp::mul(var_58, var_61);
            var_64 = wp::load(var_59);
            var_63 = safe_div_2(var_64, var_62);
            // if dm == 0.0:                                                                      <L 2527>
            var_66 = (var_63 == var_65);
            if (var_66) {
                // continue                                                                       <L 2528>
                goto start_for_0;
            }
            // n = ctx_Jaref_in[worldid, efcid0] * mu                                             <L 2530>
            var_67 = wp::address(var_ctx_Jaref_in, var_18, var_37);
            var_69 = wp::load(var_67);
            var_68 = wp::mul(var_69, var_56);
            // z01 = mu * efc_J_in[worldid, efcid0, dof1id]                                       <L 2531>
            var_70 = wp::address(var_efc_J_in, var_18, var_37, var_3);
            var_72 = wp::load(var_70);
            var_71 = wp::mul(var_56, var_72);
            // z02 = mu * efc_J_in[worldid, efcid0, dof2id]                                       <L 2532>
            var_73 = wp::address(var_efc_J_in, var_18, var_37, var_6);
            var_75 = wp::load(var_73);
            var_74 = wp::mul(var_56, var_75);
            // tt = float(0.0)                                                                    <L 2533>
            var_77 = wp::float(var_76);
            // projection1 = float(0.0)                                                           <L 2534>
            var_79 = wp::float(var_78);
            // projection2 = float(0.0)                                                           <L 2535>
            var_81 = wp::float(var_80);
            // tangent_dot = float(0.0)                                                           <L 2536>
            var_83 = wp::float(var_82);
            // for dim in range(1, condim):                                                       <L 2537>
            var_85 = wp::range(var_84, var_24);
            start_for_3:;
                if (iter_cmp(var_85) == 0) goto end_for_3;
                var_86 = wp::iter_next(var_85);
                // efcid = contact_efc_address_in[conid, dim]                                     <L 2538>
                var_87 = wp::address(var_contact_efc_address_in, var_11, var_86);
                var_89 = wp::load(var_87);
                var_88 = wp::copy(var_89);
                // if efcid >= 0:                                                                 <L 2539>
                var_91 = (var_88 >= var_90);
                if (var_91) {
                    // scale = fri[dim - 1]                                                       <L 2540>
                    var_93 = wp::sub(var_86, var_92);
                    var_94 = wp::extract(var_46, var_93);
                    // u = ctx_Jaref_in[worldid, efcid] * scale                                   <L 2541>
                    var_95 = wp::address(var_ctx_Jaref_in, var_18, var_88);
                    var_97 = wp::load(var_95);
                    var_96 = wp::mul(var_97, var_94);
                    // z1 = scale * efc_J_in[worldid, efcid, dof1id]                              <L 2542>
                    var_98 = wp::address(var_efc_J_in, var_18, var_88, var_3);
                    var_100 = wp::load(var_98);
                    var_99 = wp::mul(var_94, var_100);
                    // z2 = scale * efc_J_in[worldid, efcid, dof2id]                              <L 2543>
                    var_101 = wp::address(var_efc_J_in, var_18, var_88, var_6);
                    var_103 = wp::load(var_101);
                    var_102 = wp::mul(var_94, var_103);
                    // tt += u * u                                                                <L 2544>
                    var_104 = wp::mul(var_96, var_96);
                    var_105 = wp::add(var_77, var_104);
                    // projection1 += u * z1                                                      <L 2545>
                    var_106 = wp::mul(var_96, var_99);
                    var_107 = wp::add(var_79, var_106);
                    // projection2 += u * z2                                                      <L 2546>
                    var_108 = wp::mul(var_96, var_102);
                    var_109 = wp::add(var_81, var_108);
                    // tangent_dot += z1 * z2                                                     <L 2547>
                    var_110 = wp::mul(var_99, var_102);
                    var_111 = wp::add(var_83, var_110);
                }
                var_112 = wp::where(var_91, var_105, var_77);
                var_113 = wp::where(var_91, var_107, var_79);
                var_114 = wp::where(var_91, var_109, var_81);
                var_115 = wp::where(var_91, var_111, var_83);
                wp::assign(var_77, var_112);
                wp::assign(var_79, var_113);
                wp::assign(var_81, var_114);
                wp::assign(var_83, var_115);
                goto start_for_3;
            end_for_3:;
            // t = wp.max(wp.sqrt(tt), types.MJ_MINVAL)                                           <L 2549>
            var_116 = wp::sqrt(var_77);
            var_119 = wp::max(var_116, var_118);
            // ttt = wp.max(t * t * t, types.MJ_MINVAL)                                           <L 2550>
            var_120 = wp::mul(var_119, var_119);
            var_121 = wp::mul(var_120, var_119);
            var_124 = wp::max(var_121, var_123);
            // mu_tinv = math.safe_div(mu, t)                                                     <L 2551>
            var_125 = safe_div_2(var_56, var_119);
            // h = _elliptic_hessian_entry_from_projections(                                      <L 2552>
            // dm,                                                                                <L 2553>
            // mu_tinv,                                                                           <L 2554>
            // mu * math.safe_div(n, ttt),                                                        <L 2555>
            var_126 = safe_div_2(var_68, var_124);
            var_127 = wp::mul(var_56, var_126);
            // mu2 - n * mu_tinv,                                                                 <L 2556>
            var_128 = wp::mul(var_68, var_125);
            var_129 = wp::sub(var_58, var_128);
            // z01,                                                                               <L 2557>
            // z02,                                                                               <L 2558>
            // projection1,                                                                       <L 2559>
            // projection2,                                                                       <L 2560>
            // tangent_dot,                                                                       <L 2561>
            var_130 = _elliptic_hessian_entry_from_projections_0(var_63, var_125, var_127, var_129, var_71, var_74, var_79, var_81, var_83);
            // ctx_h_out[worldid, dof1id, dof2id] += h                                            <L 2564>
            var_131 = wp::atomic_add(var_ctx_h_out, var_18, var_3, var_6, var_130);
            goto start_for_0;
        end_for_0:;
}



extern "C" __global__ void _mul_m_sparse_compact_fbf15af2_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_M_mulm_rowadr,
    wp::array_t<wp::int32> var_M_mulm_col,
    wp::array_t<wp::int32> var_M_mulm_madr,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<wp::int32> var_dof_cdof_in,
    wp::array_t<wp::int32> var_cdof_dof_in,
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
        bool* var_2;
        bool var_3;
        bool var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        const wp::float32 var_10 = 0.0;
        const wp::float32 var_11 = 0.0;
        wp::float32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        const wp::int32 var_16 = 1;
        wp::int32 var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::range_t var_21;
        wp::int32 var_22;
        wp::int32* var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        const wp::int32 var_28 = 0;
        bool var_29;
        wp::int32* var_30;
        wp::float32* var_31;
        wp::int32 var_32;
        wp::float32* var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        //---------
        // forward
        // def _mul_m_sparse_compact(                                                             <L 3771>
        // worldid, ci = wp.tid()                                                                 <L 3787>
        builtin_tid2d(var_0, var_1);
        // if skip[worldid]:                                                                      <L 3789>
        var_2 = wp::address(var_skip, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 3790>
            return;
        }
        var_4 = wp::load(var_2);
        // dof = cdof_dof_in[worldid, ci]                                                         <L 3792>
        var_5 = wp::address(var_cdof_dof_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if dof < 0:                                                                            <L 3793>
        var_9 = (var_6 < var_8);
        if (var_9) {
            // res[worldid, ci] = 0.0                                                             <L 3794>
            wp::array_store(var_res, var_0, var_1, var_10);
            // return                                                                             <L 3795>
            return;
        }
        // acc = float(0.0)                                                                       <L 3797>
        var_12 = wp::float(var_11);
        // start = M_mulm_rowadr[dof]                                                             <L 3798>
        var_13 = wp::address(var_M_mulm_rowadr, var_6);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // end = M_mulm_rowadr[dof + 1]                                                           <L 3799>
        var_17 = wp::add(var_6, var_16);
        var_18 = wp::address(var_M_mulm_rowadr, var_17);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // for k in range(start, end):                                                            <L 3800>
        var_21 = wp::range(var_14, var_19);
        start_for_2:;
            if (iter_cmp(var_21) == 0) goto end_for_2;
            var_22 = wp::iter_next(var_21);
            // cj = dof_cdof_in[worldid, M_mulm_col[k]]                                           <L 3803>
            var_23 = wp::address(var_M_mulm_col, var_22);
            var_25 = wp::load(var_23);
            var_24 = wp::address(var_dof_cdof_in, var_0, var_25);
            var_27 = wp::load(var_24);
            var_26 = wp::copy(var_27);
            // if cj >= 0:                                                                        <L 3804>
            var_29 = (var_26 >= var_28);
            if (var_29) {
                // acc += M_in[worldid, M_mulm_madr[k]] * vec[worldid, cj]                        <L 3805>
                var_30 = wp::address(var_M_mulm_madr, var_22);
                var_32 = wp::load(var_30);
                var_31 = wp::address(var_M_in, var_0, var_32);
                var_33 = wp::address(var_vec, var_0, var_26);
                var_35 = wp::load(var_31);
                var_36 = wp::load(var_33);
                var_34 = wp::mul(var_35, var_36);
                var_37 = wp::add(var_12, var_34);
            }
            var_38 = wp::where(var_29, var_37, var_12);
            wp::assign(var_12, var_38);
            goto start_for_2;
        end_for_2:;
        // res[worldid, ci] = acc                                                                 <L 3806>
        wp::array_store(var_res, var_0, var_1, var_12);
}

