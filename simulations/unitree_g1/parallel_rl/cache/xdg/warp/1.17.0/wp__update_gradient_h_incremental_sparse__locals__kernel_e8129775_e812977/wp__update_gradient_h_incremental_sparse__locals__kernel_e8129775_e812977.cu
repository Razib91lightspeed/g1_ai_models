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



extern "C" __global__ void _update_gradient_h_incremental_sparse__locals__kernel_46c5404e_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::array_t<wp::int32> var_efc_J_colind_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::int32> var_efc_state_in,
    wp::array_t<wp::int32> var_dof_cdof_in,
    wp::array_t<wp::int32> var_quad_changed_ids_in,
    wp::array_t<wp::int32> var_quad_changed_count_in,
    wp::int32 var_slots_per_world,
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
        wp::int32 var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::range_t var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        const wp::float32 var_14 = 0.0;
        wp::float32 var_15;
        wp::int32* var_16;
        const wp::int32 var_17 = 1;
        bool var_18;
        wp::int32 var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        const wp::int32 var_29 = 1;
        wp::int32 var_30;
        wp::int32 var_31;
        const wp::int32 var_32 = 2;
        wp::int32 var_33;
        const wp::int32 var_34 = 32;
        wp::range_t var_35;
        wp::int32 var_36;
        const wp::int32 var_37 = 8;
        wp::int32 var_38;
        const wp::int32 var_39 = 1;
        wp::int32 var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        const wp::float32 var_43 = 1.0;
        wp::float32 var_44;
        const wp::float32 var_45 = 0.5;
        wp::float32 var_46;
        wp::int32 var_47;
        const wp::int32 var_48 = 1;
        wp::int32 var_49;
        wp::int32 var_50;
        const wp::int32 var_51 = 2;
        wp::int32 var_52;
        wp::int32 var_53;
        const wp::int32 var_54 = 0;
        wp::int32 var_55;
        wp::float32* var_56;
        wp::float32 var_57;
        wp::float32 var_58;
        const wp::int32 var_59 = 0;
        wp::int32 var_60;
        wp::float32* var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::float32 var_64;
        wp::float32 var_65;
        const wp::float32 var_66 = 0.0;
        bool var_67;
        const wp::int32 var_68 = 0;
        wp::int32 var_69;
        wp::int32* var_70;
        wp::int32 var_71;
        wp::int32 var_72;
        const wp::int32 var_73 = 0;
        wp::int32 var_74;
        wp::int32* var_75;
        wp::int32 var_76;
        wp::int32 var_77;
        const bool var_78 = false;
        wp::int32 var_79;
        wp::array_t<wp::float32> var_80;
        wp::int32 var_81;
        wp::float32 var_82;
        //---------
        // forward
        // def kernel(                                                                            <L 2002>
        // worldid, slot, lane = wp.tid()                                                         <L 2024>
        builtin_tid3d(var_0, var_1, var_2);
        // n_changes = quad_changed_count_in[worldid]                                             <L 2026>
        var_3 = wp::address(var_quad_changed_count_in, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // for change_idx in range(slot, n_changes, slots_per_world):                             <L 2027>
        var_6 = wp::range(var_1, var_4, var_slots_per_world);
        start_for_0:;
            if (iter_cmp(var_6) == 0) goto end_for_0;
            var_7 = wp::iter_next(var_6);
            // efcid = quad_changed_ids_in[worldid, change_idx]                                   <L 2028>
            var_8 = wp::address(var_quad_changed_ids_in, var_0, var_7);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
            // D = efc_D_in[worldid, efcid]                                                       <L 2029>
            var_11 = wp::address(var_efc_D_in, var_0, var_9);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
            // sign = float(0.0)                                                                  <L 2030>
            var_15 = wp::float(var_14);
            // if efc_state_in[worldid, efcid] == types.ConstraintState.QUADRATIC.value:          <L 2031>
            var_16 = wp::address(var_efc_state_in, var_0, var_9);
            var_19 = wp::load(var_16);
            var_18 = (var_19 == var_17);
            if (var_18) {
                // sign = D                                                                       <L 2032>
                var_20 = wp::copy(var_12);
            }
            if (!var_18) {
                // sign = -D                                                                      <L 2034>
                var_21 = wp::neg(var_12);
            }
            var_22 = wp::where(var_18, var_20, var_21);
            // rownnz = efc_J_rownnz_in[worldid, efcid]                                           <L 2036>
            var_23 = wp::address(var_efc_J_rownnz_in, var_0, var_9);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // rowadr = efc_J_rowadr_in[worldid, efcid]                                           <L 2037>
            var_26 = wp::address(var_efc_J_rowadr_in, var_0, var_9);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // n_entries = rownnz * (rownnz + 1) // 2                                             <L 2038>
            var_30 = wp::add(var_24, var_29);
            var_31 = wp::mul(var_24, var_30);
            var_33 = wp::floordiv(var_31, var_32);
            // for entry in range(lane, n_entries, wp.static(_JTDAJ_THREADS_PER_GROUP)):          <L 2040>
            var_35 = wp::range(var_2, var_33, var_34);
            start_for_2:;
                if (iter_cmp(var_35) == 0) goto end_for_2;
                var_36 = wp::iter_next(var_35);
                // ii = int((wp.sqrt(float(8 * entry + 1)) - 1.0) * 0.5)                          <L 2041>
                var_38 = wp::mul(var_37, var_36);
                var_40 = wp::add(var_38, var_39);
                var_41 = wp::float(var_40);
                var_42 = wp::sqrt(var_41);
                var_44 = wp::sub(var_42, var_43);
                var_46 = wp::mul(var_44, var_45);
                var_47 = wp::int(var_46);
                // jj = entry - ii * (ii + 1) // 2                                                <L 2042>
                var_49 = wp::add(var_47, var_48);
                var_50 = wp::mul(var_47, var_49);
                var_52 = wp::floordiv(var_50, var_51);
                var_53 = wp::sub(var_36, var_52);
                // Ji = efc_J_in[worldid, 0, rowadr + ii]                                         <L 2043>
                var_55 = wp::add(var_27, var_47);
                var_56 = wp::address(var_efc_J_in, var_0, var_54, var_55);
                var_58 = wp::load(var_56);
                var_57 = wp::copy(var_58);
                // Jj = efc_J_in[worldid, 0, rowadr + jj]                                         <L 2044>
                var_60 = wp::add(var_27, var_53);
                var_61 = wp::address(var_efc_J_in, var_0, var_59, var_60);
                var_63 = wp::load(var_61);
                var_62 = wp::copy(var_63);
                // h = sign * Ji * Jj                                                             <L 2045>
                var_64 = wp::mul(var_22, var_57);
                var_65 = wp::mul(var_64, var_62);
                // if h != 0.0:                                                                   <L 2046>
                var_67 = (var_65 != var_66);
                if (var_67) {
                    // colindi = efc_J_colind_in[worldid, 0, rowadr + ii]                         <L 2047>
                    var_69 = wp::add(var_27, var_47);
                    var_70 = wp::address(var_efc_J_colind_in, var_0, var_68, var_69);
                    var_72 = wp::load(var_70);
                    var_71 = wp::copy(var_72);
                    // colindj = efc_J_colind_in[worldid, 0, rowadr + jj]                         <L 2048>
                    var_74 = wp::add(var_27, var_53);
                    var_75 = wp::address(var_efc_J_colind_in, var_0, var_73, var_74);
                    var_77 = wp::load(var_75);
                    var_76 = wp::copy(var_77);
                    // if wp.static(COMPACT):                                                     <L 2049>
                    // wp.atomic_add(ctx_h_out[worldid, wp.min(colindi, colindj)], wp.max(colindi, colindj), h)       <L 2054>
                    var_79 = wp::min(var_71, var_76);
                    var_80 = wp::view(var_ctx_h_out, var_0, var_79);
                    var_81 = wp::max(var_71, var_76);
                    var_82 = wp::atomic_add(var_80, var_81, var_65);
                }
                goto start_for_2;
            end_for_2:;
            goto start_for_0;
        end_for_0:;
}

