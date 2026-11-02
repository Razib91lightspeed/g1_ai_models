#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 128
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



extern "C" __global__ void _JTDACJ_sparse__locals__kernel_81e4d8e1_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::float32> var_opt_impratio_invsqrt,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_efc_id_in,
    wp::array_t<wp::int32> var_efc_jtdaj_adr_in,
    wp::array_t<wp::int32> var_efc_jtdaj_nrow_in,
    wp::array_t<wp::int32> var_efc_jtdaj_nblock_in,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::array_t<wp::int32> var_efc_J_colind_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::int32> var_efc_state_in,
    wp::array_t<wp::int32> var_dof_cdof_in,
    wp::array_t<wp::float32> var_ctx_Jaref_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::int32 var_groups_per_world,
    wp::array_t<wp::float32> var_h_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
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
        const bool var_3 = false;
        const wp::int32 var_4 = 32;
        bool* var_5;
        bool var_6;
        bool var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::range_t var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::int32* var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        const wp::int32 var_25 = 1;
        wp::int32 var_26;
        wp::int32 var_27;
        const wp::int32 var_28 = 2;
        wp::int32 var_29;
        const bool var_30 = false;
        const bool var_31 = false;
        const wp::int32 var_32 = 32;
        wp::range_t var_33;
        wp::int32 var_34;
        const wp::int32 var_35 = 8;
        wp::int32 var_36;
        const wp::int32 var_37 = 1;
        wp::int32 var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        const wp::float32 var_41 = 1.0;
        wp::float32 var_42;
        const wp::float32 var_43 = 0.5;
        wp::float32 var_44;
        wp::int32 var_45;
        const wp::int32 var_46 = 1;
        wp::int32 var_47;
        wp::int32 var_48;
        const wp::int32 var_49 = 2;
        wp::int32 var_50;
        wp::int32 var_51;
        const wp::int32 var_52 = 0;
        wp::int32 var_53;
        wp::int32* var_54;
        wp::int32 var_55;
        wp::int32 var_56;
        const wp::int32 var_57 = 0;
        wp::int32 var_58;
        wp::int32* var_59;
        wp::int32 var_60;
        wp::int32 var_61;
        const wp::float32 var_62 = 0.0;
        wp::float32 var_63;
        const bool var_64 = false;
        bool var_65;
        wp::range_t var_66;
        wp::int32 var_67;
        wp::int32 var_68;
        wp::int32* var_69;
        const wp::int32 var_70 = 1;
        bool var_71;
        wp::int32 var_72;
        wp::int32* var_73;
        wp::int32 var_74;
        wp::int32 var_75;
        const wp::int32 var_76 = 0;
        wp::int32 var_77;
        wp::float32* var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        const wp::int32 var_81 = 0;
        wp::int32 var_82;
        wp::float32* var_83;
        wp::float32 var_84;
        wp::float32 var_85;
        wp::float32* var_86;
        wp::float32 var_87;
        wp::float32 var_88;
        wp::float32 var_89;
        wp::float32 var_90;
        wp::float32 var_91;
        const wp::float32 var_92 = 0.0;
        bool var_93;
        const bool var_94 = false;
        wp::int32 var_95;
        wp::array_t<wp::float32> var_96;
        wp::int32 var_97;
        wp::float32 var_98;
        //---------
        // forward
        // def kernel(                                                                            <L 2948>
        // worldid, slot, lane = wp.tid()                                                         <L 2972>
        builtin_tid3d(var_0, var_1, var_2);
        // if wp.static(ELLIPTIC):                                                                <L 2973>
        // lanes = wp.static(_JTDAJ_THREADS_PER_GROUP)                                            <L 2976>
        // if ctx_done_in[worldid]:                                                               <L 2977>
        var_5 = wp::address(var_ctx_done_in, var_0);
        var_6 = wp::load(var_5);
        if (var_6) {
            // return                                                                             <L 2978>
            continue;
        }
        var_7 = wp::load(var_5);
        // count = efc_jtdaj_nblock_in[worldid]                                                   <L 2979>
        var_8 = wp::address(var_efc_jtdaj_nblock_in, var_0);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // for groupid in range(slot, count, groups_per_world):                                   <L 2980>
        var_11 = wp::range(var_1, var_9, var_groups_per_world);
        start_for_1:;
            if (iter_cmp(var_11) == 0) goto end_for_1;
            var_12 = wp::iter_next(var_11);
            // head_row = efc_jtdaj_adr_in[worldid, groupid]                                      <L 2981>
            var_13 = wp::address(var_efc_jtdaj_adr_in, var_0, var_12);
            var_15 = wp::load(var_13);
            var_14 = wp::copy(var_15);
            // block_rows = efc_jtdaj_nrow_in[worldid, groupid]                                   <L 2982>
            var_16 = wp::address(var_efc_jtdaj_nrow_in, var_0, var_12);
            var_18 = wp::load(var_16);
            var_17 = wp::copy(var_18);
            // head_adr = efc_J_rowadr_in[worldid, head_row]                                      <L 2983>
            var_19 = wp::address(var_efc_J_rowadr_in, var_0, var_14);
            var_21 = wp::load(var_19);
            var_20 = wp::copy(var_21);
            // support = efc_J_rownnz_in[worldid, head_row]                                       <L 2984>
            var_22 = wp::address(var_efc_J_rownnz_in, var_0, var_14);
            var_24 = wp::load(var_22);
            var_23 = wp::copy(var_24);
            // n_entries = support * (support + 1) // 2                                           <L 2985>
            var_26 = wp::add(var_23, var_25);
            var_27 = wp::mul(var_23, var_26);
            var_29 = wp::floordiv(var_27, var_28);
            // is_cone = False                                                                    <L 2987>
            // if wp.static(ELLIPTIC):                                                            <L 2988>
            // for entry in range(lane, n_entries, lanes):                                        <L 3018>
            var_33 = wp::range(var_2, var_29, var_32);
            start_for_3:;
                if (iter_cmp(var_33) == 0) goto end_for_3;
                var_34 = wp::iter_next(var_33);
                // block_col = int((wp.sqrt(float(8 * entry + 1)) - 1.0) * 0.5)                   <L 3019>
                var_36 = wp::mul(var_35, var_34);
                var_38 = wp::add(var_36, var_37);
                var_39 = wp::float(var_38);
                var_40 = wp::sqrt(var_39);
                var_42 = wp::sub(var_40, var_41);
                var_44 = wp::mul(var_42, var_43);
                var_45 = wp::int(var_44);
                // block_row = entry - block_col * (block_col + 1) // 2                           <L 3020>
                var_47 = wp::add(var_45, var_46);
                var_48 = wp::mul(var_45, var_47);
                var_50 = wp::floordiv(var_48, var_49);
                var_51 = wp::sub(var_34, var_50);
                // dof_row = efc_J_colind_in[worldid, 0, head_adr + block_row]                    <L 3021>
                var_53 = wp::add(var_20, var_51);
                var_54 = wp::address(var_efc_J_colind_in, var_0, var_52, var_53);
                var_56 = wp::load(var_54);
                var_55 = wp::copy(var_56);
                // dof_col = efc_J_colind_in[worldid, 0, head_adr + block_col]                    <L 3022>
                var_58 = wp::add(var_20, var_45);
                var_59 = wp::address(var_efc_J_colind_in, var_0, var_57, var_58);
                var_61 = wp::load(var_59);
                var_60 = wp::copy(var_61);
                // hval = float(0.0)                                                              <L 3023>
                var_63 = wp::float(var_62);
                // if wp.static(ELLIPTIC):                                                        <L 3024>
                // if not is_cone:                                                                <L 3035>
                var_65 = wp::unot(var_30);
                if (var_65) {
                    // for member in range(block_rows):                                           <L 3036>
                    var_66 = wp::range(var_17);
                    start_for_5:;
                        if (iter_cmp(var_66) == 0) goto end_for_5;
                        var_67 = wp::iter_next(var_66);
                        // member_row = head_row + member                                         <L 3037>
                        var_68 = wp::add(var_14, var_67);
                        // if efc_state_in[worldid, member_row] == types.ConstraintState.QUADRATIC.value:       <L 3038>
                        var_69 = wp::address(var_efc_state_in, var_0, var_68);
                        var_72 = wp::load(var_69);
                        var_71 = (var_72 == var_70);
                        if (var_71) {
                            // member_adr = efc_J_rowadr_in[worldid, member_row]                  <L 3039>
                            var_73 = wp::address(var_efc_J_rowadr_in, var_0, var_68);
                            var_75 = wp::load(var_73);
                            var_74 = wp::copy(var_75);
                            // j_row = efc_J_in[worldid, 0, member_adr + block_row]               <L 3040>
                            var_77 = wp::add(var_74, var_51);
                            var_78 = wp::address(var_efc_J_in, var_0, var_76, var_77);
                            var_80 = wp::load(var_78);
                            var_79 = wp::copy(var_80);
                            // j_col = efc_J_in[worldid, 0, member_adr + block_col]               <L 3041>
                            var_82 = wp::add(var_74, var_45);
                            var_83 = wp::address(var_efc_J_in, var_0, var_81, var_82);
                            var_85 = wp::load(var_83);
                            var_84 = wp::copy(var_85);
                            // hval += j_row * efc_D_in[worldid, member_row] * j_col              <L 3042>
                            var_86 = wp::address(var_efc_D_in, var_0, var_68);
                            var_88 = wp::load(var_86);
                            var_87 = wp::mul(var_79, var_88);
                            var_89 = wp::mul(var_87, var_84);
                            var_90 = wp::add(var_63, var_89);
                        }
                        var_91 = wp::where(var_71, var_90, var_63);
                        wp::assign(var_63, var_91);
                        goto start_for_5;
                    end_for_5:;
                }
                // if hval != 0.0:                                                                <L 3043>
                var_93 = (var_63 != var_92);
                if (var_93) {
                    // if wp.static(COMPACT):                                                     <L 3044>
                    // wp.atomic_add(h_out[worldid, wp.min(dof_row, dof_col)], wp.max(dof_row, dof_col), hval)       <L 3049>
                    var_95 = wp::min(var_55, var_60);
                    var_96 = wp::view(var_h_out, var_0, var_95);
                    var_97 = wp::max(var_55, var_60);
                    var_98 = wp::atomic_add(var_96, var_97, var_63);
                }
                goto start_for_3;
            end_for_3:;
            goto start_for_1;
        end_for_1:;
    }
}

