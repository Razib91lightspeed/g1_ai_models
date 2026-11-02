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


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/sim/contacts.py:95
static CUDA_CALLABLE wp::vec_t<3, wp::float32> contact_surface_point_0(
    wp::transform_t<wp::float32> var_X_wb,
    wp::vec_t<3, wp::float32> var_point_local,
    wp::vec_t<3, wp::float32> var_offset_local)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    //---------
    // forward
    // def contact_surface_point(                                                             <L 96>
    // return wp.transform_point(X_wb, point_local + offset_local)                            <L 115>
    var_0 = wp::add(var_point_local, var_offset_local);
    var_1 = wp::transform_point(var_X_wb, var_0);
    return var_1;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/sim/contacts.py:95
static CUDA_CALLABLE void adj_contact_surface_point_0(
    wp::transform_t<wp::float32> var_X_wb,
    wp::vec_t<3, wp::float32> var_point_local,
    wp::vec_t<3, wp::float32> var_offset_local,
    wp::transform_t<wp::float32> & adj_X_wb,
    wp::vec_t<3, wp::float32> & adj_point_local,
    wp::vec_t<3, wp::float32> & adj_offset_local,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void accumulate_contact_forces_kernel_0d85994d_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_num_contacts,
    wp::array_t<wp::int32> var_contact_shape0,
    wp::array_t<wp::int32> var_contact_shape1,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_point0,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_point1,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_offset0,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_offset1,
    wp::array_t<wp::vec_t<6, wp::float32>> var_contact_force,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_normal,
    wp::array_t<wp::int32> var_shape_body,
    wp::array_t<wp::transform_t<wp::float32>> var_body_q,
    wp::array_t<wp::int32> var_sensing_shape_to_row,
    wp::array_t<wp::int32> var_counterpart_shape_to_col,
    wp::array_t<wp::vec_t<3, wp::float32>> var_force_matrix,
    wp::array_t<wp::vec_t<3, wp::float32>> var_total_force,
    wp::array_t<wp::vec_t<3, wp::float32>> var_force_matrix_friction,
    wp::array_t<wp::vec_t<3, wp::float32>> var_total_force_friction,
    wp::array_t<wp::vec_t<3, wp::float32>> var_position_matrix,
    wp::array_t<wp::float32> var_position_weight)
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
        wp::int32* var_2;
        bool var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        bool var_11;
        const wp::int32 var_12 = 0;
        bool var_13;
        const wp::int32 var_14 = 0;
        bool var_15;
        wp::vec_t<6, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<6, wp::float32> var_18;
        wp::vec_t<3, wp::float32>* var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::float32 var_22;
        const wp::float32 var_23 = 1.0;
        wp::float32 var_24;
        wp::float32 var_25;
        const wp::float32 var_26 = 0.0001;
        bool var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::float32 var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::vec_t<3, wp::float32> var_32;
        wp::int32* var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        wp::int32* var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        const wp::int32 var_39 = 0;
        bool var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::vec_t<3, wp::float32> var_42;
        const wp::int32 var_43 = 0;
        bool var_44;
        wp::vec_t<3, wp::float32> var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::int32* var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        wp::int32* var_52;
        wp::int32 var_53;
        wp::int32 var_54;
        bool var_55;
        const wp::int32 var_56 = 0;
        bool var_57;
        const wp::int32 var_58 = 0;
        bool var_59;
        bool var_60;
        const wp::int32 var_61 = 0;
        bool var_62;
        const wp::int32 var_63 = 0;
        bool var_64;
        wp::vec_t<3, wp::float32> var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::vec_t<3, wp::float32> var_67;
        wp::vec_t<3, wp::float32> var_68;
        wp::vec_t<3, wp::float32> var_69;
        wp::vec_t<3, wp::float32> var_70;
        wp::float32 var_71;
        bool var_72;
        const wp::float32 var_73 = 0.0;
        bool var_74;
        bool var_75;
        wp::int32* var_76;
        wp::int32 var_77;
        wp::int32 var_78;
        wp::int32* var_79;
        wp::int32 var_80;
        wp::int32 var_81;
        const wp::int32 var_82 = 0;
        bool var_83;
        const wp::int32 var_84 = 0;
        wp::int32 var_85;
        wp::transform_t<wp::float32>* var_86;
        wp::transform_t<wp::float32> var_87;
        wp::transform_t<wp::float32> var_88;
        wp::transform_t<wp::float32> var_89;
        const wp::int32 var_90 = 0;
        bool var_91;
        const wp::int32 var_92 = 0;
        wp::int32 var_93;
        wp::transform_t<wp::float32>* var_94;
        wp::transform_t<wp::float32> var_95;
        wp::transform_t<wp::float32> var_96;
        wp::transform_t<wp::float32> var_97;
        wp::vec_t<3, wp::float32>* var_98;
        wp::vec_t<3, wp::float32>* var_99;
        wp::vec_t<3, wp::float32> var_100;
        wp::vec_t<3, wp::float32> var_101;
        wp::vec_t<3, wp::float32> var_102;
        wp::vec_t<3, wp::float32>* var_103;
        wp::vec_t<3, wp::float32>* var_104;
        wp::vec_t<3, wp::float32> var_105;
        wp::vec_t<3, wp::float32> var_106;
        wp::vec_t<3, wp::float32> var_107;
        const wp::float32 var_108 = 0.5;
        wp::vec_t<3, wp::float32> var_109;
        wp::vec_t<3, wp::float32> var_110;
        wp::vec_t<3, wp::float32> var_111;
        wp::vec_t<3, wp::float32> var_112;
        wp::float32 var_113;
        wp::vec_t<3, wp::float32> var_114;
        wp::float32 var_115;
        //---------
        // forward
        // def accumulate_contact_forces_kernel(                                                  <L 68>
        // contact_index = wp.tid()                                                               <L 91>
        var_0 = builtin_tid1d();
        // if contact_index >= num_contacts[0]:                                                   <L 92>
        var_2 = wp::address(var_num_contacts, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_0 >= var_4);
        if (var_3) {
            // return                                                                             <L 93>
            continue;
        }
        // shape0 = contact_shape0[contact_index]                                                 <L 95>
        var_5 = wp::address(var_contact_shape0, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // shape1 = contact_shape1[contact_index]                                                 <L 96>
        var_8 = wp::address(var_contact_shape1, var_0);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // assert shape0 >= 0 and shape1 >= 0                                                     <L 97>
        var_13 = (var_6 >= var_12);
        var_11 = var_13;
        if (var_11) {
            var_15 = (var_9 >= var_14);
            var_11 = var_11 && var_15;
        }
        assert(((void)"assert shape0 >= 0 and shape1 >= 0",var_11));
        // force = wp.spatial_top(contact_force[contact_index])                                   <L 98>
        var_16 = wp::address(var_contact_force, var_0);
        var_18 = wp::load(var_16);
        var_17 = wp::spatial_top(var_18);
        // n = contact_normal[contact_index]                                                      <L 101>
        var_19 = wp::address(var_contact_normal, var_0);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // len_sq = wp.dot(n, n)                                                                  <L 102>
        var_22 = wp::dot(var_20, var_20);
        // if wp.abs(len_sq - 1.0) > 1.0e-4:                                                      <L 103>
        var_24 = wp::sub(var_22, var_23);
        var_25 = wp::abs(var_24);
        var_27 = (var_25 > var_26);
        if (var_27) {
            // n = wp.normalize(n)                                                                <L 104>
            var_28 = wp::normalize(var_20);
        }
        var_29 = wp::where(var_27, var_28, var_20);
        // friction = force - wp.dot(force, n) * n                                                <L 105>
        var_30 = wp::dot(var_17, var_29);
        var_31 = wp::mul(var_30, var_29);
        var_32 = wp::sub(var_17, var_31);
        // row0 = sensing_shape_to_row[shape0]                                                    <L 107>
        var_33 = wp::address(var_sensing_shape_to_row, var_6);
        var_35 = wp::load(var_33);
        var_34 = wp::copy(var_35);
        // row1 = sensing_shape_to_row[shape1]                                                    <L 108>
        var_36 = wp::address(var_sensing_shape_to_row, var_9);
        var_38 = wp::load(var_36);
        var_37 = wp::copy(var_38);
        // if total_force:                                                                        <L 111>
        if (var_total_force) {
            // assert total_force_friction                                                        <L 112>
            assert(((void)"assert total_force_friction",var_total_force_friction));
            // if row0 >= 0:                                                                      <L 113>
            var_40 = (var_34 >= var_39);
            if (var_40) {
                // wp.atomic_add(total_force, row0, force)                                        <L 114>
                var_41 = wp::atomic_add(var_total_force, var_34, var_17);
                // wp.atomic_add(total_force_friction, row0, friction)                            <L 115>
                var_42 = wp::atomic_add(var_total_force_friction, var_34, var_32);
            }
            // if row1 >= 0:                                                                      <L 116>
            var_44 = (var_37 >= var_43);
            if (var_44) {
                // wp.atomic_add(total_force, row1, -force)                                       <L 117>
                var_45 = wp::neg(var_17);
                var_46 = wp::atomic_add(var_total_force, var_37, var_45);
                // wp.atomic_add(total_force_friction, row1, -friction)                           <L 118>
                var_47 = wp::neg(var_32);
                var_48 = wp::atomic_add(var_total_force_friction, var_37, var_47);
            }
        }
        // if force_matrix:                                                                       <L 121>
        if (var_force_matrix) {
            // assert force_matrix_friction                                                       <L 122>
            assert(((void)"assert force_matrix_friction",var_force_matrix_friction));
            // col0 = counterpart_shape_to_col[shape0]                                            <L 123>
            var_49 = wp::address(var_counterpart_shape_to_col, var_6);
            var_51 = wp::load(var_49);
            var_50 = wp::copy(var_51);
            // col1 = counterpart_shape_to_col[shape1]                                            <L 124>
            var_52 = wp::address(var_counterpart_shape_to_col, var_9);
            var_54 = wp::load(var_52);
            var_53 = wp::copy(var_54);
            // matched0 = row0 >= 0 and col1 >= 0                                                 <L 125>
            var_57 = (var_34 >= var_56);
            var_55 = var_57;
            if (var_55) {
                var_59 = (var_53 >= var_58);
                var_55 = var_55 && var_59;
            }
            // matched1 = row1 >= 0 and col0 >= 0                                                 <L 126>
            var_62 = (var_37 >= var_61);
            var_60 = var_62;
            if (var_60) {
                var_64 = (var_50 >= var_63);
                var_60 = var_60 && var_64;
            }
            // if matched0:                                                                       <L 127>
            if (var_55) {
                // wp.atomic_add(force_matrix, row0, col1, force)                                 <L 128>
                var_65 = wp::atomic_add(var_force_matrix, var_34, var_53, var_17);
                // wp.atomic_add(force_matrix_friction, row0, col1, friction)                     <L 129>
                var_66 = wp::atomic_add(var_force_matrix_friction, var_34, var_53, var_32);
            }
            // if matched1:                                                                       <L 130>
            if (var_60) {
                // wp.atomic_add(force_matrix, row1, col0, -force)                                <L 131>
                var_67 = wp::neg(var_17);
                var_68 = wp::atomic_add(var_force_matrix, var_37, var_50, var_67);
                // wp.atomic_add(force_matrix_friction, row1, col0, -friction)                    <L 132>
                var_69 = wp::neg(var_32);
                var_70 = wp::atomic_add(var_force_matrix_friction, var_37, var_50, var_69);
            }
            // if position_matrix:                                                                <L 134>
            if (var_position_matrix) {
                // assert position_weight                                                         <L 135>
                assert(((void)"assert position_weight",var_position_weight));
                // weight = wp.length(force)                                                      <L 136>
                var_71 = wp::length(var_17);
                // if weight > 0.0 and (matched0 or matched1):                                    <L 137>
                var_74 = (var_71 > var_73);
                var_72 = var_74;
                if (var_72) {
                    var_75 = var_55;
                    if (!var_75) {
                        var_75 = var_75 || var_60;
                    }
                    var_72 = var_72 && var_75;
                }
                if (var_72) {
                    // body0 = shape_body[shape0]                                                 <L 138>
                    var_76 = wp::address(var_shape_body, var_6);
                    var_78 = wp::load(var_76);
                    var_77 = wp::copy(var_78);
                    // body1 = shape_body[shape1]                                                 <L 139>
                    var_79 = wp::address(var_shape_body, var_9);
                    var_81 = wp::load(var_79);
                    var_80 = wp::copy(var_81);
                    // transform0 = wp.where(body0 >= 0, body_q[wp.max(body0, 0)], wp.transform_identity())       <L 140>
                    var_83 = (var_77 >= var_82);
                    var_85 = wp::max(var_77, var_84);
                    var_86 = wp::address(var_body_q, var_85);
                    var_87 = wp::transform_identity<wp::float32>();
                    var_89 = wp::load(var_86);
                    var_88 = wp::where(var_83, var_89, var_87);
                    // transform1 = wp.where(body1 >= 0, body_q[wp.max(body1, 0)], wp.transform_identity())       <L 141>
                    var_91 = (var_80 >= var_90);
                    var_93 = wp::max(var_80, var_92);
                    var_94 = wp::address(var_body_q, var_93);
                    var_95 = wp::transform_identity<wp::float32>();
                    var_97 = wp::load(var_94);
                    var_96 = wp::where(var_91, var_97, var_95);
                    // point0_world = contact_surface_point(                                      <L 142>
                    // transform0, contact_point0[contact_index], contact_offset0[contact_index]       <L 143>
                    var_98 = wp::address(var_contact_point0, var_0);
                    var_99 = wp::address(var_contact_offset0, var_0);
                    var_101 = wp::load(var_98);
                    var_102 = wp::load(var_99);
                    var_100 = contact_surface_point_0(var_88, var_101, var_102);
                    // point1_world = contact_surface_point(                                      <L 145>
                    // transform1, contact_point1[contact_index], contact_offset1[contact_index]       <L 146>
                    var_103 = wp::address(var_contact_point1, var_0);
                    var_104 = wp::address(var_contact_offset1, var_0);
                    var_106 = wp::load(var_103);
                    var_107 = wp::load(var_104);
                    var_105 = contact_surface_point_0(var_96, var_106, var_107);
                    // midpoint = 0.5 * (point0_world + point1_world)                             <L 148>
                    var_109 = wp::add(var_100, var_105);
                    var_110 = wp::mul(var_108, var_109);
                    // weighted_midpoint = weight * midpoint                                      <L 149>
                    var_111 = wp::mul(var_71, var_110);
                    // if matched0:                                                               <L 151>
                    if (var_55) {
                        // wp.atomic_add(position_matrix, row0, col1, weighted_midpoint)          <L 152>
                        var_112 = wp::atomic_add(var_position_matrix, var_34, var_53, var_111);
                        // wp.atomic_add(position_weight, row0, col1, weight)                     <L 153>
                        var_113 = wp::atomic_add(var_position_weight, var_34, var_53, var_71);
                    }
                    // if matched1:                                                               <L 154>
                    if (var_60) {
                        // wp.atomic_add(position_matrix, row1, col0, weighted_midpoint)          <L 155>
                        var_114 = wp::atomic_add(var_position_matrix, var_37, var_50, var_111);
                        // wp.atomic_add(position_weight, row1, col0, weight)                     <L 156>
                        var_115 = wp::atomic_add(var_position_weight, var_37, var_50, var_71);
                    }
                }
            }
        }
    }
}



extern "C" __global__ void normalize_contact_positions_kernel_91927957_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_position_matrix,
    wp::array_t<wp::float32> var_position_weight)
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
        const wp::float32 var_5 = 0.0;
        bool var_6;
        wp::vec_t<3, wp::float32>* var_7;
        wp::vec_t<3, wp::float32> var_8;
        wp::vec_t<3, wp::float32> var_9;
        //---------
        // forward
        // def normalize_contact_positions_kernel(                                                <L 160>
        // row, col = wp.tid()                                                                    <L 165>
        builtin_tid2d(var_0, var_1);
        // weight = position_weight[row, col]                                                     <L 166>
        var_2 = wp::address(var_position_weight, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if weight > 0.0:                                                                       <L 167>
        var_6 = (var_3 > var_5);
        if (var_6) {
            // position_matrix[row, col] /= weight                                                <L 168>
            var_7 = wp::address(var_position_matrix, var_0, var_1);
            var_9 = wp::load(var_7);
            var_8 = wp::div(var_9, var_3);
            wp::array_store(var_position_matrix, var_0, var_1, var_8);
        }
    }
}



extern "C" __global__ void expand_body_to_shape_kernel_97bcfec4_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_body_to_row,
    wp::array_t<wp::int32> var_body_to_col,
    wp::array_t<wp::int32> var_shape_body,
    wp::array_t<wp::int32> var_shape_to_row,
    wp::array_t<wp::int32> var_shape_to_col)
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
        wp::int32* var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        const wp::int32 var_4 = -1;
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        const wp::int32 var_11 = -1;
        const wp::int32 var_12 = 0;
        bool var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        //---------
        // forward
        // def expand_body_to_shape_kernel(                                                       <L 172>
        // tid = wp.tid()                                                                         <L 181>
        var_0 = builtin_tid1d();
        // body = shape_body[tid]                                                                 <L 182>
        var_1 = wp::address(var_shape_body, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // if body_to_row:                                                                        <L 184>
        if (var_body_to_row) {
            // row = -1                                                                           <L 185>
            // if body >= 0:                                                                      <L 186>
            var_6 = (var_2 >= var_5);
            if (var_6) {
                // row = body_to_row[body]                                                        <L 187>
                var_7 = wp::address(var_body_to_row, var_2);
                var_9 = wp::load(var_7);
                var_8 = wp::copy(var_9);
            }
            var_10 = wp::where(var_6, var_8, var_4);
            // shape_to_row[tid] = row                                                            <L 188>
            wp::array_store(var_shape_to_row, var_0, var_10);
        }
        // if body_to_col:                                                                        <L 190>
        if (var_body_to_col) {
            // col = -1                                                                           <L 191>
            // if body >= 0:                                                                      <L 192>
            var_13 = (var_2 >= var_12);
            if (var_13) {
                // col = body_to_col[body]                                                        <L 193>
                var_14 = wp::address(var_body_to_col, var_2);
                var_16 = wp::load(var_14);
                var_15 = wp::copy(var_16);
            }
            var_17 = wp::where(var_13, var_15, var_11);
            // shape_to_col[tid] = col                                                            <L 194>
            wp::array_store(var_shape_to_col, var_0, var_17);
        }
    }
}



extern "C" __global__ void compute_sensing_transforms_kernel_fb4d81c9_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::int32> var_sensing_kinds,
    wp::array_t<wp::int32> var_shape_body,
    wp::array_t<wp::transform_t<wp::float32>> var_shape_transform,
    wp::array_t<wp::transform_t<wp::float32>> var_body_q,
    wp::array_t<wp::transform_t<wp::float32>> var_transforms)
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
        wp::int32* var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        const wp::int32 var_7 = 2;
        bool var_8;
        wp::transform_t<wp::float32>* var_9;
        wp::transform_t<wp::float32> var_10;
        const wp::int32 var_11 = 1;
        bool var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        const wp::int32 var_16 = 0;
        bool var_17;
        wp::transform_t<wp::float32>* var_18;
        wp::transform_t<wp::float32>* var_19;
        wp::transform_t<wp::float32> var_20;
        wp::transform_t<wp::float32> var_21;
        wp::transform_t<wp::float32> var_22;
        wp::transform_t<wp::float32>* var_23;
        wp::transform_t<wp::float32> var_24;
        //---------
        // forward
        // def compute_sensing_transforms_kernel(                                                 <L 45>
        // tid = wp.tid()                                                                         <L 54>
        var_0 = builtin_tid1d();
        // index = indices[tid]                                                                   <L 55>
        var_1 = wp::address(var_indices, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // sensing_kind = sensing_kinds[tid]                                                      <L 56>
        var_4 = wp::address(var_sensing_kinds, var_0);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // if sensing_kind == wp.static(_SENSING_KIND_BODY):                                      <L 57>
        var_8 = (var_5 == var_7);
        if (var_8) {
            // transforms[tid] = body_q[index]                                                    <L 58>
            var_9 = wp::address(var_body_q, var_2);
            var_10 = wp::load(var_9);
            wp::array_store(var_transforms, var_0, var_10);
        }
        if (!var_8) {
            // elif sensing_kind == wp.static(_SENSING_KIND_SHAPE):                               <L 59>
            var_12 = (var_5 == var_11);
            if (var_12) {
                // body_index = shape_body[index]                                                 <L 60>
                var_13 = wp::address(var_shape_body, var_2);
                var_15 = wp::load(var_13);
                var_14 = wp::copy(var_15);
                // if body_index >= 0:                                                            <L 61>
                var_17 = (var_14 >= var_16);
                if (var_17) {
                    // transforms[tid] = wp.transform_multiply(body_q[body_index], shape_transform[index])       <L 62>
                    var_18 = wp::address(var_body_q, var_14);
                    var_19 = wp::address(var_shape_transform, var_2);
                    var_21 = wp::load(var_18);
                    var_22 = wp::load(var_19);
                    var_20 = wp::transform_multiply(var_21, var_22);
                    wp::array_store(var_transforms, var_0, var_20);
                }
                if (!var_17) {
                    // transforms[tid] = shape_transform[index]                                   <L 64>
                    var_23 = wp::address(var_shape_transform, var_2);
                    var_24 = wp::load(var_23);
                    wp::array_store(var_transforms, var_0, var_24);
                }
            }
        }
    }
}

