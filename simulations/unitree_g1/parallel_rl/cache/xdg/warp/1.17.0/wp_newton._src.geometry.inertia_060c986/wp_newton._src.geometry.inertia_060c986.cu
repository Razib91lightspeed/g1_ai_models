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


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/inertia.py:301
static CUDA_CALLABLE void triangle_inertia_0(
    wp::vec_t<3, wp::float32> var_v0,
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::mat_t<3, 3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    const wp::float32 var_2 = 6.0;
    wp::float32 var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    const wp::float32 var_7 = 4.0;
    wp::vec_t<3, wp::float32> var_8;
    wp::mat_t<3, 3, wp::float32> var_9;
    wp::mat_t<3, 3, wp::float32> var_10;
    wp::mat_t<3, 3, wp::float32> var_11;
    wp::mat_t<3, 3, wp::float32> var_12;
    wp::mat_t<3, 3, wp::float32> var_13;
    wp::mat_t<3, 3, wp::float32> var_14;
    wp::mat_t<3, 3, wp::float32> var_15;
    wp::mat_t<3, 3, wp::float32> var_16;
    wp::mat_t<3, 3, wp::float32> var_17;
    const wp::float32 var_18 = 10.0;
    wp::float32 var_19;
    wp::mat_t<3, 3, wp::float32> var_20;
    wp::mat_t<3, 3, wp::float32> var_21;
    wp::mat_t<3, 3, wp::float32> var_22;
    const wp::float32 var_23 = 20.0;
    wp::float32 var_24;
    wp::mat_t<3, 3, wp::float32> var_25;
    wp::mat_t<3, 3, wp::float32> var_26;
    wp::mat_t<3, 3, wp::float32> var_27;
    wp::mat_t<3, 3, wp::float32> var_28;
    wp::mat_t<3, 3, wp::float32> var_29;
    wp::mat_t<3, 3, wp::float32> var_30;
    wp::mat_t<3, 3, wp::float32> var_31;
    //---------
    // forward
    // def triangle_inertia(                                                                  <L 302>
    // vol = wp.dot(v0, wp.cross(v1, v2)) / 6.0  # tetra volume (0,v0,v1,v2)                  <L 307>
    var_0 = wp::cross(var_v1, var_v2);
    var_1 = wp::dot(var_v0, var_0);
    var_3 = wp::div(var_1, var_2);
    // first = vol * (v0 + v1 + v2) / 4.0  # first-order integral                             <L 308>
    var_4 = wp::add(var_v0, var_v1);
    var_5 = wp::add(var_4, var_v2);
    var_6 = wp::mul(var_3, var_5);
    var_8 = wp::div(var_6, var_7);
    // o00, o11, o22 = wp.outer(v0, v0), wp.outer(v1, v1), wp.outer(v2, v2)                   <L 311>
    var_9 = wp::outer(var_v0, var_v0);
    var_10 = wp::outer(var_v1, var_v1);
    var_11 = wp::outer(var_v2, var_v2);
    // o01, o02, o12 = wp.outer(v0, v1), wp.outer(v0, v2), wp.outer(v1, v2)                   <L 312>
    var_12 = wp::outer(var_v0, var_v1);
    var_13 = wp::outer(var_v0, var_v2);
    var_14 = wp::outer(var_v1, var_v2);
    // o01t, o02t, o12t = wp.transpose(o01), wp.transpose(o02), wp.transpose(o12)             <L 313>
    var_15 = wp::transpose(var_12);
    var_16 = wp::transpose(var_13);
    var_17 = wp::transpose(var_14);
    // second = (vol / 10.0) * (o00 + o11 + o22)                                              <L 315>
    var_19 = wp::div(var_3, var_18);
    var_20 = wp::add(var_9, var_10);
    var_21 = wp::add(var_20, var_11);
    var_22 = wp::mul(var_19, var_21);
    // second += (vol / 20.0) * (o01 + o01t + o02 + o02t + o12 + o12t)                        <L 316>
    var_24 = wp::div(var_3, var_23);
    var_25 = wp::add(var_12, var_15);
    var_26 = wp::add(var_25, var_13);
    var_27 = wp::add(var_26, var_16);
    var_28 = wp::add(var_27, var_14);
    var_29 = wp::add(var_28, var_17);
    var_30 = wp::mul(var_24, var_29);
    var_31 = wp::add(var_22, var_30);
    // return vol, first, second                                                              <L 318>
    ret_0 = var_3;
    ret_1 = var_8;
    ret_2 = var_31;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/inertia.py:321
static CUDA_CALLABLE void store_mesh_inertia_sum_0(
    wp::float32 var_v_sum,
    wp::vec_t<3, wp::float32> var_f_sum,
    wp::mat_t<3, 3, wp::float32> var_s_sum,
    wp::int32 var_lane,
    wp::array_t<wp::float32> var_volume,
    wp::array_t<wp::vec_t<3, wp::float32>> var_first,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_second)
{
    //---------
    // primal vars
    wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>> var_0 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>>{};
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_1 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    const wp::int32 var_4 = 0;
    wp::float32 var_5;
    wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>> var_6 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>>{};
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_7 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
    const wp::int32 var_8 = 0;
    wp::float32 var_9;
    const wp::int32 var_10 = 1;
    wp::float32 var_11;
    wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>> var_12 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>>{};
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_13 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
    const wp::int32 var_14 = 0;
    wp::float32 var_15;
    const wp::int32 var_16 = 2;
    wp::float32 var_17;
    wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>> var_18 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>>{};
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_19 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
    const wp::int32 var_20 = 0;
    wp::float32 var_21;
    const wp::int32 var_22 = 0;
    const wp::int32 var_23 = 0;
    wp::float32 var_24;
    wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>> var_25 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>>{};
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_26 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
    const wp::int32 var_27 = 0;
    wp::float32 var_28;
    const wp::int32 var_29 = 0;
    const wp::int32 var_30 = 1;
    wp::float32 var_31;
    wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>> var_32 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>>{};
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_33 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
    const wp::int32 var_34 = 0;
    wp::float32 var_35;
    const wp::int32 var_36 = 0;
    const wp::int32 var_37 = 2;
    wp::float32 var_38;
    wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>> var_39 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>>{};
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_40 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
    const wp::int32 var_41 = 0;
    wp::float32 var_42;
    const wp::int32 var_43 = 1;
    const wp::int32 var_44 = 0;
    wp::float32 var_45;
    wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>> var_46 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>>{};
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_47 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
    const wp::int32 var_48 = 0;
    wp::float32 var_49;
    const wp::int32 var_50 = 1;
    const wp::int32 var_51 = 1;
    wp::float32 var_52;
    wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>> var_53 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>>{};
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_54 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
    const wp::int32 var_55 = 0;
    wp::float32 var_56;
    const wp::int32 var_57 = 1;
    const wp::int32 var_58 = 2;
    wp::float32 var_59;
    wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>> var_60 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>>{};
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_61 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
    const wp::int32 var_62 = 0;
    wp::float32 var_63;
    const wp::int32 var_64 = 2;
    const wp::int32 var_65 = 0;
    wp::float32 var_66;
    wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>> var_67 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>>{};
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_68 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
    const wp::int32 var_69 = 0;
    wp::float32 var_70;
    const wp::int32 var_71 = 2;
    const wp::int32 var_72 = 1;
    wp::float32 var_73;
    wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>> var_74 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>>{};
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_75 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
    const wp::int32 var_76 = 0;
    wp::float32 var_77;
    const wp::int32 var_78 = 2;
    const wp::int32 var_79 = 2;
    wp::float32 var_80;
    wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>> var_81 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>>{};
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_82 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
    const wp::int32 var_83 = 0;
    wp::float32 var_84;
    const wp::int32 var_85 = 0;
    bool var_86;
    const wp::int32 var_87 = 0;
    wp::vec_t<3, wp::float32> var_88;
    const wp::int32 var_89 = 0;
    wp::mat_t<3, 3, wp::float32> var_90;
    const wp::int32 var_91 = 0;
    //---------
    // forward
    // def store_mesh_inertia_sum(                                                            <L 322>
    // v_total = wp.tile_sum(wp.tile(v_sum))[0]                                               <L 331>
    var_0 = wp::tile<wp::float32>(var_v_sum);
    var_1 = wp::tile_sum(var_0);
    var_3 = wp::tile_extract(var_1, var_2);
    // fx_total = wp.tile_sum(wp.tile(f_sum[0]))[0]                                           <L 332>
    var_5 = wp::extract(var_f_sum, var_4);
    var_6 = wp::tile<wp::float32>(var_5);
    var_7 = wp::tile_sum(var_6);
    var_9 = wp::tile_extract(var_7, var_8);
    // fy_total = wp.tile_sum(wp.tile(f_sum[1]))[0]                                           <L 333>
    var_11 = wp::extract(var_f_sum, var_10);
    var_12 = wp::tile<wp::float32>(var_11);
    var_13 = wp::tile_sum(var_12);
    var_15 = wp::tile_extract(var_13, var_14);
    // fz_total = wp.tile_sum(wp.tile(f_sum[2]))[0]                                           <L 334>
    var_17 = wp::extract(var_f_sum, var_16);
    var_18 = wp::tile<wp::float32>(var_17);
    var_19 = wp::tile_sum(var_18);
    var_21 = wp::tile_extract(var_19, var_20);
    // s00_total = wp.tile_sum(wp.tile(s_sum[0, 0]))[0]                                       <L 335>
    var_24 = wp::extract(var_s_sum, var_22, var_23);
    var_25 = wp::tile<wp::float32>(var_24);
    var_26 = wp::tile_sum(var_25);
    var_28 = wp::tile_extract(var_26, var_27);
    // s01_total = wp.tile_sum(wp.tile(s_sum[0, 1]))[0]                                       <L 336>
    var_31 = wp::extract(var_s_sum, var_29, var_30);
    var_32 = wp::tile<wp::float32>(var_31);
    var_33 = wp::tile_sum(var_32);
    var_35 = wp::tile_extract(var_33, var_34);
    // s02_total = wp.tile_sum(wp.tile(s_sum[0, 2]))[0]                                       <L 337>
    var_38 = wp::extract(var_s_sum, var_36, var_37);
    var_39 = wp::tile<wp::float32>(var_38);
    var_40 = wp::tile_sum(var_39);
    var_42 = wp::tile_extract(var_40, var_41);
    // s10_total = wp.tile_sum(wp.tile(s_sum[1, 0]))[0]                                       <L 338>
    var_45 = wp::extract(var_s_sum, var_43, var_44);
    var_46 = wp::tile<wp::float32>(var_45);
    var_47 = wp::tile_sum(var_46);
    var_49 = wp::tile_extract(var_47, var_48);
    // s11_total = wp.tile_sum(wp.tile(s_sum[1, 1]))[0]                                       <L 339>
    var_52 = wp::extract(var_s_sum, var_50, var_51);
    var_53 = wp::tile<wp::float32>(var_52);
    var_54 = wp::tile_sum(var_53);
    var_56 = wp::tile_extract(var_54, var_55);
    // s12_total = wp.tile_sum(wp.tile(s_sum[1, 2]))[0]                                       <L 340>
    var_59 = wp::extract(var_s_sum, var_57, var_58);
    var_60 = wp::tile<wp::float32>(var_59);
    var_61 = wp::tile_sum(var_60);
    var_63 = wp::tile_extract(var_61, var_62);
    // s20_total = wp.tile_sum(wp.tile(s_sum[2, 0]))[0]                                       <L 341>
    var_66 = wp::extract(var_s_sum, var_64, var_65);
    var_67 = wp::tile<wp::float32>(var_66);
    var_68 = wp::tile_sum(var_67);
    var_70 = wp::tile_extract(var_68, var_69);
    // s21_total = wp.tile_sum(wp.tile(s_sum[2, 1]))[0]                                       <L 342>
    var_73 = wp::extract(var_s_sum, var_71, var_72);
    var_74 = wp::tile<wp::float32>(var_73);
    var_75 = wp::tile_sum(var_74);
    var_77 = wp::tile_extract(var_75, var_76);
    // s22_total = wp.tile_sum(wp.tile(s_sum[2, 2]))[0]                                       <L 343>
    var_80 = wp::extract(var_s_sum, var_78, var_79);
    var_81 = wp::tile<wp::float32>(var_80);
    var_82 = wp::tile_sum(var_81);
    var_84 = wp::tile_extract(var_82, var_83);
    // if lane == 0:                                                                          <L 345>
    var_86 = (var_lane == var_85);
    if (var_86) {
        // volume[0] = v_total                                                                <L 346>
        wp::array_store(var_volume, var_87, var_3);
        // first[0] = wp.vec3(fx_total, fy_total, fz_total)                                   <L 347>
        var_88 = wp::vec_t<3, wp::float32>(var_9, var_15, var_21);
        wp::array_store(var_first, var_89, var_88);
        // second[0] = wp.mat33(                                                              <L 348>
        // s00_total,                                                                         <L 349>
        // s01_total,                                                                         <L 350>
        // s02_total,                                                                         <L 351>
        // s10_total,                                                                         <L 352>
        // s11_total,                                                                         <L 353>
        // s12_total,                                                                         <L 354>
        // s20_total,                                                                         <L 355>
        // s21_total,                                                                         <L 356>
        // s22_total,                                                                         <L 357>
        var_90 = wp::mat_t<3, 3, wp::float32>(var_28, var_35, var_42, var_49, var_56, var_63, var_70, var_77, var_84);
        // second[0] = wp.mat33(                                                              <L 348>
        wp::array_store(var_second, var_91, var_90);
    }
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/inertia.py:301
static CUDA_CALLABLE void adj_triangle_inertia_0(
    wp::vec_t<3, wp::float32> var_v0,
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::mat_t<3, 3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_v0,
    wp::vec_t<3, wp::float32> & adj_v1,
    wp::vec_t<3, wp::float32> & adj_v2,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::mat_t<3, 3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/inertia.py:321
static CUDA_CALLABLE void adj_store_mesh_inertia_sum_0(
    wp::float32 var_v_sum,
    wp::vec_t<3, wp::float32> var_f_sum,
    wp::mat_t<3, 3, wp::float32> var_s_sum,
    wp::int32 var_lane,
    wp::array_t<wp::float32> var_volume,
    wp::array_t<wp::vec_t<3, wp::float32>> var_first,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_second,
    wp::float32 & adj_v_sum,
    wp::vec_t<3, wp::float32> & adj_f_sum,
    wp::mat_t<3, 3, wp::float32> & adj_s_sum,
    wp::int32 & adj_lane,
    wp::array_t<wp::float32> & adj_volume,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_first,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_second)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void compute_solid_mesh_inertia_7f9d593c_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertices,
    wp::int32 var_num_tris,
    wp::array_t<wp::float32> var_volume,
    wp::array_t<wp::vec_t<3, wp::float32>> var_first,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_second)
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
        const wp::float32 var_2 = 0.0;
        wp::float32 var_3;
        const wp::float32 var_4 = 0.0;
        wp::vec_t<3, wp::float32> var_5;
        const wp::float32 var_6 = 0.0;
        wp::mat_t<3, 3, wp::float32> var_7;
        wp::int32 var_8;
        wp::range_t var_9;
        wp::int32 var_10;
        const wp::int32 var_11 = 3;
        wp::int32 var_12;
        const wp::int32 var_13 = 0;
        wp::int32 var_14;
        wp::int32* var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::int32 var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        const wp::int32 var_20 = 3;
        wp::int32 var_21;
        const wp::int32 var_22 = 1;
        wp::int32 var_23;
        wp::int32* var_24;
        wp::vec_t<3, wp::float32>* var_25;
        wp::int32 var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        const wp::int32 var_29 = 3;
        wp::int32 var_30;
        const wp::int32 var_31 = 2;
        wp::int32 var_32;
        wp::int32* var_33;
        wp::vec_t<3, wp::float32>* var_34;
        wp::int32 var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::float32 var_38;
        wp::vec_t<3, wp::float32> var_39;
        wp::mat_t<3, 3, wp::float32> var_40;
        wp::float32 var_41;
        wp::vec_t<3, wp::float32> var_42;
        wp::mat_t<3, 3, wp::float32> var_43;
        //---------
        // forward
        // def compute_solid_mesh_inertia(                                                        <L 362>
        // _block_id, lane = wp.tid()                                                             <L 371>
        builtin_tid2d(var_0, var_1);
        // v_sum = float(0.0)                                                                     <L 373>
        var_3 = wp::float(var_2);
        // f_sum = wp.vec3(0.0)                                                                   <L 374>
        var_5 = wp::vec_t<3, wp::float32>(var_4);
        // s_sum = wp.mat33(0.0)                                                                  <L 375>
        var_7 = wp::mat_t<3, 3, wp::float32>(var_6);
        // for tri in range(lane, num_tris, wp.block_dim()):                                      <L 377>
        var_8 = builtin_block_dim();
        var_9 = wp::range(var_1, var_num_tris, var_8);
        start_for_0:;
            if (iter_cmp(var_9) == 0) goto end_for_0;
            var_10 = wp::iter_next(var_9);
            // p = vertices[indices[tri * 3 + 0]]                                                 <L 378>
            var_12 = wp::mul(var_10, var_11);
            var_14 = wp::add(var_12, var_13);
            var_15 = wp::address(var_indices, var_14);
            var_17 = wp::load(var_15);
            var_16 = wp::address(var_vertices, var_17);
            var_19 = wp::load(var_16);
            var_18 = wp::copy(var_19);
            // q = vertices[indices[tri * 3 + 1]]                                                 <L 379>
            var_21 = wp::mul(var_10, var_20);
            var_23 = wp::add(var_21, var_22);
            var_24 = wp::address(var_indices, var_23);
            var_26 = wp::load(var_24);
            var_25 = wp::address(var_vertices, var_26);
            var_28 = wp::load(var_25);
            var_27 = wp::copy(var_28);
            // r = vertices[indices[tri * 3 + 2]]                                                 <L 380>
            var_30 = wp::mul(var_10, var_29);
            var_32 = wp::add(var_30, var_31);
            var_33 = wp::address(var_indices, var_32);
            var_35 = wp::load(var_33);
            var_34 = wp::address(var_vertices, var_35);
            var_37 = wp::load(var_34);
            var_36 = wp::copy(var_37);
            // v, f, s = triangle_inertia(p, q, r)                                                <L 382>
            triangle_inertia_0(var_18, var_27, var_36, var_38, var_39, var_40);
            // v_sum += v                                                                         <L 383>
            var_41 = wp::add(var_3, var_38);
            // f_sum += f                                                                         <L 384>
            var_42 = wp::add(var_5, var_39);
            // s_sum += s                                                                         <L 385>
            var_43 = wp::add(var_7, var_40);
            wp::assign(var_3, var_41);
            wp::assign(var_5, var_42);
            wp::assign(var_7, var_43);
            goto start_for_0;
        end_for_0:;
        // store_mesh_inertia_sum(v_sum, f_sum, s_sum, lane, volume, first, second)               <L 387>
        store_mesh_inertia_sum_0(var_3, var_5, var_7, var_1, var_volume, var_first, var_second);
    }
}



extern "C" __global__ void compute_hollow_mesh_inertia_ebcbb4ba_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertices,
    wp::array_t<wp::float32> var_thickness,
    wp::int32 var_num_tris,
    wp::array_t<wp::float32> var_volume,
    wp::array_t<wp::vec_t<3, wp::float32>> var_first,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_second)
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
        const wp::float32 var_2 = 0.0;
        wp::float32 var_3;
        const wp::float32 var_4 = 0.0;
        wp::vec_t<3, wp::float32> var_5;
        const wp::float32 var_6 = 0.0;
        wp::mat_t<3, 3, wp::float32> var_7;
        wp::int32 var_8;
        wp::range_t var_9;
        wp::int32 var_10;
        const wp::int32 var_11 = 3;
        wp::int32 var_12;
        const wp::int32 var_13 = 0;
        wp::int32 var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        const wp::int32 var_18 = 3;
        wp::int32 var_19;
        const wp::int32 var_20 = 1;
        wp::int32 var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        const wp::int32 var_25 = 3;
        wp::int32 var_26;
        const wp::int32 var_27 = 2;
        wp::int32 var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        wp::vec_t<3, wp::float32>* var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::vec_t<3, wp::float32>* var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::vec_t<3, wp::float32>* var_38;
        wp::vec_t<3, wp::float32> var_39;
        wp::vec_t<3, wp::float32> var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::vec_t<3, wp::float32> var_42;
        wp::vec_t<3, wp::float32> var_43;
        wp::vec_t<3, wp::float32> var_44;
        wp::vec_t<3, wp::float32> var_45;
        wp::float32* var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::float32 var_48;
        wp::float32* var_49;
        wp::vec_t<3, wp::float32> var_50;
        wp::float32 var_51;
        wp::float32* var_52;
        wp::vec_t<3, wp::float32> var_53;
        wp::float32 var_54;
        wp::vec_t<3, wp::float32> var_55;
        wp::vec_t<3, wp::float32> var_56;
        wp::vec_t<3, wp::float32> var_57;
        wp::vec_t<3, wp::float32> var_58;
        wp::vec_t<3, wp::float32> var_59;
        wp::vec_t<3, wp::float32> var_60;
        const wp::float32 var_61 = 0.0;
        wp::float32 var_62;
        const wp::float32 var_63 = 0.0;
        wp::vec_t<3, wp::float32> var_64;
        const wp::float32 var_65 = 0.0;
        wp::mat_t<3, 3, wp::float32> var_66;
        wp::float32 var_67;
        wp::vec_t<3, wp::float32> var_68;
        wp::mat_t<3, 3, wp::float32> var_69;
        wp::float32 var_70;
        wp::vec_t<3, wp::float32> var_71;
        wp::mat_t<3, 3, wp::float32> var_72;
        wp::float32 var_73;
        wp::vec_t<3, wp::float32> var_74;
        wp::mat_t<3, 3, wp::float32> var_75;
        wp::float32 var_76;
        wp::vec_t<3, wp::float32> var_77;
        wp::mat_t<3, 3, wp::float32> var_78;
        wp::float32 var_79;
        wp::vec_t<3, wp::float32> var_80;
        wp::mat_t<3, 3, wp::float32> var_81;
        wp::float32 var_82;
        wp::vec_t<3, wp::float32> var_83;
        wp::mat_t<3, 3, wp::float32> var_84;
        wp::float32 var_85;
        wp::vec_t<3, wp::float32> var_86;
        wp::mat_t<3, 3, wp::float32> var_87;
        wp::float32 var_88;
        wp::vec_t<3, wp::float32> var_89;
        wp::mat_t<3, 3, wp::float32> var_90;
        wp::float32 var_91;
        wp::vec_t<3, wp::float32> var_92;
        wp::mat_t<3, 3, wp::float32> var_93;
        wp::float32 var_94;
        wp::vec_t<3, wp::float32> var_95;
        wp::mat_t<3, 3, wp::float32> var_96;
        wp::float32 var_97;
        wp::vec_t<3, wp::float32> var_98;
        wp::mat_t<3, 3, wp::float32> var_99;
        wp::float32 var_100;
        wp::vec_t<3, wp::float32> var_101;
        wp::mat_t<3, 3, wp::float32> var_102;
        wp::float32 var_103;
        wp::vec_t<3, wp::float32> var_104;
        wp::mat_t<3, 3, wp::float32> var_105;
        wp::float32 var_106;
        wp::vec_t<3, wp::float32> var_107;
        wp::mat_t<3, 3, wp::float32> var_108;
        wp::float32 var_109;
        wp::vec_t<3, wp::float32> var_110;
        wp::mat_t<3, 3, wp::float32> var_111;
        wp::float32 var_112;
        wp::vec_t<3, wp::float32> var_113;
        wp::mat_t<3, 3, wp::float32> var_114;
        wp::float32 var_115;
        wp::vec_t<3, wp::float32> var_116;
        wp::mat_t<3, 3, wp::float32> var_117;
        //---------
        // forward
        // def compute_hollow_mesh_inertia(                                                       <L 391>
        // _block_id, lane = wp.tid()                                                             <L 401>
        builtin_tid2d(var_0, var_1);
        // v_sum = float(0.0)                                                                     <L 403>
        var_3 = wp::float(var_2);
        // f_sum = wp.vec3(0.0)                                                                   <L 404>
        var_5 = wp::vec_t<3, wp::float32>(var_4);
        // s_sum = wp.mat33(0.0)                                                                  <L 405>
        var_7 = wp::mat_t<3, 3, wp::float32>(var_6);
        // for tid in range(lane, num_tris, wp.block_dim()):                                      <L 407>
        var_8 = builtin_block_dim();
        var_9 = wp::range(var_1, var_num_tris, var_8);
        start_for_0:;
            if (iter_cmp(var_9) == 0) goto end_for_0;
            var_10 = wp::iter_next(var_9);
            // i = indices[tid * 3 + 0]                                                           <L 408>
            var_12 = wp::mul(var_10, var_11);
            var_14 = wp::add(var_12, var_13);
            var_15 = wp::address(var_indices, var_14);
            var_17 = wp::load(var_15);
            var_16 = wp::copy(var_17);
            // j = indices[tid * 3 + 1]                                                           <L 409>
            var_19 = wp::mul(var_10, var_18);
            var_21 = wp::add(var_19, var_20);
            var_22 = wp::address(var_indices, var_21);
            var_24 = wp::load(var_22);
            var_23 = wp::copy(var_24);
            // k = indices[tid * 3 + 2]                                                           <L 410>
            var_26 = wp::mul(var_10, var_25);
            var_28 = wp::add(var_26, var_27);
            var_29 = wp::address(var_indices, var_28);
            var_31 = wp::load(var_29);
            var_30 = wp::copy(var_31);
            // vi = vertices[i]                                                                   <L 412>
            var_32 = wp::address(var_vertices, var_16);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            // vj = vertices[j]                                                                   <L 413>
            var_35 = wp::address(var_vertices, var_23);
            var_37 = wp::load(var_35);
            var_36 = wp::copy(var_37);
            // vk = vertices[k]                                                                   <L 414>
            var_38 = wp::address(var_vertices, var_30);
            var_40 = wp::load(var_38);
            var_39 = wp::copy(var_40);
            // normal = -wp.normalize(wp.cross(vj - vi, vk - vi))                                 <L 416>
            var_41 = wp::sub(var_36, var_33);
            var_42 = wp::sub(var_39, var_33);
            var_43 = wp::cross(var_41, var_42);
            var_44 = wp::normalize(var_43);
            var_45 = wp::neg(var_44);
            // ti = normal * thickness[i]                                                         <L 417>
            var_46 = wp::address(var_thickness, var_16);
            var_48 = wp::load(var_46);
            var_47 = wp::mul(var_45, var_48);
            // tj = normal * thickness[j]                                                         <L 418>
            var_49 = wp::address(var_thickness, var_23);
            var_51 = wp::load(var_49);
            var_50 = wp::mul(var_45, var_51);
            // tk = normal * thickness[k]                                                         <L 419>
            var_52 = wp::address(var_thickness, var_30);
            var_54 = wp::load(var_52);
            var_53 = wp::mul(var_45, var_54);
            // vi0 = vi - ti                                                                      <L 422>
            var_55 = wp::sub(var_33, var_47);
            // vi1 = vi + ti                                                                      <L 423>
            var_56 = wp::add(var_33, var_47);
            // vj0 = vj - tj                                                                      <L 424>
            var_57 = wp::sub(var_36, var_50);
            // vj1 = vj + tj                                                                      <L 425>
            var_58 = wp::add(var_36, var_50);
            // vk0 = vk - tk                                                                      <L 426>
            var_59 = wp::sub(var_39, var_53);
            // vk1 = vk + tk                                                                      <L 427>
            var_60 = wp::add(var_39, var_53);
            // v_total = float(0.0)                                                               <L 429>
            var_62 = wp::float(var_61);
            // f_total = wp.vec3(0.0)                                                             <L 430>
            var_64 = wp::vec_t<3, wp::float32>(var_63);
            // s_total = wp.mat33(0.0)                                                            <L 431>
            var_66 = wp::mat_t<3, 3, wp::float32>(var_65);
            // v, f, s = triangle_inertia(vi0, vj0, vk0)                                          <L 433>
            triangle_inertia_0(var_55, var_57, var_59, var_67, var_68, var_69);
            // v_total += v                                                                       <L 434>
            var_70 = wp::add(var_62, var_67);
            // f_total += f                                                                       <L 435>
            var_71 = wp::add(var_64, var_68);
            // s_total += s                                                                       <L 436>
            var_72 = wp::add(var_66, var_69);
            // v, f, s = triangle_inertia(vj0, vk1, vk0)                                          <L 437>
            triangle_inertia_0(var_57, var_60, var_59, var_73, var_74, var_75);
            // v_total += v                                                                       <L 438>
            var_76 = wp::add(var_70, var_73);
            // f_total += f                                                                       <L 439>
            var_77 = wp::add(var_71, var_74);
            // s_total += s                                                                       <L 440>
            var_78 = wp::add(var_72, var_75);
            // v, f, s = triangle_inertia(vj0, vj1, vk1)                                          <L 441>
            triangle_inertia_0(var_57, var_58, var_60, var_79, var_80, var_81);
            // v_total += v                                                                       <L 442>
            var_82 = wp::add(var_76, var_79);
            // f_total += f                                                                       <L 443>
            var_83 = wp::add(var_77, var_80);
            // s_total += s                                                                       <L 444>
            var_84 = wp::add(var_78, var_81);
            // v, f, s = triangle_inertia(vj0, vi1, vj1)                                          <L 445>
            triangle_inertia_0(var_57, var_56, var_58, var_85, var_86, var_87);
            // v_total += v                                                                       <L 446>
            var_88 = wp::add(var_82, var_85);
            // f_total += f                                                                       <L 447>
            var_89 = wp::add(var_83, var_86);
            // s_total += s                                                                       <L 448>
            var_90 = wp::add(var_84, var_87);
            // v, f, s = triangle_inertia(vj0, vi0, vi1)                                          <L 449>
            triangle_inertia_0(var_57, var_55, var_56, var_91, var_92, var_93);
            // v_total += v                                                                       <L 450>
            var_94 = wp::add(var_88, var_91);
            // f_total += f                                                                       <L 451>
            var_95 = wp::add(var_89, var_92);
            // s_total += s                                                                       <L 452>
            var_96 = wp::add(var_90, var_93);
            // v, f, s = triangle_inertia(vj1, vi1, vk1)                                          <L 453>
            triangle_inertia_0(var_58, var_56, var_60, var_97, var_98, var_99);
            // v_total += v                                                                       <L 454>
            var_100 = wp::add(var_94, var_97);
            // f_total += f                                                                       <L 455>
            var_101 = wp::add(var_95, var_98);
            // s_total += s                                                                       <L 456>
            var_102 = wp::add(var_96, var_99);
            // v, f, s = triangle_inertia(vi1, vi0, vk0)                                          <L 457>
            triangle_inertia_0(var_56, var_55, var_59, var_103, var_104, var_105);
            // v_total += v                                                                       <L 458>
            var_106 = wp::add(var_100, var_103);
            // f_total += f                                                                       <L 459>
            var_107 = wp::add(var_101, var_104);
            // s_total += s                                                                       <L 460>
            var_108 = wp::add(var_102, var_105);
            // v, f, s = triangle_inertia(vi1, vk0, vk1)                                          <L 461>
            triangle_inertia_0(var_56, var_59, var_60, var_109, var_110, var_111);
            // v_total += v                                                                       <L 462>
            var_112 = wp::add(var_106, var_109);
            // f_total += f                                                                       <L 463>
            var_113 = wp::add(var_107, var_110);
            // s_total += s                                                                       <L 464>
            var_114 = wp::add(var_108, var_111);
            // v_sum += v_total                                                                   <L 466>
            var_115 = wp::add(var_3, var_112);
            // f_sum += f_total                                                                   <L 467>
            var_116 = wp::add(var_5, var_113);
            // s_sum += s_total                                                                   <L 468>
            var_117 = wp::add(var_7, var_114);
            wp::assign(var_3, var_115);
            wp::assign(var_5, var_116);
            wp::assign(var_7, var_117);
            goto start_for_0;
        end_for_0:;
        // store_mesh_inertia_sum(v_sum, f_sum, s_sum, lane, volume, first, second)               <L 470>
        store_mesh_inertia_sum_0(var_3, var_5, var_7, var_1, var_volume, var_first, var_second);
    }
}

