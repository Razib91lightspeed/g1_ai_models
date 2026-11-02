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


struct GenericShapeData_ba895271
{
    wp::int32 shape_type;
    wp::vec_t<3, wp::float32> scale;
    wp::vec_t<3, wp::float32> auxiliary;
    wp::vec_t<3, wp::float32> center;
    wp::int32 shape_index;


    GenericShapeData_ba895271() = default;
    CUDA_CALLABLE GenericShapeData_ba895271(wp::int32 const& shape_type,
    wp::vec_t<3, wp::float32> const& scale = {},
    wp::vec_t<3, wp::float32> const& auxiliary = {},
    wp::vec_t<3, wp::float32> const& center = {},
    wp::int32 const& shape_index = {})
        : shape_type{shape_type}
        , scale{scale}
        , auxiliary{auxiliary}
        , center{center}
        , shape_index{shape_index}

    {
    }

    CUDA_CALLABLE GenericShapeData_ba895271& operator += (const GenericShapeData_ba895271& rhs)
    {    shape_type += rhs.shape_type;
    scale += rhs.scale;
    auxiliary += rhs.auxiliary;
    center += rhs.center;
    shape_index += rhs.shape_index;

        return *this;}

};

static CUDA_CALLABLE void adj_GenericShapeData_ba895271(wp::int32 const&,
    wp::vec_t<3, wp::float32> const&,
    wp::vec_t<3, wp::float32> const&,
    wp::vec_t<3, wp::float32> const&,
    wp::int32 const&,
    wp::int32 & adj_shape_type,
    wp::vec_t<3, wp::float32> & adj_scale,
    wp::vec_t<3, wp::float32> & adj_auxiliary,
    wp::vec_t<3, wp::float32> & adj_center,
    wp::int32 & adj_shape_index,
    GenericShapeData_ba895271 & adj_ret)
{
    adj_shape_type += adj_ret.shape_type;
    adj_scale += adj_ret.scale;
    adj_auxiliary += adj_ret.auxiliary;
    adj_center += adj_ret.center;
    adj_shape_index += adj_ret.shape_index;
}

// Required when compiling adjoints.
CUDA_CALLABLE GenericShapeData_ba895271 add(const GenericShapeData_ba895271& a, const GenericShapeData_ba895271& b)
{
    return GenericShapeData_ba895271();
}

CUDA_CALLABLE void adj_atomic_add(GenericShapeData_ba895271* p, GenericShapeData_ba895271 t)
{
    wp::adj_atomic_add(&p->shape_type, t.shape_type);
    wp::adj_atomic_add(&p->scale, t.scale);
    wp::adj_atomic_add(&p->auxiliary, t.auxiliary);
    wp::adj_atomic_add(&p->center, t.center);
    wp::adj_atomic_add(&p->shape_index, t.shape_index);
}




struct SupportMapDataProvider_e77f8b9f
{
    char _dummy_;


    
    CUDA_CALLABLE SupportMapDataProvider_e77f8b9f()
    
    {
    }

    CUDA_CALLABLE SupportMapDataProvider_e77f8b9f& operator += (const SupportMapDataProvider_e77f8b9f& rhs)
    {
        return *this;}

};

static CUDA_CALLABLE void adj_SupportMapDataProvider_e77f8b9f(SupportMapDataProvider_e77f8b9f & adj_ret)
{
}

// Required when compiling adjoints.
CUDA_CALLABLE SupportMapDataProvider_e77f8b9f add(const SupportMapDataProvider_e77f8b9f& a, const SupportMapDataProvider_e77f8b9f& b)
{
    return SupportMapDataProvider_e77f8b9f();
}

CUDA_CALLABLE void adj_atomic_add(SupportMapDataProvider_e77f8b9f* p, SupportMapDataProvider_e77f8b9f t)
{
}




// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:81
static CUDA_CALLABLE wp::vec_t<3, wp::float32> pack_mesh_ptr_0(
    wp::uint64 var_ptr)
{
    //---------
    // primal vars
    wp::uint64 var_0;
    wp::uint64 var_1;
    wp::float32 var_2;
    wp::uint64 var_3;
    wp::uint64 var_4;
    wp::uint64 var_5;
    wp::uint64 var_6;
    wp::float32 var_7;
    wp::uint64 var_8;
    wp::uint64 var_9;
    wp::uint64 var_10;
    wp::uint64 var_11;
    wp::float32 var_12;
    wp::vec_t<3, wp::float32> var_13;
    //---------
    // forward
    // def pack_mesh_ptr(ptr: wp.uint64) -> wp.vec3:                                          <L 82>
    // chunk1 = float(ptr & wp.uint64(0x3FFFFF))  # bits 0-21                                 <L 85>
    var_0 = 4194303ull;
    var_1 = wp::bit_and(var_ptr, var_0);
    var_2 = wp::float(var_1);
    // chunk2 = float((ptr >> wp.uint64(22)) & wp.uint64(0x3FFFFF))  # bits 22-43             <L 86>
    var_3 = 22ull;
    var_4 = wp::rshift(var_ptr, var_3);
    var_5 = 4194303ull;
    var_6 = wp::bit_and(var_4, var_5);
    var_7 = wp::float(var_6);
    // chunk3 = float((ptr >> wp.uint64(44)) & wp.uint64(0xFFFFF))  # bits 44-63 (20 bits)       <L 87>
    var_8 = 44ull;
    var_9 = wp::rshift(var_ptr, var_8);
    var_10 = 1048575ull;
    var_11 = wp::bit_and(var_9, var_10);
    var_12 = wp::float(var_11);
    // return wp.vec3(chunk1, chunk2, chunk3)                                                 <L 89>
    var_13 = wp::vec_t<3, wp::float32>(var_2, var_7, var_12);
    return var_13;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:92
static CUDA_CALLABLE wp::uint64 unpack_mesh_ptr_0(
    wp::vec_t<3, wp::float32> var_arr)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    wp::uint64 var_2;
    wp::uint64 var_3;
    wp::uint64 var_4;
    const wp::int32 var_5 = 1;
    wp::float32 var_6;
    wp::uint64 var_7;
    wp::uint64 var_8;
    wp::uint64 var_9;
    wp::uint64 var_10;
    wp::uint64 var_11;
    const wp::int32 var_12 = 2;
    wp::float32 var_13;
    wp::uint64 var_14;
    wp::uint64 var_15;
    wp::uint64 var_16;
    wp::uint64 var_17;
    wp::uint64 var_18;
    wp::uint64 var_19;
    wp::uint64 var_20;
    //---------
    // forward
    // def unpack_mesh_ptr(arr: wp.vec3) -> wp.uint64:                                        <L 93>
    // chunk1 = wp.uint64(arr[0]) & wp.uint64(0x3FFFFF)                                       <L 96>
    var_1 = wp::extract(var_arr, var_0);
    var_2 = wp::uint64(var_1);
    var_3 = 4194303ull;
    var_4 = wp::bit_and(var_2, var_3);
    // chunk2 = (wp.uint64(arr[1]) & wp.uint64(0x3FFFFF)) << wp.uint64(22)                    <L 97>
    var_6 = wp::extract(var_arr, var_5);
    var_7 = wp::uint64(var_6);
    var_8 = 4194303ull;
    var_9 = wp::bit_and(var_7, var_8);
    var_10 = 22ull;
    var_11 = wp::lshift(var_9, var_10);
    // chunk3 = (wp.uint64(arr[2]) & wp.uint64(0xFFFFF)) << wp.uint64(44)                     <L 98>
    var_13 = wp::extract(var_arr, var_12);
    var_14 = wp::uint64(var_13);
    var_15 = 1048575ull;
    var_16 = wp::bit_and(var_14, var_15);
    var_17 = 44ull;
    var_18 = wp::lshift(var_16, var_17);
    // return chunk1 | chunk2 | chunk3                                                        <L 100>
    var_19 = wp::bit_or(var_4, var_11);
    var_20 = wp::bit_or(var_19, var_18);
    return var_20;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:194
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _support_map_convex_mesh_exhaustive_0(
    GenericShapeData_ba895271 var_geom,
    wp::vec_t<3, wp::float32> var_direction)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32>* var_0;
    wp::uint64 var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::Mesh var_3;
    wp::vec_t<3, wp::float32>* var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    const wp::int32 var_7 = 0;
    wp::int32 var_8;
    const wp::float32 var_9 = -10000000000.0;
    wp::float32 var_10;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_11;
    wp::shape_t* var_12;
    const wp::int32 var_13 = 0;
    wp::int32 var_14;
    wp::shape_t var_15;
    wp::range_t var_16;
    wp::int32 var_17;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_18;
    wp::vec_t<3, wp::float32>* var_19;
    wp::array_t<wp::vec_t<3, wp::float32>> var_20;
    wp::float32 var_21;
    wp::vec_t<3, wp::float32> var_22;
    bool var_23;
    wp::float32 var_24;
    wp::int32 var_25;
    wp::int32 var_26;
    wp::float32 var_27;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_28;
    wp::vec_t<3, wp::float32>* var_29;
    wp::array_t<wp::vec_t<3, wp::float32>> var_30;
    wp::vec_t<3, wp::float32>* var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    //---------
    // forward
    // def _support_map_convex_mesh_exhaustive(geom: GenericShapeData, direction: wp.vec3) -> wp.vec3:       <L 195>
    // mesh = wp.mesh_get(unpack_mesh_ptr(geom.auxiliary))                                    <L 197>
    var_0 = &((var_geom).auxiliary);
    var_2 = wp::load(var_0);
    var_1 = unpack_mesh_ptr_0(var_2);
    var_3 = wp::mesh_get(var_1);
    // scaled_dir = wp.cw_mul(direction, geom.scale)                                          <L 198>
    var_4 = &((var_geom).scale);
    var_6 = wp::load(var_4);
    var_5 = wp::cw_mul(var_direction, var_6);
    // best_idx = int(0)                                                                      <L 199>
    var_8 = wp::int(var_7);
    // max_dot = float(-1.0e10)                                                               <L 200>
    var_10 = wp::float(var_9);
    // for i in range(mesh.points.shape[0]):                                                  <L 201>
    var_11 = &((var_3).points);
    var_12 = &(var_11->shape);
    var_15 = wp::load(var_12);
    var_14 = wp::extract(var_15, var_13);
    var_16 = wp::range(var_14);
    start_for_0:;
        if (iter_cmp(var_16) == 0) goto end_for_0;
        var_17 = wp::iter_next(var_16);
        // dot_val = wp.dot(mesh.points[i], scaled_dir)                                       <L 202>
        var_18 = &((var_3).points);
        var_20 = wp::load(var_18);
        var_19 = wp::address(var_20, var_17);
        var_22 = wp::load(var_19);
        var_21 = wp::dot(var_22, var_5);
        // if dot_val > max_dot:                                                              <L 203>
        var_23 = (var_21 > var_10);
        if (var_23) {
            // max_dot = dot_val                                                              <L 204>
            var_24 = wp::copy(var_21);
            // best_idx = i                                                                   <L 205>
            var_25 = wp::copy(var_17);
        }
        var_26 = wp::where(var_23, var_25, var_8);
        var_27 = wp::where(var_23, var_24, var_10);
        wp::assign(var_8, var_26);
        wp::assign(var_10, var_27);
        goto start_for_0;
    end_for_0:;
    // return wp.cw_mul(mesh.points[best_idx], geom.scale)                                    <L 206>
    var_28 = &((var_3).points);
    var_30 = wp::load(var_28);
    var_29 = wp::address(var_30, var_8);
    var_31 = &((var_geom).scale);
    var_33 = wp::load(var_29);
    var_34 = wp::load(var_31);
    var_32 = wp::cw_mul(var_33, var_34);
    return var_32;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:209
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _support_map_box_0(
    GenericShapeData_ba895271 var_geom,
    wp::vec_t<3, wp::float32> var_direction)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::int32 var_3 = 1;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::int32 var_6 = 2;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::float32 var_11 = 1e-10;
    wp::float32 var_12;
    const wp::int32 var_13 = 0;
    wp::float32 var_14;
    wp::float32 var_15;
    bool var_16;
    const wp::float32 var_17 = 1.0;
    const wp::float32 var_18 = -1.0;
    wp::float32 var_19;
    const wp::int32 var_20 = 1;
    wp::float32 var_21;
    wp::float32 var_22;
    bool var_23;
    const wp::float32 var_24 = 1.0;
    const wp::float32 var_25 = -1.0;
    wp::float32 var_26;
    const wp::int32 var_27 = 2;
    wp::float32 var_28;
    wp::float32 var_29;
    bool var_30;
    const wp::float32 var_31 = 1.0;
    const wp::float32 var_32 = -1.0;
    wp::float32 var_33;
    wp::vec_t<3, wp::float32>* var_34;
    const wp::int32 var_35 = 0;
    wp::float32 var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::float32 var_38;
    wp::vec_t<3, wp::float32>* var_39;
    const wp::int32 var_40 = 1;
    wp::float32 var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::float32 var_43;
    wp::vec_t<3, wp::float32>* var_44;
    const wp::int32 var_45 = 2;
    wp::float32 var_46;
    wp::vec_t<3, wp::float32> var_47;
    wp::float32 var_48;
    wp::vec_t<3, wp::float32> var_49;
    //---------
    // forward
    // def _support_map_box(geom: GenericShapeData, direction: wp.vec3) -> wp.vec3:           <L 210>
    // direction_scale = wp.max(wp.abs(direction[0]), wp.max(wp.abs(direction[1]), wp.abs(direction[2])))       <L 212>
    var_1 = wp::extract(var_direction, var_0);
    var_2 = wp::abs(var_1);
    var_4 = wp::extract(var_direction, var_3);
    var_5 = wp::abs(var_4);
    var_7 = wp::extract(var_direction, var_6);
    var_8 = wp::abs(var_7);
    var_9 = wp::max(var_5, var_8);
    var_10 = wp::max(var_2, var_9);
    // threshold = BOX_SUPPORT_DEADBAND * direction_scale                                     <L 213>
    var_12 = wp::mul(var_11, var_10);
    // sx = 1.0 if direction[0] >= -threshold else -1.0                                       <L 214>
    var_14 = wp::extract(var_direction, var_13);
    var_15 = wp::neg(var_12);
    var_16 = (var_14 >= var_15);
    if (var_16) {
    }
    if (!var_16) {
    }
    var_19 = wp::where(var_16, var_17, var_18);
    // sy = 1.0 if direction[1] >= -threshold else -1.0                                       <L 215>
    var_21 = wp::extract(var_direction, var_20);
    var_22 = wp::neg(var_12);
    var_23 = (var_21 >= var_22);
    if (var_23) {
    }
    if (!var_23) {
    }
    var_26 = wp::where(var_23, var_24, var_25);
    // sz = 1.0 if direction[2] >= -threshold else -1.0                                       <L 216>
    var_28 = wp::extract(var_direction, var_27);
    var_29 = wp::neg(var_12);
    var_30 = (var_28 >= var_29);
    if (var_30) {
    }
    if (!var_30) {
    }
    var_33 = wp::where(var_30, var_31, var_32);
    // return wp.vec3(sx * geom.scale[0], sy * geom.scale[1], sz * geom.scale[2])             <L 217>
    var_34 = &((var_geom).scale);
    var_37 = wp::load(var_34);
    var_36 = wp::extract(var_37, var_35);
    var_38 = wp::mul(var_19, var_36);
    var_39 = &((var_geom).scale);
    var_42 = wp::load(var_39);
    var_41 = wp::extract(var_42, var_40);
    var_43 = wp::mul(var_26, var_41);
    var_44 = &((var_geom).scale);
    var_47 = wp::load(var_44);
    var_46 = wp::extract(var_47, var_45);
    var_48 = wp::mul(var_33, var_46);
    var_49 = wp::vec_t<3, wp::float32>(var_38, var_43, var_48);
    return var_49;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:46
static CUDA_CALLABLE wp::float32 _support_rsqrt_rn_0(
    wp::float32 value)
{

#if defined(__CUDA_ARCH__)
return __frsqrt_rn(value);
#else
return 1.0f / sqrtf(value);
#endif
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:0
static CUDA_CALLABLE wp::vec_t<3, wp::float32> support_map_0(
    GenericShapeData_ba895271 var_geom,
    wp::vec_t<3, wp::float32> var_direction,
    SupportMapDataProvider_e77f8b9f var_data_provider)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 1e-12;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 0.0;
    wp::vec_t<3, wp::float32> var_4;
    wp::int32* var_5;
    const wp::int32 var_6 = 10;
    bool var_7;
    wp::int32 var_8;
    wp::vec_t<3, wp::float32> var_9;
    bool var_10;
    wp::int32* var_11;
    const wp::int32 var_12 = 1000;
    bool var_13;
    wp::int32 var_14;
    wp::int32* var_15;
    const wp::int32 var_16 = 1001;
    bool var_17;
    wp::int32 var_18;
    const wp::float32 var_19 = 0.0;
    const wp::float32 var_20 = 0.0;
    const wp::float32 var_21 = 0.0;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32>* var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32>* var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    bool var_32;
    bool var_33;
    bool var_34;
    wp::vec_t<3, wp::float32> var_35;
    bool var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::vec_t<3, wp::float32> var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::int32* var_41;
    const wp::int32 var_42 = 1001;
    bool var_43;
    wp::int32 var_44;
    const wp::int32 var_45 = 2;
    wp::float32 var_46;
    const wp::float32 var_47 = 0.0;
    bool var_48;
    const wp::float32 var_49 = 0.0;
    const wp::float32 var_50 = 0.0;
    const wp::float32 var_51 = 1.0;
    const wp::float32 var_52 = -1.0;
    wp::vec_t<3, wp::float32> var_53;
    wp::vec_t<3, wp::float32> var_54;
    wp::vec_t<3, wp::float32> var_55;
    wp::vec_t<3, wp::float32> var_56;
    wp::int32* var_57;
    const wp::int32 var_58 = 7;
    bool var_59;
    wp::int32 var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::int32* var_62;
    const wp::int32 var_63 = 3;
    bool var_64;
    wp::int32 var_65;
    wp::vec_t<3, wp::float32>* var_66;
    const wp::int32 var_67 = 0;
    wp::float32 var_68;
    wp::vec_t<3, wp::float32> var_69;
    wp::float32 var_70;
    bool var_71;
    wp::float32 var_72;
    wp::vec_t<3, wp::float32> var_73;
    const wp::float32 var_74 = 1.0;
    const wp::float32 var_75 = 0.0;
    const wp::float32 var_76 = 0.0;
    wp::vec_t<3, wp::float32> var_77;
    wp::vec_t<3, wp::float32> var_78;
    wp::vec_t<3, wp::float32> var_79;
    wp::int32* var_80;
    const wp::int32 var_81 = 4;
    bool var_82;
    wp::int32 var_83;
    wp::vec_t<3, wp::float32>* var_84;
    const wp::int32 var_85 = 0;
    wp::float32 var_86;
    wp::vec_t<3, wp::float32> var_87;
    wp::vec_t<3, wp::float32>* var_88;
    const wp::int32 var_89 = 1;
    wp::float32 var_90;
    wp::vec_t<3, wp::float32> var_91;
    wp::float32 var_92;
    bool var_93;
    wp::float32 var_94;
    wp::vec_t<3, wp::float32> var_95;
    const wp::float32 var_96 = 1.0;
    const wp::float32 var_97 = 0.0;
    const wp::float32 var_98 = 0.0;
    wp::vec_t<3, wp::float32> var_99;
    wp::vec_t<3, wp::float32> var_100;
    wp::vec_t<3, wp::float32> var_101;
    const wp::int32 var_102 = 2;
    wp::float32 var_103;
    const wp::float32 var_104 = 0.0;
    bool var_105;
    const wp::float32 var_106 = 0.0;
    const wp::float32 var_107 = 0.0;
    wp::vec_t<3, wp::float32> var_108;
    wp::vec_t<3, wp::float32> var_109;
    const wp::float32 var_110 = 0.0;
    const wp::float32 var_111 = 0.0;
    wp::float32 var_112;
    wp::vec_t<3, wp::float32> var_113;
    wp::vec_t<3, wp::float32> var_114;
    wp::vec_t<3, wp::float32> var_115;
    wp::int32* var_116;
    const wp::int32 var_117 = 5;
    bool var_118;
    wp::int32 var_119;
    wp::vec_t<3, wp::float32>* var_120;
    const wp::int32 var_121 = 0;
    wp::float32 var_122;
    wp::vec_t<3, wp::float32> var_123;
    wp::vec_t<3, wp::float32>* var_124;
    const wp::int32 var_125 = 1;
    wp::float32 var_126;
    wp::vec_t<3, wp::float32> var_127;
    wp::vec_t<3, wp::float32>* var_128;
    const wp::int32 var_129 = 2;
    wp::float32 var_130;
    wp::vec_t<3, wp::float32> var_131;
    wp::float32 var_132;
    bool var_133;
    const wp::int32 var_134 = 0;
    wp::float32 var_135;
    wp::float32 var_136;
    const wp::int32 var_137 = 1;
    wp::float32 var_138;
    wp::float32 var_139;
    const wp::int32 var_140 = 2;
    wp::float32 var_141;
    wp::float32 var_142;
    wp::float32 var_143;
    wp::float32 var_144;
    wp::float32 var_145;
    wp::float32 var_146;
    wp::float32 var_147;
    bool var_148;
    wp::float32 var_149;
    wp::float32 var_150;
    const wp::int32 var_151 = 0;
    wp::float32 var_152;
    wp::float32 var_153;
    wp::float32 var_154;
    wp::float32 var_155;
    const wp::int32 var_156 = 1;
    wp::float32 var_157;
    wp::float32 var_158;
    wp::float32 var_159;
    wp::float32 var_160;
    const wp::int32 var_161 = 2;
    wp::float32 var_162;
    wp::float32 var_163;
    wp::float32 var_164;
    wp::vec_t<3, wp::float32> var_165;
    const wp::float32 var_166 = 0.0;
    const wp::float32 var_167 = 0.0;
    wp::vec_t<3, wp::float32> var_168;
    wp::vec_t<3, wp::float32> var_169;
    const wp::float32 var_170 = 0.0;
    const wp::float32 var_171 = 0.0;
    wp::vec_t<3, wp::float32> var_172;
    wp::vec_t<3, wp::float32> var_173;
    wp::int32* var_174;
    const wp::int32 var_175 = 6;
    bool var_176;
    wp::int32 var_177;
    wp::vec_t<3, wp::float32>* var_178;
    const wp::int32 var_179 = 0;
    wp::float32 var_180;
    wp::vec_t<3, wp::float32> var_181;
    wp::vec_t<3, wp::float32>* var_182;
    const wp::int32 var_183 = 1;
    wp::float32 var_184;
    wp::vec_t<3, wp::float32> var_185;
    wp::vec_t<3, wp::float32>* var_186;
    const wp::int32 var_187 = 2;
    wp::float32 var_188;
    wp::vec_t<3, wp::float32> var_189;
    const wp::int32 var_190 = 0;
    wp::float32 var_191;
    const wp::int32 var_192 = 1;
    wp::float32 var_193;
    const wp::float32 var_194 = 0.0;
    wp::vec_t<3, wp::float32> var_195;
    wp::float32 var_196;
    const wp::float32 var_197 = 0.0;
    bool var_198;
    bool var_199;
    wp::float32 var_200;
    wp::vec_t<3, wp::float32> var_201;
    const wp::int32 var_202 = 0;
    wp::float32 var_203;
    wp::float32 var_204;
    const wp::int32 var_205 = 1;
    wp::float32 var_206;
    wp::float32 var_207;
    const wp::float32 var_208 = 0.0;
    wp::vec_t<3, wp::float32> var_209;
    const wp::float32 var_210 = 0.0;
    const wp::float32 var_211 = 0.0;
    wp::vec_t<3, wp::float32> var_212;
    wp::vec_t<3, wp::float32> var_213;
    const wp::int32 var_214 = 2;
    wp::float32 var_215;
    const wp::float32 var_216 = 0.0;
    bool var_217;
    const wp::int32 var_218 = 0;
    wp::float32 var_219;
    const wp::int32 var_220 = 1;
    wp::float32 var_221;
    wp::vec_t<3, wp::float32> var_222;
    const wp::int32 var_223 = 2;
    wp::float32 var_224;
    const wp::float32 var_225 = 0.0;
    bool var_226;
    const wp::int32 var_227 = 0;
    wp::float32 var_228;
    const wp::int32 var_229 = 1;
    wp::float32 var_230;
    wp::float32 var_231;
    wp::vec_t<3, wp::float32> var_232;
    wp::vec_t<3, wp::float32> var_233;
    wp::vec_t<3, wp::float32> var_234;
    wp::vec_t<3, wp::float32> var_235;
    bool var_236;
    wp::float32 var_237;
    wp::vec_t<3, wp::float32> var_238;
    const wp::float32 var_239 = 0.0;
    const wp::float32 var_240 = 1.0;
    const wp::float32 var_241 = 0.0;
    const wp::float32 var_242 = 0.0;
    wp::vec_t<3, wp::float32> var_243;
    wp::float32 var_244;
    wp::vec_t<3, wp::float32> var_245;
    const wp::int32 var_246 = 2;
    wp::float32 var_247;
    const wp::int32 var_248 = 2;
    wp::float32 var_249;
    wp::float32 var_250;
    wp::float32 var_251;
    wp::float32 var_252;
    const wp::float32 var_253 = 0.0;
    bool var_254;
    const wp::int32 var_255 = 2;
    wp::float32 var_256;
    wp::float32 var_257;
    wp::float32 var_258;
    wp::float32 var_259;
    wp::float32 var_260;
    wp::float32 var_261;
    wp::float32 var_262;
    wp::float32 var_263;
    wp::float32 var_264;
    wp::float32 var_265;
    wp::float32 var_266;
    wp::float32 var_267;
    const wp::float32 var_268 = 0.0;
    wp::float32 var_269;
    wp::float32 var_270;
    wp::float32 var_271;
    wp::float32 var_272;
    bool var_273;
    wp::float32 var_274;
    wp::float32 var_275;
    wp::float32 var_276;
    wp::float32 var_277;
    const wp::int32 var_278 = 0;
    wp::float32 var_279;
    wp::float32 var_280;
    const wp::int32 var_281 = 1;
    wp::float32 var_282;
    wp::float32 var_283;
    wp::vec_t<3, wp::float32> var_284;
    wp::vec_t<3, wp::float32> var_285;
    wp::vec_t<3, wp::float32> var_286;
    wp::int32* var_287;
    const wp::int32 var_288 = 9;
    bool var_289;
    wp::int32 var_290;
    wp::vec_t<3, wp::float32>* var_291;
    const wp::int32 var_292 = 0;
    wp::float32 var_293;
    wp::vec_t<3, wp::float32> var_294;
    wp::vec_t<3, wp::float32>* var_295;
    const wp::int32 var_296 = 1;
    wp::float32 var_297;
    wp::vec_t<3, wp::float32> var_298;
    const wp::float32 var_299 = 0.0;
    const wp::float32 var_300 = 0.0;
    wp::vec_t<3, wp::float32> var_301;
    const wp::int32 var_302 = 0;
    wp::float32 var_303;
    const wp::int32 var_304 = 1;
    wp::float32 var_305;
    const wp::float32 var_306 = 0.0;
    wp::vec_t<3, wp::float32> var_307;
    wp::float32 var_308;
    bool var_309;
    const wp::float32 var_310 = 2.0;
    wp::float32 var_311;
    wp::float32 var_312;
    const wp::float32 var_313 = 0.0;
    wp::float32 var_314;
    bool var_315;
    const wp::int32 var_316 = 2;
    wp::float32 var_317;
    const wp::float32 var_318 = 0.0;
    bool var_319;
    wp::vec_t<3, wp::float32> var_320;
    const wp::float32 var_321 = 0.0;
    wp::float32 var_322;
    wp::vec_t<3, wp::float32> var_323;
    wp::vec_t<3, wp::float32> var_324;
    const wp::int32 var_325 = 2;
    wp::float32 var_326;
    wp::float32 var_327;
    bool var_328;
    wp::vec_t<3, wp::float32> var_329;
    wp::vec_t<3, wp::float32> var_330;
    const wp::int32 var_331 = 0;
    wp::float32 var_332;
    wp::float32 var_333;
    const wp::int32 var_334 = 1;
    wp::float32 var_335;
    wp::float32 var_336;
    wp::float32 var_337;
    wp::vec_t<3, wp::float32> var_338;
    wp::vec_t<3, wp::float32> var_339;
    wp::vec_t<3, wp::float32> var_340;
    wp::int32* var_341;
    const wp::int32 var_342 = 1;
    bool var_343;
    wp::int32 var_344;
    wp::vec_t<3, wp::float32>* var_345;
    const wp::int32 var_346 = 0;
    wp::float32 var_347;
    wp::vec_t<3, wp::float32> var_348;
    wp::vec_t<3, wp::float32>* var_349;
    const wp::int32 var_350 = 1;
    wp::float32 var_351;
    wp::vec_t<3, wp::float32> var_352;
    const wp::int32 var_353 = 0;
    wp::float32 var_354;
    const wp::float32 var_355 = 0.0;
    bool var_356;
    const wp::float32 var_357 = 1.0;
    const wp::float32 var_358 = -1.0;
    wp::float32 var_359;
    const wp::int32 var_360 = 1;
    wp::float32 var_361;
    const wp::float32 var_362 = 0.0;
    bool var_363;
    const wp::float32 var_364 = 1.0;
    const wp::float32 var_365 = -1.0;
    wp::float32 var_366;
    wp::float32 var_367;
    wp::float32 var_368;
    const wp::float32 var_369 = 0.0;
    wp::vec_t<3, wp::float32> var_370;
    const wp::float32 var_371 = 0.0;
    const wp::float32 var_372 = 0.0;
    const wp::float32 var_373 = 0.0;
    wp::vec_t<3, wp::float32> var_374;
    wp::vec_t<3, wp::float32> var_375;
    wp::vec_t<3, wp::float32> var_376;
    wp::vec_t<3, wp::float32> var_377;
    wp::float32 var_378;
    wp::float32 var_379;
    wp::vec_t<3, wp::float32> var_380;
    wp::vec_t<3, wp::float32> var_381;
    wp::float32 var_382;
    wp::vec_t<3, wp::float32> var_383;
    wp::vec_t<3, wp::float32> var_384;
    wp::float32 var_385;
    wp::float32 var_386;
    wp::float32 var_387;
    wp::vec_t<3, wp::float32> var_388;
    wp::float32 var_389;
    wp::float32 var_390;
    wp::vec_t<3, wp::float32> var_391;
    wp::vec_t<3, wp::float32> var_392;
    wp::vec_t<3, wp::float32> var_393;
    wp::vec_t<3, wp::float32> var_394;
    //---------
    // forward
    // def support_map(geom: GenericShapeData, direction: wp.vec3, data_provider: Any) -> wp.vec3:       <L 1>
    // eps = 1.0e-12                                                                          <L 17>
    // result = wp.vec3(0.0, 0.0, 0.0)                                                        <L 19>
    var_4 = wp::vec_t<3, wp::float32>(var_1, var_2, var_3);
    // if geom.shape_type == GeoType.CONVEX_MESH:                                             <L 21>
    var_5 = &((var_geom).shape_type);
    var_8 = wp::load(var_5);
    var_7 = (var_8 == var_6);
    if (var_7) {
        // result = _support_map_convex_mesh_exhaustive(geom, direction)                      <L 22>
        var_9 = _support_map_convex_mesh_exhaustive_0(var_geom, var_direction);
    }
    if (!var_7) {
        // elif geom.shape_type == GeoTypeEx.TRIANGLE or geom.shape_type == GeoTypeEx.TRIANGLE_PRISM:       <L 24>
        var_11 = &((var_geom).shape_type);
        var_14 = wp::load(var_11);
        var_13 = (var_14 == var_12);
        var_10 = var_13;
        if (!var_10) {
            var_15 = &((var_geom).shape_type);
            var_18 = wp::load(var_15);
            var_17 = (var_18 == var_16);
            var_10 = var_10 || var_17;
        }
        if (var_10) {
            // tri_a = wp.vec3(0.0, 0.0, 0.0)                                                 <L 26>
            var_22 = wp::vec_t<3, wp::float32>(var_19, var_20, var_21);
            // tri_b = geom.scale                                                             <L 27>
            var_23 = &((var_geom).scale);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // tri_c = geom.auxiliary                                                         <L 28>
            var_26 = &((var_geom).auxiliary);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // dot_a = wp.dot(tri_a, direction)                                               <L 31>
            var_29 = wp::dot(var_22, var_direction);
            // dot_b = wp.dot(tri_b, direction)                                               <L 32>
            var_30 = wp::dot(var_24, var_direction);
            // dot_c = wp.dot(tri_c, direction)                                               <L 33>
            var_31 = wp::dot(var_27, var_direction);
            // if dot_a >= dot_b and dot_a >= dot_c:                                          <L 36>
            var_33 = (var_29 >= var_30);
            var_32 = var_33;
            if (var_32) {
                var_34 = (var_29 >= var_31);
                var_32 = var_32 && var_34;
            }
            if (var_32) {
                // result = tri_a                                                             <L 37>
                var_35 = wp::copy(var_22);
            }
            if (!var_32) {
                // elif dot_b >= dot_c:                                                       <L 38>
                var_36 = (var_30 >= var_31);
                if (var_36) {
                    // result = tri_b                                                         <L 39>
                    var_37 = wp::copy(var_24);
                }
                if (!var_36) {
                    // result = tri_c                                                         <L 41>
                    var_38 = wp::copy(var_27);
                }
                var_39 = wp::where(var_36, var_37, var_38);
            }
            var_40 = wp::where(var_32, var_35, var_39);
            // if geom.shape_type == GeoTypeEx.TRIANGLE_PRISM:                                <L 48>
            var_41 = &((var_geom).shape_type);
            var_44 = wp::load(var_41);
            var_43 = (var_44 == var_42);
            if (var_43) {
                // if direction[2] < 0.0:                                                     <L 49>
                var_46 = wp::extract(var_direction, var_45);
                var_48 = (var_46 < var_47);
                if (var_48) {
                    // result = result + wp.vec3(0.0, 0.0, -TRIANGLE_PRISM_EXTRUSION)         <L 50>
                    var_53 = wp::vec_t<3, wp::float32>(var_49, var_50, var_52);
                    var_54 = wp::add(var_40, var_53);
                }
                var_55 = wp::where(var_48, var_54, var_40);
            }
            var_56 = wp::where(var_43, var_55, var_40);
        }
        if (!var_10) {
            // elif geom.shape_type == GeoType.BOX:                                           <L 51>
            var_57 = &((var_geom).shape_type);
            var_60 = wp::load(var_57);
            var_59 = (var_60 == var_58);
            if (var_59) {
                // result = _support_map_box(geom, direction)                                 <L 58>
                var_61 = _support_map_box_0(var_geom, var_direction);
            }
            if (!var_59) {
                // elif geom.shape_type == GeoType.SPHERE:                                    <L 60>
                var_62 = &((var_geom).shape_type);
                var_65 = wp::load(var_62);
                var_64 = (var_65 == var_63);
                if (var_64) {
                    // radius = geom.scale[0]                                                 <L 61>
                    var_66 = &((var_geom).scale);
                    var_69 = wp::load(var_66);
                    var_68 = wp::extract(var_69, var_67);
                    // dir_len_sq = wp.length_sq(direction)                                   <L 62>
                    var_70 = wp::length_sq(var_direction);
                    // if dir_len_sq > eps:                                                   <L 63>
                    var_71 = (var_70 > var_0);
                    if (var_71) {
                        // n = direction * _support_rsqrt_rn(dir_len_sq)                      <L 64>
                        var_72 = _support_rsqrt_rn_0(var_70);
                        var_73 = wp::mul(var_direction, var_72);
                    }
                    if (!var_71) {
                        // n = wp.vec3(1.0, 0.0, 0.0)                                         <L 66>
                        var_77 = wp::vec_t<3, wp::float32>(var_74, var_75, var_76);
                    }
                    var_78 = wp::where(var_71, var_73, var_77);
                    // result = n * radius                                                    <L 67>
                    var_79 = wp::mul(var_78, var_68);
                }
                if (!var_64) {
                    // elif geom.shape_type == GeoType.CAPSULE:                               <L 69>
                    var_80 = &((var_geom).shape_type);
                    var_83 = wp::load(var_80);
                    var_82 = (var_83 == var_81);
                    if (var_82) {
                        // radius = geom.scale[0]                                             <L 70>
                        var_84 = &((var_geom).scale);
                        var_87 = wp::load(var_84);
                        var_86 = wp::extract(var_87, var_85);
                        // half_height = geom.scale[1]                                        <L 71>
                        var_88 = &((var_geom).scale);
                        var_91 = wp::load(var_88);
                        var_90 = wp::extract(var_91, var_89);
                        // dir_len_sq = wp.length_sq(direction)                               <L 75>
                        var_92 = wp::length_sq(var_direction);
                        // if dir_len_sq > eps:                                               <L 76>
                        var_93 = (var_92 > var_0);
                        if (var_93) {
                            // n = direction * _support_rsqrt_rn(dir_len_sq)                  <L 77>
                            var_94 = _support_rsqrt_rn_0(var_92);
                            var_95 = wp::mul(var_direction, var_94);
                        }
                        if (!var_93) {
                            // n = wp.vec3(1.0, 0.0, 0.0)                                     <L 79>
                            var_99 = wp::vec_t<3, wp::float32>(var_96, var_97, var_98);
                        }
                        var_100 = wp::where(var_93, var_95, var_99);
                        // result = n * radius                                                <L 80>
                        var_101 = wp::mul(var_100, var_86);
                        // if direction[2] >= 0.0:                                            <L 84>
                        var_103 = wp::extract(var_direction, var_102);
                        var_105 = (var_103 >= var_104);
                        if (var_105) {
                            // result = result + wp.vec3(0.0, 0.0, half_height)               <L 85>
                            var_108 = wp::vec_t<3, wp::float32>(var_106, var_107, var_90);
                            var_109 = wp::add(var_101, var_108);
                        }
                        if (!var_105) {
                            // result = result + wp.vec3(0.0, 0.0, -half_height)              <L 87>
                            var_112 = wp::neg(var_90);
                            var_113 = wp::vec_t<3, wp::float32>(var_110, var_111, var_112);
                            var_114 = wp::add(var_101, var_113);
                        }
                        var_115 = wp::where(var_105, var_109, var_114);
                    }
                    if (!var_82) {
                        // elif geom.shape_type == GeoType.ELLIPSOID:                         <L 89>
                        var_116 = &((var_geom).shape_type);
                        var_119 = wp::load(var_116);
                        var_118 = (var_119 == var_117);
                        if (var_118) {
                            // a = geom.scale[0]                                              <L 92>
                            var_120 = &((var_geom).scale);
                            var_123 = wp::load(var_120);
                            var_122 = wp::extract(var_123, var_121);
                            // b = geom.scale[1]                                              <L 93>
                            var_124 = &((var_geom).scale);
                            var_127 = wp::load(var_124);
                            var_126 = wp::extract(var_127, var_125);
                            // c = geom.scale[2]                                              <L 94>
                            var_128 = &((var_geom).scale);
                            var_131 = wp::load(var_128);
                            var_130 = wp::extract(var_131, var_129);
                            // dir_len_sq = wp.length_sq(direction)                           <L 95>
                            var_132 = wp::length_sq(var_direction);
                            // if dir_len_sq > eps:                                           <L 96>
                            var_133 = (var_132 > var_0);
                            if (var_133) {
                                // adx = a * direction[0]                                     <L 97>
                                var_135 = wp::extract(var_direction, var_134);
                                var_136 = wp::mul(var_122, var_135);
                                // bdy = b * direction[1]                                     <L 98>
                                var_138 = wp::extract(var_direction, var_137);
                                var_139 = wp::mul(var_126, var_138);
                                // cdz = c * direction[2]                                     <L 99>
                                var_141 = wp::extract(var_direction, var_140);
                                var_142 = wp::mul(var_130, var_141);
                                // denom_sq = adx * adx + bdy * bdy + cdz * cdz               <L 100>
                                var_143 = wp::mul(var_136, var_136);
                                var_144 = wp::mul(var_139, var_139);
                                var_145 = wp::add(var_143, var_144);
                                var_146 = wp::mul(var_142, var_142);
                                var_147 = wp::add(var_145, var_146);
                                // if denom_sq > eps:                                         <L 101>
                                var_148 = (var_147 > var_0);
                                if (var_148) {
                                    // inv_denom = _support_rsqrt_rn(denom_sq)                <L 102>
                                    var_149 = _support_rsqrt_rn_0(var_147);
                                    // result = wp.vec3(                                      <L 103>
                                    // (a * a) * direction[0] * inv_denom,                    <L 104>
                                    var_150 = wp::mul(var_122, var_122);
                                    var_152 = wp::extract(var_direction, var_151);
                                    var_153 = wp::mul(var_150, var_152);
                                    var_154 = wp::mul(var_153, var_149);
                                    // (b * b) * direction[1] * inv_denom,                    <L 105>
                                    var_155 = wp::mul(var_126, var_126);
                                    var_157 = wp::extract(var_direction, var_156);
                                    var_158 = wp::mul(var_155, var_157);
                                    var_159 = wp::mul(var_158, var_149);
                                    // (c * c) * direction[2] * inv_denom,                    <L 106>
                                    var_160 = wp::mul(var_130, var_130);
                                    var_162 = wp::extract(var_direction, var_161);
                                    var_163 = wp::mul(var_160, var_162);
                                    var_164 = wp::mul(var_163, var_149);
                                    var_165 = wp::vec_t<3, wp::float32>(var_154, var_159, var_164);
                                }
                                if (!var_148) {
                                    // result = wp.vec3(a, 0.0, 0.0)                          <L 109>
                                    var_168 = wp::vec_t<3, wp::float32>(var_122, var_166, var_167);
                                }
                                var_169 = wp::where(var_148, var_165, var_168);
                            }
                            if (!var_133) {
                                // result = wp.vec3(a, 0.0, 0.0)                              <L 111>
                                var_172 = wp::vec_t<3, wp::float32>(var_122, var_170, var_171);
                            }
                            var_173 = wp::where(var_133, var_169, var_172);
                        }
                        if (!var_118) {
                            // elif geom.shape_type == GeoType.CYLINDER:                      <L 113>
                            var_174 = &((var_geom).shape_type);
                            var_177 = wp::load(var_174);
                            var_176 = (var_177 == var_175);
                            if (var_176) {
                                // radius = geom.scale[0]                                     <L 114>
                                var_178 = &((var_geom).scale);
                                var_181 = wp::load(var_178);
                                var_180 = wp::extract(var_181, var_179);
                                // half_height = geom.scale[1]                                <L 115>
                                var_182 = &((var_geom).scale);
                                var_185 = wp::load(var_182);
                                var_184 = wp::extract(var_185, var_183);
                                // barrel_radius = geom.scale[2]                              <L 116>
                                var_186 = &((var_geom).scale);
                                var_189 = wp::load(var_186);
                                var_188 = wp::extract(var_189, var_187);
                                // dir_xy = wp.vec3(direction[0], direction[1], 0.0)          <L 118>
                                var_191 = wp::extract(var_direction, var_190);
                                var_193 = wp::extract(var_direction, var_192);
                                var_195 = wp::vec_t<3, wp::float32>(var_191, var_193, var_194);
                                // dir_xy_len_sq = wp.length_sq(dir_xy)                       <L 119>
                                var_196 = wp::length_sq(var_195);
                                // if barrel_radius == 0.0:                                   <L 121>
                                var_198 = (var_188 == var_197);
                                if (var_198) {
                                    // if dir_xy_len_sq > eps:                                <L 123>
                                    var_199 = (var_196 > var_0);
                                    if (var_199) {
                                        // n_xy = dir_xy * _support_rsqrt_rn(dir_xy_len_sq)       <L 124>
                                        var_200 = _support_rsqrt_rn_0(var_196);
                                        var_201 = wp::mul(var_195, var_200);
                                        // lateral_point = wp.vec3(n_xy[0] * radius, n_xy[1] * radius, 0.0)       <L 125>
                                        var_203 = wp::extract(var_201, var_202);
                                        var_204 = wp::mul(var_203, var_180);
                                        var_206 = wp::extract(var_201, var_205);
                                        var_207 = wp::mul(var_206, var_180);
                                        var_209 = wp::vec_t<3, wp::float32>(var_204, var_207, var_208);
                                    }
                                    if (!var_199) {
                                        // lateral_point = wp.vec3(radius, 0.0, 0.0)          <L 127>
                                        var_212 = wp::vec_t<3, wp::float32>(var_180, var_210, var_211);
                                    }
                                    var_213 = wp::where(var_199, var_209, var_212);
                                    // if direction[2] > 0.0:                                 <L 129>
                                    var_215 = wp::extract(var_direction, var_214);
                                    var_217 = (var_215 > var_216);
                                    if (var_217) {
                                        // result = wp.vec3(lateral_point[0], lateral_point[1], half_height)       <L 130>
                                        var_219 = wp::extract(var_213, var_218);
                                        var_221 = wp::extract(var_213, var_220);
                                        var_222 = wp::vec_t<3, wp::float32>(var_219, var_221, var_184);
                                    }
                                    if (!var_217) {
                                        // elif direction[2] < 0.0:                           <L 131>
                                        var_224 = wp::extract(var_direction, var_223);
                                        var_226 = (var_224 < var_225);
                                        if (var_226) {
                                            // result = wp.vec3(lateral_point[0], lateral_point[1], -half_height)       <L 132>
                                            var_228 = wp::extract(var_213, var_227);
                                            var_230 = wp::extract(var_213, var_229);
                                            var_231 = wp::neg(var_184);
                                            var_232 = wp::vec_t<3, wp::float32>(var_228, var_230, var_231);
                                        }
                                        if (!var_226) {
                                            // result = lateral_point                         <L 134>
                                            var_233 = wp::copy(var_213);
                                        }
                                        var_234 = wp::where(var_226, var_232, var_233);
                                    }
                                    var_235 = wp::where(var_217, var_222, var_234);
                                }
                                if (!var_198) {
                                    // if dir_xy_len_sq > eps:                                <L 136>
                                    var_236 = (var_196 > var_0);
                                    if (var_236) {
                                        // dir_xy_len = wp.sqrt(dir_xy_len_sq)                <L 137>
                                        var_237 = wp::sqrt(var_196);
                                        // n_xy = dir_xy / dir_xy_len                         <L 138>
                                        var_238 = wp::div(var_195, var_237);
                                    }
                                    if (!var_236) {
                                        // dir_xy_len = 0.0                                   <L 140>
                                        // n_xy = wp.vec3(1.0, 0.0, 0.0)                      <L 141>
                                        var_243 = wp::vec_t<3, wp::float32>(var_240, var_241, var_242);
                                    }
                                    var_244 = wp::where(var_236, var_237, var_239);
                                    var_245 = wp::where(var_236, var_238, var_243);
                                    // direction_len = wp.sqrt(dir_xy_len_sq + direction[2] * direction[2])       <L 143>
                                    var_247 = wp::extract(var_direction, var_246);
                                    var_249 = wp::extract(var_direction, var_248);
                                    var_250 = wp::mul(var_247, var_249);
                                    var_251 = wp::add(var_196, var_250);
                                    var_252 = wp::sqrt(var_251);
                                    // support_z = 0.0                                        <L 144>
                                    // if direction_len > eps:                                <L 145>
                                    var_254 = (var_252 > var_0);
                                    if (var_254) {
                                        // support_z = wp.clamp(barrel_radius * direction[2] / direction_len, -half_height, half_height)       <L 146>
                                        var_256 = wp::extract(var_direction, var_255);
                                        var_257 = wp::mul(var_188, var_256);
                                        var_258 = wp::div(var_257, var_252);
                                        var_259 = wp::neg(var_184);
                                        var_260 = wp::clamp(var_258, var_259, var_184);
                                    }
                                    var_261 = wp::where(var_254, var_260, var_253);
                                    // barrel_radius_sq = barrel_radius * barrel_radius       <L 148>
                                    var_262 = wp::mul(var_188, var_188);
                                    // half_height_sq = half_height * half_height             <L 149>
                                    var_263 = wp::mul(var_184, var_184);
                                    // support_z_sq = support_z * support_z                   <L 150>
                                    var_264 = wp::mul(var_261, var_261);
                                    // end_offset = wp.sqrt(barrel_radius_sq - half_height_sq)       <L 151>
                                    var_265 = wp::sub(var_262, var_263);
                                    var_266 = wp::sqrt(var_265);
                                    // support_offset = wp.sqrt(wp.max(barrel_radius_sq - support_z_sq, 0.0))       <L 152>
                                    var_267 = wp::sub(var_262, var_264);
                                    var_269 = wp::max(var_267, var_268);
                                    var_270 = wp::sqrt(var_269);
                                    // offset_sum = support_offset + end_offset               <L 153>
                                    var_271 = wp::add(var_270, var_266);
                                    // support_radius = radius                                <L 154>
                                    var_272 = wp::copy(var_180);
                                    // if offset_sum > eps:                                   <L 155>
                                    var_273 = (var_271 > var_0);
                                    if (var_273) {
                                        // support_radius += (half_height_sq - support_z_sq) / offset_sum       <L 156>
                                        var_274 = wp::sub(var_263, var_264);
                                        var_275 = wp::div(var_274, var_271);
                                        var_276 = wp::add(var_272, var_275);
                                    }
                                    var_277 = wp::where(var_273, var_276, var_272);
                                    // result = wp.vec3(n_xy[0] * support_radius, n_xy[1] * support_radius, support_z)       <L 157>
                                    var_279 = wp::extract(var_245, var_278);
                                    var_280 = wp::mul(var_279, var_277);
                                    var_282 = wp::extract(var_245, var_281);
                                    var_283 = wp::mul(var_282, var_277);
                                    var_284 = wp::vec_t<3, wp::float32>(var_280, var_283, var_261);
                                }
                                var_285 = wp::where(var_198, var_235, var_284);
                                var_286 = wp::where(var_198, var_201, var_245);
                            }
                            if (!var_176) {
                                // elif geom.shape_type == GeoType.CONE:                      <L 159>
                                var_287 = &((var_geom).shape_type);
                                var_290 = wp::load(var_287);
                                var_289 = (var_290 == var_288);
                                if (var_289) {
                                    // radius = geom.scale[0]                                 <L 160>
                                    var_291 = &((var_geom).scale);
                                    var_294 = wp::load(var_291);
                                    var_293 = wp::extract(var_294, var_292);
                                    // half_height = geom.scale[1]                            <L 161>
                                    var_295 = &((var_geom).scale);
                                    var_298 = wp::load(var_295);
                                    var_297 = wp::extract(var_298, var_296);
                                    // apex = wp.vec3(0.0, 0.0, half_height)                  <L 166>
                                    var_301 = wp::vec_t<3, wp::float32>(var_299, var_300, var_297);
                                    // dir_xy = wp.vec3(direction[0], direction[1], 0.0)       <L 167>
                                    var_303 = wp::extract(var_direction, var_302);
                                    var_305 = wp::extract(var_direction, var_304);
                                    var_307 = wp::vec_t<3, wp::float32>(var_303, var_305, var_306);
                                    // dir_xy_len = wp.length(dir_xy)                         <L 168>
                                    var_308 = wp::length(var_307);
                                    // k = radius / (2.0 * half_height) if half_height > eps else 0.0       <L 169>
                                    var_309 = (var_297 > var_0);
                                    if (var_309) {
                                        var_311 = wp::mul(var_310, var_297);
                                        var_312 = wp::div(var_293, var_311);
                                    }
                                    if (!var_309) {
                                    }
                                    var_314 = wp::where(var_309, var_312, var_313);
                                    // if dir_xy_len <= eps:                                  <L 171>
                                    var_315 = (var_308 <= var_0);
                                    if (var_315) {
                                        // if direction[2] >= 0.0:                            <L 173>
                                        var_317 = wp::extract(var_direction, var_316);
                                        var_319 = (var_317 >= var_318);
                                        if (var_319) {
                                            // result = apex                                  <L 174>
                                            var_320 = wp::copy(var_301);
                                        }
                                        if (!var_319) {
                                            // result = wp.vec3(radius, 0.0, -half_height)       <L 176>
                                            var_322 = wp::neg(var_297);
                                            var_323 = wp::vec_t<3, wp::float32>(var_293, var_321, var_322);
                                        }
                                        var_324 = wp::where(var_319, var_320, var_323);
                                    }
                                    if (!var_315) {
                                        // if direction[2] >= k * dir_xy_len:                 <L 178>
                                        var_326 = wp::extract(var_direction, var_325);
                                        var_327 = wp::mul(var_314, var_308);
                                        var_328 = (var_326 >= var_327);
                                        if (var_328) {
                                            // result = apex                                  <L 179>
                                            var_329 = wp::copy(var_301);
                                        }
                                        if (!var_328) {
                                            // n_xy = dir_xy / dir_xy_len                     <L 181>
                                            var_330 = wp::div(var_307, var_308);
                                            // result = wp.vec3(n_xy[0] * radius, n_xy[1] * radius, -half_height)       <L 182>
                                            var_332 = wp::extract(var_330, var_331);
                                            var_333 = wp::mul(var_332, var_293);
                                            var_335 = wp::extract(var_330, var_334);
                                            var_336 = wp::mul(var_335, var_293);
                                            var_337 = wp::neg(var_297);
                                            var_338 = wp::vec_t<3, wp::float32>(var_333, var_336, var_337);
                                        }
                                        var_339 = wp::where(var_328, var_329, var_338);
                                    }
                                    var_340 = wp::where(var_315, var_324, var_339);
                                }
                                if (!var_289) {
                                    // elif geom.shape_type == GeoType.PLANE:                 <L 184>
                                    var_341 = &((var_geom).shape_type);
                                    var_344 = wp::load(var_341);
                                    var_343 = (var_344 == var_342);
                                    if (var_343) {
                                        // half_width = geom.scale[0]                         <L 187>
                                        var_345 = &((var_geom).scale);
                                        var_348 = wp::load(var_345);
                                        var_347 = wp::extract(var_348, var_346);
                                        // half_length = geom.scale[1]                        <L 188>
                                        var_349 = &((var_geom).scale);
                                        var_352 = wp::load(var_349);
                                        var_351 = wp::extract(var_352, var_350);
                                        // sx = 1.0 if direction[0] >= 0.0 else -1.0          <L 191>
                                        var_354 = wp::extract(var_direction, var_353);
                                        var_356 = (var_354 >= var_355);
                                        if (var_356) {
                                        }
                                        if (!var_356) {
                                        }
                                        var_359 = wp::where(var_356, var_357, var_358);
                                        // sy = 1.0 if direction[1] >= 0.0 else -1.0          <L 192>
                                        var_361 = wp::extract(var_direction, var_360);
                                        var_363 = (var_361 >= var_362);
                                        if (var_363) {
                                        }
                                        if (!var_363) {
                                        }
                                        var_366 = wp::where(var_363, var_364, var_365);
                                        // result = wp.vec3(sx * half_width, sy * half_length, 0.0)       <L 195>
                                        var_367 = wp::mul(var_359, var_347);
                                        var_368 = wp::mul(var_366, var_351);
                                        var_370 = wp::vec_t<3, wp::float32>(var_367, var_368, var_369);
                                    }
                                    if (!var_343) {
                                        // result = wp.vec3(0.0, 0.0, 0.0)                    <L 199>
                                        var_374 = wp::vec_t<3, wp::float32>(var_371, var_372, var_373);
                                    }
                                    var_375 = wp::where(var_343, var_370, var_374);
                                }
                                var_376 = wp::where(var_289, var_340, var_375);
                            }
                            var_377 = wp::where(var_176, var_285, var_376);
                            var_378 = wp::where(var_176, var_180, var_293);
                            var_379 = wp::where(var_176, var_184, var_297);
                            var_380 = wp::where(var_176, var_195, var_307);
                            var_381 = wp::where(var_176, var_286, var_330);
                            var_382 = wp::where(var_176, var_244, var_308);
                        }
                        var_383 = wp::where(var_118, var_173, var_377);
                    }
                    var_384 = wp::where(var_82, var_115, var_383);
                    var_385 = wp::where(var_82, var_86, var_378);
                    var_386 = wp::where(var_82, var_90, var_379);
                    var_387 = wp::where(var_82, var_92, var_132);
                }
                var_388 = wp::where(var_64, var_79, var_384);
                var_389 = wp::where(var_64, var_68, var_385);
                var_390 = wp::where(var_64, var_70, var_387);
                var_391 = wp::where(var_64, var_78, var_100);
            }
            var_392 = wp::where(var_59, var_61, var_388);
        }
        var_393 = wp::where(var_10, var_56, var_392);
    }
    var_394 = wp::where(var_7, var_9, var_393);
    // return result                                                                          <L 201>
    return var_394;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_core.py:0
static CUDA_CALLABLE void compute_tight_aabb_from_support_0(
    GenericShapeData_ba895271 var_shape_data,
    wp::quat_t<wp::float32> var_orientation,
    wp::vec_t<3, wp::float32> var_center_pos,
    SupportMapDataProvider_e77f8b9f var_data_provider,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::mat_t<3, 3, wp::float32> var_0;
    wp::mat_t<3, 3, wp::float32> var_1;
    const wp::int32 var_2 = 0;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    const wp::int32 var_5 = 1;
    const wp::int32 var_6 = 0;
    wp::float32 var_7;
    const wp::int32 var_8 = 2;
    const wp::int32 var_9 = 0;
    wp::float32 var_10;
    wp::vec_t<3, wp::float32> var_11;
    const wp::int32 var_12 = 0;
    const wp::int32 var_13 = 1;
    wp::float32 var_14;
    const wp::int32 var_15 = 1;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    const wp::int32 var_18 = 2;
    const wp::int32 var_19 = 1;
    wp::float32 var_20;
    wp::vec_t<3, wp::float32> var_21;
    const wp::int32 var_22 = 0;
    const wp::int32 var_23 = 2;
    wp::float32 var_24;
    const wp::int32 var_25 = 1;
    const wp::int32 var_26 = 2;
    wp::float32 var_27;
    const wp::int32 var_28 = 2;
    const wp::int32 var_29 = 2;
    wp::float32 var_30;
    wp::vec_t<3, wp::float32> var_31;
    const wp::float32 var_32 = 0.0;
    wp::float32 var_33;
    const wp::float32 var_34 = 0.0;
    wp::float32 var_35;
    const wp::float32 var_36 = 0.0;
    wp::float32 var_37;
    const wp::float32 var_38 = 0.0;
    wp::float32 var_39;
    const wp::float32 var_40 = 0.0;
    wp::float32 var_41;
    const wp::float32 var_42 = 0.0;
    wp::float32 var_43;
    wp::int32* var_44;
    const wp::int32 var_45 = 10;
    bool var_46;
    wp::int32 var_47;
    wp::vec_t<3, wp::float32>* var_48;
    wp::uint64 var_49;
    wp::vec_t<3, wp::float32> var_50;
    wp::Mesh var_51;
    wp::vec_t<3, wp::float32>* var_52;
    wp::vec_t<3, wp::float32> var_53;
    wp::vec_t<3, wp::float32> var_54;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_55;
    wp::shape_t* var_56;
    const wp::int32 var_57 = 0;
    wp::int32 var_58;
    wp::shape_t var_59;
    wp::vec_t<3, wp::float32> var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::vec_t<3, wp::float32> var_62;
    const wp::float32 var_63 = 10000000000.0;
    wp::float32 var_64;
    const wp::float32 var_65 = -10000000000.0;
    wp::float32 var_66;
    const wp::float32 var_67 = 10000000000.0;
    wp::float32 var_68;
    const wp::float32 var_69 = -10000000000.0;
    wp::float32 var_70;
    const wp::float32 var_71 = 10000000000.0;
    wp::float32 var_72;
    const wp::float32 var_73 = -10000000000.0;
    wp::float32 var_74;
    wp::range_t var_75;
    wp::int32 var_76;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_77;
    wp::vec_t<3, wp::float32>* var_78;
    wp::array_t<wp::vec_t<3, wp::float32>> var_79;
    wp::vec_t<3, wp::float32> var_80;
    wp::vec_t<3, wp::float32> var_81;
    wp::float32 var_82;
    wp::float32 var_83;
    wp::float32 var_84;
    wp::float32 var_85;
    wp::float32 var_86;
    wp::float32 var_87;
    wp::float32 var_88;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::vec_t<3, wp::float32> var_91;
    wp::float32 var_92;
    wp::vec_t<3, wp::float32> var_93;
    wp::float32 var_94;
    wp::vec_t<3, wp::float32> var_95;
    wp::float32 var_96;
    wp::vec_t<3, wp::float32> var_97;
    wp::vec_t<3, wp::float32> var_98;
    wp::float32 var_99;
    wp::vec_t<3, wp::float32> var_100;
    wp::vec_t<3, wp::float32> var_101;
    wp::float32 var_102;
    wp::vec_t<3, wp::float32> var_103;
    wp::vec_t<3, wp::float32> var_104;
    wp::float32 var_105;
    wp::float32 var_106;
    wp::float32 var_107;
    wp::float32 var_108;
    wp::float32 var_109;
    wp::float32 var_110;
    wp::float32 var_111;
    wp::vec_t<3, wp::float32> var_112;
    wp::vec_t<3, wp::float32> var_113;
    wp::vec_t<3, wp::float32> var_114;
    wp::vec_t<3, wp::float32> var_115;
    //---------
    // forward
    // def compute_tight_aabb_from_support(                                                   <L 1>
    // rot_mat = wp.quat_to_matrix(orientation)                                               <L 21>
    var_0 = wp::quat_to_matrix(var_orientation);
    // rot_mat_t = wp.transpose(rot_mat)                                                      <L 22>
    var_1 = wp::transpose(var_0);
    // local_x = wp.vec3(rot_mat_t[0, 0], rot_mat_t[1, 0], rot_mat_t[2, 0])                   <L 25>
    var_4 = wp::extract(var_1, var_2, var_3);
    var_7 = wp::extract(var_1, var_5, var_6);
    var_10 = wp::extract(var_1, var_8, var_9);
    var_11 = wp::vec_t<3, wp::float32>(var_4, var_7, var_10);
    // local_y = wp.vec3(rot_mat_t[0, 1], rot_mat_t[1, 1], rot_mat_t[2, 1])                   <L 26>
    var_14 = wp::extract(var_1, var_12, var_13);
    var_17 = wp::extract(var_1, var_15, var_16);
    var_20 = wp::extract(var_1, var_18, var_19);
    var_21 = wp::vec_t<3, wp::float32>(var_14, var_17, var_20);
    // local_z = wp.vec3(rot_mat_t[0, 2], rot_mat_t[1, 2], rot_mat_t[2, 2])                   <L 27>
    var_24 = wp::extract(var_1, var_22, var_23);
    var_27 = wp::extract(var_1, var_25, var_26);
    var_30 = wp::extract(var_1, var_28, var_29);
    var_31 = wp::vec_t<3, wp::float32>(var_24, var_27, var_30);
    // min_x = float(0.0)                                                                     <L 32>
    var_33 = wp::float(var_32);
    // max_x = float(0.0)                                                                     <L 33>
    var_35 = wp::float(var_34);
    // min_y = float(0.0)                                                                     <L 34>
    var_37 = wp::float(var_36);
    // max_y = float(0.0)                                                                     <L 35>
    var_39 = wp::float(var_38);
    // min_z = float(0.0)                                                                     <L 36>
    var_41 = wp::float(var_40);
    // max_z = float(0.0)                                                                     <L 37>
    var_43 = wp::float(var_42);
    // if shape_data.shape_type == GeoType.CONVEX_MESH:                                       <L 39>
    var_44 = &((var_shape_data).shape_type);
    var_47 = wp::load(var_44);
    var_46 = (var_47 == var_45);
    if (var_46) {
        // mesh_ptr = unpack_mesh_ptr(shape_data.auxiliary)                                   <L 43>
        var_48 = &((var_shape_data).auxiliary);
        var_50 = wp::load(var_48);
        var_49 = unpack_mesh_ptr_0(var_50);
        // mesh = wp.mesh_get(mesh_ptr)                                                       <L 44>
        var_51 = wp::mesh_get(var_49);
        // mesh_scale = shape_data.scale                                                      <L 45>
        var_52 = &((var_shape_data).scale);
        var_54 = wp::load(var_52);
        var_53 = wp::copy(var_54);
        // num_verts = mesh.points.shape[0]                                                   <L 46>
        var_55 = &((var_51).points);
        var_56 = &(var_55->shape);
        var_59 = wp::load(var_56);
        var_58 = wp::extract(var_59, var_57);
        // scaled_x = wp.cw_mul(local_x, mesh_scale)                                          <L 49>
        var_60 = wp::cw_mul(var_11, var_53);
        // scaled_y = wp.cw_mul(local_y, mesh_scale)                                          <L 50>
        var_61 = wp::cw_mul(var_21, var_53);
        // scaled_z = wp.cw_mul(local_z, mesh_scale)                                          <L 51>
        var_62 = wp::cw_mul(var_31, var_53);
        // min_x = float(1.0e10)                                                              <L 53>
        var_64 = wp::float(var_63);
        // max_x = float(-1.0e10)                                                             <L 54>
        var_66 = wp::float(var_65);
        // min_y = float(1.0e10)                                                              <L 55>
        var_68 = wp::float(var_67);
        // max_y = float(-1.0e10)                                                             <L 56>
        var_70 = wp::float(var_69);
        // min_z = float(1.0e10)                                                              <L 57>
        var_72 = wp::float(var_71);
        // max_z = float(-1.0e10)                                                             <L 58>
        var_74 = wp::float(var_73);
        // for i in range(num_verts):                                                         <L 60>
        var_75 = wp::range(var_58);
        start_for_0:;
            if (iter_cmp(var_75) == 0) goto end_for_0;
            var_76 = wp::iter_next(var_75);
            // p = mesh.points[i]                                                             <L 61>
            var_77 = &((var_51).points);
            var_79 = wp::load(var_77);
            var_78 = wp::address(var_79, var_76);
            var_81 = wp::load(var_78);
            var_80 = wp::copy(var_81);
            // vx = wp.dot(p, scaled_x)                                                       <L 62>
            var_82 = wp::dot(var_80, var_60);
            // vy = wp.dot(p, scaled_y)                                                       <L 63>
            var_83 = wp::dot(var_80, var_61);
            // vz = wp.dot(p, scaled_z)                                                       <L 64>
            var_84 = wp::dot(var_80, var_62);
            // min_x = wp.min(min_x, vx)                                                      <L 65>
            var_85 = wp::min(var_64, var_82);
            // max_x = wp.max(max_x, vx)                                                      <L 66>
            var_86 = wp::max(var_66, var_82);
            // min_y = wp.min(min_y, vy)                                                      <L 67>
            var_87 = wp::min(var_68, var_83);
            // max_y = wp.max(max_y, vy)                                                      <L 68>
            var_88 = wp::max(var_70, var_83);
            // min_z = wp.min(min_z, vz)                                                      <L 69>
            var_89 = wp::min(var_72, var_84);
            // max_z = wp.max(max_z, vz)                                                      <L 70>
            var_90 = wp::max(var_74, var_84);
            wp::assign(var_64, var_85);
            wp::assign(var_66, var_86);
            wp::assign(var_68, var_87);
            wp::assign(var_70, var_88);
            wp::assign(var_72, var_89);
            wp::assign(var_74, var_90);
            goto start_for_0;
        end_for_0:;
    }
    if (!var_46) {
        // support_point = support_map(shape_data, local_x, data_provider)                    <L 73>
        var_91 = support_map_0(var_shape_data, var_11, var_data_provider);
        // max_x = wp.dot(local_x, support_point)                                             <L 74>
        var_92 = wp::dot(var_11, var_91);
        // support_point = support_map(shape_data, local_y, data_provider)                    <L 76>
        var_93 = support_map_0(var_shape_data, var_21, var_data_provider);
        // max_y = wp.dot(local_y, support_point)                                             <L 77>
        var_94 = wp::dot(var_21, var_93);
        // support_point = support_map(shape_data, local_z, data_provider)                    <L 79>
        var_95 = support_map_0(var_shape_data, var_31, var_data_provider);
        // max_z = wp.dot(local_z, support_point)                                             <L 80>
        var_96 = wp::dot(var_31, var_95);
        // support_point = support_map(shape_data, -local_x, data_provider)                   <L 82>
        var_97 = wp::neg(var_11);
        var_98 = support_map_0(var_shape_data, var_97, var_data_provider);
        // min_x = wp.dot(local_x, support_point)                                             <L 83>
        var_99 = wp::dot(var_11, var_98);
        // support_point = support_map(shape_data, -local_y, data_provider)                   <L 85>
        var_100 = wp::neg(var_21);
        var_101 = support_map_0(var_shape_data, var_100, var_data_provider);
        // min_y = wp.dot(local_y, support_point)                                             <L 86>
        var_102 = wp::dot(var_21, var_101);
        // support_point = support_map(shape_data, -local_z, data_provider)                   <L 88>
        var_103 = wp::neg(var_31);
        var_104 = support_map_0(var_shape_data, var_103, var_data_provider);
        // min_z = wp.dot(local_z, support_point)                                             <L 89>
        var_105 = wp::dot(var_31, var_104);
    }
    var_106 = wp::where(var_46, var_64, var_99);
    var_107 = wp::where(var_46, var_66, var_92);
    var_108 = wp::where(var_46, var_68, var_102);
    var_109 = wp::where(var_46, var_70, var_94);
    var_110 = wp::where(var_46, var_72, var_105);
    var_111 = wp::where(var_46, var_74, var_96);
    // aabb_min = wp.vec3(min_x, min_y, min_z) + center_pos                                   <L 92>
    var_112 = wp::vec_t<3, wp::float32>(var_106, var_108, var_110);
    var_113 = wp::add(var_112, var_center_pos);
    // aabb_max = wp.vec3(max_x, max_y, max_z) + center_pos                                   <L 93>
    var_114 = wp::vec_t<3, wp::float32>(var_107, var_109, var_111);
    var_115 = wp::add(var_114, var_center_pos);
    // return aabb_min, aabb_max                                                              <L 95>
    ret_0 = var_113;
    ret_1 = var_115;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:81
static CUDA_CALLABLE void adj_pack_mesh_ptr_0(
    wp::uint64 var_ptr,
    wp::uint64 & adj_ptr,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:92
static CUDA_CALLABLE void adj_unpack_mesh_ptr_0(
    wp::vec_t<3, wp::float32> var_arr,
    wp::vec_t<3, wp::float32> & adj_arr,
    wp::uint64 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:194
static CUDA_CALLABLE void adj__support_map_convex_mesh_exhaustive_0(
    GenericShapeData_ba895271 var_geom,
    wp::vec_t<3, wp::float32> var_direction,
    GenericShapeData_ba895271 & adj_geom,
    wp::vec_t<3, wp::float32> & adj_direction,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:209
static CUDA_CALLABLE void adj__support_map_box_0(
    GenericShapeData_ba895271 var_geom,
    wp::vec_t<3, wp::float32> var_direction,
    GenericShapeData_ba895271 & adj_geom,
    wp::vec_t<3, wp::float32> & adj_direction,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:46
static CUDA_CALLABLE void adj__support_rsqrt_rn_0(
    wp::float32 value,
    wp::float32 & adj_value,
    wp::float32 & adj_ret)
{
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:0
static CUDA_CALLABLE void adj_support_map_0(
    GenericShapeData_ba895271 var_geom,
    wp::vec_t<3, wp::float32> var_direction,
    SupportMapDataProvider_e77f8b9f var_data_provider,
    GenericShapeData_ba895271 & adj_geom,
    wp::vec_t<3, wp::float32> & adj_direction,
    SupportMapDataProvider_e77f8b9f & adj_data_provider,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_core.py:0
static CUDA_CALLABLE void adj_compute_tight_aabb_from_support_0(
    GenericShapeData_ba895271 var_shape_data,
    wp::quat_t<wp::float32> var_orientation,
    wp::vec_t<3, wp::float32> var_center_pos,
    SupportMapDataProvider_e77f8b9f var_data_provider,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    GenericShapeData_ba895271 & adj_shape_data,
    wp::quat_t<wp::float32> & adj_orientation,
    wp::vec_t<3, wp::float32> & adj_center_pos,
    SupportMapDataProvider_e77f8b9f & adj_data_provider,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void compute_shape_velocities_27284e1d_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_body_q,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_qd,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com,
    wp::array_t<wp::int32> var_shape_body,
    wp::array_t<wp::transform_t<wp::float32>> var_shape_transform,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_collision_aabb_lower,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_collision_aabb_upper,
    wp::array_t<wp::float32> var_shape_collision_radius,
    wp::array_t<wp::float32> var_shape_gap,
    wp::float32 var_collision_update_dt,
    wp::float32 var_max_speculative_extension,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_linear_velocity,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_angular_velocity,
    wp::array_t<wp::float32> var_shape_search_gap,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_displacement,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_aabb_lower,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_aabb_upper)
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
        bool var_5;
        const wp::float32 var_6 = 0.0;
        wp::vec_t<3, wp::float32> var_7;
        const wp::float32 var_8 = 0.0;
        wp::vec_t<3, wp::float32> var_9;
        wp::float32* var_10;
        wp::float32 var_11;
        const wp::float32 var_12 = 0.0;
        wp::vec_t<3, wp::float32> var_13;
        wp::transform_t<wp::float32>* var_14;
        wp::transform_t<wp::float32> var_15;
        wp::transform_t<wp::float32> var_16;
        wp::transform_t<wp::float32>* var_17;
        wp::transform_t<wp::float32> var_18;
        wp::transform_t<wp::float32> var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<6, wp::float32>* var_24;
        wp::vec_t<6, wp::float32> var_25;
        wp::vec_t<6, wp::float32> var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::vec_t<3, wp::float32>* var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::vec_t<3, wp::float32>* var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::vec_t<3, wp::float32> var_38;
        wp::vec_t<3, wp::float32> var_39;
        wp::vec_t<3, wp::float32> var_40;
        wp::float32 var_41;
        wp::float32* var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        wp::float32 var_48;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::float32* var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        wp::vec_t<3, wp::float32> var_54;
        wp::float32 var_55;
        wp::vec_t<3, wp::float32> var_56;
        wp::vec_t<3, wp::float32> var_57;
        wp::vec_t<3, wp::float32> var_58;
        wp::vec_t<3, wp::float32>* var_59;
        wp::vec_t<3, wp::float32> var_60;
        wp::vec_t<3, wp::float32> var_61;
        wp::vec_t<3, wp::float32>* var_62;
        wp::vec_t<3, wp::float32> var_63;
        wp::vec_t<3, wp::float32> var_64;
        //---------
        // forward
        // def compute_shape_velocities(                                                          <L 446>
        // shape_id = wp.tid()                                                                    <L 474>
        var_0 = builtin_tid1d();
        // body_id = shape_body[shape_id]                                                         <L 475>
        var_1 = wp::address(var_shape_body, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // if body_id == -1:                                                                      <L 476>
        var_5 = (var_2 == var_4);
        if (var_5) {
            // shape_linear_velocity[shape_id] = wp.vec3(0.0)                                     <L 477>
            var_7 = wp::vec_t<3, wp::float32>(var_6);
            wp::array_store(var_shape_linear_velocity, var_0, var_7);
            // shape_angular_velocity[shape_id] = wp.vec3(0.0)                                    <L 478>
            var_9 = wp::vec_t<3, wp::float32>(var_8);
            wp::array_store(var_shape_angular_velocity, var_0, var_9);
            // shape_search_gap[shape_id] = shape_gap[shape_id]                                   <L 479>
            var_10 = wp::address(var_shape_gap, var_0);
            var_11 = wp::load(var_10);
            wp::array_store(var_shape_search_gap, var_0, var_11);
            // shape_displacement[shape_id] = wp.vec3(0.0)                                        <L 480>
            var_13 = wp::vec_t<3, wp::float32>(var_12);
            wp::array_store(var_shape_displacement, var_0, var_13);
            // return                                                                             <L 481>
            continue;
        }
        // X_wb = body_q[body_id]                                                                 <L 483>
        var_14 = wp::address(var_body_q, var_2);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // X_ws = wp.transform_multiply(X_wb, shape_transform[shape_id])                          <L 484>
        var_17 = wp::address(var_shape_transform, var_0);
        var_19 = wp::load(var_17);
        var_18 = wp::transform_multiply(var_15, var_19);
        // shape_origin_world = wp.transform_get_translation(X_ws)                                <L 485>
        var_20 = wp::transform_get_translation(var_18);
        // com_world = wp.transform_point(X_wb, body_com[body_id])                                <L 486>
        var_21 = wp::address(var_body_com, var_2);
        var_23 = wp::load(var_21);
        var_22 = wp::transform_point(var_15, var_23);
        // twist = body_qd[body_id]                                                               <L 487>
        var_24 = wp::address(var_body_qd, var_2);
        var_26 = wp::load(var_24);
        var_25 = wp::copy(var_26);
        // com_velocity = wp.spatial_top(twist)                                                   <L 488>
        var_27 = wp::spatial_top(var_25);
        // angular_velocity = wp.spatial_bottom(twist)                                            <L 489>
        var_28 = wp::spatial_bottom(var_25);
        // shape_origin_velocity = com_velocity + wp.cross(angular_velocity, shape_origin_world - com_world)       <L 490>
        var_29 = wp::sub(var_20, var_22);
        var_30 = wp::cross(var_28, var_29);
        var_31 = wp::add(var_27, var_30);
        // shape_linear_velocity[shape_id] = shape_origin_velocity                                <L 491>
        wp::array_store(var_shape_linear_velocity, var_0, var_31);
        // shape_angular_velocity[shape_id] = angular_velocity                                    <L 492>
        wp::array_store(var_shape_angular_velocity, var_0, var_28);
        // local_lower = shape_collision_aabb_lower[shape_id]                                     <L 494>
        var_32 = wp::address(var_shape_collision_aabb_lower, var_0);
        var_34 = wp::load(var_32);
        var_33 = wp::copy(var_34);
        // local_upper = shape_collision_aabb_upper[shape_id]                                     <L 495>
        var_35 = wp::address(var_shape_collision_aabb_upper, var_0);
        var_37 = wp::load(var_35);
        var_36 = wp::copy(var_37);
        // furthest = wp.max(wp.abs(local_lower), wp.abs(local_upper))                            <L 496>
        var_38 = wp::abs(var_33);
        var_39 = wp::abs(var_36);
        var_40 = wp::max(var_38, var_39);
        // angular_radius = wp.max(wp.length(furthest), shape_collision_radius[shape_id])         <L 497>
        var_41 = wp::length(var_40);
        var_42 = wp::address(var_shape_collision_radius, var_0);
        var_44 = wp::load(var_42);
        var_43 = wp::max(var_41, var_44);
        // angular_speed_bound = wp.length(angular_velocity) * angular_radius                     <L 498>
        var_45 = wp::length(var_28);
        var_46 = wp::mul(var_45, var_43);
        // search_extension = wp.min(                                                             <L 499>
        // (wp.length(shape_origin_velocity) + angular_speed_bound) * collision_update_dt,        <L 500>
        var_47 = wp::length(var_31);
        var_48 = wp::add(var_47, var_46);
        var_49 = wp::mul(var_48, var_collision_update_dt);
        // max_speculative_extension,                                                             <L 501>
        var_50 = wp::min(var_49, var_max_speculative_extension);
        // shape_search_gap[shape_id] = shape_gap[shape_id] + search_extension                    <L 503>
        var_51 = wp::address(var_shape_gap, var_0);
        var_53 = wp::load(var_51);
        var_52 = wp::add(var_53, var_50);
        wp::array_store(var_shape_search_gap, var_0, var_52);
        // displacement = shape_origin_velocity * collision_update_dt                             <L 505>
        var_54 = wp::mul(var_31, var_collision_update_dt);
        // angular_extension = angular_speed_bound * collision_update_dt                          <L 506>
        var_55 = wp::mul(var_46, var_collision_update_dt);
        // cap = wp.vec3(max_speculative_extension)                                               <L 507>
        var_56 = wp::vec_t<3, wp::float32>(var_max_speculative_extension);
        // shape_displacement[shape_id] = displacement                                            <L 509>
        wp::array_store(var_shape_displacement, var_0, var_54);
        // angular_extension_vec = wp.min(wp.vec3(angular_extension), cap)                        <L 510>
        var_57 = wp::vec_t<3, wp::float32>(var_55);
        var_58 = wp::min(var_57, var_56);
        // shape_aabb_lower[shape_id] = shape_aabb_lower[shape_id] - angular_extension_vec        <L 511>
        var_59 = wp::address(var_shape_aabb_lower, var_0);
        var_61 = wp::load(var_59);
        var_60 = wp::sub(var_61, var_58);
        wp::array_store(var_shape_aabb_lower, var_0, var_60);
        // shape_aabb_upper[shape_id] = shape_aabb_upper[shape_id] + angular_extension_vec        <L 512>
        var_62 = wp::address(var_shape_aabb_upper, var_0);
        var_64 = wp::load(var_62);
        var_63 = wp::add(var_64, var_58);
        wp::array_store(var_shape_aabb_upper, var_0, var_63);
    }
}



extern "C" __global__ void compute_shape_aabbs_599b8929_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_body_q,
    wp::array_t<wp::transform_t<wp::float32>> var_shape_transform,
    wp::array_t<wp::int32> var_shape_body,
    wp::array_t<wp::int32> var_shape_type,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_scale,
    wp::array_t<wp::float32> var_shape_collision_radius,
    wp::array_t<wp::uint64> var_shape_source_ptr,
    wp::array_t<wp::float32> var_shape_margin,
    wp::array_t<wp::float32> var_shape_gap,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_collision_aabb_lower,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_collision_aabb_upper,
    wp::array_t<wp::int32> var_contact_counters,
    wp::array_t<wp::int32> var_contact_generation,
    wp::array_t<wp::int32> var_broad_phase_pair_count,
    wp::int32 var_num_contact_counters,
    wp::array_t<wp::vec_t<3, wp::float32>> var_aabb_lower,
    wp::array_t<wp::vec_t<3, wp::float32>> var_aabb_upper,
    wp::array_t<wp::vec_t<4, wp::float32>> var_geom_data,
    wp::array_t<wp::transform_t<wp::float32>> var_geom_xform)
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
        bool var_2;
        wp::range_t var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        const wp::int32 var_6 = 0;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        const wp::int32 var_10 = 2147483647;
        bool var_11;
        const wp::int32 var_12 = 0;
        const wp::int32 var_13 = 1;
        wp::int32 var_14;
        wp::int32 var_15;
        const wp::int32 var_16 = 0;
        const wp::int32 var_17 = 0;
        const wp::int32 var_18 = 0;
        wp::int32* var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        const wp::int32 var_25 = -1;
        bool var_26;
        wp::transform_t<wp::float32>* var_27;
        wp::transform_t<wp::float32> var_28;
        wp::transform_t<wp::float32> var_29;
        wp::transform_t<wp::float32>* var_30;
        wp::transform_t<wp::float32>* var_31;
        wp::transform_t<wp::float32> var_32;
        wp::transform_t<wp::float32> var_33;
        wp::transform_t<wp::float32> var_34;
        wp::transform_t<wp::float32> var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::quat_t<wp::float32> var_37;
        wp::float32* var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        wp::float32* var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::vec_t<3, wp::float32> var_44;
        wp::vec_t<3, wp::float32>* var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::vec_t<3, wp::float32> var_47;
        bool var_48;
        const wp::int32 var_49 = 1;
        bool var_50;
        bool var_51;
        const wp::int32 var_52 = 0;
        wp::float32 var_53;
        const wp::float32 var_54 = 0.0;
        bool var_55;
        const wp::int32 var_56 = 1;
        wp::float32 var_57;
        const wp::float32 var_58 = 0.0;
        bool var_59;
        bool var_60;
        const wp::int32 var_61 = 8;
        bool var_62;
        const wp::int32 var_63 = 2;
        bool var_64;
        const wp::int32 var_65 = 10;
        bool var_66;
        wp::vec_t<3, wp::float32> var_67;
        const wp::float32 var_68 = 0.0;
        const wp::float32 var_69 = 0.0;
        const wp::float32 var_70 = 1.0;
        wp::vec_t<3, wp::float32> var_71;
        wp::vec_t<3, wp::float32> var_72;
        const wp::float32 var_73 = 1000000.0;
        wp::vec_t<3, wp::float32> var_74;
        wp::vec_t<3, wp::float32> var_75;
        wp::vec_t<3, wp::float32> var_76;
        wp::vec_t<3, wp::float32> var_77;
        wp::vec_t<3, wp::float32> var_78;
        const wp::int32 var_79 = 0;
        wp::float32 var_80;
        wp::float32 var_81;
        const wp::float32 var_82 = 0.5;
        bool var_83;
        const wp::int32 var_84 = 1;
        wp::int32 var_85;
        const wp::int32 var_86 = 3;
        wp::int32 var_87;
        wp::float32 var_88;
        wp::float32 var_89;
        const wp::int32 var_90 = 2;
        wp::int32 var_91;
        const wp::int32 var_92 = 3;
        wp::int32 var_93;
        wp::float32 var_94;
        wp::float32 var_95;
        wp::float32 var_96;
        wp::float32 var_97;
        wp::float32 var_98;
        wp::float32 var_99;
        const wp::float32 var_100 = 0.0;
        bool var_101;
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
        const wp::int32 var_112 = 1;
        wp::float32 var_113;
        wp::float32 var_114;
        const wp::float32 var_115 = 0.5;
        bool var_116;
        const wp::int32 var_117 = 1;
        wp::int32 var_118;
        const wp::int32 var_119 = 3;
        wp::int32 var_120;
        wp::float32 var_121;
        wp::float32 var_122;
        const wp::int32 var_123 = 2;
        wp::int32 var_124;
        const wp::int32 var_125 = 3;
        wp::int32 var_126;
        wp::float32 var_127;
        wp::float32 var_128;
        wp::float32 var_129;
        wp::float32 var_130;
        wp::float32 var_131;
        wp::float32 var_132;
        const wp::float32 var_133 = 0.0;
        bool var_134;
        wp::float32 var_135;
        wp::float32 var_136;
        wp::float32 var_137;
        wp::float32 var_138;
        wp::float32 var_139;
        wp::float32 var_140;
        wp::float32 var_141;
        wp::float32 var_142;
        wp::float32 var_143;
        wp::float32 var_144;
        wp::float32 var_145;
        wp::float32 var_146;
        const wp::int32 var_147 = 2;
        wp::float32 var_148;
        wp::float32 var_149;
        const wp::float32 var_150 = 0.5;
        bool var_151;
        const wp::int32 var_152 = 1;
        wp::int32 var_153;
        const wp::int32 var_154 = 3;
        wp::int32 var_155;
        wp::float32 var_156;
        wp::float32 var_157;
        const wp::int32 var_158 = 2;
        wp::int32 var_159;
        const wp::int32 var_160 = 3;
        wp::int32 var_161;
        wp::float32 var_162;
        wp::float32 var_163;
        wp::float32 var_164;
        wp::float32 var_165;
        wp::float32 var_166;
        wp::float32 var_167;
        const wp::float32 var_168 = 0.0;
        bool var_169;
        wp::float32 var_170;
        wp::float32 var_171;
        wp::float32 var_172;
        wp::float32 var_173;
        wp::float32 var_174;
        wp::float32 var_175;
        wp::float32 var_176;
        wp::float32 var_177;
        wp::float32 var_178;
        wp::float32 var_179;
        wp::float32 var_180;
        wp::float32 var_181;
        const wp::int32 var_182 = 3;
        bool var_183;
        const wp::int32 var_184 = 0;
        wp::float32 var_185;
        wp::vec_t<3, wp::float32> var_186;
        wp::vec_t<3, wp::float32> var_187;
        wp::vec_t<3, wp::float32> var_188;
        wp::vec_t<3, wp::float32> var_189;
        wp::vec_t<3, wp::float32> var_190;
        const wp::int32 var_191 = 7;
        bool var_192;
        const wp::float32 var_193 = 1.0;
        const wp::float32 var_194 = 0.0;
        const wp::float32 var_195 = 0.0;
        wp::vec_t<3, wp::float32> var_196;
        wp::vec_t<3, wp::float32> var_197;
        const wp::float32 var_198 = 0.0;
        const wp::float32 var_199 = 1.0;
        const wp::float32 var_200 = 0.0;
        wp::vec_t<3, wp::float32> var_201;
        wp::vec_t<3, wp::float32> var_202;
        const wp::float32 var_203 = 0.0;
        const wp::float32 var_204 = 0.0;
        const wp::float32 var_205 = 1.0;
        wp::vec_t<3, wp::float32> var_206;
        wp::vec_t<3, wp::float32> var_207;
        const wp::int32 var_208 = 0;
        wp::float32 var_209;
        wp::float32 var_210;
        const wp::int32 var_211 = 0;
        wp::float32 var_212;
        wp::float32 var_213;
        const wp::int32 var_214 = 0;
        wp::float32 var_215;
        wp::float32 var_216;
        const wp::int32 var_217 = 1;
        wp::float32 var_218;
        wp::float32 var_219;
        wp::float32 var_220;
        const wp::int32 var_221 = 0;
        wp::float32 var_222;
        wp::float32 var_223;
        const wp::int32 var_224 = 2;
        wp::float32 var_225;
        wp::float32 var_226;
        wp::float32 var_227;
        const wp::int32 var_228 = 1;
        wp::float32 var_229;
        wp::float32 var_230;
        const wp::int32 var_231 = 0;
        wp::float32 var_232;
        wp::float32 var_233;
        const wp::int32 var_234 = 1;
        wp::float32 var_235;
        wp::float32 var_236;
        const wp::int32 var_237 = 1;
        wp::float32 var_238;
        wp::float32 var_239;
        wp::float32 var_240;
        const wp::int32 var_241 = 1;
        wp::float32 var_242;
        wp::float32 var_243;
        const wp::int32 var_244 = 2;
        wp::float32 var_245;
        wp::float32 var_246;
        wp::float32 var_247;
        const wp::int32 var_248 = 2;
        wp::float32 var_249;
        wp::float32 var_250;
        const wp::int32 var_251 = 0;
        wp::float32 var_252;
        wp::float32 var_253;
        const wp::int32 var_254 = 2;
        wp::float32 var_255;
        wp::float32 var_256;
        const wp::int32 var_257 = 1;
        wp::float32 var_258;
        wp::float32 var_259;
        wp::float32 var_260;
        const wp::int32 var_261 = 2;
        wp::float32 var_262;
        wp::float32 var_263;
        const wp::int32 var_264 = 2;
        wp::float32 var_265;
        wp::float32 var_266;
        wp::float32 var_267;
        wp::vec_t<3, wp::float32> var_268;
        wp::vec_t<3, wp::float32> var_269;
        wp::vec_t<3, wp::float32> var_270;
        wp::vec_t<3, wp::float32> var_271;
        wp::vec_t<3, wp::float32> var_272;
        const wp::int32 var_273 = 4;
        bool var_274;
        const wp::int32 var_275 = 0;
        wp::float32 var_276;
        const wp::int32 var_277 = 1;
        wp::float32 var_278;
        const wp::float32 var_279 = 0.0;
        const wp::float32 var_280 = 0.0;
        const wp::float32 var_281 = 1.0;
        wp::vec_t<3, wp::float32> var_282;
        wp::vec_t<3, wp::float32> var_283;
        wp::vec_t<3, wp::float32> var_284;
        wp::vec_t<3, wp::float32> var_285;
        wp::vec_t<3, wp::float32> var_286;
        wp::vec_t<3, wp::float32> var_287;
        wp::vec_t<3, wp::float32> var_288;
        wp::vec_t<3, wp::float32> var_289;
        wp::vec_t<3, wp::float32> var_290;
        wp::vec_t<3, wp::float32> var_291;
        const wp::int32 var_292 = 6;
        bool var_293;
        const wp::int32 var_294 = 0;
        wp::float32 var_295;
        const wp::int32 var_296 = 1;
        wp::float32 var_297;
        const wp::int32 var_298 = 2;
        wp::float32 var_299;
        bool var_300;
        bool var_301;
        const wp::float32 var_302 = 0.0;
        bool var_303;
        wp::float32 var_304;
        wp::float32 var_305;
        wp::float32 var_306;
        wp::float32 var_307;
        wp::float32 var_308;
        wp::float32 var_309;
        wp::float32 var_310;
        wp::float32 var_311;
        wp::float32 var_312;
        const wp::float32 var_313 = 1.0;
        const wp::float32 var_314 = 0.0;
        const wp::float32 var_315 = 0.0;
        wp::vec_t<3, wp::float32> var_316;
        wp::vec_t<3, wp::float32> var_317;
        const wp::float32 var_318 = 0.0;
        const wp::float32 var_319 = 1.0;
        const wp::float32 var_320 = 0.0;
        wp::vec_t<3, wp::float32> var_321;
        wp::vec_t<3, wp::float32> var_322;
        const wp::float32 var_323 = 0.0;
        const wp::float32 var_324 = 0.0;
        const wp::float32 var_325 = 1.0;
        wp::vec_t<3, wp::float32> var_326;
        wp::vec_t<3, wp::float32> var_327;
        const wp::int32 var_328 = 0;
        wp::float32 var_329;
        const wp::int32 var_330 = 0;
        wp::float32 var_331;
        wp::float32 var_332;
        const wp::int32 var_333 = 0;
        wp::float32 var_334;
        const wp::int32 var_335 = 0;
        wp::float32 var_336;
        wp::float32 var_337;
        wp::float32 var_338;
        wp::float32 var_339;
        wp::float32 var_340;
        const wp::int32 var_341 = 0;
        wp::float32 var_342;
        wp::float32 var_343;
        wp::float32 var_344;
        wp::float32 var_345;
        const wp::int32 var_346 = 1;
        wp::float32 var_347;
        const wp::int32 var_348 = 1;
        wp::float32 var_349;
        wp::float32 var_350;
        const wp::int32 var_351 = 1;
        wp::float32 var_352;
        const wp::int32 var_353 = 1;
        wp::float32 var_354;
        wp::float32 var_355;
        wp::float32 var_356;
        wp::float32 var_357;
        wp::float32 var_358;
        const wp::int32 var_359 = 1;
        wp::float32 var_360;
        wp::float32 var_361;
        wp::float32 var_362;
        wp::float32 var_363;
        const wp::int32 var_364 = 2;
        wp::float32 var_365;
        const wp::int32 var_366 = 2;
        wp::float32 var_367;
        wp::float32 var_368;
        const wp::int32 var_369 = 2;
        wp::float32 var_370;
        const wp::int32 var_371 = 2;
        wp::float32 var_372;
        wp::float32 var_373;
        wp::float32 var_374;
        wp::float32 var_375;
        wp::float32 var_376;
        const wp::int32 var_377 = 2;
        wp::float32 var_378;
        wp::float32 var_379;
        wp::float32 var_380;
        wp::float32 var_381;
        wp::vec_t<3, wp::float32> var_382;
        wp::vec_t<3, wp::float32> var_383;
        wp::vec_t<3, wp::float32> var_384;
        wp::vec_t<3, wp::float32> var_385;
        wp::vec_t<3, wp::float32> var_386;
        wp::vec_t<3, wp::float32>* var_387;
        wp::vec_t<3, wp::float32> var_388;
        wp::vec_t<3, wp::float32> var_389;
        wp::vec_t<3, wp::float32>* var_390;
        wp::vec_t<3, wp::float32> var_391;
        wp::vec_t<3, wp::float32> var_392;
        wp::vec_t<3, wp::float32> var_393;
        const wp::float32 var_394 = 0.5;
        wp::vec_t<3, wp::float32> var_395;
        wp::vec_t<3, wp::float32> var_396;
        const wp::float32 var_397 = 0.5;
        wp::vec_t<3, wp::float32> var_398;
        wp::vec_t<3, wp::float32> var_399;
        wp::vec_t<3, wp::float32> var_400;
        const wp::float32 var_401 = 1.0;
        const wp::float32 var_402 = 0.0;
        const wp::float32 var_403 = 0.0;
        wp::vec_t<3, wp::float32> var_404;
        wp::vec_t<3, wp::float32> var_405;
        const wp::float32 var_406 = 0.0;
        const wp::float32 var_407 = 1.0;
        const wp::float32 var_408 = 0.0;
        wp::vec_t<3, wp::float32> var_409;
        wp::vec_t<3, wp::float32> var_410;
        const wp::float32 var_411 = 0.0;
        const wp::float32 var_412 = 0.0;
        const wp::float32 var_413 = 1.0;
        wp::vec_t<3, wp::float32> var_414;
        wp::vec_t<3, wp::float32> var_415;
        const wp::int32 var_416 = 0;
        wp::float32 var_417;
        wp::float32 var_418;
        const wp::int32 var_419 = 0;
        wp::float32 var_420;
        wp::float32 var_421;
        const wp::int32 var_422 = 0;
        wp::float32 var_423;
        wp::float32 var_424;
        const wp::int32 var_425 = 1;
        wp::float32 var_426;
        wp::float32 var_427;
        wp::float32 var_428;
        const wp::int32 var_429 = 0;
        wp::float32 var_430;
        wp::float32 var_431;
        const wp::int32 var_432 = 2;
        wp::float32 var_433;
        wp::float32 var_434;
        wp::float32 var_435;
        const wp::int32 var_436 = 1;
        wp::float32 var_437;
        wp::float32 var_438;
        const wp::int32 var_439 = 0;
        wp::float32 var_440;
        wp::float32 var_441;
        const wp::int32 var_442 = 1;
        wp::float32 var_443;
        wp::float32 var_444;
        const wp::int32 var_445 = 1;
        wp::float32 var_446;
        wp::float32 var_447;
        wp::float32 var_448;
        const wp::int32 var_449 = 1;
        wp::float32 var_450;
        wp::float32 var_451;
        const wp::int32 var_452 = 2;
        wp::float32 var_453;
        wp::float32 var_454;
        wp::float32 var_455;
        const wp::int32 var_456 = 2;
        wp::float32 var_457;
        wp::float32 var_458;
        const wp::int32 var_459 = 0;
        wp::float32 var_460;
        wp::float32 var_461;
        const wp::int32 var_462 = 2;
        wp::float32 var_463;
        wp::float32 var_464;
        const wp::int32 var_465 = 1;
        wp::float32 var_466;
        wp::float32 var_467;
        wp::float32 var_468;
        const wp::int32 var_469 = 2;
        wp::float32 var_470;
        wp::float32 var_471;
        const wp::int32 var_472 = 2;
        wp::float32 var_473;
        wp::float32 var_474;
        wp::float32 var_475;
        wp::vec_t<3, wp::float32> var_476;
        wp::vec_t<3, wp::float32> var_477;
        wp::vec_t<3, wp::float32> var_478;
        wp::vec_t<3, wp::float32> var_479;
        wp::vec_t<3, wp::float32> var_480;
        GenericShapeData_ba895271 var_481;
        const wp::int32 var_482 = 1;
        bool var_483;
        const wp::int32 var_484 = 0;
        wp::float32 var_485;
        const wp::float32 var_486 = 0.5;
        wp::float32 var_487;
        const wp::int32 var_488 = 1;
        wp::float32 var_489;
        const wp::float32 var_490 = 0.5;
        wp::float32 var_491;
        const wp::float32 var_492 = 0.0;
        wp::vec_t<3, wp::float32> var_493;
        wp::vec_t<3, wp::float32> var_494;
        const wp::float32 var_495 = 0.0;
        const wp::float32 var_496 = 0.0;
        const wp::float32 var_497 = 0.0;
        wp::vec_t<3, wp::float32> var_498;
        const wp::float32 var_499 = 0.0;
        const wp::float32 var_500 = 0.0;
        const wp::float32 var_501 = 0.0;
        wp::vec_t<3, wp::float32> var_502;
        const wp::int32 var_503 = 10;
        bool var_504;
        wp::uint64* var_505;
        wp::vec_t<3, wp::float32> var_506;
        wp::uint64 var_507;
        SupportMapDataProvider_e77f8b9f var_508;
        wp::vec_t<3, wp::float32> var_509;
        wp::vec_t<3, wp::float32> var_510;
        wp::vec_t<3, wp::float32> var_511;
        wp::vec_t<3, wp::float32> var_512;
        wp::vec_t<3, wp::float32> var_513;
        wp::vec_t<3, wp::float32> var_514;
        wp::vec_t<3, wp::float32> var_515;
        wp::vec_t<3, wp::float32> var_516;
        wp::vec_t<3, wp::float32> var_517;
        wp::vec_t<3, wp::float32> var_518;
        wp::float32 var_519;
        wp::float32 var_520;
        wp::vec_t<3, wp::float32> var_521;
        wp::vec_t<3, wp::float32> var_522;
        wp::vec_t<3, wp::float32> var_523;
        wp::vec_t<3, wp::float32> var_524;
        wp::vec_t<3, wp::float32> var_525;
        wp::vec_t<3, wp::float32> var_526;
        wp::vec_t<3, wp::float32> var_527;
        wp::float32 var_528;
        wp::vec_t<3, wp::float32> var_529;
        wp::vec_t<3, wp::float32> var_530;
        wp::vec_t<3, wp::float32> var_531;
        const wp::int32 var_532 = 0;
        wp::float32 var_533;
        const wp::int32 var_534 = 1;
        wp::float32 var_535;
        const wp::int32 var_536 = 2;
        wp::float32 var_537;
        wp::vec_t<4, wp::float32> var_538;
        //---------
        // forward
        // def compute_shape_aabbs(                                                               <L 254>
        // shape_id = wp.tid()                                                                    <L 282>
        var_0 = builtin_tid1d();
        // if shape_id == 0:                                                                      <L 286>
        var_2 = (var_0 == var_1);
        if (var_2) {
            // for c in range(num_contact_counters):                                              <L 287>
            var_3 = wp::range(var_num_contact_counters);
            start_for_0:;
                if (iter_cmp(var_3) == 0) goto end_for_0;
                var_4 = wp::iter_next(var_3);
                // contact_counters[c] = 0                                                        <L 288>
                wp::array_store(var_contact_counters, var_4, var_5);
                goto start_for_0;
            end_for_0:;
            // g = contact_generation[0]                                                          <L 289>
            var_7 = wp::address(var_contact_generation, var_6);
            var_9 = wp::load(var_7);
            var_8 = wp::copy(var_9);
            // if g == 2147483647:                                                                <L 290>
            var_11 = (var_8 == var_10);
            if (var_11) {
                // g = 0                                                                          <L 291>
            }
            if (!var_11) {
                // g = g + 1                                                                      <L 293>
                var_14 = wp::add(var_8, var_13);
            }
            var_15 = wp::where(var_11, var_12, var_14);
            // contact_generation[0] = g                                                          <L 294>
            wp::array_store(var_contact_generation, var_16, var_15);
            // broad_phase_pair_count[0] = 0                                                      <L 295>
            wp::array_store(var_broad_phase_pair_count, var_18, var_17);
        }
        // rigid_id = shape_body[shape_id]                                                        <L 297>
        var_19 = wp::address(var_shape_body, var_0);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // geo_type = shape_type[shape_id]                                                        <L 298>
        var_22 = wp::address(var_shape_type, var_0);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // if rigid_id == -1:                                                                     <L 301>
        var_26 = (var_20 == var_25);
        if (var_26) {
            // X_ws = shape_transform[shape_id]                                                   <L 302>
            var_27 = wp::address(var_shape_transform, var_0);
            var_29 = wp::load(var_27);
            var_28 = wp::copy(var_29);
        }
        if (!var_26) {
            // X_ws = wp.transform_multiply(body_q[rigid_id], shape_transform[shape_id])          <L 304>
            var_30 = wp::address(var_body_q, var_20);
            var_31 = wp::address(var_shape_transform, var_0);
            var_33 = wp::load(var_30);
            var_34 = wp::load(var_31);
            var_32 = wp::transform_multiply(var_33, var_34);
        }
        var_35 = wp::where(var_26, var_28, var_32);
        // pos = wp.transform_get_translation(X_ws)                                               <L 306>
        var_36 = wp::transform_get_translation(var_35);
        // orientation = wp.transform_get_rotation(X_ws)                                          <L 307>
        var_37 = wp::transform_get_rotation(var_35);
        // margin = shape_margin[shape_id]                                                        <L 309>
        var_38 = wp::address(var_shape_margin, var_0);
        var_40 = wp::load(var_38);
        var_39 = wp::copy(var_40);
        // effective_gap = margin + shape_gap[shape_id]                                           <L 312>
        var_41 = wp::address(var_shape_gap, var_0);
        var_43 = wp::load(var_41);
        var_42 = wp::add(var_39, var_43);
        // margin_vec = wp.vec3(effective_gap, effective_gap, effective_gap)                      <L 313>
        var_44 = wp::vec_t<3, wp::float32>(var_42, var_42, var_42);
        // scale = shape_scale[shape_id]                                                          <L 316>
        var_45 = wp::address(var_shape_scale, var_0);
        var_47 = wp::load(var_45);
        var_46 = wp::copy(var_47);
        // is_infinite_plane = (geo_type == GeoType.PLANE) and (scale[0] == 0.0 and scale[1] == 0.0)       <L 317>
        var_50 = (var_23 == var_49);
        var_48 = var_50;
        if (var_48) {
            var_53 = wp::extract(var_46, var_52);
            var_55 = (var_53 == var_54);
            var_51 = var_55;
            if (var_51) {
                var_57 = wp::extract(var_46, var_56);
                var_59 = (var_57 == var_58);
                var_51 = var_51 && var_59;
            }
            var_48 = var_48 && var_51;
        }
        // has_local_aabb = geo_type == GeoType.MESH or geo_type == GeoType.HFIELD or geo_type == GeoType.CONVEX_MESH       <L 318>
        var_62 = (var_23 == var_61);
        var_60 = var_62;
        if (!var_60) {
            var_64 = (var_23 == var_63);
            var_60 = var_60 || var_64;
        }
        if (!var_60) {
            var_66 = (var_23 == var_65);
            var_60 = var_60 || var_66;
        }
        // geom_scale = scale                                                                     <L 320>
        var_67 = wp::copy(var_46);
        // if is_infinite_plane:                                                                  <L 322>
        if (var_48) {
            // normal = wp.quat_rotate(orientation, wp.vec3(0.0, 0.0, 1.0))                       <L 330>
            var_71 = wp::vec_t<3, wp::float32>(var_68, var_69, var_70);
            var_72 = wp::quat_rotate(var_37, var_71);
            // HALF_SPACE_EXTENT = 1.0e6                                                          <L 332>
            // half_extents = wp.vec3(HALF_SPACE_EXTENT, HALF_SPACE_EXTENT, HALF_SPACE_EXTENT)       <L 333>
            var_74 = wp::vec_t<3, wp::float32>(var_73, var_73, var_73);
            // lo = pos - half_extents - margin_vec                                               <L 334>
            var_75 = wp::sub(var_36, var_74);
            var_76 = wp::sub(var_75, var_44);
            // hi = pos + half_extents + margin_vec                                               <L 335>
            var_77 = wp::add(var_36, var_74);
            var_78 = wp::add(var_77, var_44);
            // for i in range(3):                                                                 <L 336>
            // n_i = normal[i]                                                                    <L 337>
            var_80 = wp::extract(var_72, var_79);
            // if wp.abs(n_i) > 0.5:                                                              <L 339>
            var_81 = wp::abs(var_80);
            var_83 = (var_81 > var_82);
            if (var_83) {
                // lateral = wp.abs(normal[(i + 1) % 3]) + wp.abs(normal[(i + 2) % 3])            <L 340>
                var_85 = wp::add(var_79, var_84);
                var_87 = wp::mod(var_85, var_86);
                var_88 = wp::extract(var_72, var_87);
                var_89 = wp::abs(var_88);
                var_91 = wp::add(var_79, var_90);
                var_93 = wp::mod(var_91, var_92);
                var_94 = wp::extract(var_72, var_93);
                var_95 = wp::abs(var_94);
                var_96 = wp::add(var_89, var_95);
                // rise = lateral * HALF_SPACE_EXTENT / wp.abs(n_i)                               <L 341>
                var_97 = wp::mul(var_96, var_73);
                var_98 = wp::abs(var_80);
                var_99 = wp::div(var_97, var_98);
                // if n_i > 0.0:                                                                  <L 342>
                var_101 = (var_80 > var_100);
                if (var_101) {
                    // hi[i] = wp.min(hi[i], pos[i] + rise + effective_gap)                       <L 343>
                    var_102 = wp::extract(var_78, var_79);
                    var_103 = wp::extract(var_36, var_79);
                    var_104 = wp::add(var_103, var_99);
                    var_105 = wp::add(var_104, var_42);
                    var_106 = wp::min(var_102, var_105);
                    wp::assign_inplace(var_78, var_79, var_106);
                }
                if (!var_101) {
                    // lo[i] = wp.max(lo[i], pos[i] - rise - effective_gap)                       <L 345>
                    var_107 = wp::extract(var_76, var_79);
                    var_108 = wp::extract(var_36, var_79);
                    var_109 = wp::sub(var_108, var_99);
                    var_110 = wp::sub(var_109, var_42);
                    var_111 = wp::max(var_107, var_110);
                    wp::assign_inplace(var_76, var_79, var_111);
                }
            }
            // n_i = normal[i]                                                                    <L 337>
            var_113 = wp::extract(var_72, var_112);
            // if wp.abs(n_i) > 0.5:                                                              <L 339>
            var_114 = wp::abs(var_113);
            var_116 = (var_114 > var_115);
            if (var_116) {
                // lateral = wp.abs(normal[(i + 1) % 3]) + wp.abs(normal[(i + 2) % 3])            <L 340>
                var_118 = wp::add(var_112, var_117);
                var_120 = wp::mod(var_118, var_119);
                var_121 = wp::extract(var_72, var_120);
                var_122 = wp::abs(var_121);
                var_124 = wp::add(var_112, var_123);
                var_126 = wp::mod(var_124, var_125);
                var_127 = wp::extract(var_72, var_126);
                var_128 = wp::abs(var_127);
                var_129 = wp::add(var_122, var_128);
                // rise = lateral * HALF_SPACE_EXTENT / wp.abs(n_i)                               <L 341>
                var_130 = wp::mul(var_129, var_73);
                var_131 = wp::abs(var_113);
                var_132 = wp::div(var_130, var_131);
                // if n_i > 0.0:                                                                  <L 342>
                var_134 = (var_113 > var_133);
                if (var_134) {
                    // hi[i] = wp.min(hi[i], pos[i] + rise + effective_gap)                       <L 343>
                    var_135 = wp::extract(var_78, var_112);
                    var_136 = wp::extract(var_36, var_112);
                    var_137 = wp::add(var_136, var_132);
                    var_138 = wp::add(var_137, var_42);
                    var_139 = wp::min(var_135, var_138);
                    wp::assign_inplace(var_78, var_112, var_139);
                }
                if (!var_134) {
                    // lo[i] = wp.max(lo[i], pos[i] - rise - effective_gap)                       <L 345>
                    var_140 = wp::extract(var_76, var_112);
                    var_141 = wp::extract(var_36, var_112);
                    var_142 = wp::sub(var_141, var_132);
                    var_143 = wp::sub(var_142, var_42);
                    var_144 = wp::max(var_140, var_143);
                    wp::assign_inplace(var_76, var_112, var_144);
                }
            }
            var_145 = wp::where(var_116, var_129, var_96);
            var_146 = wp::where(var_116, var_132, var_99);
            // n_i = normal[i]                                                                    <L 337>
            var_148 = wp::extract(var_72, var_147);
            // if wp.abs(n_i) > 0.5:                                                              <L 339>
            var_149 = wp::abs(var_148);
            var_151 = (var_149 > var_150);
            if (var_151) {
                // lateral = wp.abs(normal[(i + 1) % 3]) + wp.abs(normal[(i + 2) % 3])            <L 340>
                var_153 = wp::add(var_147, var_152);
                var_155 = wp::mod(var_153, var_154);
                var_156 = wp::extract(var_72, var_155);
                var_157 = wp::abs(var_156);
                var_159 = wp::add(var_147, var_158);
                var_161 = wp::mod(var_159, var_160);
                var_162 = wp::extract(var_72, var_161);
                var_163 = wp::abs(var_162);
                var_164 = wp::add(var_157, var_163);
                // rise = lateral * HALF_SPACE_EXTENT / wp.abs(n_i)                               <L 341>
                var_165 = wp::mul(var_164, var_73);
                var_166 = wp::abs(var_148);
                var_167 = wp::div(var_165, var_166);
                // if n_i > 0.0:                                                                  <L 342>
                var_169 = (var_148 > var_168);
                if (var_169) {
                    // hi[i] = wp.min(hi[i], pos[i] + rise + effective_gap)                       <L 343>
                    var_170 = wp::extract(var_78, var_147);
                    var_171 = wp::extract(var_36, var_147);
                    var_172 = wp::add(var_171, var_167);
                    var_173 = wp::add(var_172, var_42);
                    var_174 = wp::min(var_170, var_173);
                    wp::assign_inplace(var_78, var_147, var_174);
                }
                if (!var_169) {
                    // lo[i] = wp.max(lo[i], pos[i] - rise - effective_gap)                       <L 345>
                    var_175 = wp::extract(var_76, var_147);
                    var_176 = wp::extract(var_36, var_147);
                    var_177 = wp::sub(var_176, var_167);
                    var_178 = wp::sub(var_177, var_42);
                    var_179 = wp::max(var_175, var_178);
                    wp::assign_inplace(var_76, var_147, var_179);
                }
            }
            var_180 = wp::where(var_151, var_164, var_145);
            var_181 = wp::where(var_151, var_167, var_146);
            // aabb_lower[shape_id] = lo                                                          <L 346>
            wp::array_store(var_aabb_lower, var_0, var_76);
            // aabb_upper[shape_id] = hi                                                          <L 347>
            wp::array_store(var_aabb_upper, var_0, var_78);
        }
        if (!var_48) {
            // elif geo_type == GeoType.SPHERE:                                                   <L 348>
            var_183 = (var_23 == var_182);
            if (var_183) {
                // radius = scale[0]                                                              <L 349>
                var_185 = wp::extract(var_46, var_184);
                // half_extents = wp.vec3(radius, radius, radius)                                 <L 350>
                var_186 = wp::vec_t<3, wp::float32>(var_185, var_185, var_185);
                // aabb_lower[shape_id] = pos - half_extents - margin_vec                         <L 351>
                var_187 = wp::sub(var_36, var_186);
                var_188 = wp::sub(var_187, var_44);
                wp::array_store(var_aabb_lower, var_0, var_188);
                // aabb_upper[shape_id] = pos + half_extents + margin_vec                         <L 352>
                var_189 = wp::add(var_36, var_186);
                var_190 = wp::add(var_189, var_44);
                wp::array_store(var_aabb_upper, var_0, var_190);
            }
            if (!var_183) {
                // elif geo_type == GeoType.BOX:                                                  <L 353>
                var_192 = (var_23 == var_191);
                if (var_192) {
                    // r0 = wp.quat_rotate(orientation, wp.vec3(1.0, 0.0, 0.0))                   <L 355>
                    var_196 = wp::vec_t<3, wp::float32>(var_193, var_194, var_195);
                    var_197 = wp::quat_rotate(var_37, var_196);
                    // r1 = wp.quat_rotate(orientation, wp.vec3(0.0, 1.0, 0.0))                   <L 356>
                    var_201 = wp::vec_t<3, wp::float32>(var_198, var_199, var_200);
                    var_202 = wp::quat_rotate(var_37, var_201);
                    // r2 = wp.quat_rotate(orientation, wp.vec3(0.0, 0.0, 1.0))                   <L 357>
                    var_206 = wp::vec_t<3, wp::float32>(var_203, var_204, var_205);
                    var_207 = wp::quat_rotate(var_37, var_206);
                    // half_extents = wp.vec3(                                                    <L 358>
                    // wp.abs(r0[0]) * scale[0] + wp.abs(r1[0]) * scale[1] + wp.abs(r2[0]) * scale[2],       <L 359>
                    var_209 = wp::extract(var_197, var_208);
                    var_210 = wp::abs(var_209);
                    var_212 = wp::extract(var_46, var_211);
                    var_213 = wp::mul(var_210, var_212);
                    var_215 = wp::extract(var_202, var_214);
                    var_216 = wp::abs(var_215);
                    var_218 = wp::extract(var_46, var_217);
                    var_219 = wp::mul(var_216, var_218);
                    var_220 = wp::add(var_213, var_219);
                    var_222 = wp::extract(var_207, var_221);
                    var_223 = wp::abs(var_222);
                    var_225 = wp::extract(var_46, var_224);
                    var_226 = wp::mul(var_223, var_225);
                    var_227 = wp::add(var_220, var_226);
                    // wp.abs(r0[1]) * scale[0] + wp.abs(r1[1]) * scale[1] + wp.abs(r2[1]) * scale[2],       <L 360>
                    var_229 = wp::extract(var_197, var_228);
                    var_230 = wp::abs(var_229);
                    var_232 = wp::extract(var_46, var_231);
                    var_233 = wp::mul(var_230, var_232);
                    var_235 = wp::extract(var_202, var_234);
                    var_236 = wp::abs(var_235);
                    var_238 = wp::extract(var_46, var_237);
                    var_239 = wp::mul(var_236, var_238);
                    var_240 = wp::add(var_233, var_239);
                    var_242 = wp::extract(var_207, var_241);
                    var_243 = wp::abs(var_242);
                    var_245 = wp::extract(var_46, var_244);
                    var_246 = wp::mul(var_243, var_245);
                    var_247 = wp::add(var_240, var_246);
                    // wp.abs(r0[2]) * scale[0] + wp.abs(r1[2]) * scale[1] + wp.abs(r2[2]) * scale[2],       <L 361>
                    var_249 = wp::extract(var_197, var_248);
                    var_250 = wp::abs(var_249);
                    var_252 = wp::extract(var_46, var_251);
                    var_253 = wp::mul(var_250, var_252);
                    var_255 = wp::extract(var_202, var_254);
                    var_256 = wp::abs(var_255);
                    var_258 = wp::extract(var_46, var_257);
                    var_259 = wp::mul(var_256, var_258);
                    var_260 = wp::add(var_253, var_259);
                    var_262 = wp::extract(var_207, var_261);
                    var_263 = wp::abs(var_262);
                    var_265 = wp::extract(var_46, var_264);
                    var_266 = wp::mul(var_263, var_265);
                    var_267 = wp::add(var_260, var_266);
                    var_268 = wp::vec_t<3, wp::float32>(var_227, var_247, var_267);
                    // aabb_lower[shape_id] = pos - half_extents - margin_vec                     <L 363>
                    var_269 = wp::sub(var_36, var_268);
                    var_270 = wp::sub(var_269, var_44);
                    wp::array_store(var_aabb_lower, var_0, var_270);
                    // aabb_upper[shape_id] = pos + half_extents + margin_vec                     <L 364>
                    var_271 = wp::add(var_36, var_268);
                    var_272 = wp::add(var_271, var_44);
                    wp::array_store(var_aabb_upper, var_0, var_272);
                }
                if (!var_192) {
                    // elif geo_type == GeoType.CAPSULE:                                          <L 365>
                    var_274 = (var_23 == var_273);
                    if (var_274) {
                        // radius = scale[0]                                                      <L 366>
                        var_276 = wp::extract(var_46, var_275);
                        // half_height = scale[1]                                                 <L 367>
                        var_278 = wp::extract(var_46, var_277);
                        // axis = wp.quat_rotate(orientation, wp.vec3(0.0, 0.0, 1.0))             <L 368>
                        var_282 = wp::vec_t<3, wp::float32>(var_279, var_280, var_281);
                        var_283 = wp::quat_rotate(var_37, var_282);
                        // half_extents = wp.vec3(radius, radius, radius) + wp.abs(axis) * half_height       <L 369>
                        var_284 = wp::vec_t<3, wp::float32>(var_276, var_276, var_276);
                        var_285 = wp::abs(var_283);
                        var_286 = wp::mul(var_285, var_278);
                        var_287 = wp::add(var_284, var_286);
                        // aabb_lower[shape_id] = pos - half_extents - margin_vec                 <L 370>
                        var_288 = wp::sub(var_36, var_287);
                        var_289 = wp::sub(var_288, var_44);
                        wp::array_store(var_aabb_lower, var_0, var_289);
                        // aabb_upper[shape_id] = pos + half_extents + margin_vec                 <L 371>
                        var_290 = wp::add(var_36, var_287);
                        var_291 = wp::add(var_290, var_44);
                        wp::array_store(var_aabb_upper, var_0, var_291);
                    }
                    if (!var_274) {
                        // elif geo_type == GeoType.CYLINDER:                                     <L 372>
                        var_293 = (var_23 == var_292);
                        if (var_293) {
                            // radius = scale[0]                                                  <L 373>
                            var_295 = wp::extract(var_46, var_294);
                            // half_height = scale[1]                                             <L 374>
                            var_297 = wp::extract(var_46, var_296);
                            // barrel_radius = scale[2]                                           <L 375>
                            var_299 = wp::extract(var_46, var_298);
                            // if barrel_radius >= half_height and barrel_radius > 0.0:           <L 377>
                            var_301 = (var_299 >= var_297);
                            var_300 = var_301;
                            if (var_300) {
                                var_303 = (var_299 > var_302);
                                var_300 = var_300 && var_303;
                            }
                            if (var_300) {
                                // radius += (half_height * half_height) / (                      <L 378>
                                var_304 = wp::mul(var_297, var_297);
                                // barrel_radius + wp.sqrt(barrel_radius * barrel_radius - half_height * half_height)       <L 379>
                                var_305 = wp::mul(var_299, var_299);
                                var_306 = wp::mul(var_297, var_297);
                                var_307 = wp::sub(var_305, var_306);
                                var_308 = wp::sqrt(var_307);
                                var_309 = wp::add(var_299, var_308);
                                var_310 = wp::div(var_304, var_309);
                                // radius += (half_height * half_height) / (                      <L 378>
                                var_311 = wp::add(var_295, var_310);
                            }
                            var_312 = wp::where(var_300, var_311, var_295);
                            // r0 = wp.quat_rotate(orientation, wp.vec3(1.0, 0.0, 0.0))           <L 381>
                            var_316 = wp::vec_t<3, wp::float32>(var_313, var_314, var_315);
                            var_317 = wp::quat_rotate(var_37, var_316);
                            // r1 = wp.quat_rotate(orientation, wp.vec3(0.0, 1.0, 0.0))           <L 382>
                            var_321 = wp::vec_t<3, wp::float32>(var_318, var_319, var_320);
                            var_322 = wp::quat_rotate(var_37, var_321);
                            // r2 = wp.quat_rotate(orientation, wp.vec3(0.0, 0.0, 1.0))           <L 383>
                            var_326 = wp::vec_t<3, wp::float32>(var_323, var_324, var_325);
                            var_327 = wp::quat_rotate(var_37, var_326);
                            // half_extents = wp.vec3(                                            <L 384>
                            // radius * wp.sqrt(r0[0] * r0[0] + r1[0] * r1[0]) + half_height * wp.abs(r2[0]),       <L 385>
                            var_329 = wp::extract(var_317, var_328);
                            var_331 = wp::extract(var_317, var_330);
                            var_332 = wp::mul(var_329, var_331);
                            var_334 = wp::extract(var_322, var_333);
                            var_336 = wp::extract(var_322, var_335);
                            var_337 = wp::mul(var_334, var_336);
                            var_338 = wp::add(var_332, var_337);
                            var_339 = wp::sqrt(var_338);
                            var_340 = wp::mul(var_312, var_339);
                            var_342 = wp::extract(var_327, var_341);
                            var_343 = wp::abs(var_342);
                            var_344 = wp::mul(var_297, var_343);
                            var_345 = wp::add(var_340, var_344);
                            // radius * wp.sqrt(r0[1] * r0[1] + r1[1] * r1[1]) + half_height * wp.abs(r2[1]),       <L 386>
                            var_347 = wp::extract(var_317, var_346);
                            var_349 = wp::extract(var_317, var_348);
                            var_350 = wp::mul(var_347, var_349);
                            var_352 = wp::extract(var_322, var_351);
                            var_354 = wp::extract(var_322, var_353);
                            var_355 = wp::mul(var_352, var_354);
                            var_356 = wp::add(var_350, var_355);
                            var_357 = wp::sqrt(var_356);
                            var_358 = wp::mul(var_312, var_357);
                            var_360 = wp::extract(var_327, var_359);
                            var_361 = wp::abs(var_360);
                            var_362 = wp::mul(var_297, var_361);
                            var_363 = wp::add(var_358, var_362);
                            // radius * wp.sqrt(r0[2] * r0[2] + r1[2] * r1[2]) + half_height * wp.abs(r2[2]),       <L 387>
                            var_365 = wp::extract(var_317, var_364);
                            var_367 = wp::extract(var_317, var_366);
                            var_368 = wp::mul(var_365, var_367);
                            var_370 = wp::extract(var_322, var_369);
                            var_372 = wp::extract(var_322, var_371);
                            var_373 = wp::mul(var_370, var_372);
                            var_374 = wp::add(var_368, var_373);
                            var_375 = wp::sqrt(var_374);
                            var_376 = wp::mul(var_312, var_375);
                            var_378 = wp::extract(var_327, var_377);
                            var_379 = wp::abs(var_378);
                            var_380 = wp::mul(var_297, var_379);
                            var_381 = wp::add(var_376, var_380);
                            var_382 = wp::vec_t<3, wp::float32>(var_345, var_363, var_381);
                            // aabb_lower[shape_id] = pos - half_extents - margin_vec             <L 389>
                            var_383 = wp::sub(var_36, var_382);
                            var_384 = wp::sub(var_383, var_44);
                            wp::array_store(var_aabb_lower, var_0, var_384);
                            // aabb_upper[shape_id] = pos + half_extents + margin_vec             <L 390>
                            var_385 = wp::add(var_36, var_382);
                            var_386 = wp::add(var_385, var_44);
                            wp::array_store(var_aabb_upper, var_0, var_386);
                        }
                        if (!var_293) {
                            // elif has_local_aabb:                                               <L 391>
                            if (var_60) {
                                // local_lo = shape_collision_aabb_lower[shape_id]                <L 395>
                                var_387 = wp::address(var_shape_collision_aabb_lower, var_0);
                                var_389 = wp::load(var_387);
                                var_388 = wp::copy(var_389);
                                // local_hi = shape_collision_aabb_upper[shape_id]                <L 396>
                                var_390 = wp::address(var_shape_collision_aabb_upper, var_0);
                                var_392 = wp::load(var_390);
                                var_391 = wp::copy(var_392);
                                // center = (local_lo + local_hi) * 0.5                           <L 398>
                                var_393 = wp::add(var_388, var_391);
                                var_395 = wp::mul(var_393, var_394);
                                // half = (local_hi - local_lo) * 0.5                             <L 399>
                                var_396 = wp::sub(var_391, var_388);
                                var_398 = wp::mul(var_396, var_397);
                                // world_center = wp.quat_rotate(orientation, center) + pos       <L 402>
                                var_399 = wp::quat_rotate(var_37, var_395);
                                var_400 = wp::add(var_399, var_36);
                                // r0 = wp.quat_rotate(orientation, wp.vec3(1.0, 0.0, 0.0))       <L 405>
                                var_404 = wp::vec_t<3, wp::float32>(var_401, var_402, var_403);
                                var_405 = wp::quat_rotate(var_37, var_404);
                                // r1 = wp.quat_rotate(orientation, wp.vec3(0.0, 1.0, 0.0))       <L 406>
                                var_409 = wp::vec_t<3, wp::float32>(var_406, var_407, var_408);
                                var_410 = wp::quat_rotate(var_37, var_409);
                                // r2 = wp.quat_rotate(orientation, wp.vec3(0.0, 0.0, 1.0))       <L 407>
                                var_414 = wp::vec_t<3, wp::float32>(var_411, var_412, var_413);
                                var_415 = wp::quat_rotate(var_37, var_414);
                                // world_half = wp.vec3(                                          <L 409>
                                // wp.abs(r0[0]) * half[0] + wp.abs(r1[0]) * half[1] + wp.abs(r2[0]) * half[2],       <L 410>
                                var_417 = wp::extract(var_405, var_416);
                                var_418 = wp::abs(var_417);
                                var_420 = wp::extract(var_398, var_419);
                                var_421 = wp::mul(var_418, var_420);
                                var_423 = wp::extract(var_410, var_422);
                                var_424 = wp::abs(var_423);
                                var_426 = wp::extract(var_398, var_425);
                                var_427 = wp::mul(var_424, var_426);
                                var_428 = wp::add(var_421, var_427);
                                var_430 = wp::extract(var_415, var_429);
                                var_431 = wp::abs(var_430);
                                var_433 = wp::extract(var_398, var_432);
                                var_434 = wp::mul(var_431, var_433);
                                var_435 = wp::add(var_428, var_434);
                                // wp.abs(r0[1]) * half[0] + wp.abs(r1[1]) * half[1] + wp.abs(r2[1]) * half[2],       <L 411>
                                var_437 = wp::extract(var_405, var_436);
                                var_438 = wp::abs(var_437);
                                var_440 = wp::extract(var_398, var_439);
                                var_441 = wp::mul(var_438, var_440);
                                var_443 = wp::extract(var_410, var_442);
                                var_444 = wp::abs(var_443);
                                var_446 = wp::extract(var_398, var_445);
                                var_447 = wp::mul(var_444, var_446);
                                var_448 = wp::add(var_441, var_447);
                                var_450 = wp::extract(var_415, var_449);
                                var_451 = wp::abs(var_450);
                                var_453 = wp::extract(var_398, var_452);
                                var_454 = wp::mul(var_451, var_453);
                                var_455 = wp::add(var_448, var_454);
                                // wp.abs(r0[2]) * half[0] + wp.abs(r1[2]) * half[1] + wp.abs(r2[2]) * half[2],       <L 412>
                                var_457 = wp::extract(var_405, var_456);
                                var_458 = wp::abs(var_457);
                                var_460 = wp::extract(var_398, var_459);
                                var_461 = wp::mul(var_458, var_460);
                                var_463 = wp::extract(var_410, var_462);
                                var_464 = wp::abs(var_463);
                                var_466 = wp::extract(var_398, var_465);
                                var_467 = wp::mul(var_464, var_466);
                                var_468 = wp::add(var_461, var_467);
                                var_470 = wp::extract(var_415, var_469);
                                var_471 = wp::abs(var_470);
                                var_473 = wp::extract(var_398, var_472);
                                var_474 = wp::mul(var_471, var_473);
                                var_475 = wp::add(var_468, var_474);
                                var_476 = wp::vec_t<3, wp::float32>(var_435, var_455, var_475);
                                // aabb_lower[shape_id] = world_center - world_half - margin_vec       <L 415>
                                var_477 = wp::sub(var_400, var_476);
                                var_478 = wp::sub(var_477, var_44);
                                wp::array_store(var_aabb_lower, var_0, var_478);
                                // aabb_upper[shape_id] = world_center + world_half + margin_vec       <L 416>
                                var_479 = wp::add(var_400, var_476);
                                var_480 = wp::add(var_479, var_44);
                                wp::array_store(var_aabb_upper, var_0, var_480);
                            }
                            if (!var_60) {
                                // shape_data = GenericShapeData()                                <L 420>
                                var_481 = GenericShapeData_ba895271();
                                // shape_data.shape_type = geo_type                               <L 421>
                                var_481.shape_type = var_23;
                                // if geo_type == GeoType.PLANE:                                  <L 422>
                                var_483 = (var_23 == var_482);
                                if (var_483) {
                                    // geom_scale = wp.vec3(scale[0] * 0.5, scale[1] * 0.5, 0.0)       <L 423>
                                    var_485 = wp::extract(var_46, var_484);
                                    var_487 = wp::mul(var_485, var_486);
                                    var_489 = wp::extract(var_46, var_488);
                                    var_491 = wp::mul(var_489, var_490);
                                    var_493 = wp::vec_t<3, wp::float32>(var_487, var_491, var_492);
                                }
                                var_494 = wp::where(var_483, var_493, var_67);
                                // shape_data.scale = geom_scale                                  <L 424>
                                var_481.scale = var_494;
                                // shape_data.auxiliary = wp.vec3(0.0, 0.0, 0.0)                  <L 425>
                                var_498 = wp::vec_t<3, wp::float32>(var_495, var_496, var_497);
                                var_481.auxiliary = var_498;
                                // shape_data.center = wp.vec3(0.0, 0.0, 0.0)                     <L 426>
                                var_502 = wp::vec_t<3, wp::float32>(var_499, var_500, var_501);
                                var_481.center = var_502;
                                // if geo_type == GeoType.CONVEX_MESH:                            <L 429>
                                var_504 = (var_23 == var_503);
                                if (var_504) {
                                    // shape_data.auxiliary = pack_mesh_ptr(shape_source_ptr[shape_id])       <L 430>
                                    var_505 = wp::address(var_shape_source_ptr, var_0);
                                    var_507 = wp::load(var_505);
                                    var_506 = pack_mesh_ptr_0(var_507);
                                    var_481.auxiliary = var_506;
                                }
                                // data_provider = SupportMapDataProvider()                       <L 432>
                                var_508 = SupportMapDataProvider_e77f8b9f();
                                // aabb_min_world, aabb_max_world = compute_tight_aabb_from_support(shape_data, orientation, pos, data_provider)       <L 435>
                                compute_tight_aabb_from_support_0(var_481, var_37, var_36, var_508, var_509, var_510);
                                // aabb_lower[shape_id] = aabb_min_world - margin_vec             <L 437>
                                var_511 = wp::sub(var_509, var_44);
                                wp::array_store(var_aabb_lower, var_0, var_511);
                                // aabb_upper[shape_id] = aabb_max_world + margin_vec             <L 438>
                                var_512 = wp::add(var_510, var_44);
                                wp::array_store(var_aabb_upper, var_0, var_512);
                            }
                            var_513 = wp::where(var_60, var_67, var_494);
                        }
                        var_514 = wp::where(var_293, var_67, var_513);
                        var_515 = wp::where(var_293, var_317, var_405);
                        var_516 = wp::where(var_293, var_322, var_410);
                        var_517 = wp::where(var_293, var_327, var_415);
                    }
                    var_518 = wp::where(var_274, var_67, var_514);
                    var_519 = wp::where(var_274, var_276, var_312);
                    var_520 = wp::where(var_274, var_278, var_297);
                    var_521 = wp::where(var_274, var_287, var_382);
                }
                var_522 = wp::where(var_192, var_67, var_518);
                var_523 = wp::where(var_192, var_197, var_515);
                var_524 = wp::where(var_192, var_202, var_516);
                var_525 = wp::where(var_192, var_207, var_517);
                var_526 = wp::where(var_192, var_268, var_521);
            }
            var_527 = wp::where(var_183, var_67, var_522);
            var_528 = wp::where(var_183, var_185, var_519);
            var_529 = wp::where(var_183, var_186, var_526);
        }
        var_530 = wp::where(var_48, var_67, var_527);
        var_531 = wp::where(var_48, var_74, var_529);
        // geom_data[shape_id] = wp.vec4(geom_scale[0], geom_scale[1], geom_scale[2], margin)       <L 441>
        var_533 = wp::extract(var_530, var_532);
        var_535 = wp::extract(var_530, var_534);
        var_537 = wp::extract(var_530, var_536);
        var_538 = wp::vec_t<4, wp::float32>(var_533, var_535, var_537, var_39);
        wp::array_store(var_geom_data, var_0, var_538);
        // geom_xform[shape_id] = X_ws                                                            <L 442>
        wp::array_store(var_geom_xform, var_0, var_35);
    }
}

