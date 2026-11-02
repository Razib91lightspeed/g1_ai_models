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


struct HeightfieldData_f2b8d59a
{
    wp::int32 data_offset;
    wp::int32 nrow;
    wp::int32 ncol;
    wp::float32 hx;
    wp::float32 hy;
    wp::float32 min_z;
    wp::float32 max_z;


    HeightfieldData_f2b8d59a() = default;
    CUDA_CALLABLE HeightfieldData_f2b8d59a(wp::int32 const& data_offset,
    wp::int32 const& nrow = {},
    wp::int32 const& ncol = {},
    wp::float32 const& hx = {},
    wp::float32 const& hy = {},
    wp::float32 const& min_z = {},
    wp::float32 const& max_z = {})
        : data_offset{data_offset}
        , nrow{nrow}
        , ncol{ncol}
        , hx{hx}
        , hy{hy}
        , min_z{min_z}
        , max_z{max_z}

    {
    }

    CUDA_CALLABLE HeightfieldData_f2b8d59a& operator += (const HeightfieldData_f2b8d59a& rhs)
    {    data_offset += rhs.data_offset;
    nrow += rhs.nrow;
    ncol += rhs.ncol;
    hx += rhs.hx;
    hy += rhs.hy;
    min_z += rhs.min_z;
    max_z += rhs.max_z;

        return *this;}

};

static CUDA_CALLABLE void adj_HeightfieldData_f2b8d59a(wp::int32 const&,
    wp::int32 const&,
    wp::int32 const&,
    wp::float32 const&,
    wp::float32 const&,
    wp::float32 const&,
    wp::float32 const&,
    wp::int32 & adj_data_offset,
    wp::int32 & adj_nrow,
    wp::int32 & adj_ncol,
    wp::float32 & adj_hx,
    wp::float32 & adj_hy,
    wp::float32 & adj_min_z,
    wp::float32 & adj_max_z,
    HeightfieldData_f2b8d59a & adj_ret)
{
    adj_data_offset += adj_ret.data_offset;
    adj_nrow += adj_ret.nrow;
    adj_ncol += adj_ret.ncol;
    adj_hx += adj_ret.hx;
    adj_hy += adj_ret.hy;
    adj_min_z += adj_ret.min_z;
    adj_max_z += adj_ret.max_z;
}

// Required when compiling adjoints.
CUDA_CALLABLE HeightfieldData_f2b8d59a add(const HeightfieldData_f2b8d59a& a, const HeightfieldData_f2b8d59a& b)
{
    return HeightfieldData_f2b8d59a();
}

CUDA_CALLABLE void adj_atomic_add(HeightfieldData_f2b8d59a* p, HeightfieldData_f2b8d59a t)
{
    wp::adj_atomic_add(&p->data_offset, t.data_offset);
    wp::adj_atomic_add(&p->nrow, t.nrow);
    wp::adj_atomic_add(&p->ncol, t.ncol);
    wp::adj_atomic_add(&p->hx, t.hx);
    wp::adj_atomic_add(&p->hy, t.hy);
    wp::adj_atomic_add(&p->min_z, t.min_z);
    wp::adj_atomic_add(&p->max_z, t.max_z);
}




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




// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/utils/heightfield.py:366
static CUDA_CALLABLE void heightfield_vs_convex_midphase_0(
    wp::int32 var_hfield_shape,
    wp::int32 var_other_shape,
    HeightfieldData_f2b8d59a var_hfd,
    wp::array_t<wp::float32> var_elevation_data,
    wp::array_t<wp::transform_t<wp::float32>> var_shape_transform,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_collision_aabb_lower,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_collision_aabb_upper,
    wp::array_t<wp::vec_t<4, wp::float32>> var_shape_data,
    wp::array_t<wp::float32> var_shape_gap,
    wp::array_t<wp::vec_t<3, wp::int32>> var_triangle_pairs,
    wp::array_t<wp::int32> var_triangle_pairs_count)
{
    //---------
    // primal vars
    wp::transform_t<wp::float32>* var_0;
    wp::transform_t<wp::float32> var_1;
    wp::transform_t<wp::float32> var_2;
    wp::transform_t<wp::float32>* var_3;
    wp::transform_t<wp::float32> var_4;
    wp::transform_t<wp::float32> var_5;
    wp::transform_t<wp::float32> var_6;
    wp::transform_t<wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::quat_t<wp::float32> var_9;
    wp::vec_t<3, wp::float32>* var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32>* var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    const wp::float32 var_16 = 0.5;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<3, wp::float32> var_18;
    const wp::float32 var_19 = 0.5;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    const wp::float32 var_24 = 1.0;
    const wp::float32 var_25 = 0.0;
    const wp::float32 var_26 = 0.0;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    const wp::float32 var_29 = 0.0;
    const wp::float32 var_30 = 1.0;
    const wp::float32 var_31 = 0.0;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32> var_33;
    const wp::float32 var_34 = 0.0;
    const wp::float32 var_35 = 0.0;
    const wp::float32 var_36 = 1.0;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    const wp::int32 var_39 = 0;
    wp::float32 var_40;
    wp::float32 var_41;
    const wp::int32 var_42 = 0;
    wp::float32 var_43;
    wp::float32 var_44;
    const wp::int32 var_45 = 0;
    wp::float32 var_46;
    wp::float32 var_47;
    const wp::int32 var_48 = 1;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::int32 var_52 = 0;
    wp::float32 var_53;
    wp::float32 var_54;
    const wp::int32 var_55 = 2;
    wp::float32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    const wp::int32 var_59 = 1;
    wp::float32 var_60;
    wp::float32 var_61;
    const wp::int32 var_62 = 0;
    wp::float32 var_63;
    wp::float32 var_64;
    const wp::int32 var_65 = 1;
    wp::float32 var_66;
    wp::float32 var_67;
    const wp::int32 var_68 = 1;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    const wp::int32 var_72 = 1;
    wp::float32 var_73;
    wp::float32 var_74;
    const wp::int32 var_75 = 2;
    wp::float32 var_76;
    wp::float32 var_77;
    wp::float32 var_78;
    const wp::int32 var_79 = 2;
    wp::float32 var_80;
    wp::float32 var_81;
    const wp::int32 var_82 = 0;
    wp::float32 var_83;
    wp::float32 var_84;
    const wp::int32 var_85 = 2;
    wp::float32 var_86;
    wp::float32 var_87;
    const wp::int32 var_88 = 1;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    const wp::int32 var_92 = 2;
    wp::float32 var_93;
    wp::float32 var_94;
    const wp::int32 var_95 = 2;
    wp::float32 var_96;
    wp::float32 var_97;
    wp::float32 var_98;
    wp::vec_t<3, wp::float32> var_99;
    wp::float32* var_100;
    wp::float32* var_101;
    wp::float32 var_102;
    wp::float32 var_103;
    wp::float32 var_104;
    wp::vec_t<4, wp::float32>* var_105;
    const wp::int32 var_106 = 3;
    wp::float32 var_107;
    wp::vec_t<4, wp::float32> var_108;
    wp::vec_t<4, wp::float32>* var_109;
    const wp::int32 var_110 = 3;
    wp::float32 var_111;
    wp::vec_t<4, wp::float32> var_112;
    wp::float32 var_113;
    wp::float32 var_114;
    wp::vec_t<3, wp::float32> var_115;
    wp::vec_t<3, wp::float32> var_116;
    wp::vec_t<3, wp::float32> var_117;
    wp::vec_t<3, wp::float32> var_118;
    wp::vec_t<3, wp::float32> var_119;
    const wp::float32 var_120 = 2.0;
    wp::float32* var_121;
    wp::float32 var_122;
    wp::float32 var_123;
    wp::int32* var_124;
    const wp::int32 var_125 = 1;
    wp::int32 var_126;
    wp::int32 var_127;
    wp::float32 var_128;
    wp::float32 var_129;
    const wp::float32 var_130 = 2.0;
    wp::float32* var_131;
    wp::float32 var_132;
    wp::float32 var_133;
    wp::int32* var_134;
    const wp::int32 var_135 = 1;
    wp::int32 var_136;
    wp::int32 var_137;
    wp::float32 var_138;
    wp::float32 var_139;
    const wp::int32 var_140 = 0;
    wp::float32 var_141;
    wp::float32* var_142;
    wp::float32 var_143;
    wp::float32 var_144;
    wp::float32 var_145;
    const wp::int32 var_146 = 0;
    wp::float32 var_147;
    wp::float32* var_148;
    wp::float32 var_149;
    wp::float32 var_150;
    wp::float32 var_151;
    const wp::int32 var_152 = 1;
    wp::float32 var_153;
    wp::float32* var_154;
    wp::float32 var_155;
    wp::float32 var_156;
    wp::float32 var_157;
    const wp::int32 var_158 = 1;
    wp::float32 var_159;
    wp::float32* var_160;
    wp::float32 var_161;
    wp::float32 var_162;
    wp::float32 var_163;
    wp::float32 var_164;
    wp::int32 var_165;
    const wp::int32 var_166 = 0;
    wp::int32 var_167;
    wp::float32 var_168;
    wp::int32 var_169;
    wp::int32* var_170;
    const wp::int32 var_171 = 2;
    wp::int32 var_172;
    wp::int32 var_173;
    wp::int32 var_174;
    wp::float32 var_175;
    wp::int32 var_176;
    const wp::int32 var_177 = 0;
    wp::int32 var_178;
    wp::float32 var_179;
    wp::int32 var_180;
    wp::int32* var_181;
    const wp::int32 var_182 = 2;
    wp::int32 var_183;
    wp::int32 var_184;
    wp::int32 var_185;
    wp::int32* var_186;
    const wp::int32 var_187 = 1;
    wp::int32 var_188;
    wp::int32 var_189;
    wp::int32* var_190;
    wp::int32 var_191;
    wp::int32 var_192;
    wp::float32* var_193;
    wp::float32* var_194;
    wp::float32 var_195;
    wp::float32 var_196;
    wp::float32 var_197;
    const wp::int32 var_198 = 1;
    wp::int32 var_199;
    wp::range_t var_200;
    wp::int32 var_201;
    const wp::int32 var_202 = 1;
    wp::int32 var_203;
    wp::range_t var_204;
    wp::int32 var_205;
    wp::int32* var_206;
    wp::int32 var_207;
    wp::int32 var_208;
    wp::int32 var_209;
    wp::int32 var_210;
    wp::float32* var_211;
    wp::float32 var_212;
    wp::float32 var_213;
    wp::int32* var_214;
    wp::int32 var_215;
    wp::int32 var_216;
    wp::int32 var_217;
    wp::int32 var_218;
    const wp::int32 var_219 = 1;
    wp::int32 var_220;
    wp::float32* var_221;
    wp::float32 var_222;
    wp::float32 var_223;
    const wp::int32 var_224 = 1;
    wp::int32 var_225;
    wp::int32* var_226;
    wp::int32 var_227;
    wp::int32 var_228;
    wp::int32 var_229;
    wp::int32 var_230;
    wp::float32* var_231;
    wp::float32 var_232;
    wp::float32 var_233;
    const wp::int32 var_234 = 1;
    wp::int32 var_235;
    wp::int32* var_236;
    wp::int32 var_237;
    wp::int32 var_238;
    wp::int32 var_239;
    wp::int32 var_240;
    const wp::int32 var_241 = 1;
    wp::int32 var_242;
    wp::float32* var_243;
    wp::float32 var_244;
    wp::float32 var_245;
    wp::float32 var_246;
    wp::float32 var_247;
    wp::float32 var_248;
    const wp::float32 var_249 = 0.0;
    bool var_250;
    wp::float32 var_251;
    wp::float32 var_252;
    wp::float32 var_253;
    wp::float32 var_254;
    wp::float32* var_255;
    wp::float32 var_256;
    wp::float32 var_257;
    wp::float32 var_258;
    const wp::int32 var_259 = 2;
    wp::float32 var_260;
    bool var_261;
    const wp::int32 var_262 = 0;
    wp::int32 var_263;
    wp::int32 var_264;
    const wp::int32 var_265 = 2;
    wp::int32 var_266;
    wp::int32 var_267;
    const wp::int32 var_268 = 0;
    const wp::int32 var_269 = 1;
    wp::int32 var_270;
    wp::shape_t* var_271;
    const wp::int32 var_272 = 0;
    wp::int32 var_273;
    wp::shape_t var_274;
    bool var_275;
    wp::vec_t<3, wp::int32> var_276;
    const wp::int32 var_277 = 1;
    wp::int32 var_278;
    wp::int32 var_279;
    const wp::int32 var_280 = 2;
    wp::int32 var_281;
    wp::int32 var_282;
    const wp::int32 var_283 = 0;
    const wp::int32 var_284 = 1;
    wp::int32 var_285;
    wp::shape_t* var_286;
    const wp::int32 var_287 = 0;
    wp::int32 var_288;
    wp::shape_t var_289;
    bool var_290;
    wp::vec_t<3, wp::int32> var_291;
    //---------
    // forward
    // def heightfield_vs_convex_midphase(                                                    <L 367>
    // X_hfield_ws = shape_transform[hfield_shape]                                            <L 409>
    var_0 = wp::address(var_shape_transform, var_hfield_shape);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // X_other_ws = shape_transform[other_shape]                                              <L 410>
    var_3 = wp::address(var_shape_transform, var_other_shape);
    var_5 = wp::load(var_3);
    var_4 = wp::copy(var_5);
    // X_other_in_hfield = wp.transform_multiply(wp.transform_inverse(X_hfield_ws), X_other_ws)       <L 411>
    var_6 = wp::transform_inverse(var_1);
    var_7 = wp::transform_multiply(var_6, var_4);
    // other_pos = wp.transform_get_translation(X_other_in_hfield)                            <L 413>
    var_8 = wp::transform_get_translation(var_7);
    // other_rot = wp.transform_get_rotation(X_other_in_hfield)                               <L 414>
    var_9 = wp::transform_get_rotation(var_7);
    // local_lo = shape_collision_aabb_lower[other_shape]                                     <L 416>
    var_10 = wp::address(var_shape_collision_aabb_lower, var_other_shape);
    var_12 = wp::load(var_10);
    var_11 = wp::copy(var_12);
    // local_hi = shape_collision_aabb_upper[other_shape]                                     <L 417>
    var_13 = wp::address(var_shape_collision_aabb_upper, var_other_shape);
    var_15 = wp::load(var_13);
    var_14 = wp::copy(var_15);
    // local_center = 0.5 * (local_lo + local_hi)                                             <L 418>
    var_17 = wp::add(var_11, var_14);
    var_18 = wp::mul(var_16, var_17);
    // local_half = 0.5 * (local_hi - local_lo)                                               <L 419>
    var_20 = wp::sub(var_14, var_11);
    var_21 = wp::mul(var_19, var_20);
    // center_in_hfield = wp.quat_rotate(other_rot, local_center) + other_pos                 <L 421>
    var_22 = wp::quat_rotate(var_9, var_18);
    var_23 = wp::add(var_22, var_8);
    // r0 = wp.quat_rotate(other_rot, wp.vec3(1.0, 0.0, 0.0))                                 <L 426>
    var_27 = wp::vec_t<3, wp::float32>(var_24, var_25, var_26);
    var_28 = wp::quat_rotate(var_9, var_27);
    // r1 = wp.quat_rotate(other_rot, wp.vec3(0.0, 1.0, 0.0))                                 <L 427>
    var_32 = wp::vec_t<3, wp::float32>(var_29, var_30, var_31);
    var_33 = wp::quat_rotate(var_9, var_32);
    // r2 = wp.quat_rotate(other_rot, wp.vec3(0.0, 0.0, 1.0))                                 <L 428>
    var_37 = wp::vec_t<3, wp::float32>(var_34, var_35, var_36);
    var_38 = wp::quat_rotate(var_9, var_37);
    // half_in_hfield = wp.vec3(                                                              <L 429>
    // wp.abs(r0[0]) * local_half[0] + wp.abs(r1[0]) * local_half[1] + wp.abs(r2[0]) * local_half[2],       <L 430>
    var_40 = wp::extract(var_28, var_39);
    var_41 = wp::abs(var_40);
    var_43 = wp::extract(var_21, var_42);
    var_44 = wp::mul(var_41, var_43);
    var_46 = wp::extract(var_33, var_45);
    var_47 = wp::abs(var_46);
    var_49 = wp::extract(var_21, var_48);
    var_50 = wp::mul(var_47, var_49);
    var_51 = wp::add(var_44, var_50);
    var_53 = wp::extract(var_38, var_52);
    var_54 = wp::abs(var_53);
    var_56 = wp::extract(var_21, var_55);
    var_57 = wp::mul(var_54, var_56);
    var_58 = wp::add(var_51, var_57);
    // wp.abs(r0[1]) * local_half[0] + wp.abs(r1[1]) * local_half[1] + wp.abs(r2[1]) * local_half[2],       <L 431>
    var_60 = wp::extract(var_28, var_59);
    var_61 = wp::abs(var_60);
    var_63 = wp::extract(var_21, var_62);
    var_64 = wp::mul(var_61, var_63);
    var_66 = wp::extract(var_33, var_65);
    var_67 = wp::abs(var_66);
    var_69 = wp::extract(var_21, var_68);
    var_70 = wp::mul(var_67, var_69);
    var_71 = wp::add(var_64, var_70);
    var_73 = wp::extract(var_38, var_72);
    var_74 = wp::abs(var_73);
    var_76 = wp::extract(var_21, var_75);
    var_77 = wp::mul(var_74, var_76);
    var_78 = wp::add(var_71, var_77);
    // wp.abs(r0[2]) * local_half[0] + wp.abs(r1[2]) * local_half[1] + wp.abs(r2[2]) * local_half[2],       <L 432>
    var_80 = wp::extract(var_28, var_79);
    var_81 = wp::abs(var_80);
    var_83 = wp::extract(var_21, var_82);
    var_84 = wp::mul(var_81, var_83);
    var_86 = wp::extract(var_33, var_85);
    var_87 = wp::abs(var_86);
    var_89 = wp::extract(var_21, var_88);
    var_90 = wp::mul(var_87, var_89);
    var_91 = wp::add(var_84, var_90);
    var_93 = wp::extract(var_38, var_92);
    var_94 = wp::abs(var_93);
    var_96 = wp::extract(var_21, var_95);
    var_97 = wp::mul(var_94, var_96);
    var_98 = wp::add(var_91, var_97);
    var_99 = wp::vec_t<3, wp::float32>(var_58, var_78, var_98);
    // gap_sum = shape_gap[hfield_shape] + shape_gap[other_shape]                             <L 435>
    var_100 = wp::address(var_shape_gap, var_hfield_shape);
    var_101 = wp::address(var_shape_gap, var_other_shape);
    var_103 = wp::load(var_100);
    var_104 = wp::load(var_101);
    var_102 = wp::add(var_103, var_104);
    // margin_sum = shape_data[hfield_shape][3] + shape_data[other_shape][3]                  <L 436>
    var_105 = wp::address(var_shape_data, var_hfield_shape);
    var_108 = wp::load(var_105);
    var_107 = wp::extract(var_108, var_106);
    var_109 = wp::address(var_shape_data, var_other_shape);
    var_112 = wp::load(var_109);
    var_111 = wp::extract(var_112, var_110);
    var_113 = wp::add(var_107, var_111);
    // contact_threshold = gap_sum + margin_sum                                               <L 437>
    var_114 = wp::add(var_102, var_113);
    // threshold_vec = wp.vec3(contact_threshold, contact_threshold, contact_threshold)       <L 438>
    var_115 = wp::vec_t<3, wp::float32>(var_114, var_114, var_114);
    // aabb_lower = center_in_hfield - half_in_hfield - threshold_vec                         <L 440>
    var_116 = wp::sub(var_23, var_99);
    var_117 = wp::sub(var_116, var_115);
    // aabb_upper = center_in_hfield + half_in_hfield + threshold_vec                         <L 441>
    var_118 = wp::add(var_23, var_99);
    var_119 = wp::add(var_118, var_115);
    // dx = 2.0 * hfd.hx / wp.float32(hfd.ncol - 1)                                           <L 444>
    var_121 = &((var_hfd).hx);
    var_123 = wp::load(var_121);
    var_122 = wp::mul(var_120, var_123);
    var_124 = &((var_hfd).ncol);
    var_127 = wp::load(var_124);
    var_126 = wp::sub(var_127, var_125);
    var_128 = wp::float32(var_126);
    var_129 = wp::div(var_122, var_128);
    // dy = 2.0 * hfd.hy / wp.float32(hfd.nrow - 1)                                           <L 445>
    var_131 = &((var_hfd).hy);
    var_133 = wp::load(var_131);
    var_132 = wp::mul(var_130, var_133);
    var_134 = &((var_hfd).nrow);
    var_137 = wp::load(var_134);
    var_136 = wp::sub(var_137, var_135);
    var_138 = wp::float32(var_136);
    var_139 = wp::div(var_132, var_138);
    // col_min_f = (aabb_lower[0] + hfd.hx) / dx                                              <L 447>
    var_141 = wp::extract(var_117, var_140);
    var_142 = &((var_hfd).hx);
    var_144 = wp::load(var_142);
    var_143 = wp::add(var_141, var_144);
    var_145 = wp::div(var_143, var_129);
    // col_max_f = (aabb_upper[0] + hfd.hx) / dx                                              <L 448>
    var_147 = wp::extract(var_119, var_146);
    var_148 = &((var_hfd).hx);
    var_150 = wp::load(var_148);
    var_149 = wp::add(var_147, var_150);
    var_151 = wp::div(var_149, var_129);
    // row_min_f = (aabb_lower[1] + hfd.hy) / dy                                              <L 449>
    var_153 = wp::extract(var_117, var_152);
    var_154 = &((var_hfd).hy);
    var_156 = wp::load(var_154);
    var_155 = wp::add(var_153, var_156);
    var_157 = wp::div(var_155, var_139);
    // row_max_f = (aabb_upper[1] + hfd.hy) / dy                                              <L 450>
    var_159 = wp::extract(var_119, var_158);
    var_160 = &((var_hfd).hy);
    var_162 = wp::load(var_160);
    var_161 = wp::add(var_159, var_162);
    var_163 = wp::div(var_161, var_139);
    // col_min = wp.max(wp.int32(wp.floor(col_min_f)), 0)                                     <L 452>
    var_164 = wp::floor(var_145);
    var_165 = wp::int32(var_164);
    var_167 = wp::max(var_165, var_166);
    // col_max = wp.min(wp.int32(wp.floor(col_max_f)), hfd.ncol - 2)                          <L 453>
    var_168 = wp::floor(var_151);
    var_169 = wp::int32(var_168);
    var_170 = &((var_hfd).ncol);
    var_173 = wp::load(var_170);
    var_172 = wp::sub(var_173, var_171);
    var_174 = wp::min(var_169, var_172);
    // row_min = wp.max(wp.int32(wp.floor(row_min_f)), 0)                                     <L 454>
    var_175 = wp::floor(var_157);
    var_176 = wp::int32(var_175);
    var_178 = wp::max(var_176, var_177);
    // row_max = wp.min(wp.int32(wp.floor(row_max_f)), hfd.nrow - 2)                          <L 455>
    var_179 = wp::floor(var_163);
    var_180 = wp::int32(var_179);
    var_181 = &((var_hfd).nrow);
    var_184 = wp::load(var_181);
    var_183 = wp::sub(var_184, var_182);
    var_185 = wp::min(var_180, var_183);
    // cols = hfd.ncol - 1                                                                    <L 457>
    var_186 = &((var_hfd).ncol);
    var_189 = wp::load(var_186);
    var_188 = wp::sub(var_189, var_187);
    // base = hfd.data_offset                                                                 <L 458>
    var_190 = &((var_hfd).data_offset);
    var_192 = wp::load(var_190);
    var_191 = wp::copy(var_192);
    // z_range = hfd.max_z - hfd.min_z                                                        <L 459>
    var_193 = &((var_hfd).max_z);
    var_194 = &((var_hfd).min_z);
    var_196 = wp::load(var_193);
    var_197 = wp::load(var_194);
    var_195 = wp::sub(var_196, var_197);
    // for r in range(row_min, row_max + 1):                                                  <L 460>
    var_199 = wp::add(var_185, var_198);
    var_200 = wp::range(var_178, var_199);
    start_for_0:;
        if (iter_cmp(var_200) == 0) goto end_for_0;
        var_201 = wp::iter_next(var_200);
        // for c in range(col_min, col_max + 1):                                              <L 461>
        var_203 = wp::add(var_174, var_202);
        var_204 = wp::range(var_167, var_203);
        start_for_2:;
            if (iter_cmp(var_204) == 0) goto end_for_2;
            var_205 = wp::iter_next(var_204);
            // h00 = elevation_data[base + r * hfd.ncol + c]                                  <L 464>
            var_206 = &((var_hfd).ncol);
            var_208 = wp::load(var_206);
            var_207 = wp::mul(var_201, var_208);
            var_209 = wp::add(var_191, var_207);
            var_210 = wp::add(var_209, var_205);
            var_211 = wp::address(var_elevation_data, var_210);
            var_213 = wp::load(var_211);
            var_212 = wp::copy(var_213);
            // h10 = elevation_data[base + r * hfd.ncol + c + 1]                              <L 465>
            var_214 = &((var_hfd).ncol);
            var_216 = wp::load(var_214);
            var_215 = wp::mul(var_201, var_216);
            var_217 = wp::add(var_191, var_215);
            var_218 = wp::add(var_217, var_205);
            var_220 = wp::add(var_218, var_219);
            var_221 = wp::address(var_elevation_data, var_220);
            var_223 = wp::load(var_221);
            var_222 = wp::copy(var_223);
            // h01 = elevation_data[base + (r + 1) * hfd.ncol + c]                            <L 466>
            var_225 = wp::add(var_201, var_224);
            var_226 = &((var_hfd).ncol);
            var_228 = wp::load(var_226);
            var_227 = wp::mul(var_225, var_228);
            var_229 = wp::add(var_191, var_227);
            var_230 = wp::add(var_229, var_205);
            var_231 = wp::address(var_elevation_data, var_230);
            var_233 = wp::load(var_231);
            var_232 = wp::copy(var_233);
            // h11 = elevation_data[base + (r + 1) * hfd.ncol + c + 1]                        <L 467>
            var_235 = wp::add(var_201, var_234);
            var_236 = &((var_hfd).ncol);
            var_238 = wp::load(var_236);
            var_237 = wp::mul(var_235, var_238);
            var_239 = wp::add(var_191, var_237);
            var_240 = wp::add(var_239, var_205);
            var_242 = wp::add(var_240, var_241);
            var_243 = wp::address(var_elevation_data, var_242);
            var_245 = wp::load(var_243);
            var_244 = wp::copy(var_245);
            // corner_sample = wp.max(wp.max(h00, h10), wp.max(h01, h11))                     <L 468>
            var_246 = wp::max(var_212, var_222);
            var_247 = wp::max(var_232, var_244);
            var_248 = wp::max(var_246, var_247);
            // if z_range < 0.0:                                                              <L 469>
            var_250 = (var_195 < var_249);
            if (var_250) {
                // corner_sample = wp.min(wp.min(h00, h10), wp.min(h01, h11))                 <L 470>
                var_251 = wp::min(var_212, var_222);
                var_252 = wp::min(var_232, var_244);
                var_253 = wp::min(var_251, var_252);
            }
            var_254 = wp::where(var_250, var_253, var_248);
            // cell_max_z = hfd.min_z + corner_sample * z_range                               <L 471>
            var_255 = &((var_hfd).min_z);
            var_256 = wp::mul(var_254, var_195);
            var_258 = wp::load(var_255);
            var_257 = wp::add(var_258, var_256);
            // if aabb_lower[2] <= cell_max_z:                                                <L 472>
            var_260 = wp::extract(var_117, var_259);
            var_261 = (var_260 <= var_257);
            if (var_261) {
                // for tri_sub in range(2):                                                   <L 473>
                // tri_idx = (r * cols + c) * 2 + tri_sub                                     <L 474>
                var_263 = wp::mul(var_201, var_188);
                var_264 = wp::add(var_263, var_205);
                var_266 = wp::mul(var_264, var_265);
                var_267 = wp::add(var_266, var_262);
                // out_idx = wp.atomic_add(triangle_pairs_count, 0, 1)                        <L 475>
                var_270 = wp::atomic_add(var_triangle_pairs_count, var_268, var_269);
                // if out_idx < triangle_pairs.shape[0]:                                      <L 476>
                var_271 = &(var_triangle_pairs.shape);
                var_274 = wp::load(var_271);
                var_273 = wp::extract(var_274, var_272);
                var_275 = (var_270 < var_273);
                if (var_275) {
                    // triangle_pairs[out_idx] = wp.vec3i(hfield_shape, other_shape, tri_idx)       <L 477>
                    var_276 = wp::vec_t<3, wp::int32>(var_hfield_shape, var_other_shape, var_267);
                    wp::array_store(var_triangle_pairs, var_270, var_276);
                }
                // tri_idx = (r * cols + c) * 2 + tri_sub                                     <L 474>
                var_278 = wp::mul(var_201, var_188);
                var_279 = wp::add(var_278, var_205);
                var_281 = wp::mul(var_279, var_280);
                var_282 = wp::add(var_281, var_277);
                // out_idx = wp.atomic_add(triangle_pairs_count, 0, 1)                        <L 475>
                var_285 = wp::atomic_add(var_triangle_pairs_count, var_283, var_284);
                // if out_idx < triangle_pairs.shape[0]:                                      <L 476>
                var_286 = &(var_triangle_pairs.shape);
                var_289 = wp::load(var_286);
                var_288 = wp::extract(var_289, var_287);
                var_290 = (var_285 < var_288);
                if (var_290) {
                    // triangle_pairs[out_idx] = wp.vec3i(hfield_shape, other_shape, tri_idx)       <L 477>
                    var_291 = wp::vec_t<3, wp::int32>(var_hfield_shape, var_other_shape, var_282);
                    wp::array_store(var_triangle_pairs, var_285, var_291);
                }
            }
            goto start_for_2;
        end_for_2:;
        goto start_for_0;
    end_for_0:;
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


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_core.py:893
static CUDA_CALLABLE void aabb_to_unscaled_0(
    wp::vec_t<3, wp::float32> var_aabb_lower,
    wp::vec_t<3, wp::float32> var_aabb_upper,
    wp::vec_t<3, wp::float32> var_scale,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 1e-12;
    wp::float32 var_1;
    const wp::float32 var_2 = 1.0;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    wp::float32 var_5;
    bool var_6;
    const wp::int32 var_7 = 0;
    wp::float32 var_8;
    const wp::int32 var_9 = 0;
    wp::float32 var_10;
    const wp::float32 var_11 = 0.0;
    bool var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::float32 var_17 = 1.0;
    const wp::int32 var_18 = 1;
    wp::float32 var_19;
    wp::float32 var_20;
    bool var_21;
    const wp::int32 var_22 = 1;
    wp::float32 var_23;
    const wp::int32 var_24 = 1;
    wp::float32 var_25;
    const wp::float32 var_26 = 0.0;
    bool var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    const wp::float32 var_32 = 1.0;
    const wp::int32 var_33 = 2;
    wp::float32 var_34;
    wp::float32 var_35;
    bool var_36;
    const wp::int32 var_37 = 2;
    wp::float32 var_38;
    const wp::int32 var_39 = 2;
    wp::float32 var_40;
    const wp::float32 var_41 = 0.0;
    bool var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    const wp::int32 var_47 = 0;
    wp::float32 var_48;
    wp::float32 var_49;
    const wp::int32 var_50 = 0;
    wp::float32 var_51;
    wp::float32 var_52;
    const wp::int32 var_53 = 1;
    wp::float32 var_54;
    wp::float32 var_55;
    const wp::int32 var_56 = 1;
    wp::float32 var_57;
    wp::float32 var_58;
    const wp::int32 var_59 = 2;
    wp::float32 var_60;
    wp::float32 var_61;
    const wp::int32 var_62 = 2;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::vec_t<3, wp::float32> var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::vec_t<3, wp::float32> var_72;
    wp::vec_t<3, wp::float32> var_73;
    //---------
    // forward
    // def aabb_to_unscaled(                                                                  <L 894>
    // eps = float(1.0e-12)                                                                   <L 912>
    var_1 = wp::float(var_0);
    // inv_x = 1.0 / wp.where(wp.abs(scale[0]) > eps, scale[0], wp.where(scale[0] >= 0.0, eps, -eps))       <L 913>
    var_4 = wp::extract(var_scale, var_3);
    var_5 = wp::abs(var_4);
    var_6 = (var_5 > var_1);
    var_8 = wp::extract(var_scale, var_7);
    var_10 = wp::extract(var_scale, var_9);
    var_12 = (var_10 >= var_11);
    var_13 = wp::neg(var_1);
    var_14 = wp::where(var_12, var_1, var_13);
    var_15 = wp::where(var_6, var_8, var_14);
    var_16 = wp::div(var_2, var_15);
    // inv_y = 1.0 / wp.where(wp.abs(scale[1]) > eps, scale[1], wp.where(scale[1] >= 0.0, eps, -eps))       <L 914>
    var_19 = wp::extract(var_scale, var_18);
    var_20 = wp::abs(var_19);
    var_21 = (var_20 > var_1);
    var_23 = wp::extract(var_scale, var_22);
    var_25 = wp::extract(var_scale, var_24);
    var_27 = (var_25 >= var_26);
    var_28 = wp::neg(var_1);
    var_29 = wp::where(var_27, var_1, var_28);
    var_30 = wp::where(var_21, var_23, var_29);
    var_31 = wp::div(var_17, var_30);
    // inv_z = 1.0 / wp.where(wp.abs(scale[2]) > eps, scale[2], wp.where(scale[2] >= 0.0, eps, -eps))       <L 915>
    var_34 = wp::extract(var_scale, var_33);
    var_35 = wp::abs(var_34);
    var_36 = (var_35 > var_1);
    var_38 = wp::extract(var_scale, var_37);
    var_40 = wp::extract(var_scale, var_39);
    var_42 = (var_40 >= var_41);
    var_43 = wp::neg(var_1);
    var_44 = wp::where(var_42, var_1, var_43);
    var_45 = wp::where(var_36, var_38, var_44);
    var_46 = wp::div(var_32, var_45);
    // lx0 = aabb_lower[0] * inv_x                                                            <L 917>
    var_48 = wp::extract(var_aabb_lower, var_47);
    var_49 = wp::mul(var_48, var_16);
    // lx1 = aabb_upper[0] * inv_x                                                            <L 918>
    var_51 = wp::extract(var_aabb_upper, var_50);
    var_52 = wp::mul(var_51, var_16);
    // ly0 = aabb_lower[1] * inv_y                                                            <L 919>
    var_54 = wp::extract(var_aabb_lower, var_53);
    var_55 = wp::mul(var_54, var_31);
    // ly1 = aabb_upper[1] * inv_y                                                            <L 920>
    var_57 = wp::extract(var_aabb_upper, var_56);
    var_58 = wp::mul(var_57, var_31);
    // lz0 = aabb_lower[2] * inv_z                                                            <L 921>
    var_60 = wp::extract(var_aabb_lower, var_59);
    var_61 = wp::mul(var_60, var_46);
    // lz1 = aabb_upper[2] * inv_z                                                            <L 922>
    var_63 = wp::extract(var_aabb_upper, var_62);
    var_64 = wp::mul(var_63, var_46);
    // out_lower = wp.vec3(wp.min(lx0, lx1), wp.min(ly0, ly1), wp.min(lz0, lz1))              <L 924>
    var_65 = wp::min(var_49, var_52);
    var_66 = wp::min(var_55, var_58);
    var_67 = wp::min(var_61, var_64);
    var_68 = wp::vec_t<3, wp::float32>(var_65, var_66, var_67);
    // out_upper = wp.vec3(wp.max(lx0, lx1), wp.max(ly0, ly1), wp.max(lz0, lz1))              <L 925>
    var_69 = wp::max(var_49, var_52);
    var_70 = wp::max(var_55, var_58);
    var_71 = wp::max(var_61, var_64);
    var_72 = wp::vec_t<3, wp::float32>(var_69, var_70, var_71);
    // return out_lower, out_upper, wp.vec3(inv_x, inv_y, inv_z)                              <L 926>
    var_73 = wp::vec_t<3, wp::float32>(var_16, var_31, var_46);
    ret_0 = var_68;
    ret_1 = var_72;
    ret_2 = var_73;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_core.py:965
static CUDA_CALLABLE void _compute_mesh_vs_convex_query_aabb_0(
    wp::int32 var_mesh_shape,
    wp::int32 var_non_mesh_shape,
    wp::transform_t<wp::float32> var_X_mesh_ws,
    wp::transform_t<wp::float32> var_X_ws,
    wp::array_t<wp::int32> var_shape_type,
    wp::array_t<wp::vec_t<4, wp::float32>> var_shape_data,
    wp::array_t<wp::uint64> var_shape_source_ptr,
    wp::float32 var_contact_threshold,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::transform_t<wp::float32> var_0;
    wp::transform_t<wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::quat_t<wp::float32> var_3;
    wp::int32* var_4;
    wp::int32 var_5;
    wp::int32 var_6;
    wp::vec_t<4, wp::float32>* var_7;
    wp::vec_t<4, wp::float32> var_8;
    wp::vec_t<4, wp::float32> var_9;
    const wp::int32 var_10 = 0;
    wp::float32 var_11;
    const wp::int32 var_12 = 1;
    wp::float32 var_13;
    const wp::int32 var_14 = 2;
    wp::float32 var_15;
    wp::vec_t<3, wp::float32> var_16;
    GenericShapeData_ba895271 var_17;
    const wp::float32 var_18 = 0.0;
    const wp::float32 var_19 = 0.0;
    const wp::float32 var_20 = 0.0;
    wp::vec_t<3, wp::float32> var_21;
    const wp::float32 var_22 = 0.0;
    const wp::float32 var_23 = 0.0;
    const wp::float32 var_24 = 0.0;
    wp::vec_t<3, wp::float32> var_25;
    const wp::int32 var_26 = 10;
    bool var_27;
    wp::uint64* var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::uint64 var_30;
    SupportMapDataProvider_e77f8b9f var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<4, wp::float32>* var_34;
    wp::vec_t<4, wp::float32> var_35;
    wp::vec_t<4, wp::float32> var_36;
    const wp::int32 var_37 = 0;
    wp::float32 var_38;
    const wp::int32 var_39 = 1;
    wp::float32 var_40;
    const wp::int32 var_41 = 2;
    wp::float32 var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::vec_t<3, wp::float32> var_45;
    wp::vec_t<3, wp::float32> var_46;
    const wp::int32 var_47 = 0;
    wp::float32 var_48;
    wp::float32 var_49;
    const wp::float32 var_50 = 1e-12;
    wp::float32 var_51;
    wp::float32 var_52;
    const wp::int32 var_53 = 1;
    wp::float32 var_54;
    wp::float32 var_55;
    const wp::float32 var_56 = 1e-12;
    wp::float32 var_57;
    wp::float32 var_58;
    const wp::int32 var_59 = 2;
    wp::float32 var_60;
    wp::float32 var_61;
    const wp::float32 var_62 = 1e-12;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::vec_t<3, wp::float32> var_68;
    //---------
    // forward
    // def _compute_mesh_vs_convex_query_aabb(                                                <L 966>
    // X_mesh_sw = wp.transform_inverse(X_mesh_ws)                                            <L 977>
    var_0 = wp::transform_inverse(var_X_mesh_ws);
    // X_mesh_shape = wp.transform_multiply(X_mesh_sw, X_ws)                                  <L 978>
    var_1 = wp::transform_multiply(var_0, var_X_ws);
    // pos_in_mesh = wp.transform_get_translation(X_mesh_shape)                               <L 979>
    var_2 = wp::transform_get_translation(var_1);
    // orientation_in_mesh = wp.transform_get_rotation(X_mesh_shape)                          <L 980>
    var_3 = wp::transform_get_rotation(var_1);
    // geo_type = shape_type[non_mesh_shape]                                                  <L 982>
    var_4 = wp::address(var_shape_type, var_non_mesh_shape);
    var_6 = wp::load(var_4);
    var_5 = wp::copy(var_6);
    // data_vec4 = shape_data[non_mesh_shape]                                                 <L 983>
    var_7 = wp::address(var_shape_data, var_non_mesh_shape);
    var_9 = wp::load(var_7);
    var_8 = wp::copy(var_9);
    // scale = wp.vec3(data_vec4[0], data_vec4[1], data_vec4[2])                              <L 984>
    var_11 = wp::extract(var_8, var_10);
    var_13 = wp::extract(var_8, var_12);
    var_15 = wp::extract(var_8, var_14);
    var_16 = wp::vec_t<3, wp::float32>(var_11, var_13, var_15);
    // generic_shape_data = GenericShapeData()                                                <L 986>
    var_17 = GenericShapeData_ba895271();
    // generic_shape_data.shape_type = geo_type                                               <L 987>
    var_17.shape_type = var_5;
    // generic_shape_data.scale = scale                                                       <L 988>
    var_17.scale = var_16;
    // generic_shape_data.auxiliary = wp.vec3(0.0, 0.0, 0.0)                                  <L 989>
    var_21 = wp::vec_t<3, wp::float32>(var_18, var_19, var_20);
    var_17.auxiliary = var_21;
    // generic_shape_data.center = wp.vec3(0.0, 0.0, 0.0)                                     <L 990>
    var_25 = wp::vec_t<3, wp::float32>(var_22, var_23, var_24);
    var_17.center = var_25;
    // if geo_type == GeoType.CONVEX_MESH:                                                    <L 991>
    var_27 = (var_5 == var_26);
    if (var_27) {
        // generic_shape_data.auxiliary = pack_mesh_ptr(shape_source_ptr[non_mesh_shape])       <L 992>
        var_28 = wp::address(var_shape_source_ptr, var_non_mesh_shape);
        var_30 = wp::load(var_28);
        var_29 = pack_mesh_ptr_0(var_30);
        var_17.auxiliary = var_29;
    }
    // data_provider = SupportMapDataProvider()                                               <L 994>
    var_31 = SupportMapDataProvider_e77f8b9f();
    // aabb_lower, aabb_upper = compute_tight_aabb_from_support(                              <L 996>
    // generic_shape_data, orientation_in_mesh, pos_in_mesh, data_provider                    <L 997>
    compute_tight_aabb_from_support_0(var_17, var_3, var_2, var_31, var_32, var_33);
    // mesh_scale_vec4 = shape_data[mesh_shape]                                               <L 1001>
    var_34 = wp::address(var_shape_data, var_mesh_shape);
    var_36 = wp::load(var_34);
    var_35 = wp::copy(var_36);
    // mesh_scale = wp.vec3(mesh_scale_vec4[0], mesh_scale_vec4[1], mesh_scale_vec4[2])       <L 1002>
    var_38 = wp::extract(var_35, var_37);
    var_40 = wp::extract(var_35, var_39);
    var_42 = wp::extract(var_35, var_41);
    var_43 = wp::vec_t<3, wp::float32>(var_38, var_40, var_42);
    // aabb_lower_bvh, aabb_upper_bvh, inv_scale = aabb_to_unscaled(aabb_lower, aabb_upper, mesh_scale)       <L 1003>
    aabb_to_unscaled_0(var_32, var_33, var_43, var_44, var_45, var_46);
    // margin_vec = wp.vec3(                                                                  <L 1004>
    // contact_threshold / wp.max(wp.abs(mesh_scale[0]), 1.0e-12),                            <L 1005>
    var_48 = wp::extract(var_43, var_47);
    var_49 = wp::abs(var_48);
    var_51 = wp::max(var_49, var_50);
    var_52 = wp::div(var_contact_threshold, var_51);
    // contact_threshold / wp.max(wp.abs(mesh_scale[1]), 1.0e-12),                            <L 1006>
    var_54 = wp::extract(var_43, var_53);
    var_55 = wp::abs(var_54);
    var_57 = wp::max(var_55, var_56);
    var_58 = wp::div(var_contact_threshold, var_57);
    // contact_threshold / wp.max(wp.abs(mesh_scale[2]), 1.0e-12),                            <L 1007>
    var_60 = wp::extract(var_43, var_59);
    var_61 = wp::abs(var_60);
    var_63 = wp::max(var_61, var_62);
    var_64 = wp::div(var_contact_threshold, var_63);
    var_65 = wp::vec_t<3, wp::float32>(var_52, var_58, var_64);
    // center_in_bvh = wp.cw_mul(pos_in_mesh, inv_scale)                                      <L 1009>
    var_66 = wp::cw_mul(var_2, var_46);
    // return aabb_lower_bvh - margin_vec, aabb_upper_bvh + margin_vec, center_in_bvh         <L 1010>
    var_67 = wp::sub(var_44, var_65);
    var_68 = wp::add(var_45, var_65);
    ret_0 = var_67;
    ret_1 = var_68;
    ret_2 = var_66;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_core.py:1013
static CUDA_CALLABLE bool _mesh_triangle_is_front_facing_local_0(
    wp::uint64 var_mesh_id,
    wp::vec_t<3, wp::float32> var_center_in_bvh,
    wp::int32 var_tri_idx)
{
    //---------
    // primal vars
    wp::Mesh var_0;
    wp::array_t<wp::int32>* var_1;
    const wp::int32 var_2 = 3;
    wp::int32 var_3;
    const wp::int32 var_4 = 0;
    wp::int32 var_5;
    wp::int32* var_6;
    wp::array_t<wp::int32> var_7;
    wp::int32 var_8;
    wp::int32 var_9;
    wp::array_t<wp::int32>* var_10;
    const wp::int32 var_11 = 3;
    wp::int32 var_12;
    const wp::int32 var_13 = 1;
    wp::int32 var_14;
    wp::int32* var_15;
    wp::array_t<wp::int32> var_16;
    wp::int32 var_17;
    wp::int32 var_18;
    wp::array_t<wp::int32>* var_19;
    const wp::int32 var_20 = 3;
    wp::int32 var_21;
    const wp::int32 var_22 = 2;
    wp::int32 var_23;
    wp::int32* var_24;
    wp::array_t<wp::int32> var_25;
    wp::int32 var_26;
    wp::int32 var_27;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_28;
    wp::vec_t<3, wp::float32>* var_29;
    wp::array_t<wp::vec_t<3, wp::float32>> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_33;
    wp::vec_t<3, wp::float32>* var_34;
    wp::array_t<wp::vec_t<3, wp::float32>> var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_38;
    wp::vec_t<3, wp::float32>* var_39;
    wp::array_t<wp::vec_t<3, wp::float32>> var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::vec_t<3, wp::float32> var_45;
    wp::vec_t<3, wp::float32> var_46;
    wp::float32 var_47;
    const wp::float32 var_48 = 0.0;
    bool var_49;
    bool var_50;
    //---------
    // forward
    // def _mesh_triangle_is_front_facing_local(                                              <L 1014>
    // mesh = wp.mesh_get(mesh_id)                                                            <L 1020>
    var_0 = wp::mesh_get(var_mesh_id);
    // idx0 = mesh.indices[tri_idx * 3 + 0]                                                   <L 1021>
    var_1 = &((var_0).indices);
    var_3 = wp::mul(var_tri_idx, var_2);
    var_5 = wp::add(var_3, var_4);
    var_7 = wp::load(var_1);
    var_6 = wp::address(var_7, var_5);
    var_9 = wp::load(var_6);
    var_8 = wp::copy(var_9);
    // idx1 = mesh.indices[tri_idx * 3 + 1]                                                   <L 1022>
    var_10 = &((var_0).indices);
    var_12 = wp::mul(var_tri_idx, var_11);
    var_14 = wp::add(var_12, var_13);
    var_16 = wp::load(var_10);
    var_15 = wp::address(var_16, var_14);
    var_18 = wp::load(var_15);
    var_17 = wp::copy(var_18);
    // idx2 = mesh.indices[tri_idx * 3 + 2]                                                   <L 1023>
    var_19 = &((var_0).indices);
    var_21 = wp::mul(var_tri_idx, var_20);
    var_23 = wp::add(var_21, var_22);
    var_25 = wp::load(var_19);
    var_24 = wp::address(var_25, var_23);
    var_27 = wp::load(var_24);
    var_26 = wp::copy(var_27);
    // v0 = mesh.points[idx0]                                                                 <L 1025>
    var_28 = &((var_0).points);
    var_30 = wp::load(var_28);
    var_29 = wp::address(var_30, var_8);
    var_32 = wp::load(var_29);
    var_31 = wp::copy(var_32);
    // v1 = mesh.points[idx1]                                                                 <L 1026>
    var_33 = &((var_0).points);
    var_35 = wp::load(var_33);
    var_34 = wp::address(var_35, var_17);
    var_37 = wp::load(var_34);
    var_36 = wp::copy(var_37);
    // v2 = mesh.points[idx2]                                                                 <L 1027>
    var_38 = &((var_0).points);
    var_40 = wp::load(var_38);
    var_39 = wp::address(var_40, var_26);
    var_42 = wp::load(var_39);
    var_41 = wp::copy(var_42);
    // face_normal = wp.cross(v1 - v0, v2 - v0)                                               <L 1028>
    var_43 = wp::sub(var_36, var_31);
    var_44 = wp::sub(var_41, var_31);
    var_45 = wp::cross(var_43, var_44);
    // center_dist = wp.dot(face_normal, center_in_bvh - v0)                                  <L 1029>
    var_46 = wp::sub(var_center_in_bvh, var_31);
    var_47 = wp::dot(var_45, var_46);
    // return not (center_dist < 0.0)                                                         <L 1030>
    var_49 = (var_47 < var_48);
    var_50 = wp::unot(var_49);
    return var_50;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_core.py:1033
static CUDA_CALLABLE void mesh_vs_convex_midphase_0(
    wp::int32 var_idx_in_thread_block,
    wp::int32 var_mesh_shape,
    wp::int32 var_non_mesh_shape,
    wp::transform_t<wp::float32> var_X_mesh_ws,
    wp::transform_t<wp::float32> var_X_ws,
    wp::uint64 var_mesh_id,
    wp::array_t<wp::int32> var_shape_type,
    wp::array_t<wp::vec_t<4, wp::float32>> var_shape_data,
    wp::array_t<wp::uint64> var_shape_source_ptr,
    wp::float32 var_contact_threshold,
    wp::array_t<wp::vec_t<3, wp::int32>> var_triangle_pairs,
    wp::array_t<wp::int32> var_triangle_pairs_count)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    wp::vec_t<3, wp::float32> var_1;
    const wp::float32 var_2 = 0.0;
    wp::vec_t<3, wp::float32> var_3;
    const wp::float32 var_4 = 0.0;
    wp::vec_t<3, wp::float32> var_5;
    const bool var_6 = true;
    const wp::int32 var_7 = 0;
    bool var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    const wp::int32 var_15 = 0;
    wp::float32 var_16;
    const wp::int32 var_17 = 1;
    wp::float32 var_18;
    const wp::int32 var_19 = 2;
    wp::float32 var_20;
    const wp::int32 var_21 = 0;
    wp::float32 var_22;
    const wp::int32 var_23 = 1;
    wp::float32 var_24;
    const wp::int32 var_25 = 2;
    wp::float32 var_26;
    const wp::int32 var_27 = 0;
    wp::float32 var_28;
    const wp::int32 var_29 = 1;
    wp::float32 var_30;
    const wp::int32 var_31 = 2;
    wp::float32 var_32;
    wp::mat_t<3, 3, wp::float32> var_33;
    const wp::int32 var_34 = 1;
    wp::tuple_t<wp::int32> var_35;
    const wp::str var_36 = "shared";
    wp::tile_shared_t<wp::mat_t<3, 3, wp::float32>,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_37 = wp::tile_alloc_empty<wp::mat_t<3, 3, wp::float32>,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
    const wp::int32 var_38 = 0;
    const wp::int32 var_39 = 0;
    bool var_40;
    const wp::int32 var_41 = 0;
    wp::mat_t<3, 3, wp::float32> var_42;
    const wp::int32 var_43 = 0;
    const wp::int32 var_44 = 0;
    wp::float32 var_45;
    const wp::int32 var_46 = 0;
    const wp::int32 var_47 = 1;
    wp::float32 var_48;
    const wp::int32 var_49 = 0;
    const wp::int32 var_50 = 2;
    wp::float32 var_51;
    wp::vec_t<3, wp::float32> var_52;
    const wp::int32 var_53 = 1;
    const wp::int32 var_54 = 0;
    wp::float32 var_55;
    const wp::int32 var_56 = 1;
    const wp::int32 var_57 = 1;
    wp::float32 var_58;
    const wp::int32 var_59 = 1;
    const wp::int32 var_60 = 2;
    wp::float32 var_61;
    wp::vec_t<3, wp::float32> var_62;
    const wp::int32 var_63 = 2;
    const wp::int32 var_64 = 0;
    wp::float32 var_65;
    const wp::int32 var_66 = 2;
    const wp::int32 var_67 = 1;
    wp::float32 var_68;
    const wp::int32 var_69 = 2;
    const wp::int32 var_70 = 2;
    wp::float32 var_71;
    wp::vec_t<3, wp::float32> var_72;
    const bool var_73 = true;
    wp::mesh_query_aabb_thread_block_t var_74;
    bool var_75;
    wp::tile_register_t<wp::int32,wp::tile_layout_register_t<wp::tile_shape_t<256>>> var_76 = wp::tile_register_t<wp::int32,wp::tile_layout_register_t<wp::tile_shape_t<256>>>{};
    wp::int32 var_77;
    bool var_78;
    const wp::int32 var_79 = 0;
    bool var_80;
    bool var_81;
    bool var_82;
    const wp::int32 var_83 = -1;
    wp::int32 var_84;
    const wp::int32 var_85 = 0;
    const wp::int32 var_86 = 0;
    bool var_87;
    const wp::int32 var_88 = 1;
    wp::int32 var_89;
    wp::tile_register_t<wp::int32,wp::tile_layout_register_t<wp::tile_shape_t<256>>> var_90 = wp::tile_register_t<wp::int32,wp::tile_layout_register_t<wp::tile_shape_t<256>>>{};
    wp::tile_shared_t<wp::int32,wp::tile_layout_strided_t<wp::tile_shape_t<256>, wp::tile_stride_t<1>>, true> var_91 = wp::tile_alloc_empty<wp::int32,wp::tile_shape_t<256>,wp::tile_stride_t<1>,false>();
    const wp::int32 var_92 = 0;
    wp::int32 var_93;
    const wp::int32 var_94 = 1;
    wp::int32 var_95;
    bool var_96;
    const wp::int32 var_97 = 0;
    wp::int32 var_98;
    const wp::int32 var_99 = 1;
    wp::int32 var_100;
    const wp::int32 var_101 = 0;
    const wp::int32 var_102 = 256;
    bool var_103;
    wp::int32 var_104;
    wp::int32 var_105;
    wp::int32 var_106;
    wp::int32 var_107;
    wp::int32 var_108;
    wp::tile_shared_t<wp::int32,wp::tile_layout_strided_t<wp::tile_shape_t<256>, wp::tile_stride_t<1>>, true> var_109 = wp::tile_alloc_empty<wp::int32,wp::tile_shape_t<256>,wp::tile_stride_t<1>,false>();
    wp::int32 var_110;
    const wp::int32 var_111 = 1;
    wp::int32 var_112;
    const wp::int32 var_113 = 0;
    const wp::int32 var_114 = 256;
    bool var_115;
    wp::int32 var_116;
    wp::int32 var_117;
    wp::int32 var_118;
    const wp::int32 var_119 = 0;
    bool var_120;
    const wp::int32 var_121 = 0;
    const wp::int32 var_122 = 256;
    bool var_123;
    wp::int32 var_124;
    wp::int32 var_125;
    wp::int32 var_126;
    wp::int32 var_127;
    wp::int32 var_128;
    wp::shape_t* var_129;
    const wp::int32 var_130 = 0;
    wp::int32 var_131;
    wp::shape_t var_132;
    bool var_133;
    wp::vec_t<3, wp::int32> var_134;
    //---------
    // forward
    // def mesh_vs_convex_midphase(                                                           <L 1034>
    // aabb_lower = wp.vec3(0.0)                                                              <L 1069>
    var_1 = wp::vec_t<3, wp::float32>(var_0);
    // aabb_upper = wp.vec3(0.0)                                                              <L 1070>
    var_3 = wp::vec_t<3, wp::float32>(var_2);
    // center_in_bvh = wp.vec3(0.0)                                                           <L 1071>
    var_5 = wp::vec_t<3, wp::float32>(var_4);
    // if wp.static(ENABLE_TILE_BVH_QUERY):                                                   <L 1072>
    // if idx_in_thread_block == 0:                                                           <L 1074>
    var_8 = (var_idx_in_thread_block == var_7);
    if (var_8) {
        // aabb_lower, aabb_upper, center_in_bvh = _compute_mesh_vs_convex_query_aabb(        <L 1075>
        // mesh_shape,                                                                        <L 1076>
        // non_mesh_shape,                                                                    <L 1077>
        // X_mesh_ws,                                                                         <L 1078>
        // X_ws,                                                                              <L 1079>
        // shape_type,                                                                        <L 1080>
        // shape_data,                                                                        <L 1081>
        // shape_source_ptr,                                                                  <L 1082>
        // contact_threshold,                                                                 <L 1083>
        _compute_mesh_vs_convex_query_aabb_0(var_mesh_shape, var_non_mesh_shape, var_X_mesh_ws, var_X_ws, var_shape_type, var_shape_data, var_shape_source_ptr, var_contact_threshold, var_9, var_10, var_11);
    }
    var_12 = wp::where(var_8, var_9, var_1);
    var_13 = wp::where(var_8, var_10, var_3);
    var_14 = wp::where(var_8, var_11, var_5);
    // bounds = wp.mat33(                                                                     <L 1085>
    // aabb_lower[0],                                                                         <L 1086>
    var_16 = wp::extract(var_12, var_15);
    // aabb_lower[1],                                                                         <L 1087>
    var_18 = wp::extract(var_12, var_17);
    // aabb_lower[2],                                                                         <L 1088>
    var_20 = wp::extract(var_12, var_19);
    // aabb_upper[0],                                                                         <L 1089>
    var_22 = wp::extract(var_13, var_21);
    // aabb_upper[1],                                                                         <L 1090>
    var_24 = wp::extract(var_13, var_23);
    // aabb_upper[2],                                                                         <L 1091>
    var_26 = wp::extract(var_13, var_25);
    // center_in_bvh[0],                                                                      <L 1092>
    var_28 = wp::extract(var_14, var_27);
    // center_in_bvh[1],                                                                      <L 1093>
    var_30 = wp::extract(var_14, var_29);
    // center_in_bvh[2],                                                                      <L 1094>
    var_32 = wp::extract(var_14, var_31);
    var_33 = wp::mat_t<3, 3, wp::float32>(var_16, var_18, var_20, var_22, var_24, var_26, var_28, var_30, var_32);
    // bounds_tile = wp.tile_zeros(shape=(1,), dtype=wp.mat33, storage="shared")              <L 1096>
    var_35 = wp::tuple(var_34);
    var_37 = wp::tile_zeros<wp::mat_t<3, 3, wp::float32>, 1>();
    // wp.tile_scatter_masked(bounds_tile, 0, bounds, idx_in_thread_block == 0)               <L 1097>
    var_40 = (var_idx_in_thread_block == var_39);
    wp::tile_scatter_masked(var_37, var_38, var_33, var_40);
    // bounds = wp.tile_extract(bounds_tile, 0)                                               <L 1098>
    var_42 = wp::tile_extract(var_37, var_41);
    // aabb_lower = wp.vec3(bounds[0, 0], bounds[0, 1], bounds[0, 2])                         <L 1099>
    var_45 = wp::extract(var_42, var_43, var_44);
    var_48 = wp::extract(var_42, var_46, var_47);
    var_51 = wp::extract(var_42, var_49, var_50);
    var_52 = wp::vec_t<3, wp::float32>(var_45, var_48, var_51);
    // aabb_upper = wp.vec3(bounds[1, 0], bounds[1, 1], bounds[1, 2])                         <L 1100>
    var_55 = wp::extract(var_42, var_53, var_54);
    var_58 = wp::extract(var_42, var_56, var_57);
    var_61 = wp::extract(var_42, var_59, var_60);
    var_62 = wp::vec_t<3, wp::float32>(var_55, var_58, var_61);
    // center_in_bvh = wp.vec3(bounds[2, 0], bounds[2, 1], bounds[2, 2])                      <L 1101>
    var_65 = wp::extract(var_42, var_63, var_64);
    var_68 = wp::extract(var_42, var_66, var_67);
    var_71 = wp::extract(var_42, var_69, var_70);
    var_72 = wp::vec_t<3, wp::float32>(var_65, var_68, var_71);
    // if wp.static(ENABLE_TILE_BVH_QUERY):                                                   <L 1114>
    // query = wp.tile_mesh_query_aabb(mesh_id, aabb_lower, aabb_upper)                       <L 1116>
    var_74 = wp::tile_mesh_query_aabb(var_mesh_id, var_52, var_62);
    // while wp.tile_query_valid(query):                                                      <L 1118>
    start_while_0:;
    var_75 = wp::tile_query_valid(var_74);
    if ((var_75) == false) goto end_while_0;
        // result_tile = wp.tile_mesh_query_aabb_next(query)                                  <L 1119>
        var_76 = wp::tile_mesh_query_aabb_next(var_74);
        // tri_index = wp.untile(result_tile)                                                 <L 1120>
        var_77 = wp::untile(var_76);
        // if tri_index >= 0 and not _mesh_triangle_is_front_facing_local(mesh_id, center_in_bvh, tri_index):       <L 1121>
        var_80 = (var_77 >= var_79);
        var_78 = var_80;
        if (var_78) {
            var_81 = _mesh_triangle_is_front_facing_local_0(var_mesh_id, var_72, var_77);
            var_82 = wp::unot(var_81);
            var_78 = var_78 && var_82;
        }
        if (var_78) {
            // tri_index = -1                                                                 <L 1122>
        }
        var_84 = wp::where(var_78, var_83, var_77);
        // has_tri = 0                                                                        <L 1126>
        // if tri_index >= 0:                                                                 <L 1127>
        var_87 = (var_84 >= var_86);
        if (var_87) {
            // has_tri = 1                                                                    <L 1128>
        }
        var_89 = wp::where(var_87, var_88, var_85);
        // count_tile = wp.tile(has_tri)                                                      <L 1129>
        var_90 = wp::tile<wp::int32>(var_89);
        // inclusive_scan = wp.tile_scan_inclusive(count_tile)                                <L 1130>
        var_91 = wp::tile_scan_inclusive(var_90);
        // offset = 0                                                                         <L 1131>
        // if idx_in_thread_block == wp.block_dim() - 1:                                      <L 1132>
        var_93 = builtin_block_dim();
        var_95 = wp::sub(var_93, var_94);
        var_96 = (var_idx_in_thread_block == var_95);
        if (var_96) {
            // offset = wp.atomic_add(triangle_pairs_count, 0, inclusive_scan[wp.block_dim() - 1])       <L 1133>
            var_98 = builtin_block_dim();
            var_100 = wp::sub(var_98, var_99);
            var_103 = (var_100 < var_101);
            var_104 = wp::add(var_100, var_102);
            var_105 = wp::where(var_103, var_104, var_100);
            var_106 = wp::tile_extract(var_91, var_105);
            var_107 = wp::atomic_add(var_triangle_pairs_count, var_97, var_106);
        }
        var_108 = wp::where(var_96, var_107, var_92);
        // offset_broadcast_tile = wp.tile(offset)                                            <L 1134>
        var_109 = wp::tile<wp::int32>(var_108);
        // offset_broadcast = offset_broadcast_tile[wp.block_dim() - 1]                       <L 1135>
        var_110 = builtin_block_dim();
        var_112 = wp::sub(var_110, var_111);
        var_115 = (var_112 < var_113);
        var_116 = wp::add(var_112, var_114);
        var_117 = wp::where(var_115, var_116, var_112);
        var_118 = wp::tile_extract(var_109, var_117);
        // if tri_index >= 0:                                                                 <L 1137>
        var_120 = (var_84 >= var_119);
        if (var_120) {
            // out_idx = offset_broadcast + inclusive_scan[idx_in_thread_block] - has_tri       <L 1138>
            var_123 = (var_idx_in_thread_block < var_121);
            var_124 = wp::add(var_idx_in_thread_block, var_122);
            var_125 = wp::where(var_123, var_124, var_idx_in_thread_block);
            var_126 = wp::tile_extract(var_91, var_125);
            var_127 = wp::add(var_118, var_126);
            var_128 = wp::sub(var_127, var_89);
            // if out_idx < triangle_pairs.shape[0]:                                          <L 1139>
            var_129 = &(var_triangle_pairs.shape);
            var_132 = wp::load(var_129);
            var_131 = wp::extract(var_132, var_130);
            var_133 = (var_128 < var_131);
            if (var_133) {
                // triangle_pairs[out_idx] = wp.vec3i(mesh_shape, non_mesh_shape, tri_index)       <L 1140>
                var_134 = wp::vec_t<3, wp::int32>(var_mesh_shape, var_non_mesh_shape, var_84);
                wp::array_store(var_triangle_pairs, var_128, var_134);
            }
        }
    goto start_while_0;
    end_while_0:;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/utils/heightfield.py:366
static CUDA_CALLABLE void adj_heightfield_vs_convex_midphase_0(
    wp::int32 var_hfield_shape,
    wp::int32 var_other_shape,
    HeightfieldData_f2b8d59a var_hfd,
    wp::array_t<wp::float32> var_elevation_data,
    wp::array_t<wp::transform_t<wp::float32>> var_shape_transform,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_collision_aabb_lower,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_collision_aabb_upper,
    wp::array_t<wp::vec_t<4, wp::float32>> var_shape_data,
    wp::array_t<wp::float32> var_shape_gap,
    wp::array_t<wp::vec_t<3, wp::int32>> var_triangle_pairs,
    wp::array_t<wp::int32> var_triangle_pairs_count,
    wp::int32 & adj_hfield_shape,
    wp::int32 & adj_other_shape,
    HeightfieldData_f2b8d59a & adj_hfd,
    wp::array_t<wp::float32> & adj_elevation_data,
    wp::array_t<wp::transform_t<wp::float32>> & adj_shape_transform,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_shape_collision_aabb_lower,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_shape_collision_aabb_upper,
    wp::array_t<wp::vec_t<4, wp::float32>> & adj_shape_data,
    wp::array_t<wp::float32> & adj_shape_gap,
    wp::array_t<wp::vec_t<3, wp::int32>> & adj_triangle_pairs,
    wp::array_t<wp::int32> & adj_triangle_pairs_count)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
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


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_core.py:893
static CUDA_CALLABLE void adj_aabb_to_unscaled_0(
    wp::vec_t<3, wp::float32> var_aabb_lower,
    wp::vec_t<3, wp::float32> var_aabb_upper,
    wp::vec_t<3, wp::float32> var_scale,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_aabb_lower,
    wp::vec_t<3, wp::float32> & adj_aabb_upper,
    wp::vec_t<3, wp::float32> & adj_scale,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_core.py:965
static CUDA_CALLABLE void adj__compute_mesh_vs_convex_query_aabb_0(
    wp::int32 var_mesh_shape,
    wp::int32 var_non_mesh_shape,
    wp::transform_t<wp::float32> var_X_mesh_ws,
    wp::transform_t<wp::float32> var_X_ws,
    wp::array_t<wp::int32> var_shape_type,
    wp::array_t<wp::vec_t<4, wp::float32>> var_shape_data,
    wp::array_t<wp::uint64> var_shape_source_ptr,
    wp::float32 var_contact_threshold,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::int32 & adj_mesh_shape,
    wp::int32 & adj_non_mesh_shape,
    wp::transform_t<wp::float32> & adj_X_mesh_ws,
    wp::transform_t<wp::float32> & adj_X_ws,
    wp::array_t<wp::int32> & adj_shape_type,
    wp::array_t<wp::vec_t<4, wp::float32>> & adj_shape_data,
    wp::array_t<wp::uint64> & adj_shape_source_ptr,
    wp::float32 & adj_contact_threshold,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_core.py:1013
static CUDA_CALLABLE void adj__mesh_triangle_is_front_facing_local_0(
    wp::uint64 var_mesh_id,
    wp::vec_t<3, wp::float32> var_center_in_bvh,
    wp::int32 var_tri_idx,
    wp::uint64 & adj_mesh_id,
    wp::vec_t<3, wp::float32> & adj_center_in_bvh,
    wp::int32 & adj_tri_idx,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_core.py:1033
static CUDA_CALLABLE void adj_mesh_vs_convex_midphase_0(
    wp::int32 var_idx_in_thread_block,
    wp::int32 var_mesh_shape,
    wp::int32 var_non_mesh_shape,
    wp::transform_t<wp::float32> var_X_mesh_ws,
    wp::transform_t<wp::float32> var_X_ws,
    wp::uint64 var_mesh_id,
    wp::array_t<wp::int32> var_shape_type,
    wp::array_t<wp::vec_t<4, wp::float32>> var_shape_data,
    wp::array_t<wp::uint64> var_shape_source_ptr,
    wp::float32 var_contact_threshold,
    wp::array_t<wp::vec_t<3, wp::int32>> var_triangle_pairs,
    wp::array_t<wp::int32> var_triangle_pairs_count,
    wp::int32 & adj_idx_in_thread_block,
    wp::int32 & adj_mesh_shape,
    wp::int32 & adj_non_mesh_shape,
    wp::transform_t<wp::float32> & adj_X_mesh_ws,
    wp::transform_t<wp::float32> & adj_X_ws,
    wp::uint64 & adj_mesh_id,
    wp::array_t<wp::int32> & adj_shape_type,
    wp::array_t<wp::vec_t<4, wp::float32>> & adj_shape_data,
    wp::array_t<wp::uint64> & adj_shape_source_ptr,
    wp::float32 & adj_contact_threshold,
    wp::array_t<wp::vec_t<3, wp::int32>> & adj_triangle_pairs,
    wp::array_t<wp::int32> & adj_triangle_pairs_count)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void compute_mesh_plane_vert_counts_1b66b83b_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<2, wp::int32>> var_shape_pairs_mesh_plane,
    wp::array_t<wp::int32> var_shape_pairs_mesh_plane_count,
    wp::array_t<wp::uint64> var_shape_source,
    wp::array_t<wp::int32> var_vert_counts,
    wp::array_t<wp::int32> var_total_vert_count,
    wp::int32 var_total_num_threads)
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
        const wp::int32 var_0 = 0;
        wp::int32* var_1;
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::range_t var_9;
        wp::int32 var_10;
        wp::vec_t<2, wp::int32>* var_11;
        wp::vec_t<2, wp::int32> var_12;
        wp::vec_t<2, wp::int32> var_13;
        const wp::int32 var_14 = 0;
        wp::int32 var_15;
        wp::uint64* var_16;
        wp::uint64 var_17;
        wp::uint64 var_18;
        const wp::int32 var_19 = 0;
        wp::int32 var_20;
        wp::uint64 var_21;
        bool var_22;
        wp::Mesh var_23;
        wp::array_t<wp::vec_t<3, wp::float32>>* var_24;
        wp::shape_t* var_25;
        const wp::int32 var_26 = 0;
        wp::int32 var_27;
        wp::shape_t var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        const wp::int32 var_31 = 0;
        wp::int32 var_32;
        //---------
        // forward
        // def compute_mesh_plane_vert_counts(                                                    <L 1751>
        // pair_count = wp.min(shape_pairs_mesh_plane_count[0], shape_pairs_mesh_plane.shape[0])       <L 1764>
        var_1 = wp::address(var_shape_pairs_mesh_plane_count, var_0);
        var_2 = &(var_shape_pairs_mesh_plane.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_7 = wp::load(var_1);
        var_6 = wp::min(var_7, var_4);
        // for i in range(wp.tid(), pair_count, total_num_threads):                               <L 1765>
        var_8 = builtin_tid1d();
        var_9 = wp::range(var_8, var_6, var_total_num_threads);
        start_for_0:;
            if (iter_cmp(var_9) == 0) goto end_for_0;
            var_10 = wp::iter_next(var_9);
            // pair = shape_pairs_mesh_plane[i]                                                   <L 1766>
            var_11 = wp::address(var_shape_pairs_mesh_plane, var_10);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
            // mesh_shape = pair[0]                                                               <L 1767>
            var_15 = wp::extract(var_12, var_14);
            // mesh_id = shape_source[mesh_shape]                                                 <L 1768>
            var_16 = wp::address(var_shape_source, var_15);
            var_18 = wp::load(var_16);
            var_17 = wp::copy(var_18);
            // pair_verts = int(0)  # noqa: RUF046, RUF100 - explicit cast required by Warp codegen       <L 1769>
            var_20 = wp::int(var_19);
            // if mesh_id != wp.uint64(0):                                                        <L 1770>
            var_21 = 0ull;
            var_22 = (var_17 != var_21);
            if (var_22) {
                // pair_verts = wp.mesh_get(mesh_id).points.shape[0]                              <L 1771>
                var_23 = wp::mesh_get(var_17);
                var_24 = &(var_23.points);
                var_25 = &(var_24->shape);
                var_28 = wp::load(var_25);
                var_27 = wp::extract(var_28, var_26);
            }
            var_29 = wp::where(var_22, var_27, var_20);
            // vert_counts[i] = wp.int32(pair_verts)                                              <L 1772>
            var_30 = wp::int32(var_29);
            wp::array_store(var_vert_counts, var_10, var_30);
            // wp.atomic_add(total_vert_count, 0, pair_verts)                                     <L 1773>
            var_32 = wp::atomic_add(var_total_vert_count, var_31, var_29);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void verify_narrow_phase_buffers_77619ee7_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_broad_phase_count,
    wp::int32 var_max_broad_phase,
    wp::array_t<wp::int32> var_split_query_count,
    wp::int32 var_max_split_query,
    wp::array_t<wp::int32> var_gjk_count,
    wp::int32 var_max_gjk,
    wp::array_t<wp::int32> var_split_gjk_count,
    wp::int32 var_max_split_gjk,
    wp::array_t<wp::int32> var_split_manifold_count,
    wp::int32 var_max_split_manifold,
    wp::array_t<wp::int32> var_mesh_count,
    wp::int32 var_max_mesh,
    wp::array_t<wp::int32> var_triangle_count,
    wp::int32 var_max_triangle,
    wp::array_t<wp::int32> var_mesh_plane_count,
    wp::int32 var_max_mesh_plane,
    wp::array_t<wp::int32> var_mesh_mesh_count,
    wp::int32 var_max_mesh_mesh,
    wp::array_t<wp::int32> var_sdf_sdf_count,
    wp::int32 var_max_sdf_sdf,
    wp::array_t<wp::int32> var_contact_count,
    wp::int32 var_max_contacts,
    wp::array_t<wp::int32> var_reduction_ht_active_slots,
    wp::int32 var_reduction_ht_capacity,
    wp::array_t<wp::int32> var_reduction_ht_insert_failures,
    wp::int32 var_reduction_ht_warn_load_percent)
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
        const wp::int32 var_0 = 0;
        wp::int32* var_1;
        bool var_2;
        wp::int32 var_3;
        const wp::str var_4 = "Warning: Broad phase pair buffer overflowed %d > %d.\n";
        const wp::int32 var_5 = 0;
        wp::int32* var_6;
        wp::int32 var_7;
        bool var_8;
        const wp::int32 var_9 = 0;
        bool var_10;
        const wp::int32 var_11 = 0;
        wp::int32* var_12;
        bool var_13;
        wp::int32 var_14;
        const wp::str var_15 = "Warning: Split query-result buffer overflowed %d > %d.\n";
        const wp::int32 var_16 = 0;
        wp::int32* var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 0;
        wp::int32* var_20;
        bool var_21;
        wp::int32 var_22;
        const wp::str var_23 = "Warning: GJK candidate pair buffer overflowed %d > %d.\n";
        const wp::int32 var_24 = 0;
        wp::int32* var_25;
        wp::int32 var_26;
        bool var_27;
        const wp::int32 var_28 = 0;
        bool var_29;
        const wp::int32 var_30 = 0;
        wp::int32* var_31;
        bool var_32;
        wp::int32 var_33;
        const wp::str var_34 = "Warning: Split GJK work-item buffer overflowed %d > %d.\n";
        const wp::int32 var_35 = 0;
        wp::int32* var_36;
        wp::int32 var_37;
        bool var_38;
        const wp::int32 var_39 = 0;
        bool var_40;
        const wp::int32 var_41 = 0;
        wp::int32* var_42;
        bool var_43;
        wp::int32 var_44;
        const wp::str var_45 = "Warning: Split manifold work-item buffer overflowed %d > %d.\n";
        const wp::int32 var_46 = 0;
        wp::int32* var_47;
        wp::int32 var_48;
        const wp::int32 var_49 = 0;
        wp::int32* var_50;
        bool var_51;
        wp::int32 var_52;
        const wp::str var_53 = "Warning: Mesh-convex shape pair buffer overflowed %d > %d.\n";
        const wp::int32 var_54 = 0;
        wp::int32* var_55;
        wp::int32 var_56;
        const wp::int32 var_57 = 0;
        wp::int32* var_58;
        bool var_59;
        wp::int32 var_60;
        const wp::str var_61 = "Warning: Triangle pair buffer overflowed %d > %d.\n";
        const wp::int32 var_62 = 0;
        wp::int32* var_63;
        wp::int32 var_64;
        const wp::int32 var_65 = 0;
        wp::int32* var_66;
        bool var_67;
        wp::int32 var_68;
        const wp::str var_69 = "Warning: Mesh-plane shape pair buffer overflowed %d > %d.\n";
        const wp::int32 var_70 = 0;
        wp::int32* var_71;
        wp::int32 var_72;
        const wp::int32 var_73 = 0;
        wp::int32* var_74;
        bool var_75;
        wp::int32 var_76;
        const wp::str var_77 = "Warning: Mesh-mesh shape pair buffer overflowed %d > %d.\n";
        const wp::int32 var_78 = 0;
        wp::int32* var_79;
        wp::int32 var_80;
        const wp::int32 var_81 = 0;
        wp::int32* var_82;
        bool var_83;
        wp::int32 var_84;
        const wp::str var_85 = "Warning: SDF-SDF shape pair buffer overflowed %d > %d.\n";
        const wp::int32 var_86 = 0;
        wp::int32* var_87;
        wp::int32 var_88;
        const wp::int32 var_89 = 0;
        wp::int32* var_90;
        bool var_91;
        wp::int32 var_92;
        const wp::str var_93 = "Warning: Contact buffer overflowed %d > %d.\n";
        const wp::int32 var_94 = 0;
        wp::int32* var_95;
        wp::int32 var_96;
        const wp::int32 var_97 = 0;
        bool var_98;
        wp::int32* var_99;
        wp::int32 var_100;
        wp::int32 var_101;
        const wp::int32 var_102 = 100;
        wp::int32 var_103;
        wp::int32 var_104;
        bool var_105;
        const wp::str var_106 = "Warning: Contact reduction hashtable fill ratio exceeded %d%% (%d / %d). Increase contact_reduction_hashtable_size_factor or max_triangle_pairs.\n";
        const wp::int32 var_107 = 0;
        wp::int32* var_108;
        const wp::int32 var_109 = 0;
        bool var_110;
        wp::int32 var_111;
        const wp::str var_112 = "Warning: Contact reduction hashtable insert failures %d. Increase contact_reduction_hashtable_size_factor or max_triangle_pairs.\n";
        const wp::int32 var_113 = 0;
        wp::int32* var_114;
        wp::int32 var_115;
        //---------
        // forward
        // def verify_narrow_phase_buffers(                                                       <L 2083>
        // if broad_phase_count[0] > max_broad_phase:                                             <L 2112>
        var_1 = wp::address(var_broad_phase_count, var_0);
        var_3 = wp::load(var_1);
        var_2 = (var_3 > var_max_broad_phase);
        if (var_2) {
            // wp.printf(                                                                         <L 2113>
            // "Warning: Broad phase pair buffer overflowed %d > %d.\n",                          <L 2114>
            // broad_phase_count[0],                                                              <L 2115>
            var_6 = wp::address(var_broad_phase_count, var_5);
            // max_broad_phase,                                                                   <L 2116>
            var_7 = wp::load(var_6);
            printf(var_4, var_7, var_max_broad_phase);
        }
        // if max_split_query >= 0 and split_query_count[0] > max_split_query:                    <L 2118>
        var_10 = (var_max_split_query >= var_9);
        var_8 = var_10;
        if (var_8) {
            var_12 = wp::address(var_split_query_count, var_11);
            var_14 = wp::load(var_12);
            var_13 = (var_14 > var_max_split_query);
            var_8 = var_8 && var_13;
        }
        if (var_8) {
            // wp.printf(                                                                         <L 2119>
            // "Warning: Split query-result buffer overflowed %d > %d.\n",                        <L 2120>
            // split_query_count[0],                                                              <L 2121>
            var_17 = wp::address(var_split_query_count, var_16);
            // max_split_query,                                                                   <L 2122>
            var_18 = wp::load(var_17);
            printf(var_15, var_18, var_max_split_query);
        }
        // if gjk_count[0] > max_gjk:                                                             <L 2124>
        var_20 = wp::address(var_gjk_count, var_19);
        var_22 = wp::load(var_20);
        var_21 = (var_22 > var_max_gjk);
        if (var_21) {
            // wp.printf(                                                                         <L 2125>
            // "Warning: GJK candidate pair buffer overflowed %d > %d.\n",                        <L 2126>
            // gjk_count[0],                                                                      <L 2127>
            var_25 = wp::address(var_gjk_count, var_24);
            // max_gjk,                                                                           <L 2128>
            var_26 = wp::load(var_25);
            printf(var_23, var_26, var_max_gjk);
        }
        // if max_split_gjk >= 0 and split_gjk_count[0] > max_split_gjk:                          <L 2130>
        var_29 = (var_max_split_gjk >= var_28);
        var_27 = var_29;
        if (var_27) {
            var_31 = wp::address(var_split_gjk_count, var_30);
            var_33 = wp::load(var_31);
            var_32 = (var_33 > var_max_split_gjk);
            var_27 = var_27 && var_32;
        }
        if (var_27) {
            // wp.printf(                                                                         <L 2131>
            // "Warning: Split GJK work-item buffer overflowed %d > %d.\n",                       <L 2132>
            // split_gjk_count[0],                                                                <L 2133>
            var_36 = wp::address(var_split_gjk_count, var_35);
            // max_split_gjk,                                                                     <L 2134>
            var_37 = wp::load(var_36);
            printf(var_34, var_37, var_max_split_gjk);
        }
        // if max_split_manifold >= 0 and split_manifold_count[0] > max_split_manifold:           <L 2136>
        var_40 = (var_max_split_manifold >= var_39);
        var_38 = var_40;
        if (var_38) {
            var_42 = wp::address(var_split_manifold_count, var_41);
            var_44 = wp::load(var_42);
            var_43 = (var_44 > var_max_split_manifold);
            var_38 = var_38 && var_43;
        }
        if (var_38) {
            // wp.printf(                                                                         <L 2137>
            // "Warning: Split manifold work-item buffer overflowed %d > %d.\n",                  <L 2138>
            // split_manifold_count[0],                                                           <L 2139>
            var_47 = wp::address(var_split_manifold_count, var_46);
            // max_split_manifold,                                                                <L 2140>
            var_48 = wp::load(var_47);
            printf(var_45, var_48, var_max_split_manifold);
        }
        // if mesh_count:                                                                         <L 2142>
        if (var_mesh_count) {
            // if mesh_count[0] > max_mesh:                                                       <L 2143>
            var_50 = wp::address(var_mesh_count, var_49);
            var_52 = wp::load(var_50);
            var_51 = (var_52 > var_max_mesh);
            if (var_51) {
                // wp.printf(                                                                     <L 2144>
                // "Warning: Mesh-convex shape pair buffer overflowed %d > %d.\n",                <L 2145>
                // mesh_count[0],                                                                 <L 2146>
                var_55 = wp::address(var_mesh_count, var_54);
                // max_mesh,                                                                      <L 2147>
                var_56 = wp::load(var_55);
                printf(var_53, var_56, var_max_mesh);
            }
        }
        // if triangle_count:                                                                     <L 2149>
        if (var_triangle_count) {
            // if triangle_count[0] > max_triangle:                                               <L 2150>
            var_58 = wp::address(var_triangle_count, var_57);
            var_60 = wp::load(var_58);
            var_59 = (var_60 > var_max_triangle);
            if (var_59) {
                // wp.printf(                                                                     <L 2151>
                // "Warning: Triangle pair buffer overflowed %d > %d.\n",                         <L 2152>
                // triangle_count[0],                                                             <L 2153>
                var_63 = wp::address(var_triangle_count, var_62);
                // max_triangle,                                                                  <L 2154>
                var_64 = wp::load(var_63);
                printf(var_61, var_64, var_max_triangle);
            }
        }
        // if mesh_plane_count:                                                                   <L 2156>
        if (var_mesh_plane_count) {
            // if mesh_plane_count[0] > max_mesh_plane:                                           <L 2157>
            var_66 = wp::address(var_mesh_plane_count, var_65);
            var_68 = wp::load(var_66);
            var_67 = (var_68 > var_max_mesh_plane);
            if (var_67) {
                // wp.printf(                                                                     <L 2158>
                // "Warning: Mesh-plane shape pair buffer overflowed %d > %d.\n",                 <L 2159>
                // mesh_plane_count[0],                                                           <L 2160>
                var_71 = wp::address(var_mesh_plane_count, var_70);
                // max_mesh_plane,                                                                <L 2161>
                var_72 = wp::load(var_71);
                printf(var_69, var_72, var_max_mesh_plane);
            }
        }
        // if mesh_mesh_count:                                                                    <L 2163>
        if (var_mesh_mesh_count) {
            // if mesh_mesh_count[0] > max_mesh_mesh:                                             <L 2164>
            var_74 = wp::address(var_mesh_mesh_count, var_73);
            var_76 = wp::load(var_74);
            var_75 = (var_76 > var_max_mesh_mesh);
            if (var_75) {
                // wp.printf(                                                                     <L 2165>
                // "Warning: Mesh-mesh shape pair buffer overflowed %d > %d.\n",                  <L 2166>
                // mesh_mesh_count[0],                                                            <L 2167>
                var_79 = wp::address(var_mesh_mesh_count, var_78);
                // max_mesh_mesh,                                                                 <L 2168>
                var_80 = wp::load(var_79);
                printf(var_77, var_80, var_max_mesh_mesh);
            }
        }
        // if sdf_sdf_count:                                                                      <L 2170>
        if (var_sdf_sdf_count) {
            // if sdf_sdf_count[0] > max_sdf_sdf:                                                 <L 2171>
            var_82 = wp::address(var_sdf_sdf_count, var_81);
            var_84 = wp::load(var_82);
            var_83 = (var_84 > var_max_sdf_sdf);
            if (var_83) {
                // wp.printf(                                                                     <L 2172>
                // "Warning: SDF-SDF shape pair buffer overflowed %d > %d.\n",                    <L 2173>
                // sdf_sdf_count[0],                                                              <L 2174>
                var_87 = wp::address(var_sdf_sdf_count, var_86);
                // max_sdf_sdf,                                                                   <L 2175>
                var_88 = wp::load(var_87);
                printf(var_85, var_88, var_max_sdf_sdf);
            }
        }
        // if contact_count[0] > max_contacts:                                                    <L 2177>
        var_90 = wp::address(var_contact_count, var_89);
        var_92 = wp::load(var_90);
        var_91 = (var_92 > var_max_contacts);
        if (var_91) {
            // wp.printf(                                                                         <L 2178>
            // "Warning: Contact buffer overflowed %d > %d.\n",                                   <L 2179>
            // contact_count[0],                                                                  <L 2180>
            var_95 = wp::address(var_contact_count, var_94);
            // max_contacts,                                                                      <L 2181>
            var_96 = wp::load(var_95);
            printf(var_93, var_96, var_max_contacts);
        }
        // if reduction_ht_capacity > 0:                                                          <L 2183>
        var_98 = (var_reduction_ht_capacity > var_97);
        if (var_98) {
            // reduction_ht_active_count = reduction_ht_active_slots[reduction_ht_capacity]       <L 2184>
            var_99 = wp::address(var_reduction_ht_active_slots, var_reduction_ht_capacity);
            var_101 = wp::load(var_99);
            var_100 = wp::copy(var_101);
            // if reduction_ht_active_count * 100 >= reduction_ht_capacity * reduction_ht_warn_load_percent:       <L 2185>
            var_103 = wp::mul(var_100, var_102);
            var_104 = wp::mul(var_reduction_ht_capacity, var_reduction_ht_warn_load_percent);
            var_105 = (var_103 >= var_104);
            if (var_105) {
                // wp.printf(                                                                     <L 2186>
                // "Warning: Contact reduction hashtable fill ratio exceeded %d%% (%d / %d). "       <L 2187>
                // reduction_ht_warn_load_percent,                                                <L 2189>
                // reduction_ht_active_count,                                                     <L 2190>
                // reduction_ht_capacity,                                                         <L 2191>
                printf(var_106, var_reduction_ht_warn_load_percent, var_100, var_reduction_ht_capacity);
            }
            // if reduction_ht_insert_failures[0] > 0:                                            <L 2193>
            var_108 = wp::address(var_reduction_ht_insert_failures, var_107);
            var_111 = wp::load(var_108);
            var_110 = (var_111 > var_109);
            if (var_110) {
                // wp.printf(                                                                     <L 2194>
                // "Warning: Contact reduction hashtable insert failures %d. "                    <L 2195>
                // reduction_ht_insert_failures[0],                                               <L 2197>
                var_114 = wp::address(var_reduction_ht_insert_failures, var_113);
                var_115 = wp::load(var_114);
                printf(var_112, var_115);
            }
        }
    }
}



extern "C" __global__ void narrow_phase_find_mesh_triangle_overlaps_kernel_ec87a852_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_shape_types,
    wp::array_t<wp::transform_t<wp::float32>> var_shape_transform,
    wp::array_t<wp::uint64> var_shape_source,
    wp::array_t<wp::float32> var_shape_gap,
    wp::array_t<wp::vec_t<4, wp::float32>> var_shape_data,
    wp::array_t<wp::float32> var_shape_collision_radius,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_collision_aabb_lower,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_collision_aabb_upper,
    wp::array_t<wp::int32> var_shape_heightfield_index,
    wp::array_t<HeightfieldData_f2b8d59a> var_heightfield_data,
    wp::array_t<wp::float32> var_heightfield_elevations,
    wp::array_t<wp::vec_t<2, wp::int32>> var_shape_pairs_mesh,
    wp::array_t<wp::int32> var_shape_pairs_mesh_count,
    wp::int32 var_total_num_threads,
    wp::array_t<wp::vec_t<3, wp::int32>> var_triangle_pairs,
    wp::array_t<wp::int32> var_triangle_pairs_count)
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
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::range_t var_6;
        wp::int32 var_7;
        wp::vec_t<2, wp::int32>* var_8;
        wp::vec_t<2, wp::int32> var_9;
        wp::vec_t<2, wp::int32> var_10;
        const wp::int32 var_11 = 0;
        wp::int32 var_12;
        const wp::int32 var_13 = 1;
        wp::int32 var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 2;
        bool var_22;
        const wp::int32 var_23 = 0;
        bool var_24;
        wp::int32* var_25;
        HeightfieldData_f2b8d59a* var_26;
        wp::int32 var_27;
        HeightfieldData_f2b8d59a var_28;
        HeightfieldData_f2b8d59a var_29;
        const wp::int32 var_30 = -1;
        const wp::int32 var_31 = -1;
        bool var_32;
        const wp::int32 var_33 = 8;
        bool var_34;
        const wp::int32 var_35 = 8;
        bool var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        bool var_39;
        const wp::int32 var_40 = 8;
        bool var_41;
        const wp::int32 var_42 = 8;
        bool var_43;
        wp::int32 var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        wp::int32 var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::uint64* var_50;
        wp::uint64 var_51;
        wp::uint64 var_52;
        wp::uint64 var_53;
        bool var_54;
        wp::transform_t<wp::float32>* var_55;
        wp::transform_t<wp::float32> var_56;
        wp::transform_t<wp::float32> var_57;
        wp::transform_t<wp::float32>* var_58;
        wp::transform_t<wp::float32> var_59;
        wp::transform_t<wp::float32> var_60;
        wp::float32* var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::float32* var_64;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::vec_t<4, wp::float32>* var_68;
        const wp::int32 var_69 = 3;
        wp::float32 var_70;
        wp::vec_t<4, wp::float32> var_71;
        wp::vec_t<4, wp::float32>* var_72;
        const wp::int32 var_73 = 3;
        wp::float32 var_74;
        wp::vec_t<4, wp::float32> var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        //---------
        // forward
        // def narrow_phase_find_mesh_triangle_overlaps_kernel(                                   <L 1509>
        // tid, j = wp.tid()                                                                      <L 1536>
        builtin_tid2d(var_0, var_1);
        // num_mesh_pairs = shape_pairs_mesh_count[0]                                             <L 1538>
        var_3 = wp::address(var_shape_pairs_mesh_count, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // for i in range(tid, num_mesh_pairs, total_num_threads):                                <L 1541>
        var_6 = wp::range(var_0, var_4, var_total_num_threads);
        start_for_0:;
            if (iter_cmp(var_6) == 0) goto end_for_0;
            var_7 = wp::iter_next(var_6);
            // pair = shape_pairs_mesh[i]                                                         <L 1542>
            var_8 = wp::address(var_shape_pairs_mesh, var_7);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
            // shape_a = pair[0]                                                                  <L 1543>
            var_12 = wp::extract(var_9, var_11);
            // shape_b = pair[1]                                                                  <L 1544>
            var_14 = wp::extract(var_9, var_13);
            // type_a = shape_types[shape_a]                                                      <L 1546>
            var_15 = wp::address(var_shape_types, var_12);
            var_17 = wp::load(var_15);
            var_16 = wp::copy(var_17);
            // type_b = shape_types[shape_b]                                                      <L 1547>
            var_18 = wp::address(var_shape_types, var_14);
            var_20 = wp::load(var_18);
            var_19 = wp::copy(var_20);
            // if type_a == GeoType.HFIELD:                                                       <L 1553>
            var_22 = (var_16 == var_21);
            if (var_22) {
                // if j != 0:                                                                     <L 1555>
                var_24 = (var_1 != var_23);
                if (var_24) {
                    // continue                                                                   <L 1556>
                    goto start_for_0;
                }
                // hfd = heightfield_data[shape_heightfield_index[shape_a]]                       <L 1557>
                var_25 = wp::address(var_shape_heightfield_index, var_12);
                var_27 = wp::load(var_25);
                var_26 = wp::address(var_heightfield_data, var_27);
                var_29 = wp::load(var_26);
                var_28 = wp::copy(var_29);
                // heightfield_vs_convex_midphase(                                                <L 1558>
                // shape_a,                                                                       <L 1559>
                // shape_b,                                                                       <L 1560>
                // hfd,                                                                           <L 1561>
                // heightfield_elevations,                                                        <L 1562>
                // shape_transform,                                                               <L 1563>
                // shape_collision_aabb_lower,                                                    <L 1564>
                // shape_collision_aabb_upper,                                                    <L 1565>
                // shape_data,                                                                    <L 1566>
                // shape_gap,                                                                     <L 1567>
                // triangle_pairs,                                                                <L 1568>
                // triangle_pairs_count,                                                          <L 1569>
                heightfield_vs_convex_midphase_0(var_12, var_14, var_28, var_heightfield_elevations, var_shape_transform, var_shape_collision_aabb_lower, var_shape_collision_aabb_upper, var_shape_data, var_shape_gap, var_triangle_pairs, var_triangle_pairs_count);
                // continue                                                                       <L 1571>
                goto start_for_0;
            }
            // mesh_shape = -1                                                                    <L 1576>
            // non_mesh_shape = -1                                                                <L 1577>
            // if type_a == GeoType.MESH and type_b != GeoType.MESH:                              <L 1579>
            var_34 = (var_16 == var_33);
            var_32 = var_34;
            if (var_32) {
                var_36 = (var_19 != var_35);
                var_32 = var_32 && var_36;
            }
            if (var_32) {
                // mesh_shape = shape_a                                                           <L 1580>
                var_37 = wp::copy(var_12);
                // non_mesh_shape = shape_b                                                       <L 1581>
                var_38 = wp::copy(var_14);
            }
            if (!var_32) {
                // elif type_b == GeoType.MESH and type_a != GeoType.MESH:                        <L 1582>
                var_41 = (var_19 == var_40);
                var_39 = var_41;
                if (var_39) {
                    var_43 = (var_16 != var_42);
                    var_39 = var_39 && var_43;
                }
                if (var_39) {
                    // mesh_shape = shape_b                                                       <L 1583>
                    var_44 = wp::copy(var_14);
                    // non_mesh_shape = shape_a                                                   <L 1584>
                    var_45 = wp::copy(var_12);
                }
                if (!var_39) {
                    // continue                                                                   <L 1587>
                    goto start_for_0;
                }
                var_46 = wp::where(var_39, var_44, var_30);
                var_47 = wp::where(var_39, var_45, var_31);
            }
            var_48 = wp::where(var_32, var_37, var_46);
            var_49 = wp::where(var_32, var_38, var_47);
            // mesh_id = shape_source[mesh_shape]                                                 <L 1590>
            var_50 = wp::address(var_shape_source, var_48);
            var_52 = wp::load(var_50);
            var_51 = wp::copy(var_52);
            // if mesh_id == wp.uint64(0):                                                        <L 1591>
            var_53 = 0ull;
            var_54 = (var_51 == var_53);
            if (var_54) {
                // continue                                                                       <L 1592>
                goto start_for_0;
            }
            // X_mesh_ws = shape_transform[mesh_shape]                                            <L 1595>
            var_55 = wp::address(var_shape_transform, var_48);
            var_57 = wp::load(var_55);
            var_56 = wp::copy(var_57);
            // X_ws = shape_transform[non_mesh_shape]                                             <L 1598>
            var_58 = wp::address(var_shape_transform, var_49);
            var_60 = wp::load(var_58);
            var_59 = wp::copy(var_60);
            // gap_non_mesh = shape_gap[non_mesh_shape]                                           <L 1602>
            var_61 = wp::address(var_shape_gap, var_49);
            var_63 = wp::load(var_61);
            var_62 = wp::copy(var_63);
            // gap_mesh = shape_gap[mesh_shape]                                                   <L 1603>
            var_64 = wp::address(var_shape_gap, var_48);
            var_66 = wp::load(var_64);
            var_65 = wp::copy(var_66);
            // gap_sum = gap_non_mesh + gap_mesh                                                  <L 1604>
            var_67 = wp::add(var_62, var_65);
            // margin_non_mesh = shape_data[non_mesh_shape][3]                                    <L 1605>
            var_68 = wp::address(var_shape_data, var_49);
            var_71 = wp::load(var_68);
            var_70 = wp::extract(var_71, var_69);
            // margin_mesh = shape_data[mesh_shape][3]                                            <L 1606>
            var_72 = wp::address(var_shape_data, var_48);
            var_75 = wp::load(var_72);
            var_74 = wp::extract(var_75, var_73);
            // contact_threshold = gap_sum + margin_non_mesh + margin_mesh                        <L 1607>
            var_76 = wp::add(var_67, var_70);
            var_77 = wp::add(var_76, var_74);
            // mesh_vs_convex_midphase(                                                           <L 1610>
            // j,                                                                                 <L 1611>
            // mesh_shape,                                                                        <L 1612>
            // non_mesh_shape,                                                                    <L 1613>
            // X_mesh_ws,                                                                         <L 1614>
            // X_ws,                                                                              <L 1615>
            // mesh_id,                                                                           <L 1616>
            // shape_types,                                                                       <L 1617>
            // shape_data,                                                                        <L 1618>
            // shape_source,                                                                      <L 1619>
            // contact_threshold,                                                                 <L 1620>
            // triangle_pairs,                                                                    <L 1621>
            // triangle_pairs_count,                                                              <L 1622>
            mesh_vs_convex_midphase_0(var_1, var_48, var_49, var_56, var_59, var_51, var_shape_types, var_shape_data, var_shape_source, var_77, var_triangle_pairs, var_triangle_pairs_count);
            goto start_for_0;
        end_for_0:;
    }
}

