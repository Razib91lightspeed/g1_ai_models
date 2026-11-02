#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 32
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


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/sim/articulation.py:36
static CUDA_CALLABLE void transform_2d_rotational_axes_0(
    wp::vec_t<3, wp::float32> var_axis_0,
    wp::vec_t<3, wp::float32> var_axis_1,
    wp::float32 var_q0,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::mat_t<3, 3, wp::float32> var_1;
    wp::quat_t<wp::float32> var_2;
    const wp::float32 var_3 = 1.0;
    const wp::float32 var_4 = 0.0;
    const wp::float32 var_5 = 0.0;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    const wp::float32 var_8 = 0.0;
    const wp::float32 var_9 = 1.0;
    const wp::float32 var_10 = 0.0;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::quat_t<wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    //---------
    // forward
    // def transform_2d_rotational_axes(                                                      <L 37>
    // q_off = wp.quat_from_matrix(wp.matrix_from_cols(axis_0, axis_1, wp.cross(axis_0, axis_1)))       <L 48>
    var_0 = wp::cross(var_axis_0, var_axis_1);
    var_1 = wp::matrix_from_cols<wp::float32>(var_axis_0, var_axis_1, var_0);
    var_2 = wp::quat_from_matrix(var_1);
    // local_0 = wp.quat_rotate(q_off, wp.vec3(1.0, 0.0, 0.0))                                <L 51>
    var_6 = wp::vec_t<3, wp::float32>(var_3, var_4, var_5);
    var_7 = wp::quat_rotate(var_2, var_6);
    // local_1 = wp.quat_rotate(q_off, wp.vec3(0.0, 1.0, 0.0))                                <L 52>
    var_11 = wp::vec_t<3, wp::float32>(var_8, var_9, var_10);
    var_12 = wp::quat_rotate(var_2, var_11);
    // a0 = local_0                                                                           <L 54>
    var_13 = wp::copy(var_7);
    // q_0 = wp.quat_from_axis_angle(a0, q0)                                                  <L 55>
    var_14 = wp::quat_from_axis_angle(var_13, var_q0);
    // a1 = wp.quat_rotate(q_0, local_1)                                                      <L 56>
    var_15 = wp::quat_rotate(var_14, var_12);
    // return a0, a1                                                                          <L 58>
    ret_0 = var_13;
    ret_1 = var_15;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/sim/articulation.py:61
static CUDA_CALLABLE void compute_2d_rotational_dofs_0(
    wp::vec_t<3, wp::float32> var_axis_0,
    wp::vec_t<3, wp::float32> var_axis_1,
    wp::float32 var_q0,
    wp::float32 var_q1,
    wp::float32 var_qd0,
    wp::float32 var_qd1,
    wp::quat_t<wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::quat_t<wp::float32> var_2;
    wp::quat_t<wp::float32> var_3;
    wp::quat_t<wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    //---------
    // forward
    // def compute_2d_rotational_dofs(                                                        <L 62>
    // a0, a1 = transform_2d_rotational_axes(axis_0, axis_1, q0)                              <L 73>
    transform_2d_rotational_axes_0(var_axis_0, var_axis_1, var_q0, var_0, var_1);
    // q_0 = wp.quat_from_axis_angle(a0, q0)                                                  <L 75>
    var_2 = wp::quat_from_axis_angle(var_0, var_q0);
    // q_1 = wp.quat_from_axis_angle(a1, q1)                                                  <L 76>
    var_3 = wp::quat_from_axis_angle(var_1, var_q1);
    // rot = q_1 * q_0                                                                        <L 78>
    var_4 = wp::mul(var_3, var_2);
    // vel = a0 * qd0 + a1 * qd1                                                              <L 80>
    var_5 = wp::mul(var_0, var_qd0);
    var_6 = wp::mul(var_1, var_qd1);
    var_7 = wp::add(var_5, var_6);
    // return rot, vel                                                                        <L 82>
    ret_0 = var_4;
    ret_1 = var_7;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/sim/articulation.py:127
static CUDA_CALLABLE void transform_3d_rotational_axes_0(
    wp::vec_t<3, wp::float32> var_axis_0,
    wp::vec_t<3, wp::float32> var_axis_1,
    wp::vec_t<3, wp::float32> var_axis_2,
    wp::float32 var_q0,
    wp::float32 var_q1,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::quat_t<wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::quat_t<wp::float32> var_2;
    wp::quat_t<wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    //---------
    // forward
    // def transform_3d_rotational_axes(                                                      <L 128>
    // q_0 = wp.quat_from_axis_angle(axis_0, q0)                                              <L 141>
    var_0 = wp::quat_from_axis_angle(var_axis_0, var_q0);
    // axis_1_w = wp.quat_rotate(q_0, axis_1)                                                 <L 143>
    var_1 = wp::quat_rotate(var_0, var_axis_1);
    // q_1 = wp.quat_from_axis_angle(axis_1_w, q1)                                            <L 144>
    var_2 = wp::quat_from_axis_angle(var_1, var_q1);
    // axis_2_w = wp.quat_rotate(q_1 * q_0, axis_2)                                           <L 146>
    var_3 = wp::mul(var_2, var_0);
    var_4 = wp::quat_rotate(var_3, var_axis_2);
    // return axis_0, axis_1_w, axis_2_w                                                      <L 148>
    ret_0 = var_axis_0;
    ret_1 = var_1;
    ret_2 = var_4;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/sim/articulation.py:151
static CUDA_CALLABLE void compute_3d_rotational_dofs_0(
    wp::vec_t<3, wp::float32> var_axis_0,
    wp::vec_t<3, wp::float32> var_axis_1,
    wp::vec_t<3, wp::float32> var_axis_2,
    wp::float32 var_q0,
    wp::float32 var_q1,
    wp::float32 var_q2,
    wp::float32 var_qd0,
    wp::float32 var_qd1,
    wp::float32 var_qd2,
    wp::quat_t<wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::quat_t<wp::float32> var_3;
    wp::quat_t<wp::float32> var_4;
    wp::quat_t<wp::float32> var_5;
    wp::quat_t<wp::float32> var_6;
    wp::quat_t<wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    //---------
    // forward
    // def compute_3d_rotational_dofs(                                                        <L 152>
    // axis_0_w, axis_1_w, axis_2_w = transform_3d_rotational_axes(axis_0, axis_1, axis_2, q0, q1)       <L 169>
    transform_3d_rotational_axes_0(var_axis_0, var_axis_1, var_axis_2, var_q0, var_q1, var_0, var_1, var_2);
    // q_0 = wp.quat_from_axis_angle(axis_0_w, q0)                                            <L 171>
    var_3 = wp::quat_from_axis_angle(var_0, var_q0);
    // q_1 = wp.quat_from_axis_angle(axis_1_w, q1)                                            <L 172>
    var_4 = wp::quat_from_axis_angle(var_1, var_q1);
    // q_2 = wp.quat_from_axis_angle(axis_2_w, q2)                                            <L 173>
    var_5 = wp::quat_from_axis_angle(var_2, var_q2);
    // rot = q_2 * q_1 * q_0                                                                  <L 175>
    var_6 = wp::mul(var_5, var_4);
    var_7 = wp::mul(var_6, var_3);
    // vel = axis_0_w * qd0 + axis_1_w * qd1 + axis_2_w * qd2                                 <L 176>
    var_8 = wp::mul(var_0, var_qd0);
    var_9 = wp::mul(var_1, var_qd1);
    var_10 = wp::add(var_8, var_9);
    var_11 = wp::mul(var_2, var_qd2);
    var_12 = wp::add(var_10, var_11);
    // return rot, vel                                                                        <L 178>
    ret_0 = var_7;
    ret_1 = var_12;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/sim/articulation.py:236
static CUDA_CALLABLE void eval_joint_motion_0(
    wp::int32 var_type,
    wp::int32 var_q_start,
    wp::int32 var_qd_start,
    wp::int32 var_lin_axis_count,
    wp::int32 var_ang_axis_count,
    wp::array_t<wp::float32> var_joint_q,
    wp::array_t<wp::float32> var_joint_qd,
    wp::array_t<wp::vec_t<3, wp::float32>> var_joint_axis,
    wp::transform_t<wp::float32> & ret_0,
    wp::vec_t<6, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::transform_t<wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<6, wp::float32> var_3;
    const wp::int32 var_4 = 0;
    bool var_5;
    wp::vec_t<3, wp::float32>* var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::float32* var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::float32 var_11;
    wp::quat_t<wp::float32> var_12;
    wp::transform_t<wp::float32> var_13;
    wp::float32* var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::float32 var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<6, wp::float32> var_18;
    wp::transform_t<wp::float32> var_19;
    wp::vec_t<6, wp::float32> var_20;
    const wp::int32 var_21 = 1;
    bool var_22;
    wp::vec_t<3, wp::float32>* var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::float32* var_27;
    wp::quat_t<wp::float32> var_28;
    wp::float32 var_29;
    wp::transform_t<wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::float32* var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::float32 var_34;
    wp::vec_t<6, wp::float32> var_35;
    wp::transform_t<wp::float32> var_36;
    wp::vec_t<6, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    const wp::int32 var_39 = 2;
    bool var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::float32* var_42;
    const wp::int32 var_43 = 1;
    wp::int32 var_44;
    wp::float32* var_45;
    const wp::int32 var_46 = 2;
    wp::int32 var_47;
    wp::float32* var_48;
    const wp::int32 var_49 = 3;
    wp::int32 var_50;
    wp::float32* var_51;
    wp::quat_t<wp::float32> var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::transform_t<wp::float32> var_57;
    wp::vec_t<3, wp::float32> var_58;
    wp::float32* var_59;
    const wp::int32 var_60 = 1;
    wp::int32 var_61;
    wp::float32* var_62;
    const wp::int32 var_63 = 2;
    wp::int32 var_64;
    wp::float32* var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::vec_t<6, wp::float32> var_70;
    wp::transform_t<wp::float32> var_71;
    wp::vec_t<6, wp::float32> var_72;
    bool var_73;
    const wp::int32 var_74 = 4;
    bool var_75;
    const wp::int32 var_76 = 5;
    bool var_77;
    wp::float32* var_78;
    const wp::int32 var_79 = 1;
    wp::int32 var_80;
    wp::float32* var_81;
    const wp::int32 var_82 = 2;
    wp::int32 var_83;
    wp::float32* var_84;
    wp::vec_t<3, wp::float32> var_85;
    wp::float32 var_86;
    wp::float32 var_87;
    wp::float32 var_88;
    const wp::int32 var_89 = 3;
    wp::int32 var_90;
    wp::float32* var_91;
    const wp::int32 var_92 = 4;
    wp::int32 var_93;
    wp::float32* var_94;
    const wp::int32 var_95 = 5;
    wp::int32 var_96;
    wp::float32* var_97;
    const wp::int32 var_98 = 6;
    wp::int32 var_99;
    wp::float32* var_100;
    wp::quat_t<wp::float32> var_101;
    wp::float32 var_102;
    wp::float32 var_103;
    wp::float32 var_104;
    wp::float32 var_105;
    wp::transform_t<wp::float32> var_106;
    wp::float32* var_107;
    const wp::int32 var_108 = 1;
    wp::int32 var_109;
    wp::float32* var_110;
    const wp::int32 var_111 = 2;
    wp::int32 var_112;
    wp::float32* var_113;
    wp::vec_t<3, wp::float32> var_114;
    wp::float32 var_115;
    wp::float32 var_116;
    wp::float32 var_117;
    const wp::int32 var_118 = 3;
    wp::int32 var_119;
    wp::float32* var_120;
    const wp::int32 var_121 = 4;
    wp::int32 var_122;
    wp::float32* var_123;
    const wp::int32 var_124 = 5;
    wp::int32 var_125;
    wp::float32* var_126;
    wp::vec_t<3, wp::float32> var_127;
    wp::float32 var_128;
    wp::float32 var_129;
    wp::float32 var_130;
    wp::vec_t<6, wp::float32> var_131;
    wp::transform_t<wp::float32> var_132;
    wp::vec_t<6, wp::float32> var_133;
    const wp::int32 var_134 = 6;
    bool var_135;
    const wp::float32 var_136 = 0.0;
    wp::vec_t<3, wp::float32> var_137;
    wp::quat_t<wp::float32> var_138;
    const wp::float32 var_139 = 0.0;
    wp::vec_t<3, wp::float32> var_140;
    const wp::float32 var_141 = 0.0;
    wp::vec_t<3, wp::float32> var_142;
    const wp::int32 var_143 = 0;
    bool var_144;
    wp::vec_t<3, wp::float32>* var_145;
    wp::vec_t<3, wp::float32> var_146;
    wp::vec_t<3, wp::float32> var_147;
    wp::float32* var_148;
    wp::vec_t<3, wp::float32> var_149;
    wp::float32 var_150;
    wp::vec_t<3, wp::float32> var_151;
    wp::float32* var_152;
    wp::vec_t<3, wp::float32> var_153;
    wp::float32 var_154;
    wp::vec_t<3, wp::float32> var_155;
    wp::vec_t<3, wp::float32> var_156;
    wp::vec_t<3, wp::float32> var_157;
    wp::vec_t<3, wp::float32> var_158;
    const wp::int32 var_159 = 1;
    bool var_160;
    const wp::int32 var_161 = 1;
    wp::int32 var_162;
    wp::vec_t<3, wp::float32>* var_163;
    wp::vec_t<3, wp::float32> var_164;
    wp::vec_t<3, wp::float32> var_165;
    const wp::int32 var_166 = 1;
    wp::int32 var_167;
    wp::float32* var_168;
    wp::vec_t<3, wp::float32> var_169;
    wp::float32 var_170;
    wp::vec_t<3, wp::float32> var_171;
    const wp::int32 var_172 = 1;
    wp::int32 var_173;
    wp::float32* var_174;
    wp::vec_t<3, wp::float32> var_175;
    wp::float32 var_176;
    wp::vec_t<3, wp::float32> var_177;
    wp::vec_t<3, wp::float32> var_178;
    wp::vec_t<3, wp::float32> var_179;
    wp::vec_t<3, wp::float32> var_180;
    const wp::int32 var_181 = 2;
    bool var_182;
    const wp::int32 var_183 = 2;
    wp::int32 var_184;
    wp::vec_t<3, wp::float32>* var_185;
    wp::vec_t<3, wp::float32> var_186;
    wp::vec_t<3, wp::float32> var_187;
    const wp::int32 var_188 = 2;
    wp::int32 var_189;
    wp::float32* var_190;
    wp::vec_t<3, wp::float32> var_191;
    wp::float32 var_192;
    wp::vec_t<3, wp::float32> var_193;
    const wp::int32 var_194 = 2;
    wp::int32 var_195;
    wp::float32* var_196;
    wp::vec_t<3, wp::float32> var_197;
    wp::float32 var_198;
    wp::vec_t<3, wp::float32> var_199;
    wp::vec_t<3, wp::float32> var_200;
    wp::vec_t<3, wp::float32> var_201;
    wp::vec_t<3, wp::float32> var_202;
    wp::int32 var_203;
    wp::int32 var_204;
    const wp::int32 var_205 = 1;
    bool var_206;
    wp::vec_t<3, wp::float32>* var_207;
    wp::vec_t<3, wp::float32> var_208;
    wp::vec_t<3, wp::float32> var_209;
    wp::float32* var_210;
    wp::quat_t<wp::float32> var_211;
    wp::float32 var_212;
    wp::float32* var_213;
    wp::vec_t<3, wp::float32> var_214;
    wp::float32 var_215;
    wp::vec_t<3, wp::float32> var_216;
    wp::quat_t<wp::float32> var_217;
    wp::vec_t<3, wp::float32> var_218;
    const wp::int32 var_219 = 2;
    bool var_220;
    wp::vec_t<3, wp::float32>* var_221;
    const wp::int32 var_222 = 1;
    wp::int32 var_223;
    wp::vec_t<3, wp::float32>* var_224;
    wp::float32* var_225;
    const wp::int32 var_226 = 1;
    wp::int32 var_227;
    wp::float32* var_228;
    wp::float32* var_229;
    const wp::int32 var_230 = 1;
    wp::int32 var_231;
    wp::float32* var_232;
    wp::quat_t<wp::float32> var_233;
    wp::vec_t<3, wp::float32> var_234;
    wp::vec_t<3, wp::float32> var_235;
    wp::vec_t<3, wp::float32> var_236;
    wp::float32 var_237;
    wp::float32 var_238;
    wp::float32 var_239;
    wp::float32 var_240;
    wp::quat_t<wp::float32> var_241;
    wp::vec_t<3, wp::float32> var_242;
    const wp::int32 var_243 = 3;
    bool var_244;
    wp::vec_t<3, wp::float32>* var_245;
    const wp::int32 var_246 = 1;
    wp::int32 var_247;
    wp::vec_t<3, wp::float32>* var_248;
    const wp::int32 var_249 = 2;
    wp::int32 var_250;
    wp::vec_t<3, wp::float32>* var_251;
    wp::float32* var_252;
    const wp::int32 var_253 = 1;
    wp::int32 var_254;
    wp::float32* var_255;
    const wp::int32 var_256 = 2;
    wp::int32 var_257;
    wp::float32* var_258;
    wp::float32* var_259;
    const wp::int32 var_260 = 1;
    wp::int32 var_261;
    wp::float32* var_262;
    const wp::int32 var_263 = 2;
    wp::int32 var_264;
    wp::float32* var_265;
    wp::quat_t<wp::float32> var_266;
    wp::vec_t<3, wp::float32> var_267;
    wp::vec_t<3, wp::float32> var_268;
    wp::vec_t<3, wp::float32> var_269;
    wp::vec_t<3, wp::float32> var_270;
    wp::float32 var_271;
    wp::float32 var_272;
    wp::float32 var_273;
    wp::float32 var_274;
    wp::float32 var_275;
    wp::float32 var_276;
    wp::quat_t<wp::float32> var_277;
    wp::vec_t<3, wp::float32> var_278;
    wp::transform_t<wp::float32> var_279;
    wp::vec_t<6, wp::float32> var_280;
    wp::transform_t<wp::float32> var_281;
    wp::vec_t<6, wp::float32> var_282;
    wp::vec_t<3, wp::float32> var_283;
    //---------
    // forward
    // def eval_joint_motion(                                                                 <L 237>
    // X_j = wp.transform_identity()                                                          <L 247>
    var_0 = wp::transform_identity<wp::float32>();
    // v_j = wp.spatial_vector(wp.vec3(), wp.vec3())                                          <L 248>
    var_1 = wp::vec_t<3, wp::float32>();
    var_2 = wp::vec_t<3, wp::float32>();
    var_3 = wp::vec_t<6, wp::float32>(var_1, var_2);
    // if type == JointType.PRISMATIC:                                                        <L 250>
    var_5 = (var_type == var_4);
    if (var_5) {
        // axis = joint_axis[qd_start]                                                        <L 251>
        var_6 = wp::address(var_joint_axis, var_qd_start);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // X_j = wp.transform(axis * joint_q[q_start], wp.quat_identity())                    <L 252>
        var_9 = wp::address(var_joint_q, var_q_start);
        var_11 = wp::load(var_9);
        var_10 = wp::mul(var_7, var_11);
        var_12 = wp::quat_identity<wp::float32>();
        var_13 = wp::transform_t<wp::float32>(var_10, var_12);
        // v_j = wp.spatial_vector(axis * joint_qd[qd_start], wp.vec3())                      <L 253>
        var_14 = wp::address(var_joint_qd, var_qd_start);
        var_16 = wp::load(var_14);
        var_15 = wp::mul(var_7, var_16);
        var_17 = wp::vec_t<3, wp::float32>();
        var_18 = wp::vec_t<6, wp::float32>(var_15, var_17);
    }
    var_19 = wp::where(var_5, var_13, var_0);
    var_20 = wp::where(var_5, var_18, var_3);
    // if type == JointType.REVOLUTE:                                                         <L 255>
    var_22 = (var_type == var_21);
    if (var_22) {
        // axis = joint_axis[qd_start]                                                        <L 256>
        var_23 = wp::address(var_joint_axis, var_qd_start);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // X_j = wp.transform(wp.vec3(), wp.quat_from_axis_angle(axis, joint_q[q_start]))       <L 257>
        var_26 = wp::vec_t<3, wp::float32>();
        var_27 = wp::address(var_joint_q, var_q_start);
        var_29 = wp::load(var_27);
        var_28 = wp::quat_from_axis_angle(var_24, var_29);
        var_30 = wp::transform_t<wp::float32>(var_26, var_28);
        // v_j = wp.spatial_vector(wp.vec3(), axis * joint_qd[qd_start])                      <L 258>
        var_31 = wp::vec_t<3, wp::float32>();
        var_32 = wp::address(var_joint_qd, var_qd_start);
        var_34 = wp::load(var_32);
        var_33 = wp::mul(var_24, var_34);
        var_35 = wp::vec_t<6, wp::float32>(var_31, var_33);
    }
    var_36 = wp::where(var_22, var_30, var_19);
    var_37 = wp::where(var_22, var_35, var_20);
    var_38 = wp::where(var_22, var_24, var_7);
    // if type == JointType.BALL:                                                             <L 260>
    var_40 = (var_type == var_39);
    if (var_40) {
        // X_j = wp.transform(                                                                <L 261>
        // wp.vec3(),                                                                         <L 262>
        var_41 = wp::vec_t<3, wp::float32>();
        // wp.quat(joint_q[q_start], joint_q[q_start + 1], joint_q[q_start + 2], joint_q[q_start + 3]),       <L 263>
        var_42 = wp::address(var_joint_q, var_q_start);
        var_44 = wp::add(var_q_start, var_43);
        var_45 = wp::address(var_joint_q, var_44);
        var_47 = wp::add(var_q_start, var_46);
        var_48 = wp::address(var_joint_q, var_47);
        var_50 = wp::add(var_q_start, var_49);
        var_51 = wp::address(var_joint_q, var_50);
        var_53 = wp::load(var_42);
        var_54 = wp::load(var_45);
        var_55 = wp::load(var_48);
        var_56 = wp::load(var_51);
        var_52 = wp::quat_t<wp::float32>(var_53, var_54, var_55, var_56);
        var_57 = wp::transform_t<wp::float32>(var_41, var_52);
        // v_j = wp.spatial_vector(                                                           <L 265>
        // wp.vec3(),                                                                         <L 266>
        var_58 = wp::vec_t<3, wp::float32>();
        // wp.vec3(joint_qd[qd_start], joint_qd[qd_start + 1], joint_qd[qd_start + 2]),       <L 267>
        var_59 = wp::address(var_joint_qd, var_qd_start);
        var_61 = wp::add(var_qd_start, var_60);
        var_62 = wp::address(var_joint_qd, var_61);
        var_64 = wp::add(var_qd_start, var_63);
        var_65 = wp::address(var_joint_qd, var_64);
        var_67 = wp::load(var_59);
        var_68 = wp::load(var_62);
        var_69 = wp::load(var_65);
        var_66 = wp::vec_t<3, wp::float32>(var_67, var_68, var_69);
        var_70 = wp::vec_t<6, wp::float32>(var_58, var_66);
    }
    var_71 = wp::where(var_40, var_57, var_36);
    var_72 = wp::where(var_40, var_70, var_37);
    // if type == JointType.FREE or type == JointType.DISTANCE:                               <L 270>
    var_75 = (var_type == var_74);
    var_73 = var_75;
    if (!var_73) {
        var_77 = (var_type == var_76);
        var_73 = var_73 || var_77;
    }
    if (var_73) {
        // X_j = wp.transform(                                                                <L 271>
        // wp.vec3(joint_q[q_start], joint_q[q_start + 1], joint_q[q_start + 2]),             <L 272>
        var_78 = wp::address(var_joint_q, var_q_start);
        var_80 = wp::add(var_q_start, var_79);
        var_81 = wp::address(var_joint_q, var_80);
        var_83 = wp::add(var_q_start, var_82);
        var_84 = wp::address(var_joint_q, var_83);
        var_86 = wp::load(var_78);
        var_87 = wp::load(var_81);
        var_88 = wp::load(var_84);
        var_85 = wp::vec_t<3, wp::float32>(var_86, var_87, var_88);
        // wp.quat(                                                                           <L 273>
        // joint_q[q_start + 3],                                                              <L 274>
        var_90 = wp::add(var_q_start, var_89);
        var_91 = wp::address(var_joint_q, var_90);
        // joint_q[q_start + 4],                                                              <L 275>
        var_93 = wp::add(var_q_start, var_92);
        var_94 = wp::address(var_joint_q, var_93);
        // joint_q[q_start + 5],                                                              <L 276>
        var_96 = wp::add(var_q_start, var_95);
        var_97 = wp::address(var_joint_q, var_96);
        // joint_q[q_start + 6],                                                              <L 277>
        var_99 = wp::add(var_q_start, var_98);
        var_100 = wp::address(var_joint_q, var_99);
        var_102 = wp::load(var_91);
        var_103 = wp::load(var_94);
        var_104 = wp::load(var_97);
        var_105 = wp::load(var_100);
        var_101 = wp::quat_t<wp::float32>(var_102, var_103, var_104, var_105);
        var_106 = wp::transform_t<wp::float32>(var_85, var_101);
        // v_j = wp.spatial_vector(                                                           <L 280>
        // wp.vec3(joint_qd[qd_start], joint_qd[qd_start + 1], joint_qd[qd_start + 2]),       <L 281>
        var_107 = wp::address(var_joint_qd, var_qd_start);
        var_109 = wp::add(var_qd_start, var_108);
        var_110 = wp::address(var_joint_qd, var_109);
        var_112 = wp::add(var_qd_start, var_111);
        var_113 = wp::address(var_joint_qd, var_112);
        var_115 = wp::load(var_107);
        var_116 = wp::load(var_110);
        var_117 = wp::load(var_113);
        var_114 = wp::vec_t<3, wp::float32>(var_115, var_116, var_117);
        // wp.vec3(joint_qd[qd_start + 3], joint_qd[qd_start + 4], joint_qd[qd_start + 5]),       <L 282>
        var_119 = wp::add(var_qd_start, var_118);
        var_120 = wp::address(var_joint_qd, var_119);
        var_122 = wp::add(var_qd_start, var_121);
        var_123 = wp::address(var_joint_qd, var_122);
        var_125 = wp::add(var_qd_start, var_124);
        var_126 = wp::address(var_joint_qd, var_125);
        var_128 = wp::load(var_120);
        var_129 = wp::load(var_123);
        var_130 = wp::load(var_126);
        var_127 = wp::vec_t<3, wp::float32>(var_128, var_129, var_130);
        var_131 = wp::vec_t<6, wp::float32>(var_114, var_127);
    }
    var_132 = wp::where(var_73, var_106, var_71);
    var_133 = wp::where(var_73, var_131, var_72);
    // if type == JointType.D6:                                                               <L 285>
    var_135 = (var_type == var_134);
    if (var_135) {
        // pos = wp.vec3(0.0)                                                                 <L 286>
        var_137 = wp::vec_t<3, wp::float32>(var_136);
        // rot = wp.quat_identity()                                                           <L 287>
        var_138 = wp::quat_identity<wp::float32>();
        // vel_v = wp.vec3(0.0)                                                               <L 288>
        var_140 = wp::vec_t<3, wp::float32>(var_139);
        // vel_w = wp.vec3(0.0)                                                               <L 289>
        var_142 = wp::vec_t<3, wp::float32>(var_141);
        // if lin_axis_count > 0:                                                             <L 292>
        var_144 = (var_lin_axis_count > var_143);
        if (var_144) {
            // axis = joint_axis[qd_start]                                                    <L 293>
            var_145 = wp::address(var_joint_axis, var_qd_start);
            var_147 = wp::load(var_145);
            var_146 = wp::copy(var_147);
            // pos += axis * joint_q[q_start]                                                 <L 294>
            var_148 = wp::address(var_joint_q, var_q_start);
            var_150 = wp::load(var_148);
            var_149 = wp::mul(var_146, var_150);
            var_151 = wp::add(var_137, var_149);
            // vel_v += axis * joint_qd[qd_start]                                             <L 295>
            var_152 = wp::address(var_joint_qd, var_qd_start);
            var_154 = wp::load(var_152);
            var_153 = wp::mul(var_146, var_154);
            var_155 = wp::add(var_140, var_153);
        }
        var_156 = wp::where(var_144, var_146, var_38);
        var_157 = wp::where(var_144, var_151, var_137);
        var_158 = wp::where(var_144, var_155, var_140);
        // if lin_axis_count > 1:                                                             <L 296>
        var_160 = (var_lin_axis_count > var_159);
        if (var_160) {
            // axis = joint_axis[qd_start + 1]                                                <L 297>
            var_162 = wp::add(var_qd_start, var_161);
            var_163 = wp::address(var_joint_axis, var_162);
            var_165 = wp::load(var_163);
            var_164 = wp::copy(var_165);
            // pos += axis * joint_q[q_start + 1]                                             <L 298>
            var_167 = wp::add(var_q_start, var_166);
            var_168 = wp::address(var_joint_q, var_167);
            var_170 = wp::load(var_168);
            var_169 = wp::mul(var_164, var_170);
            var_171 = wp::add(var_157, var_169);
            // vel_v += axis * joint_qd[qd_start + 1]                                         <L 299>
            var_173 = wp::add(var_qd_start, var_172);
            var_174 = wp::address(var_joint_qd, var_173);
            var_176 = wp::load(var_174);
            var_175 = wp::mul(var_164, var_176);
            var_177 = wp::add(var_158, var_175);
        }
        var_178 = wp::where(var_160, var_164, var_156);
        var_179 = wp::where(var_160, var_171, var_157);
        var_180 = wp::where(var_160, var_177, var_158);
        // if lin_axis_count > 2:                                                             <L 300>
        var_182 = (var_lin_axis_count > var_181);
        if (var_182) {
            // axis = joint_axis[qd_start + 2]                                                <L 301>
            var_184 = wp::add(var_qd_start, var_183);
            var_185 = wp::address(var_joint_axis, var_184);
            var_187 = wp::load(var_185);
            var_186 = wp::copy(var_187);
            // pos += axis * joint_q[q_start + 2]                                             <L 302>
            var_189 = wp::add(var_q_start, var_188);
            var_190 = wp::address(var_joint_q, var_189);
            var_192 = wp::load(var_190);
            var_191 = wp::mul(var_186, var_192);
            var_193 = wp::add(var_179, var_191);
            // vel_v += axis * joint_qd[qd_start + 2]                                         <L 303>
            var_195 = wp::add(var_qd_start, var_194);
            var_196 = wp::address(var_joint_qd, var_195);
            var_198 = wp::load(var_196);
            var_197 = wp::mul(var_186, var_198);
            var_199 = wp::add(var_180, var_197);
        }
        var_200 = wp::where(var_182, var_186, var_178);
        var_201 = wp::where(var_182, var_193, var_179);
        var_202 = wp::where(var_182, var_199, var_180);
        // iq = q_start + lin_axis_count                                                      <L 305>
        var_203 = wp::add(var_q_start, var_lin_axis_count);
        // iqd = qd_start + lin_axis_count                                                    <L 306>
        var_204 = wp::add(var_qd_start, var_lin_axis_count);
        // if ang_axis_count == 1:                                                            <L 307>
        var_206 = (var_ang_axis_count == var_205);
        if (var_206) {
            // axis = joint_axis[iqd]                                                         <L 308>
            var_207 = wp::address(var_joint_axis, var_204);
            var_209 = wp::load(var_207);
            var_208 = wp::copy(var_209);
            // rot = wp.quat_from_axis_angle(axis, joint_q[iq])                               <L 309>
            var_210 = wp::address(var_joint_q, var_203);
            var_212 = wp::load(var_210);
            var_211 = wp::quat_from_axis_angle(var_208, var_212);
            // vel_w = joint_qd[iqd] * axis                                                   <L 310>
            var_213 = wp::address(var_joint_qd, var_204);
            var_215 = wp::load(var_213);
            var_214 = wp::mul(var_215, var_208);
        }
        var_216 = wp::where(var_206, var_208, var_200);
        var_217 = wp::where(var_206, var_211, var_138);
        var_218 = wp::where(var_206, var_214, var_142);
        // if ang_axis_count == 2:                                                            <L 311>
        var_220 = (var_ang_axis_count == var_219);
        if (var_220) {
            // rot, vel_w = compute_2d_rotational_dofs(                                       <L 312>
            // joint_axis[iqd],                                                               <L 313>
            var_221 = wp::address(var_joint_axis, var_204);
            // joint_axis[iqd + 1],                                                           <L 314>
            var_223 = wp::add(var_204, var_222);
            var_224 = wp::address(var_joint_axis, var_223);
            // joint_q[iq],                                                                   <L 315>
            var_225 = wp::address(var_joint_q, var_203);
            // joint_q[iq + 1],                                                               <L 316>
            var_227 = wp::add(var_203, var_226);
            var_228 = wp::address(var_joint_q, var_227);
            // joint_qd[iqd],                                                                 <L 317>
            var_229 = wp::address(var_joint_qd, var_204);
            // joint_qd[iqd + 1],                                                             <L 318>
            var_231 = wp::add(var_204, var_230);
            var_232 = wp::address(var_joint_qd, var_231);
            var_235 = wp::load(var_221);
            var_236 = wp::load(var_224);
            var_237 = wp::load(var_225);
            var_238 = wp::load(var_228);
            var_239 = wp::load(var_229);
            var_240 = wp::load(var_232);
            compute_2d_rotational_dofs_0(var_235, var_236, var_237, var_238, var_239, var_240, var_233, var_234);
        }
        var_241 = wp::where(var_220, var_233, var_217);
        var_242 = wp::where(var_220, var_234, var_218);
        // if ang_axis_count == 3:                                                            <L 320>
        var_244 = (var_ang_axis_count == var_243);
        if (var_244) {
            // rot, vel_w = compute_3d_rotational_dofs(                                       <L 321>
            // joint_axis[iqd],                                                               <L 322>
            var_245 = wp::address(var_joint_axis, var_204);
            // joint_axis[iqd + 1],                                                           <L 323>
            var_247 = wp::add(var_204, var_246);
            var_248 = wp::address(var_joint_axis, var_247);
            // joint_axis[iqd + 2],                                                           <L 324>
            var_250 = wp::add(var_204, var_249);
            var_251 = wp::address(var_joint_axis, var_250);
            // joint_q[iq],                                                                   <L 325>
            var_252 = wp::address(var_joint_q, var_203);
            // joint_q[iq + 1],                                                               <L 326>
            var_254 = wp::add(var_203, var_253);
            var_255 = wp::address(var_joint_q, var_254);
            // joint_q[iq + 2],                                                               <L 327>
            var_257 = wp::add(var_203, var_256);
            var_258 = wp::address(var_joint_q, var_257);
            // joint_qd[iqd],                                                                 <L 328>
            var_259 = wp::address(var_joint_qd, var_204);
            // joint_qd[iqd + 1],                                                             <L 329>
            var_261 = wp::add(var_204, var_260);
            var_262 = wp::address(var_joint_qd, var_261);
            // joint_qd[iqd + 2],                                                             <L 330>
            var_264 = wp::add(var_204, var_263);
            var_265 = wp::address(var_joint_qd, var_264);
            var_268 = wp::load(var_245);
            var_269 = wp::load(var_248);
            var_270 = wp::load(var_251);
            var_271 = wp::load(var_252);
            var_272 = wp::load(var_255);
            var_273 = wp::load(var_258);
            var_274 = wp::load(var_259);
            var_275 = wp::load(var_262);
            var_276 = wp::load(var_265);
            compute_3d_rotational_dofs_0(var_268, var_269, var_270, var_271, var_272, var_273, var_274, var_275, var_276, var_266, var_267);
        }
        var_277 = wp::where(var_244, var_266, var_241);
        var_278 = wp::where(var_244, var_267, var_242);
        // X_j = wp.transform(pos, rot)                                                       <L 333>
        var_279 = wp::transform_t<wp::float32>(var_201, var_277);
        // v_j = wp.spatial_vector(vel_v, vel_w)                                              <L 334>
        var_280 = wp::vec_t<6, wp::float32>(var_202, var_278);
    }
    var_281 = wp::where(var_135, var_279, var_132);
    var_282 = wp::where(var_135, var_280, var_133);
    var_283 = wp::where(var_135, var_216, var_38);
    // return X_j, v_j                                                                        <L 336>
    ret_0 = var_281;
    ret_1 = var_282;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/warp/_src/math.py:131
static CUDA_CALLABLE wp::vec_t<3, wp::float32> velocity_at_point_0(
    wp::vec_t<6, wp::float32> var_qd,
    wp::vec_t<3, wp::float32> var_r)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    //---------
    // forward
    // def velocity_at_point(qd: wp.spatial_vector, r: wp.vec3) -> wp.vec3:                   <L 132>
    // return wp.spatial_bottom(qd) + wp.cross(wp.spatial_top(qd), r)                         <L 148>
    var_0 = wp::spatial_bottom(var_qd);
    var_1 = wp::spatial_top(var_qd);
    var_2 = wp::cross(var_1, var_r);
    var_3 = wp::add(var_0, var_2);
    return var_3;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/math/spatial.py:53
static CUDA_CALLABLE wp::vec_t<3, wp::float32> velocity_at_point_1(
    wp::vec_t<6, wp::float32> var_qd,
    wp::vec_t<3, wp::float32> var_r)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<6, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    //---------
    // forward
    // def velocity_at_point(qd: wp.spatial_vector, r: wp.vec3) -> wp.vec3:                   <L 54>
    // qd_wp = wp.spatial_vector(wp.spatial_bottom(qd), wp.spatial_top(qd))                   <L 77>
    var_0 = wp::spatial_bottom(var_qd);
    var_1 = wp::spatial_top(var_qd);
    var_2 = wp::vec_t<6, wp::float32>(var_0, var_1);
    // return wp.velocity_at_point(qd_wp, r)                                                  <L 78>
    var_3 = velocity_at_point_0(var_2, var_r);
    return var_3;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/sim/articulation.py:14
static CUDA_CALLABLE wp::vec_t<3, wp::float32> com_twist_to_point_velocity_0(
    wp::vec_t<6, wp::float32> var_qd,
    wp::transform_t<wp::float32> var_X_wb,
    wp::vec_t<3, wp::float32> var_body_com,
    wp::vec_t<3, wp::float32> var_point)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    //---------
    // forward
    // def com_twist_to_point_velocity(qd: wp.spatial_vector, X_wb: wp.transform, body_com: wp.vec3, point: wp.vec3):       <L 15>
    // return velocity_at_point(qd, point - wp.transform_point(X_wb, body_com))               <L 17>
    var_0 = wp::transform_point(var_X_wb, var_body_com);
    var_1 = wp::sub(var_point, var_0);
    var_2 = velocity_at_point_1(var_qd, var_1);
    return var_2;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/sim/articulation.py:28
static CUDA_CALLABLE wp::vec_t<6, wp::float32> com_twist_to_origin_twist_0(
    wp::vec_t<6, wp::float32> var_qd,
    wp::transform_t<wp::float32> var_X_wb,
    wp::vec_t<3, wp::float32> var_body_com)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<6, wp::float32> var_5;
    //---------
    // forward
    // def com_twist_to_origin_twist(qd: wp.spatial_vector, X_wb: wp.transform, body_com: wp.vec3):       <L 29>
    // omega = wp.spatial_bottom(qd)                                                          <L 31>
    var_0 = wp::spatial_bottom(var_qd);
    // v_origin = wp.spatial_top(qd) - wp.cross(omega, wp.transform_vector(X_wb, body_com))       <L 32>
    var_1 = wp::spatial_top(var_qd);
    var_2 = wp::transform_vector(var_X_wb, var_body_com);
    var_3 = wp::cross(var_0, var_2);
    var_4 = wp::sub(var_1, var_3);
    // return wp.spatial_vector(v_origin, omega)                                              <L 33>
    var_5 = wp::vec_t<6, wp::float32>(var_4, var_0);
    return var_5;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/sim/articulation.py:20
static CUDA_CALLABLE wp::vec_t<6, wp::float32> origin_twist_to_com_twist_0(
    wp::vec_t<6, wp::float32> var_qd,
    wp::transform_t<wp::float32> var_X_wb,
    wp::vec_t<3, wp::float32> var_body_com)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<6, wp::float32> var_3;
    //---------
    // forward
    // def origin_twist_to_com_twist(qd: wp.spatial_vector, X_wb: wp.transform, body_com: wp.vec3):       <L 21>
    // omega = wp.spatial_bottom(qd)                                                          <L 23>
    var_0 = wp::spatial_bottom(var_qd);
    // v_com = velocity_at_point(qd, wp.transform_vector(X_wb, body_com))                     <L 24>
    var_1 = wp::transform_vector(var_X_wb, var_body_com);
    var_2 = velocity_at_point_1(var_qd, var_1);
    // return wp.spatial_vector(v_com, omega)                                                 <L 25>
    var_3 = wp::vec_t<6, wp::float32>(var_2, var_0);
    return var_3;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/sim/articulation.py:339
static CUDA_CALLABLE void eval_joint_child_state_0(
    wp::int32 var_type,
    wp::int32 var_parent,
    wp::int32 var_child,
    wp::transform_t<wp::float32> var_X_wp,
    wp::transform_t<wp::float32> var_X_pj,
    wp::transform_t<wp::float32> var_X_cj,
    wp::transform_t<wp::float32> var_X_j,
    wp::vec_t<6, wp::float32> var_v_j,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_qd,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com,
    wp::transform_t<wp::float32> & ret_0,
    wp::vec_t<6, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::transform_t<wp::float32> var_0;
    const wp::int32 var_1 = 0;
    bool var_2;
    wp::transform_t<wp::float32> var_3;
    wp::transform_t<wp::float32> var_4;
    wp::transform_t<wp::float32> var_5;
    wp::transform_t<wp::float32> var_6;
    wp::transform_t<wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    const wp::int32 var_11 = 0;
    bool var_12;
    wp::vec_t<6, wp::float32>* var_13;
    wp::vec_t<6, wp::float32> var_14;
    wp::vec_t<6, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::float32>* var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    bool var_26;
    const wp::int32 var_27 = 4;
    bool var_28;
    const wp::int32 var_29 = 5;
    bool var_30;
    wp::vec_t<6, wp::float32> var_31;
    wp::vec_t<3, wp::float32>* var_32;
    wp::vec_t<6, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::vec_t<3, wp::float32> var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::vec_t<3, wp::float32> var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::vec_t<6, wp::float32> var_43;
    wp::vec_t<3, wp::float32>* var_44;
    wp::vec_t<6, wp::float32> var_45;
    wp::vec_t<3, wp::float32> var_46;
    //---------
    // forward
    // def eval_joint_child_state(                                                            <L 340>
    // X_wpj = X_pj                                                                           <L 352>
    var_0 = wp::copy(var_X_pj);
    // if parent >= 0:                                                                        <L 353>
    var_2 = (var_parent >= var_1);
    if (var_2) {
        // X_wpj = X_wp * X_pj                                                                <L 354>
        var_3 = wp::mul(var_X_wp, var_X_pj);
    }
    var_4 = wp::where(var_2, var_3, var_0);
    // X_wcj = X_wpj * X_j                                                                    <L 356>
    var_5 = wp::mul(var_4, var_X_j);
    // X_wc = X_wcj * wp.transform_inverse(X_cj)                                              <L 357>
    var_6 = wp::transform_inverse(var_X_cj);
    var_7 = wp::mul(var_5, var_6);
    // x_child_origin = wp.transform_get_translation(X_wc)                                    <L 358>
    var_8 = wp::transform_get_translation(var_7);
    // v_parent_origin = wp.vec3()                                                            <L 360>
    var_9 = wp::vec_t<3, wp::float32>();
    // w_parent = wp.vec3()                                                                   <L 361>
    var_10 = wp::vec_t<3, wp::float32>();
    // if parent >= 0:                                                                        <L 362>
    var_12 = (var_parent >= var_11);
    if (var_12) {
        // v_wp = body_qd[parent]                                                             <L 363>
        var_13 = wp::address(var_body_qd, var_parent);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // w_parent = wp.spatial_bottom(v_wp)                                                 <L 364>
        var_16 = wp::spatial_bottom(var_14);
        // v_parent_origin = com_twist_to_point_velocity(v_wp, X_wp, body_com[parent], x_child_origin)       <L 365>
        var_17 = wp::address(var_body_com, var_parent);
        var_19 = wp::load(var_17);
        var_18 = com_twist_to_point_velocity_0(var_14, var_X_wp, var_19, var_8);
    }
    var_20 = wp::where(var_12, var_18, var_9);
    var_21 = wp::where(var_12, var_16, var_10);
    // linear_joint_world = wp.transform_vector(X_wpj, wp.spatial_top(v_j))                   <L 367>
    var_22 = wp::spatial_top(var_v_j);
    var_23 = wp::transform_vector(var_4, var_22);
    // angular_joint_world = wp.transform_vector(X_wpj, wp.spatial_bottom(v_j))               <L 368>
    var_24 = wp::spatial_bottom(var_v_j);
    var_25 = wp::transform_vector(var_4, var_24);
    // if type == JointType.FREE or type == JointType.DISTANCE:                               <L 369>
    var_28 = (var_type == var_27);
    var_26 = var_28;
    if (!var_26) {
        var_30 = (var_type == var_29);
        var_26 = var_26 || var_30;
    }
    if (var_26) {
        // v_joint_origin = com_twist_to_origin_twist(                                        <L 371>
        // wp.spatial_vector(linear_joint_world, angular_joint_world),                        <L 372>
        var_31 = wp::vec_t<6, wp::float32>(var_23, var_25);
        // X_wc,                                                                              <L 373>
        // body_com[child],                                                                   <L 374>
        var_32 = wp::address(var_body_com, var_child);
        var_34 = wp::load(var_32);
        var_33 = com_twist_to_origin_twist_0(var_31, var_7, var_34);
        // linear_joint_origin = wp.spatial_top(v_joint_origin)                               <L 376>
        var_35 = wp::spatial_top(var_33);
    }
    if (!var_26) {
        // child_origin_offset_world = x_child_origin - wp.transform_get_translation(X_wcj)       <L 378>
        var_36 = wp::transform_get_translation(var_5);
        var_37 = wp::sub(var_8, var_36);
        // linear_joint_origin = linear_joint_world + wp.cross(angular_joint_world, child_origin_offset_world)       <L 379>
        var_38 = wp::cross(var_25, var_37);
        var_39 = wp::add(var_23, var_38);
    }
    var_40 = wp::where(var_26, var_35, var_39);
    // v_wc_origin = wp.spatial_vector(                                                       <L 381>
    // v_parent_origin + linear_joint_origin,                                                 <L 382>
    var_41 = wp::add(var_20, var_40);
    // w_parent + angular_joint_world,                                                        <L 383>
    var_42 = wp::add(var_21, var_25);
    var_43 = wp::vec_t<6, wp::float32>(var_41, var_42);
    // return X_wc, origin_twist_to_com_twist(v_wc_origin, X_wc, body_com[child])             <L 385>
    var_44 = wp::address(var_body_com, var_child);
    var_46 = wp::load(var_44);
    var_45 = origin_twist_to_com_twist_0(var_43, var_7, var_46);
    ret_0 = var_7;
    ret_1 = var_45;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/sim/articulation.py:36
static CUDA_CALLABLE void adj_transform_2d_rotational_axes_0(
    wp::vec_t<3, wp::float32> var_axis_0,
    wp::vec_t<3, wp::float32> var_axis_1,
    wp::float32 var_q0,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_axis_0,
    wp::vec_t<3, wp::float32> & adj_axis_1,
    wp::float32 & adj_q0,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/sim/articulation.py:61
static CUDA_CALLABLE void adj_compute_2d_rotational_dofs_0(
    wp::vec_t<3, wp::float32> var_axis_0,
    wp::vec_t<3, wp::float32> var_axis_1,
    wp::float32 var_q0,
    wp::float32 var_q1,
    wp::float32 var_qd0,
    wp::float32 var_qd1,
    wp::quat_t<wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_axis_0,
    wp::vec_t<3, wp::float32> & adj_axis_1,
    wp::float32 & adj_q0,
    wp::float32 & adj_q1,
    wp::float32 & adj_qd0,
    wp::float32 & adj_qd1,
    wp::quat_t<wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/sim/articulation.py:127
static CUDA_CALLABLE void adj_transform_3d_rotational_axes_0(
    wp::vec_t<3, wp::float32> var_axis_0,
    wp::vec_t<3, wp::float32> var_axis_1,
    wp::vec_t<3, wp::float32> var_axis_2,
    wp::float32 var_q0,
    wp::float32 var_q1,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_axis_0,
    wp::vec_t<3, wp::float32> & adj_axis_1,
    wp::vec_t<3, wp::float32> & adj_axis_2,
    wp::float32 & adj_q0,
    wp::float32 & adj_q1,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/sim/articulation.py:151
static CUDA_CALLABLE void adj_compute_3d_rotational_dofs_0(
    wp::vec_t<3, wp::float32> var_axis_0,
    wp::vec_t<3, wp::float32> var_axis_1,
    wp::vec_t<3, wp::float32> var_axis_2,
    wp::float32 var_q0,
    wp::float32 var_q1,
    wp::float32 var_q2,
    wp::float32 var_qd0,
    wp::float32 var_qd1,
    wp::float32 var_qd2,
    wp::quat_t<wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_axis_0,
    wp::vec_t<3, wp::float32> & adj_axis_1,
    wp::vec_t<3, wp::float32> & adj_axis_2,
    wp::float32 & adj_q0,
    wp::float32 & adj_q1,
    wp::float32 & adj_q2,
    wp::float32 & adj_qd0,
    wp::float32 & adj_qd1,
    wp::float32 & adj_qd2,
    wp::quat_t<wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/sim/articulation.py:236
static CUDA_CALLABLE void adj_eval_joint_motion_0(
    wp::int32 var_type,
    wp::int32 var_q_start,
    wp::int32 var_qd_start,
    wp::int32 var_lin_axis_count,
    wp::int32 var_ang_axis_count,
    wp::array_t<wp::float32> var_joint_q,
    wp::array_t<wp::float32> var_joint_qd,
    wp::array_t<wp::vec_t<3, wp::float32>> var_joint_axis,
    wp::transform_t<wp::float32> & ret_0,
    wp::vec_t<6, wp::float32> & ret_1,
    wp::int32 & adj_type,
    wp::int32 & adj_q_start,
    wp::int32 & adj_qd_start,
    wp::int32 & adj_lin_axis_count,
    wp::int32 & adj_ang_axis_count,
    wp::array_t<wp::float32> & adj_joint_q,
    wp::array_t<wp::float32> & adj_joint_qd,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_joint_axis,
    wp::transform_t<wp::float32> & adj_ret_0,
    wp::vec_t<6, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/warp/_src/math.py:131
static CUDA_CALLABLE void adj_velocity_at_point_0(
    wp::vec_t<6, wp::float32> var_qd,
    wp::vec_t<3, wp::float32> var_r,
    wp::vec_t<6, wp::float32> & adj_qd,
    wp::vec_t<3, wp::float32> & adj_r,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/math/spatial.py:53
static CUDA_CALLABLE void adj_velocity_at_point_1(
    wp::vec_t<6, wp::float32> var_qd,
    wp::vec_t<3, wp::float32> var_r,
    wp::vec_t<6, wp::float32> & adj_qd,
    wp::vec_t<3, wp::float32> & adj_r,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/sim/articulation.py:14
static CUDA_CALLABLE void adj_com_twist_to_point_velocity_0(
    wp::vec_t<6, wp::float32> var_qd,
    wp::transform_t<wp::float32> var_X_wb,
    wp::vec_t<3, wp::float32> var_body_com,
    wp::vec_t<3, wp::float32> var_point,
    wp::vec_t<6, wp::float32> & adj_qd,
    wp::transform_t<wp::float32> & adj_X_wb,
    wp::vec_t<3, wp::float32> & adj_body_com,
    wp::vec_t<3, wp::float32> & adj_point,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/sim/articulation.py:28
static CUDA_CALLABLE void adj_com_twist_to_origin_twist_0(
    wp::vec_t<6, wp::float32> var_qd,
    wp::transform_t<wp::float32> var_X_wb,
    wp::vec_t<3, wp::float32> var_body_com,
    wp::vec_t<6, wp::float32> & adj_qd,
    wp::transform_t<wp::float32> & adj_X_wb,
    wp::vec_t<3, wp::float32> & adj_body_com,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/sim/articulation.py:20
static CUDA_CALLABLE void adj_origin_twist_to_com_twist_0(
    wp::vec_t<6, wp::float32> var_qd,
    wp::transform_t<wp::float32> var_X_wb,
    wp::vec_t<3, wp::float32> var_body_com,
    wp::vec_t<6, wp::float32> & adj_qd,
    wp::transform_t<wp::float32> & adj_X_wb,
    wp::vec_t<3, wp::float32> & adj_body_com,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/sim/articulation.py:339
static CUDA_CALLABLE void adj_eval_joint_child_state_0(
    wp::int32 var_type,
    wp::int32 var_parent,
    wp::int32 var_child,
    wp::transform_t<wp::float32> var_X_wp,
    wp::transform_t<wp::float32> var_X_pj,
    wp::transform_t<wp::float32> var_X_cj,
    wp::transform_t<wp::float32> var_X_j,
    wp::vec_t<6, wp::float32> var_v_j,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_qd,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com,
    wp::transform_t<wp::float32> & ret_0,
    wp::vec_t<6, wp::float32> & ret_1,
    wp::int32 & adj_type,
    wp::int32 & adj_parent,
    wp::int32 & adj_child,
    wp::transform_t<wp::float32> & adj_X_wp,
    wp::transform_t<wp::float32> & adj_X_pj,
    wp::transform_t<wp::float32> & adj_X_cj,
    wp::transform_t<wp::float32> & adj_X_j,
    wp::vec_t<6, wp::float32> & adj_v_j,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_body_qd,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_body_com,
    wp::transform_t<wp::float32> & adj_ret_0,
    wp::vec_t<6, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void create_eval_articulation_fk_tile__locals__eval_articulation_fk_tile_fddaa244_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_articulation_level_start,
    wp::array_t<wp::int32> var_level_joint_start,
    wp::array_t<wp::int32> var_level_joints,
    wp::array_t<wp::int32> var_level_parent_pos,
    wp::int32 var_articulation_count,
    wp::array_t<bool> var_articulation_mask,
    wp::array_t<wp::int32> var_articulation_indices,
    wp::array_t<wp::float32> var_joint_q,
    wp::array_t<wp::float32> var_joint_qd,
    wp::array_t<wp::int32> var_joint_q_start,
    wp::array_t<wp::int32> var_joint_qd_start,
    wp::array_t<wp::int32> var_joint_type,
    wp::array_t<wp::int32> var_joint_parent,
    wp::array_t<wp::int32> var_joint_child,
    wp::array_t<wp::transform_t<wp::float32>> var_joint_X_p,
    wp::array_t<wp::transform_t<wp::float32>> var_joint_X_c,
    wp::array_t<wp::vec_t<3, wp::float32>> var_joint_axis,
    wp::array_t<wp::int32> var_joint_dof_dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com,
    wp::array_t<wp::int32> var_body_flags,
    wp::int32 var_body_flag_filter,
    wp::array_t<wp::transform_t<wp::float32>> var_body_q,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_qd)
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
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        bool var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        bool var_9;
        wp::int32 var_10;
        bool var_11;
        bool* var_12;
        bool var_13;
        bool var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        const wp::int32 var_18 = 1;
        wp::int32 var_19;
        wp::int32* var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        const wp::int32 var_23 = 12;
        const wp::str var_24 = "shared";
        wp::tile_shared_t<wp::transform_t<wp::float32>,wp::tile_layout_strided_t<wp::tile_shape_t<12>, wp::tile_stride_t<1>>, true> var_25 = wp::tile_alloc_empty<wp::transform_t<wp::float32>,wp::tile_shape_t<12>,wp::tile_stride_t<1>,false>();
        wp::range_t var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        const wp::int32 var_32 = 1;
        wp::int32 var_33;
        wp::int32* var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        const wp::int32 var_37 = 1;
        wp::int32 var_38;
        const wp::int32 var_39 = 6;
        wp::int32 var_40;
        const wp::int32 var_41 = 1;
        const wp::int32 var_42 = 1;
        wp::int32 var_43;
        wp::int32 var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        wp::int32 var_47;
        const wp::int32 var_48 = 1;
        wp::int32 var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        wp::range_t var_52;
        wp::int32 var_53;
        wp::int32 var_54;
        wp::int32 var_55;
        wp::int32 var_56;
        bool var_57;
        const wp::int32 var_58 = 0;
        wp::int32 var_59;
        wp::transform_t<wp::float32> var_60;
        wp::int32 var_61;
        wp::int32 var_62;
        wp::int32* var_63;
        wp::int32 var_64;
        wp::int32 var_65;
        wp::int32* var_66;
        wp::int32 var_67;
        wp::int32 var_68;
        wp::int32* var_69;
        wp::int32 var_70;
        wp::int32 var_71;
        const bool var_72 = true;
        bool var_73;
        const bool var_74 = false;
        const bool var_75 = false;
        bool var_76;
        wp::int32* var_77;
        wp::int32 var_78;
        wp::int32 var_79;
        wp::int32 var_80;
        wp::int32* var_81;
        wp::int32 var_82;
        wp::int32 var_83;
        wp::transform_t<wp::float32>* var_84;
        wp::transform_t<wp::float32> var_85;
        wp::transform_t<wp::float32> var_86;
        wp::transform_t<wp::float32>* var_87;
        wp::transform_t<wp::float32> var_88;
        wp::transform_t<wp::float32> var_89;
        wp::int32* var_90;
        wp::int32 var_91;
        wp::int32 var_92;
        wp::int32* var_93;
        wp::int32 var_94;
        wp::int32 var_95;
        const wp::int32 var_96 = 0;
        wp::int32* var_97;
        wp::int32 var_98;
        wp::int32 var_99;
        const wp::int32 var_100 = 1;
        wp::int32* var_101;
        wp::int32 var_102;
        wp::int32 var_103;
        wp::transform_t<wp::float32> var_104;
        wp::vec_t<6, wp::float32> var_105;
        wp::transform_t<wp::float32> var_106;
        const wp::int32 var_107 = 0;
        bool var_108;
        wp::int32 var_109;
        const wp::int32 var_110 = 0;
        const wp::int32 var_111 = 12;
        bool var_112;
        wp::int32 var_113;
        wp::int32 var_114;
        wp::transform_t<wp::float32> var_115;
        const wp::int32 var_116 = 0;
        bool var_117;
        wp::transform_t<wp::float32>* var_118;
        wp::transform_t<wp::float32> var_119;
        wp::transform_t<wp::float32> var_120;
        wp::transform_t<wp::float32> var_121;
        wp::transform_t<wp::float32> var_122;
        wp::transform_t<wp::float32> var_123;
        wp::vec_t<6, wp::float32> var_124;
        const bool var_125 = true;
        bool var_126;
        const bool var_127 = false;
        wp::transform_t<wp::float32> var_128;
        bool var_129;
        const bool var_130 = false;
        wp::int32 var_131;
        wp::transform_t<wp::float32> var_132;
        //---------
        // forward
        // def eval_articulation_fk_tile(                                                         <L 25>
        // block, thread = wp.tid()                                                               <L 50>
        builtin_tid2d(var_0, var_1);
        // articulation = block                                                                   <L 52>
        var_2 = wp::copy(var_0);
        // if articulation_indices:                                                               <L 53>
        if (var_articulation_indices) {
            // articulation = articulation_indices[block]                                         <L 54>
            var_3 = wp::address(var_articulation_indices, var_0);
            var_5 = wp::load(var_3);
            var_4 = wp::copy(var_5);
            // if articulation < 0 or articulation >= articulation_count:                         <L 55>
            var_8 = (var_4 < var_7);
            var_6 = var_8;
            if (!var_6) {
                var_9 = (var_4 >= var_articulation_count);
                var_6 = var_6 || var_9;
            }
            if (var_6) {
                // return                                                                         <L 56>
                continue;
            }
        }
        var_10 = wp::where(var_articulation_indices, var_4, var_2);
        // if articulation_mask and not articulation_mask[articulation]:                          <L 58>
        var_11 = var_articulation_mask;
        if (var_11) {
            var_12 = wp::address(var_articulation_mask, var_10);
            var_14 = wp::load(var_12);
            var_13 = wp::unot(var_14);
            var_11 = var_11 && var_13;
        }
        if (var_11) {
            // return                                                                             <L 59>
            continue;
        }
        // level_begin = articulation_level_start[articulation]                                   <L 61>
        var_15 = wp::address(var_articulation_level_start, var_10);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // level_count = articulation_level_start[articulation + 1] - level_begin                 <L 62>
        var_19 = wp::add(var_10, var_18);
        var_20 = wp::address(var_articulation_level_start, var_19);
        var_22 = wp::load(var_20);
        var_21 = wp::sub(var_22, var_16);
        // body_q_work = wp.tile_zeros(shape=buffer_capacity, dtype=wp.transform, storage="shared")       <L 67>
        var_25 = wp::tile_zeros<wp::transform_t<wp::float32>, 12>();
        // for local_level in range(level_count):                                                 <L 69>
        var_26 = wp::range(var_21);
        start_for_2:;
            if (iter_cmp(var_26) == 0) goto end_for_2;
            var_27 = wp::iter_next(var_26);
            // level = level_begin + local_level                                                  <L 70>
            var_28 = wp::add(var_16, var_27);
            // level_joint_begin = level_joint_start[level]                                       <L 71>
            var_29 = wp::address(var_level_joint_start, var_28);
            var_31 = wp::load(var_29);
            var_30 = wp::copy(var_31);
            // level_joint_count = level_joint_start[level + 1] - level_joint_begin               <L 72>
            var_33 = wp::add(var_28, var_32);
            var_34 = wp::address(var_level_joint_start, var_33);
            var_36 = wp::load(var_34);
            var_35 = wp::sub(var_36, var_30);
            // cur_offset = (local_level & 1) * level_capacity                                    <L 73>
            var_38 = wp::bit_and(var_27, var_37);
            var_40 = wp::mul(var_38, var_39);
            // prev_offset = (1 - (local_level & 1)) * level_capacity                             <L 74>
            var_43 = wp::bit_and(var_27, var_42);
            var_44 = wp::sub(var_41, var_43);
            var_45 = wp::mul(var_44, var_39);
            // chunk_count = (level_joint_count + wp.block_dim() - 1) // wp.block_dim()           <L 75>
            var_46 = builtin_block_dim();
            var_47 = wp::add(var_35, var_46);
            var_49 = wp::sub(var_47, var_48);
            var_50 = builtin_block_dim();
            var_51 = wp::floordiv(var_49, var_50);
            // for chunk in range(chunk_count):                                                   <L 77>
            var_52 = wp::range(var_51);
            start_for_4:;
                if (iter_cmp(var_52) == 0) goto end_for_4;
                var_53 = wp::iter_next(var_52);
                // scheduled_joint = chunk * wp.block_dim() + thread                              <L 78>
                var_54 = builtin_block_dim();
                var_55 = wp::mul(var_53, var_54);
                var_56 = wp::add(var_55, var_1);
                // active = scheduled_joint < level_joint_count                                   <L 79>
                var_57 = (var_56 < var_35);
                // child_slot = int(0)                                                            <L 80>
                var_59 = wp::int(var_58);
                // X_wc = wp.transform_identity()                                                 <L 81>
                var_60 = wp::transform_identity<wp::float32>();
                // if active:                                                                     <L 83>
                if (var_57) {
                    // child_slot = cur_offset + scheduled_joint                                  <L 84>
                    var_61 = wp::add(var_40, var_56);
                    // joint = level_joints[level_joint_begin + scheduled_joint]                  <L 85>
                    var_62 = wp::add(var_30, var_56);
                    var_63 = wp::address(var_level_joints, var_62);
                    var_65 = wp::load(var_63);
                    var_64 = wp::copy(var_65);
                    // type = joint_type[joint]                                                   <L 86>
                    var_66 = wp::address(var_joint_type, var_64);
                    var_68 = wp::load(var_66);
                    var_67 = wp::copy(var_68);
                    // child = joint_child[joint]                                                 <L 87>
                    var_69 = wp::address(var_joint_child, var_64);
                    var_71 = wp::load(var_69);
                    var_70 = wp::copy(var_71);
                    // evaluate_joint = bool(True)                                                <L 88>
                    var_73 = bool(var_72);
                    // if wp.static(has_rod):                                                     <L 89>
                    // write_child = bool(False)                                                  <L 92>
                    var_76 = bool(var_75);
                    // if evaluate_joint:                                                         <L 93>
                    if (var_73) {
                        // parent = joint_parent[joint]                                           <L 94>
                        var_77 = wp::address(var_joint_parent, var_64);
                        var_79 = wp::load(var_77);
                        var_78 = wp::copy(var_79);
                        // parent_pos = level_parent_pos[level_joint_begin + scheduled_joint]       <L 95>
                        var_80 = wp::add(var_30, var_56);
                        var_81 = wp::address(var_level_parent_pos, var_80);
                        var_83 = wp::load(var_81);
                        var_82 = wp::copy(var_83);
                        // X_pj = joint_X_p[joint]                                                <L 97>
                        var_84 = wp::address(var_joint_X_p, var_64);
                        var_86 = wp::load(var_84);
                        var_85 = wp::copy(var_86);
                        // X_cj = joint_X_c[joint]                                                <L 98>
                        var_87 = wp::address(var_joint_X_c, var_64);
                        var_89 = wp::load(var_87);
                        var_88 = wp::copy(var_89);
                        // q_start = joint_q_start[joint]                                         <L 99>
                        var_90 = wp::address(var_joint_q_start, var_64);
                        var_92 = wp::load(var_90);
                        var_91 = wp::copy(var_92);
                        // qd_start = joint_qd_start[joint]                                       <L 100>
                        var_93 = wp::address(var_joint_qd_start, var_64);
                        var_95 = wp::load(var_93);
                        var_94 = wp::copy(var_95);
                        // lin_axis_count = joint_dof_dim[joint, 0]                               <L 101>
                        var_97 = wp::address(var_joint_dof_dim, var_64, var_96);
                        var_99 = wp::load(var_97);
                        var_98 = wp::copy(var_99);
                        // ang_axis_count = joint_dof_dim[joint, 1]                               <L 102>
                        var_101 = wp::address(var_joint_dof_dim, var_64, var_100);
                        var_103 = wp::load(var_101);
                        var_102 = wp::copy(var_103);
                        // X_j, v_j = eval_joint_motion(                                          <L 103>
                        // type,                                                                  <L 104>
                        // q_start,                                                               <L 105>
                        // qd_start,                                                              <L 106>
                        // lin_axis_count,                                                        <L 107>
                        // ang_axis_count,                                                        <L 108>
                        // joint_q,                                                               <L 109>
                        // joint_qd,                                                              <L 110>
                        // joint_axis,                                                            <L 111>
                        eval_joint_motion_0(var_67, var_91, var_94, var_98, var_102, var_joint_q, var_joint_qd, var_joint_axis, var_104, var_105);
                        // X_wp = wp.transform_identity()                                         <L 114>
                        var_106 = wp::transform_identity<wp::float32>();
                        // if parent_pos >= 0:                                                    <L 115>
                        var_108 = (var_82 >= var_107);
                        if (var_108) {
                            // X_wp = body_q_work[prev_offset + parent_pos]                       <L 116>
                            var_109 = wp::add(var_45, var_82);
                            var_112 = (var_109 < var_110);
                            var_113 = wp::add(var_109, var_111);
                            var_114 = wp::where(var_112, var_113, var_109);
                            var_115 = wp::tile_extract(var_25, var_114);
                        }
                        if (!var_108) {
                            // elif parent >= 0:                                                  <L 117>
                            var_117 = (var_78 >= var_116);
                            if (var_117) {
                                // X_wp = body_q[parent]                                          <L 118>
                                var_118 = wp::address(var_body_q, var_78);
                                var_120 = wp::load(var_118);
                                var_119 = wp::copy(var_120);
                            }
                            var_121 = wp::where(var_117, var_119, var_106);
                        }
                        var_122 = wp::where(var_108, var_115, var_121);
                        // X_wc, v_wc = eval_joint_child_state(                                   <L 119>
                        // type,                                                                  <L 120>
                        // parent,                                                                <L 121>
                        // child,                                                                 <L 122>
                        // X_wp,                                                                  <L 123>
                        // X_pj,                                                                  <L 124>
                        // X_cj,                                                                  <L 125>
                        // X_j,                                                                   <L 126>
                        // v_j,                                                                   <L 127>
                        // body_qd,                                                               <L 128>
                        // body_com,                                                              <L 129>
                        eval_joint_child_state_0(var_67, var_78, var_70, var_122, var_85, var_88, var_104, var_105, var_body_qd, var_body_com, var_123, var_124);
                        // write_child = bool(True)                                               <L 132>
                        var_126 = bool(var_125);
                        // if wp.static(not write_all):                                           <L 133>
                        // if write_child:                                                        <L 135>
                        if (var_126) {
                            // body_q[child] = X_wc                                               <L 136>
                            wp::array_store(var_body_q, var_70, var_123);
                            // body_qd[child] = v_wc                                              <L 137>
                            wp::array_store(var_body_qd, var_70, var_124);
                        }
                    }
                    var_128 = wp::where(var_73, var_123, var_60);
                    var_129 = wp::where(var_73, var_126, var_76);
                    // if wp.static(preserve_body_q):                                             <L 139>
                }
                var_131 = wp::where(var_57, var_61, var_59);
                var_132 = wp::where(var_57, var_128, var_60);
                // wp.tile_scatter_masked(body_q_work, child_slot, X_wc, active)                  <L 144>
                wp::tile_scatter_masked(var_25, var_131, var_132, var_57);
                goto start_for_4;
            end_for_4:;
            goto start_for_2;
        end_for_2:;
    }
}

