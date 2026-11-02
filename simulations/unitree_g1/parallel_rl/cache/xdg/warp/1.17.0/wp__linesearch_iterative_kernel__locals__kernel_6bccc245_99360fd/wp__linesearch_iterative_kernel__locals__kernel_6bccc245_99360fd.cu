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


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:125
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _eval_pt_direct_alpha_zero_0(
    wp::float32 var_jaref,
    wp::float32 var_jv,
    wp::float32 var_d)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 0.5;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::vec_t<3, wp::float32> var_7;
    //---------
    // forward
    // def _eval_pt_direct_alpha_zero(jaref: float, jv: float, d: float) -> wp.vec3:          <L 126>
    // jvD = jv * d                                                                           <L 128>
    var_0 = wp::mul(var_jv, var_d);
    // return wp.vec3(0.5 * d * jaref * jaref, jvD * jaref, jv * jvD)                         <L 129>
    var_2 = wp::mul(var_1, var_d);
    var_3 = wp::mul(var_2, var_jaref);
    var_4 = wp::mul(var_3, var_jaref);
    var_5 = wp::mul(var_0, var_jaref);
    var_6 = wp::mul(var_jv, var_0);
    var_7 = wp::vec_t<3, wp::float32>(var_4, var_5, var_6);
    return var_7;
}


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


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:226
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _eval_frictionloss_pt_0(
    wp::float32 var_x,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_jv,
    wp::float32 var_d)
{
    //---------
    // primal vars
    bool var_0;
    wp::float32 var_1;
    bool var_2;
    bool var_3;
    wp::float32 var_4;
    const wp::float32 var_5 = 0.5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::float32 var_12;
    bool var_13;
    const wp::float32 var_14 = -0.5;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    const wp::float32 var_20 = 0.0;
    wp::vec_t<3, wp::float32> var_21;
    const wp::float32 var_22 = -0.5;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    const wp::float32 var_27 = 0.0;
    wp::vec_t<3, wp::float32> var_28;
    //---------
    // forward
    // def _eval_frictionloss_pt(x: float, f: float, rf: float, jv: float, d: float) -> wp.vec3:       <L 227>
    // if (-rf < x) and (x < rf):                                                             <L 229>
    var_1 = wp::neg(var_rf);
    var_2 = (var_1 < var_x);
    var_0 = var_2;
    if (var_0) {
        var_3 = (var_x < var_rf);
        var_0 = var_0 && var_3;
    }
    if (var_0) {
        // jvD = jv * d                                                                       <L 230>
        var_4 = wp::mul(var_jv, var_d);
        // return wp.vec3(0.5 * d * x * x, jvD * x, jv * jvD)                                 <L 231>
        var_6 = wp::mul(var_5, var_d);
        var_7 = wp::mul(var_6, var_x);
        var_8 = wp::mul(var_7, var_x);
        var_9 = wp::mul(var_4, var_x);
        var_10 = wp::mul(var_jv, var_4);
        var_11 = wp::vec_t<3, wp::float32>(var_8, var_9, var_10);
        return var_11;
    }
    if (!var_0) {
        // elif x <= -rf:                                                                     <L 232>
        var_12 = wp::neg(var_rf);
        var_13 = (var_x <= var_12);
        if (var_13) {
            // return wp.vec3(f * (-0.5 * rf - x), -f * jv, 0.0)                              <L 233>
            var_15 = wp::mul(var_14, var_rf);
            var_16 = wp::sub(var_15, var_x);
            var_17 = wp::mul(var_f, var_16);
            var_18 = wp::neg(var_f);
            var_19 = wp::mul(var_18, var_jv);
            var_21 = wp::vec_t<3, wp::float32>(var_17, var_19, var_20);
            return var_21;
        }
        if (!var_13) {
            // return wp.vec3(f * (-0.5 * rf + x), f * jv, 0.0)                               <L 235>
            var_23 = wp::mul(var_22, var_rf);
            var_24 = wp::add(var_23, var_x);
            var_25 = wp::mul(var_f, var_24);
            var_26 = wp::mul(var_f, var_jv);
            var_28 = wp::vec_t<3, wp::float32>(var_25, var_26, var_27);
            return var_28;
        }
    }
    return {};
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:577
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _compute_efc_eval_pt_alpha_zero_pyramidal_0(
    wp::int32 var_efcid,
    wp::int32 var_ne,
    wp::int32 var_nf,
    wp::float32 var_efc_D,
    wp::array_t<wp::float32> var_efc_frictionloss,
    wp::float32 var_ctx_Jaref,
    wp::float32 var_ctx_jv)
{
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    bool var_3;
    wp::vec_t<3, wp::float32> var_4;
    const wp::float32 var_5 = 0.0;
    wp::vec_t<3, wp::float32> var_6;
    bool var_7;
    wp::float32* var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    //---------
    // forward
    // def _compute_efc_eval_pt_alpha_zero_pyramidal(                                         <L 578>
    // if efcid >= ne + nf:                                                                   <L 590>
    var_0 = wp::add(var_ne, var_nf);
    var_1 = (var_efcid >= var_0);
    if (var_1) {
        // if ctx_Jaref < 0.0:                                                                <L 591>
        var_3 = (var_ctx_Jaref < var_2);
        if (var_3) {
            // return _eval_pt_direct_alpha_zero(ctx_Jaref, ctx_jv, efc_D)                    <L 592>
            var_4 = _eval_pt_direct_alpha_zero_0(var_ctx_Jaref, var_ctx_jv, var_efc_D);
            return var_4;
        }
        // return wp.vec3(0.0)                                                                <L 593>
        var_6 = wp::vec_t<3, wp::float32>(var_5);
        return var_6;
    }
    // if efcid >= ne:                                                                        <L 596>
    var_7 = (var_efcid >= var_ne);
    if (var_7) {
        // f = efc_frictionloss[efcid]                                                        <L 597>
        var_8 = wp::address(var_efc_frictionloss, var_efcid);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // rf = math.safe_div(f, efc_D)                                                       <L 598>
        var_11 = safe_div_2(var_9, var_efc_D);
        // return _eval_frictionloss_pt(ctx_Jaref, f, rf, ctx_jv, efc_D)                      <L 599>
        var_12 = _eval_frictionloss_pt_0(var_ctx_Jaref, var_9, var_11, var_ctx_jv, var_efc_D);
        return var_12;
    }
    // return _eval_pt_direct_alpha_zero(ctx_Jaref, ctx_jv, efc_D)                            <L 602>
    var_13 = _eval_pt_direct_alpha_zero_0(var_ctx_Jaref, var_ctx_jv, var_efc_D);
    return var_13;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:140
static CUDA_CALLABLE wp::float32 _eval_pt_direct_cost_alpha_zero_0(
    wp::float32 var_jaref,
    wp::float32 var_d)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.5;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    //---------
    // forward
    // def _eval_pt_direct_cost_alpha_zero(jaref: float, d: float) -> float:                  <L 141>
    // return 0.5 * d * jaref * jaref                                                         <L 142>
    var_1 = wp::mul(var_0, var_d);
    var_2 = wp::mul(var_1, var_jaref);
    var_3 = wp::mul(var_2, var_jaref);
    return var_3;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:145
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _eval_pt_direct_shifted_0(
    wp::float32 var_jaref,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::float32 var_alpha,
    wp::float32 var_offset)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    const wp::float32 var_4 = 0.5;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::vec_t<3, wp::float32> var_11;
    //---------
    // forward
    // def _eval_pt_direct_shifted(jaref: float, jv: float, d: float, alpha: float, offset: float) -> wp.vec3:       <L 146>
    // jvD = jv * d                                                                           <L 148>
    var_0 = wp::mul(var_jv, var_d);
    // hessian = jv * jvD                                                                     <L 149>
    var_1 = wp::mul(var_jv, var_0);
    // alpha_h = alpha * hessian                                                              <L 150>
    var_2 = wp::mul(var_alpha, var_1);
    // return wp.vec3(alpha * (jvD * jaref + 0.5 * alpha_h) + offset, jvD * jaref + alpha_h, hessian)       <L 151>
    var_3 = wp::mul(var_0, var_jaref);
    var_5 = wp::mul(var_4, var_2);
    var_6 = wp::add(var_3, var_5);
    var_7 = wp::mul(var_alpha, var_6);
    var_8 = wp::add(var_7, var_offset);
    var_9 = wp::mul(var_0, var_jaref);
    var_10 = wp::add(var_9, var_2);
    var_11 = wp::vec_t<3, wp::float32>(var_8, var_10, var_1);
    return var_11;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:238
static CUDA_CALLABLE wp::float32 _eval_frictionloss_cost_0(
    wp::float32 var_x,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_d)
{
    //---------
    // primal vars
    bool var_0;
    wp::float32 var_1;
    bool var_2;
    bool var_3;
    const wp::float32 var_4 = 0.5;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    bool var_9;
    const wp::float32 var_10 = -0.5;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    const wp::float32 var_14 = -0.5;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    //---------
    // forward
    // def _eval_frictionloss_cost(x: float, f: float, rf: float, d: float) -> float:         <L 239>
    // if (-rf < x) and (x < rf):                                                             <L 240>
    var_1 = wp::neg(var_rf);
    var_2 = (var_1 < var_x);
    var_0 = var_2;
    if (var_0) {
        var_3 = (var_x < var_rf);
        var_0 = var_0 && var_3;
    }
    if (var_0) {
        // return 0.5 * d * x * x                                                             <L 241>
        var_5 = wp::mul(var_4, var_d);
        var_6 = wp::mul(var_5, var_x);
        var_7 = wp::mul(var_6, var_x);
        return var_7;
    }
    if (!var_0) {
        // elif x <= -rf:                                                                     <L 242>
        var_8 = wp::neg(var_rf);
        var_9 = (var_x <= var_8);
        if (var_9) {
            // return f * (-0.5 * rf - x)                                                     <L 243>
            var_11 = wp::mul(var_10, var_rf);
            var_12 = wp::sub(var_11, var_x);
            var_13 = wp::mul(var_f, var_12);
            return var_13;
        }
    }
    // return f * (-0.5 * rf + x)                                                             <L 244>
    var_15 = wp::mul(var_14, var_rf);
    var_16 = wp::add(var_15, var_x);
    var_17 = wp::mul(var_f, var_16);
    return var_17;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:221
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _shift_cost_0(
    wp::vec_t<3, wp::float32> var_pt,
    wp::float32 var_cost0)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::int32 var_3 = 1;
    wp::float32 var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    wp::vec_t<3, wp::float32> var_7;
    //---------
    // forward
    // def _shift_cost(pt: wp.vec3, cost0: float) -> wp.vec3:                                 <L 222>
    // return wp.vec3(pt[0] - cost0, pt[1], pt[2])                                            <L 223>
    var_1 = wp::extract(var_pt, var_0);
    var_2 = wp::sub(var_1, var_cost0);
    var_4 = wp::extract(var_pt, var_3);
    var_6 = wp::extract(var_pt, var_5);
    var_7 = wp::vec_t<3, wp::float32>(var_2, var_4, var_6);
    return var_7;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:481
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _compute_efc_eval_pt_pyramidal_0(
    wp::int32 var_efcid,
    wp::float32 var_alpha,
    wp::int32 var_ne,
    wp::int32 var_nf,
    wp::float32 var_efc_D,
    wp::array_t<wp::float32> var_efc_frictionloss,
    wp::float32 var_ctx_Jaref,
    wp::float32 var_ctx_jv)
{
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::float32 var_5 = 0.0;
    bool var_6;
    const wp::float32 var_7 = 0.0;
    wp::float32 var_8;
    wp::float32 var_9;
    const wp::float32 var_10 = 0.0;
    bool var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::float32 var_13;
    const wp::float32 var_14 = 0.0;
    const wp::float32 var_15 = 0.0;
    wp::vec_t<3, wp::float32> var_16;
    bool var_17;
    wp::float32* var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::float32 var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::float32 var_27;
    const wp::float32 var_28 = 0.0;
    wp::vec_t<3, wp::float32> var_29;
    //---------
    // forward
    // def _compute_efc_eval_pt_pyramidal(                                                    <L 482>
    // if efcid >= ne + nf:                                                                   <L 498>
    var_0 = wp::add(var_ne, var_nf);
    var_1 = (var_efcid >= var_0);
    if (var_1) {
        // x = ctx_Jaref + alpha * ctx_jv                                                     <L 499>
        var_2 = wp::mul(var_alpha, var_ctx_jv);
        var_3 = wp::add(var_ctx_Jaref, var_2);
        // quad0 = _eval_pt_direct_cost_alpha_zero(ctx_Jaref, efc_D)                          <L 500>
        var_4 = _eval_pt_direct_cost_alpha_zero_0(var_ctx_Jaref, var_efc_D);
        // cost0 = wp.where(ctx_Jaref < 0.0, quad0, 0.0)                                      <L 501>
        var_6 = (var_ctx_Jaref < var_5);
        var_8 = wp::where(var_6, var_4, var_7);
        // offset = quad0 - cost0                                                             <L 504>
        var_9 = wp::sub(var_4, var_8);
        // if x < 0.0:                                                                        <L 505>
        var_11 = (var_3 < var_10);
        if (var_11) {
            // return _eval_pt_direct_shifted(ctx_Jaref, ctx_jv, efc_D, alpha, offset)        <L 506>
            var_12 = _eval_pt_direct_shifted_0(var_ctx_Jaref, var_ctx_jv, var_efc_D, var_alpha, var_9);
            return var_12;
        }
        // return wp.vec3(-cost0, 0.0, 0.0)                                                   <L 507>
        var_13 = wp::neg(var_8);
        var_16 = wp::vec_t<3, wp::float32>(var_13, var_14, var_15);
        return var_16;
    }
    // if efcid >= ne:                                                                        <L 510>
    var_17 = (var_efcid >= var_ne);
    if (var_17) {
        // f = efc_frictionloss[efcid]                                                        <L 511>
        var_18 = wp::address(var_efc_frictionloss, var_efcid);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // x = ctx_Jaref + alpha * ctx_jv                                                     <L 512>
        var_21 = wp::mul(var_alpha, var_ctx_jv);
        var_22 = wp::add(var_ctx_Jaref, var_21);
        // rf = math.safe_div(f, efc_D)                                                       <L 513>
        var_23 = safe_div_2(var_19, var_efc_D);
        // return _shift_cost(_eval_frictionloss_pt(x, f, rf, ctx_jv, efc_D), _eval_frictionloss_cost(ctx_Jaref, f, rf, efc_D))       <L 514>
        var_24 = _eval_frictionloss_pt_0(var_22, var_19, var_23, var_ctx_jv, var_efc_D);
        var_25 = _eval_frictionloss_cost_0(var_ctx_Jaref, var_19, var_23, var_efc_D);
        var_26 = _shift_cost_0(var_24, var_25);
        return var_26;
    }
    var_27 = wp::where(var_17, var_22, var_3);
    // return _eval_pt_direct_shifted(ctx_Jaref, ctx_jv, efc_D, alpha, 0.0)                   <L 517>
    var_29 = _eval_pt_direct_shifted_0(var_ctx_Jaref, var_ctx_jv, var_efc_D, var_alpha, var_28);
    return var_29;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:195
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _eval_pt_0(
    wp::vec_t<3, wp::float32> var_quad,
    wp::float32 var_alpha)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 2;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    const wp::int32 var_4 = 1;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::int32 var_8 = 0;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::float32 var_11 = 2.0;
    wp::float32 var_12;
    const wp::int32 var_13 = 1;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::float32 var_16 = 2.0;
    const wp::int32 var_17 = 2;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    //---------
    // forward
    // def _eval_pt(quad: wp.vec3, alpha: float) -> wp.vec3:                                  <L 196>
    // aq2 = alpha * quad[2]                                                                  <L 198>
    var_1 = wp::extract(var_quad, var_0);
    var_2 = wp::mul(var_alpha, var_1);
    // return wp.vec3(                                                                        <L 199>
    // alpha * aq2 + alpha * quad[1] + quad[0],                                               <L 200>
    var_3 = wp::mul(var_alpha, var_2);
    var_5 = wp::extract(var_quad, var_4);
    var_6 = wp::mul(var_alpha, var_5);
    var_7 = wp::add(var_3, var_6);
    var_9 = wp::extract(var_quad, var_8);
    var_10 = wp::add(var_7, var_9);
    // 2.0 * aq2 + quad[1],                                                                   <L 201>
    var_12 = wp::mul(var_11, var_2);
    var_14 = wp::extract(var_quad, var_13);
    var_15 = wp::add(var_12, var_14);
    // 2.0 * quad[2],                                                                         <L 202>
    var_18 = wp::extract(var_quad, var_17);
    var_19 = wp::mul(var_16, var_18);
    var_20 = wp::vec_t<3, wp::float32>(var_10, var_15, var_19);
    return var_20;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:172
static CUDA_CALLABLE void _eval_pt_direct_shifted_3alphas_0(
    wp::float32 var_jaref,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::float32 var_lo_alpha,
    wp::float32 var_hi_alpha,
    wp::float32 var_mid_alpha,
    wp::float32 var_offset,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::float32 var_6 = 0.5;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::vec_t<3, wp::float32> var_12;
    const wp::float32 var_13 = 0.5;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::vec_t<3, wp::float32> var_19;
    const wp::float32 var_20 = 0.5;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::vec_t<3, wp::float32> var_26;
    //---------
    // forward
    // def _eval_pt_direct_shifted_3alphas(                                                   <L 173>
    // jvD = jv * d                                                                           <L 177>
    var_0 = wp::mul(var_jv, var_d);
    // grad0 = jvD * jaref                                                                    <L 178>
    var_1 = wp::mul(var_0, var_jaref);
    // hessian = jv * jvD                                                                     <L 179>
    var_2 = wp::mul(var_jv, var_0);
    // lo_ah = lo_alpha * hessian                                                             <L 180>
    var_3 = wp::mul(var_lo_alpha, var_2);
    // hi_ah = hi_alpha * hessian                                                             <L 181>
    var_4 = wp::mul(var_hi_alpha, var_2);
    // mid_ah = mid_alpha * hessian                                                           <L 182>
    var_5 = wp::mul(var_mid_alpha, var_2);
    // return (                                                                               <L 183>
    // wp.vec3(lo_alpha * (grad0 + 0.5 * lo_ah) + offset, grad0 + lo_ah, hessian),            <L 184>
    var_7 = wp::mul(var_6, var_3);
    var_8 = wp::add(var_1, var_7);
    var_9 = wp::mul(var_lo_alpha, var_8);
    var_10 = wp::add(var_9, var_offset);
    var_11 = wp::add(var_1, var_3);
    var_12 = wp::vec_t<3, wp::float32>(var_10, var_11, var_2);
    // wp.vec3(hi_alpha * (grad0 + 0.5 * hi_ah) + offset, grad0 + hi_ah, hessian),            <L 185>
    var_14 = wp::mul(var_13, var_4);
    var_15 = wp::add(var_1, var_14);
    var_16 = wp::mul(var_hi_alpha, var_15);
    var_17 = wp::add(var_16, var_offset);
    var_18 = wp::add(var_1, var_4);
    var_19 = wp::vec_t<3, wp::float32>(var_17, var_18, var_2);
    // wp.vec3(mid_alpha * (grad0 + 0.5 * mid_ah) + offset, grad0 + mid_ah, hessian),         <L 186>
    var_21 = wp::mul(var_20, var_5);
    var_22 = wp::add(var_1, var_21);
    var_23 = wp::mul(var_mid_alpha, var_22);
    var_24 = wp::add(var_23, var_offset);
    var_25 = wp::add(var_1, var_5);
    var_26 = wp::vec_t<3, wp::float32>(var_24, var_25, var_2);
    ret_0 = var_12;
    ret_1 = var_19;
    ret_2 = var_26;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:247
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _eval_frictionloss_pt_one_0(
    wp::float32 var_x,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_half_d,
    wp::float32 var_jvD,
    wp::float32 var_hessian,
    wp::float32 var_f_jv)
{
    //---------
    // primal vars
    bool var_0;
    wp::float32 var_1;
    bool var_2;
    bool var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::float32 var_8;
    bool var_9;
    const wp::float32 var_10 = -0.5;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::float32 var_15 = 0.0;
    wp::vec_t<3, wp::float32> var_16;
    const wp::float32 var_17 = -0.5;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    const wp::float32 var_21 = 0.0;
    wp::vec_t<3, wp::float32> var_22;
    //---------
    // forward
    // def _eval_frictionloss_pt_one(x: float, f: float, rf: float, half_d: float, jvD: float, hessian: float, f_jv: float) -> wp.vec3:       <L 248>
    // if (-rf < x) and (x < rf):                                                             <L 250>
    var_1 = wp::neg(var_rf);
    var_2 = (var_1 < var_x);
    var_0 = var_2;
    if (var_0) {
        var_3 = (var_x < var_rf);
        var_0 = var_0 && var_3;
    }
    if (var_0) {
        // return wp.vec3(half_d * x * x, jvD * x, hessian)                                   <L 251>
        var_4 = wp::mul(var_half_d, var_x);
        var_5 = wp::mul(var_4, var_x);
        var_6 = wp::mul(var_jvD, var_x);
        var_7 = wp::vec_t<3, wp::float32>(var_5, var_6, var_hessian);
        return var_7;
    }
    if (!var_0) {
        // elif x <= -rf:                                                                     <L 252>
        var_8 = wp::neg(var_rf);
        var_9 = (var_x <= var_8);
        if (var_9) {
            // return wp.vec3(f * (-0.5 * rf - x), -f_jv, 0.0)                                <L 253>
            var_11 = wp::mul(var_10, var_rf);
            var_12 = wp::sub(var_11, var_x);
            var_13 = wp::mul(var_f, var_12);
            var_14 = wp::neg(var_f_jv);
            var_16 = wp::vec_t<3, wp::float32>(var_13, var_14, var_15);
            return var_16;
        }
        if (!var_9) {
            // return wp.vec3(f * (-0.5 * rf + x), f_jv, 0.0)                                 <L 255>
            var_18 = wp::mul(var_17, var_rf);
            var_19 = wp::add(var_18, var_x);
            var_20 = wp::mul(var_f, var_19);
            var_22 = wp::vec_t<3, wp::float32>(var_20, var_f_jv, var_21);
            return var_22;
        }
    }
    return {};
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:258
static CUDA_CALLABLE void _eval_frictionloss_pt_3alphas_0(
    wp::float32 var_x_lo,
    wp::float32 var_x_hi,
    wp::float32 var_x_mid,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 0.5;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    //---------
    // forward
    // def _eval_frictionloss_pt_3alphas(                                                     <L 259>
    // jvD = jv * d                                                                           <L 263>
    var_0 = wp::mul(var_jv, var_d);
    // half_d = 0.5 * d                                                                       <L 264>
    var_2 = wp::mul(var_1, var_d);
    // hessian = jv * jvD                                                                     <L 265>
    var_3 = wp::mul(var_jv, var_0);
    // f_jv = f * jv                                                                          <L 266>
    var_4 = wp::mul(var_f, var_jv);
    // return (                                                                               <L 267>
    // _eval_frictionloss_pt_one(x_lo, f, rf, half_d, jvD, hessian, f_jv),                    <L 268>
    var_5 = _eval_frictionloss_pt_one_0(var_x_lo, var_f, var_rf, var_2, var_0, var_3, var_4);
    // _eval_frictionloss_pt_one(x_hi, f, rf, half_d, jvD, hessian, f_jv),                    <L 269>
    var_6 = _eval_frictionloss_pt_one_0(var_x_hi, var_f, var_rf, var_2, var_0, var_3, var_4);
    // _eval_frictionloss_pt_one(x_mid, f, rf, half_d, jvD, hessian, f_jv),                   <L 270>
    var_7 = _eval_frictionloss_pt_one_0(var_x_mid, var_f, var_rf, var_2, var_0, var_3, var_4);
    ret_0 = var_5;
    ret_1 = var_6;
    ret_2 = var_7;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:649
static CUDA_CALLABLE void _compute_efc_eval_pt_3alphas_pyramidal_0(
    wp::int32 var_efcid,
    wp::float32 var_lo_alpha,
    wp::float32 var_hi_alpha,
    wp::float32 var_mid_alpha,
    wp::int32 var_ne,
    wp::int32 var_nf,
    wp::float32 var_efc_D,
    wp::array_t<wp::float32> var_efc_frictionloss,
    wp::float32 var_ctx_Jaref,
    wp::float32 var_ctx_jv,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    const wp::float32 var_9 = 0.0;
    bool var_10;
    const wp::float32 var_11 = 0.0;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::float32 var_17;
    const wp::float32 var_18 = 0.0;
    const wp::float32 var_19 = 0.0;
    wp::vec_t<3, wp::float32> var_20;
    const wp::float32 var_21 = 0.0;
    bool var_22;
    wp::vec_t<3, wp::float32> var_23;
    const wp::float32 var_24 = 0.0;
    bool var_25;
    wp::vec_t<3, wp::float32> var_26;
    const wp::float32 var_27 = 0.0;
    bool var_28;
    wp::vec_t<3, wp::float32> var_29;
    bool var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::float32 var_36;
    wp::float32* var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::vec_t<3, wp::float32> var_45;
    wp::vec_t<3, wp::float32> var_46;
    wp::vec_t<3, wp::float32> var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::float32 var_52 = 0.0;
    wp::vec_t<3, wp::float32> var_53;
    wp::vec_t<3, wp::float32> var_54;
    wp::vec_t<3, wp::float32> var_55;
    //---------
    // forward
    // def _compute_efc_eval_pt_3alphas_pyramidal(                                            <L 650>
    // if efcid >= ne + nf:                                                                   <L 669>
    var_0 = wp::add(var_ne, var_nf);
    var_1 = (var_efcid >= var_0);
    if (var_1) {
        // x_lo = ctx_Jaref + lo_alpha * ctx_jv                                               <L 670>
        var_2 = wp::mul(var_lo_alpha, var_ctx_jv);
        var_3 = wp::add(var_ctx_Jaref, var_2);
        // x_hi = ctx_Jaref + hi_alpha * ctx_jv                                               <L 671>
        var_4 = wp::mul(var_hi_alpha, var_ctx_jv);
        var_5 = wp::add(var_ctx_Jaref, var_4);
        // x_mid = ctx_Jaref + mid_alpha * ctx_jv                                             <L 672>
        var_6 = wp::mul(var_mid_alpha, var_ctx_jv);
        var_7 = wp::add(var_ctx_Jaref, var_6);
        // quad0 = _eval_pt_direct_cost_alpha_zero(ctx_Jaref, efc_D)                          <L 673>
        var_8 = _eval_pt_direct_cost_alpha_zero_0(var_ctx_Jaref, var_efc_D);
        // cost0 = wp.where(ctx_Jaref < 0.0, quad0, 0.0)                                      <L 674>
        var_10 = (var_ctx_Jaref < var_9);
        var_12 = wp::where(var_10, var_8, var_11);
        // offset = quad0 - cost0                                                             <L 676>
        var_13 = wp::sub(var_8, var_12);
        // pt_lo, pt_hi, pt_mid = _eval_pt_direct_shifted_3alphas(ctx_Jaref, ctx_jv, efc_D, lo_alpha, hi_alpha, mid_alpha, offset)       <L 677>
        _eval_pt_direct_shifted_3alphas_0(var_ctx_Jaref, var_ctx_jv, var_efc_D, var_lo_alpha, var_hi_alpha, var_mid_alpha, var_13, var_14, var_15, var_16);
        // inactive = wp.vec3(-cost0, 0.0, 0.0)                                               <L 678>
        var_17 = wp::neg(var_12);
        var_20 = wp::vec_t<3, wp::float32>(var_17, var_18, var_19);
        // return (                                                                           <L 679>
        // wp.where(x_lo < 0.0, pt_lo, inactive),                                             <L 680>
        var_22 = (var_3 < var_21);
        var_23 = wp::where(var_22, var_14, var_20);
        // wp.where(x_hi < 0.0, pt_hi, inactive),                                             <L 681>
        var_25 = (var_5 < var_24);
        var_26 = wp::where(var_25, var_15, var_20);
        // wp.where(x_mid < 0.0, pt_mid, inactive),                                           <L 682>
        var_28 = (var_7 < var_27);
        var_29 = wp::where(var_28, var_16, var_20);
        ret_0 = var_23;
        ret_1 = var_26;
        ret_2 = var_29;
        return;
    }
    // if efcid >= ne:                                                                        <L 686>
    var_30 = (var_efcid >= var_ne);
    if (var_30) {
        // x_lo = ctx_Jaref + lo_alpha * ctx_jv                                               <L 687>
        var_31 = wp::mul(var_lo_alpha, var_ctx_jv);
        var_32 = wp::add(var_ctx_Jaref, var_31);
        // x_hi = ctx_Jaref + hi_alpha * ctx_jv                                               <L 688>
        var_33 = wp::mul(var_hi_alpha, var_ctx_jv);
        var_34 = wp::add(var_ctx_Jaref, var_33);
        // x_mid = ctx_Jaref + mid_alpha * ctx_jv                                             <L 689>
        var_35 = wp::mul(var_mid_alpha, var_ctx_jv);
        var_36 = wp::add(var_ctx_Jaref, var_35);
        // f = efc_frictionloss[efcid]                                                        <L 690>
        var_37 = wp::address(var_efc_frictionloss, var_efcid);
        var_39 = wp::load(var_37);
        var_38 = wp::copy(var_39);
        // rf = math.safe_div(f, efc_D)                                                       <L 691>
        var_40 = safe_div_2(var_38, var_efc_D);
        // cost0 = _eval_frictionloss_cost(ctx_Jaref, f, rf, efc_D)                           <L 692>
        var_41 = _eval_frictionloss_cost_0(var_ctx_Jaref, var_38, var_40, var_efc_D);
        // lo, hi, mid = _eval_frictionloss_pt_3alphas(x_lo, x_hi, x_mid, f, rf, ctx_jv, efc_D)       <L 693>
        _eval_frictionloss_pt_3alphas_0(var_32, var_34, var_36, var_38, var_40, var_ctx_jv, var_efc_D, var_42, var_43, var_44);
        // return (_shift_cost(lo, cost0), _shift_cost(hi, cost0), _shift_cost(mid, cost0))       <L 694>
        var_45 = _shift_cost_0(var_42, var_41);
        var_46 = _shift_cost_0(var_43, var_41);
        var_47 = _shift_cost_0(var_44, var_41);
        ret_0 = var_45;
        ret_1 = var_46;
        ret_2 = var_47;
        return;
    }
    var_48 = wp::where(var_30, var_32, var_3);
    var_49 = wp::where(var_30, var_34, var_5);
    var_50 = wp::where(var_30, var_36, var_7);
    var_51 = wp::where(var_30, var_41, var_12);
    // return _eval_pt_direct_shifted_3alphas(ctx_Jaref, ctx_jv, efc_D, lo_alpha, hi_alpha, mid_alpha, 0.0)       <L 697>
    _eval_pt_direct_shifted_3alphas_0(var_ctx_Jaref, var_ctx_jv, var_efc_D, var_lo_alpha, var_hi_alpha, var_mid_alpha, var_52, var_53, var_54, var_55);
    ret_0 = var_53;
    ret_1 = var_54;
    ret_2 = var_55;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:206
static CUDA_CALLABLE void _eval_pt_3alphas_0(
    wp::vec_t<3, wp::float32> var_quad,
    wp::float32 var_lo_alpha,
    wp::float32 var_hi_alpha,
    wp::float32 var_mid_alpha,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 2;
    wp::float32 var_5;
    const wp::float32 var_6 = 2.0;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::float32 var_15 = 2.0;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::float32 var_23 = 2.0;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    const wp::float32 var_31 = 2.0;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::vec_t<3, wp::float32> var_34;
    //---------
    // forward
    // def _eval_pt_3alphas(quad: wp.vec3, lo_alpha: float, hi_alpha: float, mid_alpha: float) -> tuple[wp.vec3, wp.vec3, wp.vec3]:       <L 207>
    // q0, q1, q2 = quad[0], quad[1], quad[2]                                                 <L 209>
    var_1 = wp::extract(var_quad, var_0);
    var_3 = wp::extract(var_quad, var_2);
    var_5 = wp::extract(var_quad, var_4);
    // hessian = 2.0 * q2                                                                     <L 210>
    var_7 = wp::mul(var_6, var_5);
    // lo_aq2 = lo_alpha * q2                                                                 <L 211>
    var_8 = wp::mul(var_lo_alpha, var_5);
    // hi_aq2 = hi_alpha * q2                                                                 <L 212>
    var_9 = wp::mul(var_hi_alpha, var_5);
    // mid_aq2 = mid_alpha * q2                                                               <L 213>
    var_10 = wp::mul(var_mid_alpha, var_5);
    // return (                                                                               <L 214>
    // wp.vec3(lo_alpha * lo_aq2 + lo_alpha * q1 + q0, 2.0 * lo_aq2 + q1, hessian),           <L 215>
    var_11 = wp::mul(var_lo_alpha, var_8);
    var_12 = wp::mul(var_lo_alpha, var_3);
    var_13 = wp::add(var_11, var_12);
    var_14 = wp::add(var_13, var_1);
    var_16 = wp::mul(var_15, var_8);
    var_17 = wp::add(var_16, var_3);
    var_18 = wp::vec_t<3, wp::float32>(var_14, var_17, var_7);
    // wp.vec3(hi_alpha * hi_aq2 + hi_alpha * q1 + q0, 2.0 * hi_aq2 + q1, hessian),           <L 216>
    var_19 = wp::mul(var_hi_alpha, var_9);
    var_20 = wp::mul(var_hi_alpha, var_3);
    var_21 = wp::add(var_19, var_20);
    var_22 = wp::add(var_21, var_1);
    var_24 = wp::mul(var_23, var_9);
    var_25 = wp::add(var_24, var_3);
    var_26 = wp::vec_t<3, wp::float32>(var_22, var_25, var_7);
    // wp.vec3(mid_alpha * mid_aq2 + mid_alpha * q1 + q0, 2.0 * mid_aq2 + q1, hessian),       <L 217>
    var_27 = wp::mul(var_mid_alpha, var_10);
    var_28 = wp::mul(var_mid_alpha, var_3);
    var_29 = wp::add(var_27, var_28);
    var_30 = wp::add(var_29, var_1);
    var_32 = wp::mul(var_31, var_10);
    var_33 = wp::add(var_32, var_3);
    var_34 = wp::vec_t<3, wp::float32>(var_30, var_33, var_7);
    ret_0 = var_18;
    ret_1 = var_26;
    ret_2 = var_34;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:120
static CUDA_CALLABLE bool _in_bracket_0(
    wp::vec_t<3, wp::float32> var_x,
    wp::vec_t<3, wp::float32> var_y)
{
    //---------
    // primal vars
    bool var_0;
    bool var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 1;
    wp::float32 var_5;
    bool var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    const wp::float32 var_9 = 0.0;
    bool var_10;
    bool var_11;
    const wp::int32 var_12 = 1;
    wp::float32 var_13;
    const wp::int32 var_14 = 1;
    wp::float32 var_15;
    bool var_16;
    const wp::int32 var_17 = 1;
    wp::float32 var_18;
    const wp::float32 var_19 = 0.0;
    bool var_20;
    //---------
    // forward
    // def _in_bracket(x: wp.vec3, y: wp.vec3) -> bool:                                       <L 121>
    // return (x[1] < y[1] and y[1] < 0.0) or (x[1] > y[1] and y[1] > 0.0)                    <L 122>
    var_3 = wp::extract(var_x, var_2);
    var_5 = wp::extract(var_y, var_4);
    var_6 = (var_3 < var_5);
    var_1 = var_6;
    if (var_1) {
        var_8 = wp::extract(var_y, var_7);
        var_10 = (var_8 < var_9);
        var_1 = var_1 && var_10;
    }
    var_0 = var_1;
    if (!var_0) {
        var_13 = wp::extract(var_x, var_12);
        var_15 = wp::extract(var_y, var_14);
        var_16 = (var_13 > var_15);
        var_11 = var_16;
        if (var_11) {
            var_18 = wp::extract(var_y, var_17);
            var_20 = (var_18 > var_19);
            var_11 = var_11 && var_20;
        }
        var_0 = var_0 || var_11;
    }
    return var_0;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:125
static CUDA_CALLABLE void adj__eval_pt_direct_alpha_zero_0(
    wp::float32 var_jaref,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::float32 & adj_jaref,
    wp::float32 & adj_jv,
    wp::float32 & adj_d,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/math.py:0
static CUDA_CALLABLE void adj_safe_div_2(
    wp::float32 var_x,
    wp::float32 var_y,
    wp::float32 & adj_x,
    wp::float32 & adj_y,
    wp::float32 & adj_ret)
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
    // dual vars
    wp::float32 adj_0 = {};
    bool adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::float32 adj_5 = {};
    //---------
    // forward
    // def safe_div(x: Any, y: Any) -> Any:                                                   <L 1>
    // return x / wp.where(y != 0.0, y, types.MJ_MINVAL)                                      <L 2>
    var_1 = (var_y != var_0);
    var_4 = wp::where(var_1, var_y, var_3);
    var_5 = wp::div(var_x, var_4);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_5 += adj_ret;
    wp::adj_div(var_x, var_4, var_5, adj_x, adj_4, adj_5);
    wp::adj_where(var_1, var_y, var_3, adj_1, adj_y, adj_3, adj_4);
    // adj: return x / wp.where(y != 0.0, y, types.MJ_MINVAL)                                 <L 2>
    // adj: def safe_div(x: Any, y: Any) -> Any:                                              <L 1>
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:226
static CUDA_CALLABLE void adj__eval_frictionloss_pt_0(
    wp::float32 var_x,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::float32 & adj_x,
    wp::float32 & adj_f,
    wp::float32 & adj_rf,
    wp::float32 & adj_jv,
    wp::float32 & adj_d,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:577
static CUDA_CALLABLE void adj__compute_efc_eval_pt_alpha_zero_pyramidal_0(
    wp::int32 var_efcid,
    wp::int32 var_ne,
    wp::int32 var_nf,
    wp::float32 var_efc_D,
    wp::array_t<wp::float32> var_efc_frictionloss,
    wp::float32 var_ctx_Jaref,
    wp::float32 var_ctx_jv,
    wp::int32 & adj_efcid,
    wp::int32 & adj_ne,
    wp::int32 & adj_nf,
    wp::float32 & adj_efc_D,
    wp::array_t<wp::float32> & adj_efc_frictionloss,
    wp::float32 & adj_ctx_Jaref,
    wp::float32 & adj_ctx_jv,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:140
static CUDA_CALLABLE void adj__eval_pt_direct_cost_alpha_zero_0(
    wp::float32 var_jaref,
    wp::float32 var_d,
    wp::float32 & adj_jaref,
    wp::float32 & adj_d,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:145
static CUDA_CALLABLE void adj__eval_pt_direct_shifted_0(
    wp::float32 var_jaref,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::float32 var_alpha,
    wp::float32 var_offset,
    wp::float32 & adj_jaref,
    wp::float32 & adj_jv,
    wp::float32 & adj_d,
    wp::float32 & adj_alpha,
    wp::float32 & adj_offset,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:238
static CUDA_CALLABLE void adj__eval_frictionloss_cost_0(
    wp::float32 var_x,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_d,
    wp::float32 & adj_x,
    wp::float32 & adj_f,
    wp::float32 & adj_rf,
    wp::float32 & adj_d,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:221
static CUDA_CALLABLE void adj__shift_cost_0(
    wp::vec_t<3, wp::float32> var_pt,
    wp::float32 var_cost0,
    wp::vec_t<3, wp::float32> & adj_pt,
    wp::float32 & adj_cost0,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:481
static CUDA_CALLABLE void adj__compute_efc_eval_pt_pyramidal_0(
    wp::int32 var_efcid,
    wp::float32 var_alpha,
    wp::int32 var_ne,
    wp::int32 var_nf,
    wp::float32 var_efc_D,
    wp::array_t<wp::float32> var_efc_frictionloss,
    wp::float32 var_ctx_Jaref,
    wp::float32 var_ctx_jv,
    wp::int32 & adj_efcid,
    wp::float32 & adj_alpha,
    wp::int32 & adj_ne,
    wp::int32 & adj_nf,
    wp::float32 & adj_efc_D,
    wp::array_t<wp::float32> & adj_efc_frictionloss,
    wp::float32 & adj_ctx_Jaref,
    wp::float32 & adj_ctx_jv,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:195
static CUDA_CALLABLE void adj__eval_pt_0(
    wp::vec_t<3, wp::float32> var_quad,
    wp::float32 var_alpha,
    wp::vec_t<3, wp::float32> & adj_quad,
    wp::float32 & adj_alpha,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:172
static CUDA_CALLABLE void adj__eval_pt_direct_shifted_3alphas_0(
    wp::float32 var_jaref,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::float32 var_lo_alpha,
    wp::float32 var_hi_alpha,
    wp::float32 var_mid_alpha,
    wp::float32 var_offset,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::float32 & adj_jaref,
    wp::float32 & adj_jv,
    wp::float32 & adj_d,
    wp::float32 & adj_lo_alpha,
    wp::float32 & adj_hi_alpha,
    wp::float32 & adj_mid_alpha,
    wp::float32 & adj_offset,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:247
static CUDA_CALLABLE void adj__eval_frictionloss_pt_one_0(
    wp::float32 var_x,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_half_d,
    wp::float32 var_jvD,
    wp::float32 var_hessian,
    wp::float32 var_f_jv,
    wp::float32 & adj_x,
    wp::float32 & adj_f,
    wp::float32 & adj_rf,
    wp::float32 & adj_half_d,
    wp::float32 & adj_jvD,
    wp::float32 & adj_hessian,
    wp::float32 & adj_f_jv,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:258
static CUDA_CALLABLE void adj__eval_frictionloss_pt_3alphas_0(
    wp::float32 var_x_lo,
    wp::float32 var_x_hi,
    wp::float32 var_x_mid,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::float32 & adj_x_lo,
    wp::float32 & adj_x_hi,
    wp::float32 & adj_x_mid,
    wp::float32 & adj_f,
    wp::float32 & adj_rf,
    wp::float32 & adj_jv,
    wp::float32 & adj_d,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:649
static CUDA_CALLABLE void adj__compute_efc_eval_pt_3alphas_pyramidal_0(
    wp::int32 var_efcid,
    wp::float32 var_lo_alpha,
    wp::float32 var_hi_alpha,
    wp::float32 var_mid_alpha,
    wp::int32 var_ne,
    wp::int32 var_nf,
    wp::float32 var_efc_D,
    wp::array_t<wp::float32> var_efc_frictionloss,
    wp::float32 var_ctx_Jaref,
    wp::float32 var_ctx_jv,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::int32 & adj_efcid,
    wp::float32 & adj_lo_alpha,
    wp::float32 & adj_hi_alpha,
    wp::float32 & adj_mid_alpha,
    wp::int32 & adj_ne,
    wp::int32 & adj_nf,
    wp::float32 & adj_efc_D,
    wp::array_t<wp::float32> & adj_efc_frictionloss,
    wp::float32 & adj_ctx_Jaref,
    wp::float32 & adj_ctx_jv,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:206
static CUDA_CALLABLE void adj__eval_pt_3alphas_0(
    wp::vec_t<3, wp::float32> var_quad,
    wp::float32 var_lo_alpha,
    wp::float32 var_hi_alpha,
    wp::float32 var_mid_alpha,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_quad,
    wp::float32 & adj_lo_alpha,
    wp::float32 & adj_hi_alpha,
    wp::float32 & adj_mid_alpha,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:120
static CUDA_CALLABLE void adj__in_bracket_0(
    wp::vec_t<3, wp::float32> var_x,
    wp::vec_t<3, wp::float32> var_y,
    wp::vec_t<3, wp::float32> & adj_x,
    wp::vec_t<3, wp::float32> & adj_y,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void _linesearch_iterative_kernel__locals__kernel_4fc7dd2c_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_opt_tolerance,
    wp::array_t<wp::float32> var_opt_ls_tolerance,
    wp::array_t<wp::float32> var_opt_impratio_invsqrt,
    wp::array_t<wp::float32> var_stat_meaninertia,
    wp::array_t<wp::int32> var_ne_in,
    wp::array_t<wp::int32> var_nf_in,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::float32> var_qfrc_smooth_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_efc_type_in,
    wp::array_t<wp::int32> var_efc_id_in,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::array_t<wp::int32> var_efc_J_colind_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::float32> var_efc_frictionloss_in,
    wp::int32 var_njmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<bool> var_ctx_search_unchanged_in,
    wp::array_t<wp::float32> var_ctx_Jaref_in,
    wp::array_t<wp::float32> var_ctx_search_in,
    wp::array_t<wp::float32> var_ctx_search_dot_in,
    wp::array_t<wp::float32> var_ctx_mv_in,
    wp::array_t<wp::float32> var_ctx_jv_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ctx_quad_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_qacc_out,
    wp::array_t<wp::float32> var_efc_Ma_out,
    wp::array_t<wp::int32> var_overflow_out,
    wp::array_t<wp::float32> var_ctx_Jaref_out,
    wp::array_t<wp::float32> var_ctx_jv_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ctx_quad_out,
    wp::array_t<wp::float32> var_ctx_improvement_out,
    wp::array_t<wp::float32> var_ctx_alpha_out,
    wp::array_t<bool> var_ctx_ls_exhausted_out)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        const bool var_14 = false;
        const bool var_15 = false;
        wp::shape_t* var_16;
        const wp::int32 var_17 = 0;
        wp::int32 var_18;
        wp::shape_t var_19;
        wp::int32 var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::shape_t* var_24;
        const wp::int32 var_25 = 0;
        wp::int32 var_26;
        wp::shape_t var_27;
        wp::int32 var_28;
        wp::float32* var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::float32* var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::shape_t* var_35;
        const wp::int32 var_36 = 0;
        wp::int32 var_37;
        wp::shape_t var_38;
        wp::int32 var_39;
        wp::float32* var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        const wp::float32 var_48 = 1e-06;
        wp::float32 var_49;
        const wp::float32 var_50 = 0.0;
        wp::vec_t<3, wp::float32> var_51;
        wp::int32 var_52;
        wp::range_t var_53;
        wp::int32 var_54;
        const bool var_55 = false;
        wp::float32* var_56;
        wp::array_t<wp::float32> var_57;
        wp::float32* var_58;
        wp::float32* var_59;
        wp::vec_t<3, wp::float32> var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::vec_t<3, wp::float32> var_64;
        const bool var_65 = true;
        wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_66 = wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_shared_t<wp::vec_t<3, wp::float32>,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_67 = wp::tile_alloc_empty<wp::vec_t<3, wp::float32>,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        const wp::float32 var_68 = 0.0;
        wp::vec_t<2, wp::float32> var_69;
        wp::int32 var_70;
        wp::range_t var_71;
        wp::int32 var_72;
        wp::float32* var_73;
        wp::float32 var_74;
        wp::float32 var_75;
        wp::float32* var_76;
        wp::float32* var_77;
        wp::float32 var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        const wp::float32 var_82 = 0.5;
        wp::float32 var_83;
        wp::float32* var_84;
        wp::float32 var_85;
        wp::float32 var_86;
        wp::vec_t<2, wp::float32> var_87;
        wp::vec_t<2, wp::float32> var_88;
        const bool var_89 = true;
        wp::tile_register_t<wp::vec_t<2, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_90 = wp::tile_register_t<wp::vec_t<2, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_shared_t<wp::vec_t<2, wp::float32>,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_91 = wp::tile_alloc_empty<wp::vec_t<2, wp::float32>,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        const wp::int32 var_92 = 0;
        wp::vec_t<2, wp::float32> var_93;
        const wp::float32 var_94 = 0.0;
        const wp::int32 var_95 = 0;
        wp::float32 var_96;
        const wp::int32 var_97 = 1;
        wp::float32 var_98;
        wp::vec_t<3, wp::float32> var_99;
        const wp::int32 var_100 = 0;
        wp::float32 var_101;
        const wp::int32 var_102 = 1;
        wp::float32 var_103;
        const wp::float32 var_104 = 2.0;
        const wp::int32 var_105 = 2;
        wp::float32 var_106;
        wp::float32 var_107;
        wp::vec_t<3, wp::float32> var_108;
        const wp::int32 var_109 = 0;
        wp::vec_t<3, wp::float32> var_110;
        wp::vec_t<3, wp::float32> var_111;
        const wp::float32 var_112 = 0.0;
        const wp::int32 var_113 = 1;
        wp::float32 var_114;
        const wp::int32 var_115 = 2;
        wp::float32 var_116;
        wp::vec_t<3, wp::float32> var_117;
        const wp::float32 var_118 = 0.0;
        wp::float32 var_119;
        const bool var_120 = true;
        const wp::int32 var_121 = 0;
        wp::vec_t<3, wp::float32> var_122;
        const wp::float32 var_123 = 2.0;
        const wp::int32 var_124 = 0;
        wp::float32 var_125;
        const wp::float32 var_126 = 0.0;
        wp::float32 var_127;
        wp::float32 var_128;
        const wp::int32 var_129 = 2;
        wp::float32 var_130;
        const wp::float32 var_131 = 0.0;
        wp::float32 var_132;
        wp::float32 var_133;
        wp::float32 var_134;
        const wp::int32 var_135 = 1;
        wp::float32 var_136;
        wp::float32 var_137;
        wp::float32 var_138;
        const wp::float32 var_139 = 9.5367432e-07;
        const wp::float32 var_140 = 1.0;
        const wp::int32 var_141 = 2;
        wp::float32 var_142;
        wp::float32 var_143;
        wp::float32 var_144;
        wp::float32 var_145;
        const wp::int32 var_146 = 1;
        wp::float32 var_147;
        const wp::int32 var_148 = 2;
        wp::float32 var_149;
        wp::float32 var_150;
        wp::float32 var_151;
        const wp::float32 var_152 = 0.0;
        wp::vec_t<3, wp::float32> var_153;
        wp::int32 var_154;
        wp::range_t var_155;
        wp::int32 var_156;
        const bool var_157 = false;
        wp::float32* var_158;
        wp::array_t<wp::float32> var_159;
        wp::float32* var_160;
        wp::float32* var_161;
        wp::vec_t<3, wp::float32> var_162;
        wp::float32 var_163;
        wp::float32 var_164;
        wp::float32 var_165;
        wp::vec_t<3, wp::float32> var_166;
        const bool var_167 = true;
        wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_168 = wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_shared_t<wp::vec_t<3, wp::float32>,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_169 = wp::tile_alloc_empty<wp::vec_t<3, wp::float32>,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        wp::vec_t<3, wp::float32> var_170;
        const wp::int32 var_171 = 0;
        wp::vec_t<3, wp::float32> var_172;
        wp::vec_t<3, wp::float32> var_173;
        bool var_174;
        const wp::int32 var_175 = 1;
        wp::float32 var_176;
        wp::float32 var_177;
        bool var_178;
        const wp::int32 var_179 = 0;
        wp::float32 var_180;
        const wp::float32 var_181 = 0.0;
        bool var_182;
        bool var_183;
        bool var_184;
        const wp::float32 var_185 = 0.0;
        wp::float32 var_186;
        const wp::float32 var_187 = 0.0;
        wp::float32 var_188;
        const wp::int32 var_189 = 1;
        wp::float32 var_190;
        const wp::int32 var_191 = 1;
        wp::float32 var_192;
        bool var_193;
        wp::vec_t<3, wp::float32> var_194;
        const wp::float32 var_195 = 0.0;
        wp::float32 var_196;
        wp::vec_t<3, wp::float32> var_197;
        const wp::float32 var_198 = 0.0;
        wp::float32 var_199;
        const wp::int32 var_200 = 50;
        wp::range_t var_201;
        wp::int32 var_202;
        const wp::int32 var_203 = 1;
        wp::float32 var_204;
        const wp::int32 var_205 = 2;
        wp::float32 var_206;
        wp::float32 var_207;
        wp::float32 var_208;
        const wp::int32 var_209 = 1;
        wp::float32 var_210;
        const wp::int32 var_211 = 2;
        wp::float32 var_212;
        wp::float32 var_213;
        wp::float32 var_214;
        const wp::float32 var_215 = 0.5;
        wp::float32 var_216;
        wp::float32 var_217;
        const wp::float32 var_218 = 0.0;
        wp::vec_t<3, wp::float32> var_219;
        const wp::float32 var_220 = 0.0;
        wp::vec_t<3, wp::float32> var_221;
        const wp::float32 var_222 = 0.0;
        wp::vec_t<3, wp::float32> var_223;
        wp::int32 var_224;
        wp::range_t var_225;
        wp::int32 var_226;
        const bool var_227 = false;
        wp::float32* var_228;
        wp::array_t<wp::float32> var_229;
        wp::float32* var_230;
        wp::float32* var_231;
        wp::vec_t<3, wp::float32> var_232;
        wp::vec_t<3, wp::float32> var_233;
        wp::vec_t<3, wp::float32> var_234;
        wp::float32 var_235;
        wp::float32 var_236;
        wp::float32 var_237;
        wp::vec_t<3, wp::float32> var_238;
        wp::vec_t<3, wp::float32> var_239;
        wp::vec_t<3, wp::float32> var_240;
        const wp::int32 var_241 = 0;
        wp::float32 var_242;
        const wp::int32 var_243 = 0;
        wp::float32 var_244;
        const wp::int32 var_245 = 0;
        wp::float32 var_246;
        const wp::int32 var_247 = 1;
        wp::float32 var_248;
        const wp::int32 var_249 = 1;
        wp::float32 var_250;
        const wp::int32 var_251 = 1;
        wp::float32 var_252;
        const wp::int32 var_253 = 2;
        wp::float32 var_254;
        const wp::int32 var_255 = 2;
        wp::float32 var_256;
        const wp::int32 var_257 = 2;
        wp::float32 var_258;
        wp::mat_t<3, 3, wp::float32> var_259;
        const bool var_260 = true;
        wp::tile_register_t<wp::mat_t<3, 3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_261 = wp::tile_register_t<wp::mat_t<3, 3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_shared_t<wp::mat_t<3, 3, wp::float32>,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_262 = wp::tile_alloc_empty<wp::mat_t<3, 3, wp::float32>,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        const wp::int32 var_263 = 0;
        wp::mat_t<3, 3, wp::float32> var_264;
        wp::vec_t<3, wp::float32> var_265;
        wp::vec_t<3, wp::float32> var_266;
        wp::vec_t<3, wp::float32> var_267;
        const wp::int32 var_268 = 0;
        const wp::int32 var_269 = 0;
        wp::float32 var_270;
        const wp::int32 var_271 = 1;
        const wp::int32 var_272 = 0;
        wp::float32 var_273;
        const wp::int32 var_274 = 2;
        const wp::int32 var_275 = 0;
        wp::float32 var_276;
        wp::vec_t<3, wp::float32> var_277;
        wp::vec_t<3, wp::float32> var_278;
        const wp::int32 var_279 = 0;
        const wp::int32 var_280 = 1;
        wp::float32 var_281;
        const wp::int32 var_282 = 1;
        const wp::int32 var_283 = 1;
        wp::float32 var_284;
        const wp::int32 var_285 = 2;
        const wp::int32 var_286 = 1;
        wp::float32 var_287;
        wp::vec_t<3, wp::float32> var_288;
        wp::vec_t<3, wp::float32> var_289;
        const wp::int32 var_290 = 0;
        const wp::int32 var_291 = 2;
        wp::float32 var_292;
        const wp::int32 var_293 = 1;
        const wp::int32 var_294 = 2;
        wp::float32 var_295;
        const wp::int32 var_296 = 2;
        const wp::int32 var_297 = 2;
        wp::float32 var_298;
        wp::vec_t<3, wp::float32> var_299;
        wp::vec_t<3, wp::float32> var_300;
        bool var_301;
        wp::vec_t<3, wp::float32> var_302;
        wp::float32 var_303;
        bool var_304;
        wp::vec_t<3, wp::float32> var_305;
        wp::float32 var_306;
        bool var_307;
        wp::vec_t<3, wp::float32> var_308;
        wp::float32 var_309;
        bool var_310;
        bool var_311;
        wp::vec_t<3, wp::float32> var_312;
        wp::float32 var_313;
        bool var_314;
        wp::vec_t<3, wp::float32> var_315;
        wp::float32 var_316;
        bool var_317;
        wp::vec_t<3, wp::float32> var_318;
        wp::float32 var_319;
        bool var_320;
        bool var_321;
        bool var_322;
        bool var_323;
        bool var_324;
        bool var_325;
        const wp::int32 var_326 = 0;
        wp::float32 var_327;
        const wp::float32 var_328 = 0.0;
        bool var_329;
        const wp::int32 var_330 = 1;
        wp::float32 var_331;
        const wp::float32 var_332 = 0.0;
        bool var_333;
        const wp::int32 var_334 = 1;
        wp::float32 var_335;
        wp::float32 var_336;
        bool var_337;
        bool var_338;
        const wp::int32 var_339 = 0;
        wp::float32 var_340;
        const wp::float32 var_341 = 0.0;
        bool var_342;
        const wp::int32 var_343 = 1;
        wp::float32 var_344;
        const wp::float32 var_345 = 0.0;
        bool var_346;
        const wp::int32 var_347 = 1;
        wp::float32 var_348;
        bool var_349;
        bool var_350;
        const wp::int32 var_351 = 0;
        wp::float32 var_352;
        const wp::float32 var_353 = 0.0;
        bool var_354;
        const wp::int32 var_355 = 0;
        wp::float32 var_356;
        const wp::float32 var_357 = 0.0;
        bool var_358;
        const wp::int32 var_359 = 0;
        wp::float32 var_360;
        const wp::int32 var_361 = 0;
        wp::float32 var_362;
        bool var_363;
        wp::float32 var_364;
        const wp::int32 var_365 = 0;
        wp::float32 var_366;
        const wp::int32 var_367 = 0;
        wp::float32 var_368;
        wp::float32 var_369;
        wp::float32 var_370;
        wp::float32 var_371;
        wp::float32 var_372;
        const bool var_373 = true;
        wp::int32 var_374;
        wp::float32 var_375;
        wp::float32 var_376;
        wp::vec_t<3, wp::float32> var_377;
        wp::float32 var_378;
        wp::vec_t<3, wp::float32> var_379;
        wp::float32 var_380;
        wp::float32 var_381;
        const wp::int32 var_382 = 0;
        wp::float32 var_383;
        wp::float32 var_384;
        wp::float32 var_385;
        wp::float32 var_386;
        wp::int32 var_387;
        wp::range_t var_388;
        wp::int32 var_389;
        wp::float32* var_390;
        wp::float32 var_391;
        wp::float32 var_392;
        wp::float32 var_393;
        wp::float32* var_394;
        wp::float32 var_395;
        wp::float32 var_396;
        wp::float32 var_397;
        wp::int32 var_398;
        wp::range_t var_399;
        wp::int32 var_400;
        wp::float32* var_401;
        wp::float32 var_402;
        wp::float32 var_403;
        wp::float32 var_404;
        const wp::int32 var_405 = 0;
        bool var_406;
        const bool var_407 = true;
        wp::float32 var_408;
        bool var_409;
        bool var_410;
        const bool var_411 = true;
        const wp::str var_412 = "linesearch iterations limit reached - please increase ls_iterations to %u\n";
        const wp::int32 var_413 = 50;
        const wp::int32 var_414 = 1024;
        wp::int32 var_415;
        //---------
        // forward
        // def kernel(                                                                            <L 876>
        // worldid, tid = wp.tid()                                                                <L 922>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 924>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 925>
            continue;
        }
        var_4 = wp::load(var_2);
        // ne = ne_in[worldid]                                                                    <L 927>
        var_5 = wp::address(var_ne_in, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // nf = nf_in[worldid]                                                                    <L 928>
        var_8 = wp::address(var_nf_in, var_0);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // nefc = wp.min(njmax_in, nefc_in[worldid])                                              <L 929>
        var_11 = wp::address(var_nefc_in, var_0);
        var_13 = wp::load(var_11);
        var_12 = wp::min(var_njmax_in, var_13);
        // if wp.static(FUSE_JV):                                                                 <L 933>
        // if wp.static(IS_ELLIPTIC):                                                             <L 957>
        // tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]                            <L 1018>
        var_16 = &(var_opt_tolerance.shape);
        var_19 = wp::load(var_16);
        var_18 = wp::extract(var_19, var_17);
        var_20 = wp::mod(var_0, var_18);
        var_21 = wp::address(var_opt_tolerance, var_20);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // ls_tolerance = opt_ls_tolerance[worldid % opt_ls_tolerance.shape[0]]                   <L 1019>
        var_24 = &(var_opt_ls_tolerance.shape);
        var_27 = wp::load(var_24);
        var_26 = wp::extract(var_27, var_25);
        var_28 = wp::mod(var_0, var_26);
        var_29 = wp::address(var_opt_ls_tolerance, var_28);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // snorm = wp.sqrt(ctx_search_dot_in[worldid])                                            <L 1020>
        var_32 = wp::address(var_ctx_search_dot_in, var_0);
        var_34 = wp::load(var_32);
        var_33 = wp::sqrt(var_34);
        // meaninertia = stat_meaninertia[worldid % stat_meaninertia.shape[0]]                    <L 1021>
        var_35 = &(var_stat_meaninertia.shape);
        var_38 = wp::load(var_35);
        var_37 = wp::extract(var_38, var_36);
        var_39 = wp::mod(var_0, var_37);
        var_40 = wp::address(var_stat_meaninertia, var_39);
        var_42 = wp::load(var_40);
        var_41 = wp::copy(var_42);
        // scale = meaninertia * wp.float(nv)                                                     <L 1022>
        var_43 = wp::float(var_nv);
        var_44 = wp::mul(var_41, var_43);
        // gtol = wp.max(tolerance * ls_tolerance * snorm * scale, 1e-6)                          <L 1023>
        var_45 = wp::mul(var_22, var_30);
        var_46 = wp::mul(var_45, var_33);
        var_47 = wp::mul(var_46, var_44);
        var_49 = wp::max(var_47, var_48);
        // local_p0 = wp.vec3(0.0)                                                                <L 1026>
        var_51 = wp::vec_t<3, wp::float32>(var_50);
        // for efcid in range(tid, nefc, wp.block_dim()):                                         <L 1027>
        var_52 = builtin_block_dim();
        var_53 = wp::range(var_1, var_12, var_52);
        start_for_1:;
            if (iter_cmp(var_53) == 0) goto end_for_1;
            var_54 = wp::iter_next(var_53);
            // if wp.static(IS_ELLIPTIC):                                                         <L 1028>
            // local_p0 += _compute_efc_eval_pt_alpha_zero(                                       <L 1065>
            // efcid,                                                                             <L 1066>
            // ne,                                                                                <L 1067>
            // nf,                                                                                <L 1068>
            // efc_D_in[worldid, efcid],                                                          <L 1069>
            var_56 = wp::address(var_efc_D_in, var_0, var_54);
            // efc_frictionloss_in[worldid],                                                      <L 1070>
            var_57 = wp::view(var_efc_frictionloss_in, var_0);
            // ctx_Jaref_in[worldid, efcid],                                                      <L 1071>
            var_58 = wp::address(var_ctx_Jaref_in, var_0, var_54);
            // ctx_jv_in[worldid, efcid],                                                         <L 1072>
            var_59 = wp::address(var_ctx_jv_in, var_0, var_54);
            var_61 = wp::load(var_56);
            var_62 = wp::load(var_58);
            var_63 = wp::load(var_59);
            var_60 = _compute_efc_eval_pt_alpha_zero_pyramidal_0(var_54, var_6, var_9, var_61, var_57, var_62, var_63);
            // local_p0 += _compute_efc_eval_pt_alpha_zero(                                       <L 1065>
            var_64 = wp::add(var_51, var_60);
            wp::assign(var_51, var_64);
            goto start_for_1;
        end_for_1:;
        // p0_tile = wp.tile(local_p0, preserve_type=True)                                        <L 1078>
        var_66 = wp::tile<wp::vec_t<3, wp::float32>>(var_51);
        // p0_sum = wp.tile_reduce(wp.add, p0_tile)                                               <L 1079>
        var_67 = wp::tile_reduce(wp::add, var_66);
        // local_gauss = wp.vec2(0.0)                                                             <L 1082>
        var_69 = wp::vec_t<2, wp::float32>(var_68);
        // for dofid in range(tid, nv, wp.block_dim()):                                           <L 1083>
        var_70 = builtin_block_dim();
        var_71 = wp::range(var_1, var_nv, var_70);
        start_for_3:;
            if (iter_cmp(var_71) == 0) goto end_for_3;
            var_72 = wp::iter_next(var_71);
            // search = ctx_search_in[worldid, dofid]                                             <L 1084>
            var_73 = wp::address(var_ctx_search_in, var_0, var_72);
            var_75 = wp::load(var_73);
            var_74 = wp::copy(var_75);
            // local_gauss += wp.vec2(                                                            <L 1085>
            // search * (efc_Ma_out[worldid, dofid] - qfrc_smooth_in[worldid, dofid]),            <L 1086>
            var_76 = wp::address(var_efc_Ma_out, var_0, var_72);
            var_77 = wp::address(var_qfrc_smooth_in, var_0, var_72);
            var_79 = wp::load(var_76);
            var_80 = wp::load(var_77);
            var_78 = wp::sub(var_79, var_80);
            var_81 = wp::mul(var_74, var_78);
            // 0.5 * search * ctx_mv_in[worldid, dofid],                                          <L 1087>
            var_83 = wp::mul(var_82, var_74);
            var_84 = wp::address(var_ctx_mv_in, var_0, var_72);
            var_86 = wp::load(var_84);
            var_85 = wp::mul(var_83, var_86);
            var_87 = wp::vec_t<2, wp::float32>(var_81, var_85);
            // local_gauss += wp.vec2(                                                            <L 1085>
            var_88 = wp::add(var_69, var_87);
            wp::assign(var_69, var_88);
            goto start_for_3;
        end_for_3:;
        // gauss_tile = wp.tile(local_gauss, preserve_type=True)                                  <L 1090>
        var_90 = wp::tile<wp::vec_t<2, wp::float32>>(var_69);
        // gauss_sum = wp.tile_reduce(wp.add, gauss_tile)                                         <L 1091>
        var_91 = wp::tile_reduce(wp::add, var_90);
        // gauss_reduced = gauss_sum[0]                                                           <L 1092>
        var_93 = wp::tile_extract(var_91, var_92);
        // ctx_quad_gauss = wp.vec3(0.0, gauss_reduced[0], gauss_reduced[1])                      <L 1093>
        var_96 = wp::extract(var_93, var_95);
        var_98 = wp::extract(var_93, var_97);
        var_99 = wp::vec_t<3, wp::float32>(var_94, var_96, var_98);
        // p0 = wp.vec3(ctx_quad_gauss[0], ctx_quad_gauss[1], 2.0 * ctx_quad_gauss[2]) + p0_sum[0]       <L 1096>
        var_101 = wp::extract(var_99, var_100);
        var_103 = wp::extract(var_99, var_102);
        var_106 = wp::extract(var_99, var_105);
        var_107 = wp::mul(var_104, var_106);
        var_108 = wp::vec_t<3, wp::float32>(var_101, var_103, var_107);
        var_110 = wp::tile_extract(var_67, var_109);
        var_111 = wp::add(var_108, var_110);
        // p0_delta = wp.vec3(0.0, p0[1], p0[2])                                                  <L 1097>
        var_114 = wp::extract(var_111, var_113);
        var_116 = wp::extract(var_111, var_115);
        var_117 = wp::vec_t<3, wp::float32>(var_112, var_114, var_116);
        // noise_floor = float(0.0)                                                               <L 1107>
        var_119 = wp::float(var_118);
        // if wp.static(INCREMENTAL):                                                             <L 1108>
        // rows = p0_sum[0]                                                                       <L 1109>
        var_122 = wp::tile_extract(var_67, var_121);
        // q1_abs = wp.sqrt(2.0 * wp.max(rows[0], 0.0) * wp.max(rows[2], 0.0)) + wp.abs(ctx_quad_gauss[1])       <L 1110>
        var_125 = wp::extract(var_122, var_124);
        var_127 = wp::max(var_125, var_126);
        var_128 = wp::mul(var_123, var_127);
        var_130 = wp::extract(var_122, var_129);
        var_132 = wp::max(var_130, var_131);
        var_133 = wp::mul(var_128, var_132);
        var_134 = wp::sqrt(var_133);
        var_136 = wp::extract(var_99, var_135);
        var_137 = wp::abs(var_136);
        var_138 = wp::add(var_134, var_137);
        // noise_floor = _ALPHA_NOISE_EPS * wp.max(1.0, math.safe_div(q1_abs, p0[2]))             <L 1111>
        var_142 = wp::extract(var_111, var_141);
        var_143 = safe_div_2(var_138, var_142);
        var_144 = wp::max(var_140, var_143);
        var_145 = wp::mul(var_139, var_144);
        // lo_alpha_in = -math.safe_div(p0[1], p0[2])                                             <L 1114>
        var_147 = wp::extract(var_111, var_146);
        var_149 = wp::extract(var_111, var_148);
        var_150 = safe_div_2(var_147, var_149);
        var_151 = wp::neg(var_150);
        // local_lo_in = wp.vec3(0.0)                                                             <L 1116>
        var_153 = wp::vec_t<3, wp::float32>(var_152);
        // for efcid in range(tid, nefc, wp.block_dim()):                                         <L 1117>
        var_154 = builtin_block_dim();
        var_155 = wp::range(var_1, var_12, var_154);
        start_for_5:;
            if (iter_cmp(var_155) == 0) goto end_for_5;
            var_156 = wp::iter_next(var_155);
            // if wp.static(IS_ELLIPTIC):                                                         <L 1118>
            // local_lo_in += _compute_efc_eval_pt(                                               <L 1156>
            // efcid,                                                                             <L 1157>
            // lo_alpha_in,                                                                       <L 1158>
            // ne,                                                                                <L 1159>
            // nf,                                                                                <L 1160>
            // efc_D_in[worldid, efcid],                                                          <L 1161>
            var_158 = wp::address(var_efc_D_in, var_0, var_156);
            // efc_frictionloss_in[worldid],                                                      <L 1162>
            var_159 = wp::view(var_efc_frictionloss_in, var_0);
            // ctx_Jaref_in[worldid, efcid],                                                      <L 1163>
            var_160 = wp::address(var_ctx_Jaref_in, var_0, var_156);
            // ctx_jv_in[worldid, efcid],                                                         <L 1164>
            var_161 = wp::address(var_ctx_jv_in, var_0, var_156);
            var_163 = wp::load(var_158);
            var_164 = wp::load(var_160);
            var_165 = wp::load(var_161);
            var_162 = _compute_efc_eval_pt_pyramidal_0(var_156, var_151, var_6, var_9, var_163, var_159, var_164, var_165);
            // local_lo_in += _compute_efc_eval_pt(                                               <L 1156>
            var_166 = wp::add(var_153, var_162);
            wp::assign(var_153, var_166);
            goto start_for_5;
        end_for_5:;
        // lo_in_tile = wp.tile(local_lo_in, preserve_type=True)                                  <L 1167>
        var_168 = wp::tile<wp::vec_t<3, wp::float32>>(var_153);
        // lo_in_sum = wp.tile_reduce(wp.add, lo_in_tile)                                         <L 1168>
        var_169 = wp::tile_reduce(wp::add, var_168);
        // lo_in = _eval_pt(ctx_quad_gauss, lo_alpha_in) + lo_in_sum[0]                           <L 1169>
        var_170 = _eval_pt_0(var_99, var_151);
        var_172 = wp::tile_extract(var_169, var_171);
        var_173 = wp::add(var_170, var_172);
        // initial_converged = wp.abs(lo_in[1]) < gtol and lo_in[0] < 0.0                         <L 1172>
        var_176 = wp::extract(var_173, var_175);
        var_177 = wp::abs(var_176);
        var_178 = (var_177 < var_49);
        var_174 = var_178;
        if (var_174) {
            var_180 = wp::extract(var_173, var_179);
            var_182 = (var_180 < var_181);
            var_174 = var_174 && var_182;
        }
        // ls_converged = initial_converged                                                       <L 1173>
        var_183 = wp::copy(var_174);
        // if not initial_converged:                                                              <L 1176>
        var_184 = wp::unot(var_174);
        if (var_184) {
            // alpha = float(0.0)                                                                 <L 1177>
            var_186 = wp::float(var_185);
            // improvement = float(0.0)                                                           <L 1178>
            var_188 = wp::float(var_187);
            // lo_less = lo_in[1] < p0[1]                                                         <L 1181>
            var_190 = wp::extract(var_173, var_189);
            var_192 = wp::extract(var_111, var_191);
            var_193 = (var_190 < var_192);
            // lo = wp.where(lo_less, lo_in, p0_delta)                                            <L 1182>
            var_194 = wp::where(var_193, var_173, var_117);
            // lo_alpha = wp.where(lo_less, lo_alpha_in, 0.0)                                     <L 1183>
            var_196 = wp::where(var_193, var_151, var_195);
            // hi = wp.where(lo_less, p0_delta, lo_in)                                            <L 1184>
            var_197 = wp::where(var_193, var_117, var_173);
            // hi_alpha = wp.where(lo_less, 0.0, lo_alpha_in)                                     <L 1185>
            var_199 = wp::where(var_193, var_198, var_151);
            // for _ in range(LS_ITERATIONS):                                                     <L 1187>
            var_201 = wp::range(var_200);
            start_for_7:;
                if (iter_cmp(var_201) == 0) goto end_for_7;
                var_202 = wp::iter_next(var_201);
                // lo_next_alpha = lo_alpha - math.safe_div(lo[1], lo[2])                         <L 1188>
                var_204 = wp::extract(var_194, var_203);
                var_206 = wp::extract(var_194, var_205);
                var_207 = safe_div_2(var_204, var_206);
                var_208 = wp::sub(var_196, var_207);
                // hi_next_alpha = hi_alpha - math.safe_div(hi[1], hi[2])                         <L 1189>
                var_210 = wp::extract(var_197, var_209);
                var_212 = wp::extract(var_197, var_211);
                var_213 = safe_div_2(var_210, var_212);
                var_214 = wp::sub(var_199, var_213);
                // mid_alpha = 0.5 * (lo_alpha + hi_alpha)                                        <L 1190>
                var_216 = wp::add(var_196, var_199);
                var_217 = wp::mul(var_215, var_216);
                // local_lo = wp.vec3(0.0)                                                        <L 1192>
                var_219 = wp::vec_t<3, wp::float32>(var_218);
                // local_hi = wp.vec3(0.0)                                                        <L 1193>
                var_221 = wp::vec_t<3, wp::float32>(var_220);
                // local_mid = wp.vec3(0.0)                                                       <L 1194>
                var_223 = wp::vec_t<3, wp::float32>(var_222);
                // for efcid in range(tid, nefc, wp.block_dim()):                                 <L 1196>
                var_224 = builtin_block_dim();
                var_225 = wp::range(var_1, var_12, var_224);
                start_for_9:;
                    if (iter_cmp(var_225) == 0) goto end_for_9;
                    var_226 = wp::iter_next(var_225);
                    // if wp.static(IS_ELLIPTIC):                                                 <L 1197>
                    // r_lo, r_hi, r_mid = _compute_efc_eval_pt_3alphas(                          <L 1237>
                    // efcid,                                                                     <L 1238>
                    // lo_next_alpha,                                                             <L 1239>
                    // hi_next_alpha,                                                             <L 1240>
                    // mid_alpha,                                                                 <L 1241>
                    // ne,                                                                        <L 1242>
                    // nf,                                                                        <L 1243>
                    // efc_D_in[worldid, efcid],                                                  <L 1244>
                    var_228 = wp::address(var_efc_D_in, var_0, var_226);
                    // efc_frictionloss_in[worldid],                                              <L 1245>
                    var_229 = wp::view(var_efc_frictionloss_in, var_0);
                    // ctx_Jaref_in[worldid, efcid],                                              <L 1246>
                    var_230 = wp::address(var_ctx_Jaref_in, var_0, var_226);
                    // ctx_jv_in[worldid, efcid],                                                 <L 1247>
                    var_231 = wp::address(var_ctx_jv_in, var_0, var_226);
                    var_235 = wp::load(var_228);
                    var_236 = wp::load(var_230);
                    var_237 = wp::load(var_231);
                    _compute_efc_eval_pt_3alphas_pyramidal_0(var_226, var_208, var_214, var_217, var_6, var_9, var_235, var_229, var_236, var_237, var_232, var_233, var_234);
                    // local_lo += r_lo                                                           <L 1249>
                    var_238 = wp::add(var_219, var_232);
                    // local_hi += r_hi                                                           <L 1250>
                    var_239 = wp::add(var_221, var_233);
                    // local_mid += r_mid                                                         <L 1251>
                    var_240 = wp::add(var_223, var_234);
                    wp::assign(var_219, var_238);
                    wp::assign(var_221, var_239);
                    wp::assign(var_223, var_240);
                    goto start_for_9;
                end_for_9:;
                // local_combined = wp.mat33(                                                     <L 1254>
                // local_lo[0],                                                                   <L 1255>
                var_242 = wp::extract(var_219, var_241);
                // local_hi[0],                                                                   <L 1256>
                var_244 = wp::extract(var_221, var_243);
                // local_mid[0],                                                                  <L 1257>
                var_246 = wp::extract(var_223, var_245);
                // local_lo[1],                                                                   <L 1258>
                var_248 = wp::extract(var_219, var_247);
                // local_hi[1],                                                                   <L 1259>
                var_250 = wp::extract(var_221, var_249);
                // local_mid[1],                                                                  <L 1260>
                var_252 = wp::extract(var_223, var_251);
                // local_lo[2],                                                                   <L 1261>
                var_254 = wp::extract(var_219, var_253);
                // local_hi[2],                                                                   <L 1262>
                var_256 = wp::extract(var_221, var_255);
                // local_mid[2],                                                                  <L 1263>
                var_258 = wp::extract(var_223, var_257);
                var_259 = wp::mat_t<3, 3, wp::float32>(var_242, var_244, var_246, var_248, var_250, var_252, var_254, var_256, var_258);
                // combined_tile = wp.tile(local_combined, preserve_type=True)                    <L 1268>
                var_261 = wp::tile<wp::mat_t<3, 3, wp::float32>>(var_259);
                // combined_sum = wp.tile_reduce(wp.add, combined_tile)                           <L 1269>
                var_262 = wp::tile_reduce(wp::add, var_261);
                // result = combined_sum[0]                                                       <L 1270>
                var_264 = wp::tile_extract(var_262, var_263);
                // gauss_lo, gauss_hi, gauss_mid = _eval_pt_3alphas(ctx_quad_gauss, lo_next_alpha, hi_next_alpha, mid_alpha)       <L 1273>
                _eval_pt_3alphas_0(var_99, var_208, var_214, var_217, var_265, var_266, var_267);
                // lo_next = gauss_lo + wp.vec3(result[0, 0], result[1, 0], result[2, 0])         <L 1274>
                var_270 = wp::extract(var_264, var_268, var_269);
                var_273 = wp::extract(var_264, var_271, var_272);
                var_276 = wp::extract(var_264, var_274, var_275);
                var_277 = wp::vec_t<3, wp::float32>(var_270, var_273, var_276);
                var_278 = wp::add(var_265, var_277);
                // hi_next = gauss_hi + wp.vec3(result[0, 1], result[1, 1], result[2, 1])         <L 1275>
                var_281 = wp::extract(var_264, var_279, var_280);
                var_284 = wp::extract(var_264, var_282, var_283);
                var_287 = wp::extract(var_264, var_285, var_286);
                var_288 = wp::vec_t<3, wp::float32>(var_281, var_284, var_287);
                var_289 = wp::add(var_266, var_288);
                // mid = gauss_mid + wp.vec3(result[0, 2], result[1, 2], result[2, 2])            <L 1276>
                var_292 = wp::extract(var_264, var_290, var_291);
                var_295 = wp::extract(var_264, var_293, var_294);
                var_298 = wp::extract(var_264, var_296, var_297);
                var_299 = wp::vec_t<3, wp::float32>(var_292, var_295, var_298);
                var_300 = wp::add(var_267, var_299);
                // swap_lo_lo_next = _in_bracket(lo, lo_next)                                     <L 1280>
                var_301 = _in_bracket_0(var_194, var_278);
                // lo = wp.where(swap_lo_lo_next, lo_next, lo)                                    <L 1281>
                var_302 = wp::where(var_301, var_278, var_194);
                // lo_alpha = wp.where(swap_lo_lo_next, lo_next_alpha, lo_alpha)                  <L 1282>
                var_303 = wp::where(var_301, var_208, var_196);
                // swap_lo_mid = _in_bracket(lo, mid)                                             <L 1283>
                var_304 = _in_bracket_0(var_302, var_300);
                // lo = wp.where(swap_lo_mid, mid, lo)                                            <L 1284>
                var_305 = wp::where(var_304, var_300, var_302);
                // lo_alpha = wp.where(swap_lo_mid, mid_alpha, lo_alpha)                          <L 1285>
                var_306 = wp::where(var_304, var_217, var_303);
                // swap_lo_hi_next = _in_bracket(lo, hi_next)                                     <L 1286>
                var_307 = _in_bracket_0(var_305, var_289);
                // lo = wp.where(swap_lo_hi_next, hi_next, lo)                                    <L 1287>
                var_308 = wp::where(var_307, var_289, var_305);
                // lo_alpha = wp.where(swap_lo_hi_next, hi_next_alpha, lo_alpha)                  <L 1288>
                var_309 = wp::where(var_307, var_214, var_306);
                // swap_lo = swap_lo_lo_next or swap_lo_mid or swap_lo_hi_next                    <L 1289>
                var_310 = var_301;
                if (!var_310) {
                    var_310 = var_310 || var_304;
                }
                if (!var_310) {
                    var_310 = var_310 || var_307;
                }
                // swap_hi_hi_next = _in_bracket(hi, hi_next)                                     <L 1292>
                var_311 = _in_bracket_0(var_197, var_289);
                // hi = wp.where(swap_hi_hi_next, hi_next, hi)                                    <L 1293>
                var_312 = wp::where(var_311, var_289, var_197);
                // hi_alpha = wp.where(swap_hi_hi_next, hi_next_alpha, hi_alpha)                  <L 1294>
                var_313 = wp::where(var_311, var_214, var_199);
                // swap_hi_mid = _in_bracket(hi, mid)                                             <L 1295>
                var_314 = _in_bracket_0(var_312, var_300);
                // hi = wp.where(swap_hi_mid, mid, hi)                                            <L 1296>
                var_315 = wp::where(var_314, var_300, var_312);
                // hi_alpha = wp.where(swap_hi_mid, mid_alpha, hi_alpha)                          <L 1297>
                var_316 = wp::where(var_314, var_217, var_313);
                // swap_hi_lo_next = _in_bracket(hi, lo_next)                                     <L 1298>
                var_317 = _in_bracket_0(var_315, var_278);
                // hi = wp.where(swap_hi_lo_next, lo_next, hi)                                    <L 1299>
                var_318 = wp::where(var_317, var_278, var_315);
                // hi_alpha = wp.where(swap_hi_lo_next, lo_next_alpha, hi_alpha)                  <L 1300>
                var_319 = wp::where(var_317, var_208, var_316);
                // swap_hi = swap_hi_hi_next or swap_hi_mid or swap_hi_lo_next                    <L 1301>
                var_320 = var_311;
                if (!var_320) {
                    var_320 = var_320 || var_314;
                }
                if (!var_320) {
                    var_320 = var_320 || var_317;
                }
                // ls_done = (                                                                    <L 1304>
                // (not swap_lo and not swap_hi)                                                  <L 1305>
                var_323 = wp::unot(var_310);
                var_322 = var_323;
                if (var_322) {
                    var_324 = wp::unot(var_320);
                    var_322 = var_322 && var_324;
                }
                var_321 = var_322;
                if (!var_321) {
                    // or (lo[0] < 0.0 and lo[1] < 0.0 and lo[1] > -gtol)                         <L 1306>
                    var_327 = wp::extract(var_308, var_326);
                    var_329 = (var_327 < var_328);
                    var_325 = var_329;
                    if (var_325) {
                        var_331 = wp::extract(var_308, var_330);
                        var_333 = (var_331 < var_332);
                        var_325 = var_325 && var_333;
                    }
                    if (var_325) {
                        var_335 = wp::extract(var_308, var_334);
                        var_336 = wp::neg(var_49);
                        var_337 = (var_335 > var_336);
                        var_325 = var_325 && var_337;
                    }
                    var_321 = var_321 || var_325;
                }
                if (!var_321) {
                    // or (hi[0] < 0.0 and hi[1] > 0.0 and hi[1] < gtol)                          <L 1307>
                    var_340 = wp::extract(var_318, var_339);
                    var_342 = (var_340 < var_341);
                    var_338 = var_342;
                    if (var_338) {
                        var_344 = wp::extract(var_318, var_343);
                        var_346 = (var_344 > var_345);
                        var_338 = var_338 && var_346;
                    }
                    if (var_338) {
                        var_348 = wp::extract(var_318, var_347);
                        var_349 = (var_348 < var_49);
                        var_338 = var_338 && var_349;
                    }
                    var_321 = var_321 || var_338;
                }
                // improved = lo[0] < 0.0 or hi[0] < 0.0                                          <L 1311>
                var_352 = wp::extract(var_308, var_351);
                var_354 = (var_352 < var_353);
                var_350 = var_354;
                if (!var_350) {
                    var_356 = wp::extract(var_318, var_355);
                    var_358 = (var_356 < var_357);
                    var_350 = var_350 || var_358;
                }
                // lo_better = lo[0] < hi[0]                                                      <L 1312>
                var_360 = wp::extract(var_308, var_359);
                var_362 = wp::extract(var_318, var_361);
                var_363 = (var_360 < var_362);
                // best_alpha = wp.where(lo_better, lo_alpha, hi_alpha)                           <L 1313>
                var_364 = wp::where(var_363, var_309, var_319);
                // best_delta = wp.where(lo_better, lo[0], hi[0])                                 <L 1314>
                var_366 = wp::extract(var_308, var_365);
                var_368 = wp::extract(var_318, var_367);
                var_369 = wp::where(var_363, var_366, var_368);
                // alpha = wp.where(improved, best_alpha, alpha)                                  <L 1315>
                var_370 = wp::where(var_350, var_364, var_186);
                // improvement = wp.where(improved, -best_delta, improvement)                     <L 1316>
                var_371 = wp::neg(var_369);
                var_372 = wp::where(var_350, var_371, var_188);
                // if ls_done:                                                                    <L 1318>
                if (var_321) {
                    // ls_converged = True                                                        <L 1319>
                    // break                                                                      <L 1320>
                    wp::assign(var_156, var_226);
                    wp::assign(var_183, var_373);
                    wp::assign(var_186, var_370);
                    wp::assign(var_188, var_372);
                    wp::assign(var_194, var_308);
                    wp::assign(var_196, var_309);
                    wp::assign(var_197, var_318);
                    wp::assign(var_199, var_319);
                    goto end_for_7;
                }
                var_374 = wp::where(var_321, var_156, var_226);
                var_375 = wp::where(var_321, var_186, var_370);
                var_376 = wp::where(var_321, var_188, var_372);
                var_377 = wp::where(var_321, var_194, var_308);
                var_378 = wp::where(var_321, var_196, var_309);
                var_379 = wp::where(var_321, var_197, var_318);
                var_380 = wp::where(var_321, var_199, var_319);
                wp::assign(var_156, var_374);
                wp::assign(var_186, var_375);
                wp::assign(var_188, var_376);
                wp::assign(var_194, var_377);
                wp::assign(var_196, var_378);
                wp::assign(var_197, var_379);
                wp::assign(var_199, var_380);
                goto start_for_7;
            end_for_7:;
        }
        if (!var_184) {
            // alpha = lo_alpha_in                                                                <L 1322>
            var_381 = wp::copy(var_151);
            // improvement = -lo_in[0]                                                            <L 1323>
            var_383 = wp::extract(var_173, var_382);
            var_384 = wp::neg(var_383);
        }
        var_385 = wp::where(var_184, var_186, var_381);
        var_386 = wp::where(var_184, var_188, var_384);
        // for dofid in range(tid, nv, wp.block_dim()):                                           <L 1326>
        var_387 = builtin_block_dim();
        var_388 = wp::range(var_1, var_nv, var_387);
        start_for_11:;
            if (iter_cmp(var_388) == 0) goto end_for_11;
            var_389 = wp::iter_next(var_388);
            // qacc_out[worldid, dofid] += alpha * ctx_search_in[worldid, dofid]                  <L 1327>
            var_390 = wp::address(var_ctx_search_in, var_0, var_389);
            var_392 = wp::load(var_390);
            var_391 = wp::mul(var_385, var_392);
            var_393 = wp::atomic_add(var_qacc_out, var_0, var_389, var_391);
            // efc_Ma_out[worldid, dofid] += alpha * ctx_mv_in[worldid, dofid]                    <L 1328>
            var_394 = wp::address(var_ctx_mv_in, var_0, var_389);
            var_396 = wp::load(var_394);
            var_395 = wp::mul(var_385, var_396);
            var_397 = wp::atomic_add(var_efc_Ma_out, var_0, var_389, var_395);
            goto start_for_11;
        end_for_11:;
        // for efcid in range(tid, nefc, wp.block_dim()):                                         <L 1331>
        var_398 = builtin_block_dim();
        var_399 = wp::range(var_1, var_12, var_398);
        start_for_13:;
            if (iter_cmp(var_399) == 0) goto end_for_13;
            var_400 = wp::iter_next(var_399);
            // ctx_Jaref_out[worldid, efcid] += alpha * ctx_jv_in[worldid, efcid]                 <L 1332>
            var_401 = wp::address(var_ctx_jv_in, var_0, var_400);
            var_403 = wp::load(var_401);
            var_402 = wp::mul(var_385, var_403);
            var_404 = wp::atomic_add(var_ctx_Jaref_out, var_0, var_400, var_402);
            goto start_for_13;
        end_for_13:;
        // if tid == 0:                                                                           <L 1334>
        var_406 = (var_1 == var_405);
        if (var_406) {
            // ctx_improvement_out[worldid] = improvement                                         <L 1335>
            wp::array_store(var_ctx_improvement_out, var_0, var_386);
            // ctx_alpha_out[worldid] = alpha                                                     <L 1336>
            wp::array_store(var_ctx_alpha_out, var_0, var_385);
            // if wp.static(INCREMENTAL):                                                         <L 1337>
            // ctx_ls_exhausted_out[worldid] = wp.abs(alpha) < noise_floor                        <L 1338>
            var_408 = wp::abs(var_385);
            var_409 = (var_408 < var_145);
            wp::array_store(var_ctx_ls_exhausted_out, var_0, var_409);
            // if not ls_converged:                                                               <L 1339>
            var_410 = wp::unot(var_183);
            if (var_410) {
                // if wp.static(warn_overflow):                                                   <L 1340>
                // wp.printf(                                                                     <L 1341>
                // "linesearch iterations limit reached - please increase ls_iterations to %u\n",       <L 1342>
                // wp.static(LS_ITERATIONS),                                                      <L 1343>
                printf(var_412, var_413);
                // wp.atomic_or(overflow_out, worldid, wp.static(OverflowType.LS_ITERATIONS))       <L 1345>
                var_415 = wp::atomic_or(var_overflow_out, var_0, var_414);
            }
        }
    }
}

