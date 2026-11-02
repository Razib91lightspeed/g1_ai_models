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


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:405
static CUDA_CALLABLE wp::vec_t<2, wp::float32> _eval_elliptic_middle_0(
    wp::float32 var_N,
    wp::float32 var_T,
    wp::float32 var_D0,
    wp::float32 var_mu,
    wp::float32 var_ufrictionj,
    bool var_is_normal)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 1.0;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::float32 var_11 = 0.5;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::vec_t<2, wp::float32> var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    const wp::float32 var_19 = 0.0;
    wp::vec_t<2, wp::float32> var_20;
    //---------
    // forward
    // def _eval_elliptic_middle(                                                             <L 406>
    // dm = math.safe_div(D0, mu * mu * (1.0 + mu * mu))                                      <L 416>
    var_0 = wp::mul(var_mu, var_mu);
    var_2 = wp::mul(var_mu, var_mu);
    var_3 = wp::add(var_1, var_2);
    var_4 = wp::mul(var_0, var_3);
    var_5 = safe_div_2(var_D0, var_4);
    // nmt = N - mu * T                                                                       <L 417>
    var_6 = wp::mul(var_mu, var_T);
    var_7 = wp::sub(var_N, var_6);
    // force_normal = -dm * nmt * mu                                                          <L 418>
    var_8 = wp::neg(var_5);
    var_9 = wp::mul(var_8, var_7);
    var_10 = wp::mul(var_9, var_mu);
    // if is_normal:                                                                          <L 419>
    if (var_is_normal) {
        // return wp.vec2(force_normal, 0.5 * dm * nmt * nmt)                                 <L 420>
        var_12 = wp::mul(var_11, var_5);
        var_13 = wp::mul(var_12, var_7);
        var_14 = wp::mul(var_13, var_7);
        var_15 = wp::vec_t<2, wp::float32>(var_10, var_14);
        return var_15;
    }
    // return wp.vec2(-math.safe_div(force_normal, T) * ufrictionj, 0.0)                      <L 421>
    var_16 = safe_div_2(var_10, var_T);
    var_17 = wp::neg(var_16);
    var_18 = wp::mul(var_17, var_ufrictionj);
    var_20 = wp::vec_t<2, wp::float32>(var_18, var_19);
    return var_20;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:424
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _eval_constraint_0(
    bool var_is_equality,
    bool var_is_friction,
    bool var_is_elliptic,
    wp::float32 var_jaref,
    wp::float32 var_D,
    wp::float32 var_frictionloss,
    wp::int32 var_efcid,
    wp::int32 var_efcid0,
    wp::float32 var_jaref0,
    wp::float32 var_D0,
    wp::float32 var_mu,
    wp::float32 var_ufrictionj,
    wp::float32 var_TT)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    const wp::float32 var_2 = 0.5;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::int32 var_6 = 1;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    bool var_12;
    const wp::int32 var_13 = 2;
    const wp::int32 var_14 = 2;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::float32 var_17 = 0.5;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::vec_t<3, wp::float32> var_21;
    bool var_22;
    wp::float32 var_23;
    const wp::int32 var_24 = 3;
    const wp::int32 var_25 = 3;
    wp::float32 var_26;
    wp::float32 var_27;
    const wp::float32 var_28 = 0.5;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    const wp::int32 var_35 = 1;
    const wp::int32 var_36 = 1;
    wp::float32 var_37;
    const wp::float32 var_38 = 0.5;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::float32 var_43;
    const wp::float32 var_44 = 0.0;
    bool var_45;
    const wp::float32 var_46 = 0.0;
    wp::float32 var_47;
    wp::float32 var_48;
    bool var_49;
    wp::float32 var_50;
    bool var_51;
    bool var_52;
    const wp::float32 var_53 = 0.0;
    bool var_54;
    const wp::float32 var_55 = 0.0;
    bool var_56;
    const wp::float32 var_57 = 0.0;
    const wp::int32 var_58 = 0;
    const wp::int32 var_59 = 0;
    wp::float32 var_60;
    const wp::float32 var_61 = 0.0;
    wp::vec_t<3, wp::float32> var_62;
    bool var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    const wp::float32 var_66 = 0.0;
    bool var_67;
    bool var_68;
    const wp::float32 var_69 = 0.0;
    bool var_70;
    const wp::float32 var_71 = 0.0;
    bool var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    const wp::int32 var_75 = 1;
    const wp::int32 var_76 = 1;
    wp::float32 var_77;
    const wp::float32 var_78 = 0.5;
    wp::float32 var_79;
    wp::float32 var_80;
    wp::float32 var_81;
    wp::vec_t<3, wp::float32> var_82;
    bool var_83;
    wp::vec_t<2, wp::float32> var_84;
    const wp::int32 var_85 = 0;
    wp::float32 var_86;
    const wp::int32 var_87 = 4;
    const wp::int32 var_88 = 4;
    wp::float32 var_89;
    const wp::int32 var_90 = 1;
    wp::float32 var_91;
    wp::vec_t<3, wp::float32> var_92;
    const wp::float32 var_93 = 0.0;
    bool var_94;
    const wp::float32 var_95 = 0.0;
    const wp::int32 var_96 = 0;
    const wp::int32 var_97 = 0;
    wp::float32 var_98;
    const wp::float32 var_99 = 0.0;
    wp::vec_t<3, wp::float32> var_100;
    wp::float32 var_101;
    wp::float32 var_102;
    const wp::int32 var_103 = 1;
    const wp::int32 var_104 = 1;
    wp::float32 var_105;
    const wp::float32 var_106 = 0.5;
    wp::float32 var_107;
    wp::float32 var_108;
    wp::float32 var_109;
    wp::vec_t<3, wp::float32> var_110;
    //---------
    // forward
    // def _eval_constraint(                                                                  <L 425>
    // if is_equality:                                                                        <L 441>
    if (var_is_equality) {
        // force = -D * jaref                                                                 <L 442>
        var_0 = wp::neg(var_D);
        var_1 = wp::mul(var_0, var_jaref);
        // cost = 0.5 * D * jaref * jaref                                                     <L 443>
        var_3 = wp::mul(var_2, var_D);
        var_4 = wp::mul(var_3, var_jaref);
        var_5 = wp::mul(var_4, var_jaref);
        // return wp.vec3(force, float(types.ConstraintState.QUADRATIC.value), cost)          <L 444>
        var_8 = wp::float(var_7);
        var_9 = wp::vec_t<3, wp::float32>(var_1, var_8, var_5);
        return var_9;
    }
    // if is_friction:                                                                        <L 446>
    if (var_is_friction) {
        // rf = math.safe_div(frictionloss, D)                                                <L 447>
        var_10 = safe_div_2(var_frictionloss, var_D);
        // if jaref <= -rf:                                                                   <L 448>
        var_11 = wp::neg(var_10);
        var_12 = (var_jaref <= var_11);
        if (var_12) {
            // return wp.vec3(frictionloss, float(types.ConstraintState.LINEARNEG.value), -frictionloss * (0.5 * rf + jaref))       <L 449>
            var_15 = wp::float(var_14);
            var_16 = wp::neg(var_frictionloss);
            var_18 = wp::mul(var_17, var_10);
            var_19 = wp::add(var_18, var_jaref);
            var_20 = wp::mul(var_16, var_19);
            var_21 = wp::vec_t<3, wp::float32>(var_frictionloss, var_15, var_20);
            return var_21;
        }
        if (!var_12) {
            // elif jaref >= rf:                                                              <L 450>
            var_22 = (var_jaref >= var_10);
            if (var_22) {
                // return wp.vec3(-frictionloss, float(types.ConstraintState.LINEARPOS.value), -frictionloss * (0.5 * rf - jaref))       <L 451>
                var_23 = wp::neg(var_frictionloss);
                var_26 = wp::float(var_25);
                var_27 = wp::neg(var_frictionloss);
                var_29 = wp::mul(var_28, var_10);
                var_30 = wp::sub(var_29, var_jaref);
                var_31 = wp::mul(var_27, var_30);
                var_32 = wp::vec_t<3, wp::float32>(var_23, var_26, var_31);
                return var_32;
            }
            if (!var_22) {
                // return wp.vec3(-D * jaref, float(types.ConstraintState.QUADRATIC.value), 0.5 * D * jaref * jaref)       <L 453>
                var_33 = wp::neg(var_D);
                var_34 = wp::mul(var_33, var_jaref);
                var_37 = wp::float(var_36);
                var_39 = wp::mul(var_38, var_D);
                var_40 = wp::mul(var_39, var_jaref);
                var_41 = wp::mul(var_40, var_jaref);
                var_42 = wp::vec_t<3, wp::float32>(var_34, var_37, var_41);
                return var_42;
            }
        }
    }
    // if is_elliptic:                                                                        <L 455>
    if (var_is_elliptic) {
        // N = jaref0 * mu                                                                    <L 456>
        var_43 = wp::mul(var_jaref0, var_mu);
        // if TT <= 0.0:                                                                      <L 457>
        var_45 = (var_TT <= var_44);
        if (var_45) {
            // T = 0.0                                                                        <L 458>
        }
        if (!var_45) {
            // T = wp.sqrt(TT)                                                                <L 460>
            var_47 = wp::sqrt(var_TT);
        }
        var_48 = wp::where(var_45, var_46, var_47);
        // if (N >= mu * T) or ((T <= 0.0) and (N >= 0.0)):                                   <L 463>
        var_50 = wp::mul(var_mu, var_48);
        var_51 = (var_43 >= var_50);
        var_49 = var_51;
        if (!var_49) {
            var_54 = (var_48 <= var_53);
            var_52 = var_54;
            if (var_52) {
                var_56 = (var_43 >= var_55);
                var_52 = var_52 && var_56;
            }
            var_49 = var_49 || var_52;
        }
        if (var_49) {
            // return wp.vec3(0.0, float(types.ConstraintState.SATISFIED.value), 0.0)         <L 464>
            var_60 = wp::float(var_59);
            var_62 = wp::vec_t<3, wp::float32>(var_57, var_60, var_61);
            return var_62;
        }
        if (!var_49) {
            // elif (mu * N + T <= 0.0) or ((T <= 0.0) and (N < 0.0)):                        <L 466>
            var_64 = wp::mul(var_mu, var_43);
            var_65 = wp::add(var_64, var_48);
            var_67 = (var_65 <= var_66);
            var_63 = var_67;
            if (!var_63) {
                var_70 = (var_48 <= var_69);
                var_68 = var_70;
                if (var_68) {
                    var_72 = (var_43 < var_71);
                    var_68 = var_68 && var_72;
                }
                var_63 = var_63 || var_68;
            }
            if (var_63) {
                // return wp.vec3(-D * jaref, float(types.ConstraintState.QUADRATIC.value), 0.5 * D * jaref * jaref)       <L 467>
                var_73 = wp::neg(var_D);
                var_74 = wp::mul(var_73, var_jaref);
                var_77 = wp::float(var_76);
                var_79 = wp::mul(var_78, var_D);
                var_80 = wp::mul(var_79, var_jaref);
                var_81 = wp::mul(var_80, var_jaref);
                var_82 = wp::vec_t<3, wp::float32>(var_74, var_77, var_81);
                return var_82;
            }
            if (!var_63) {
                // is_normal = efcid == efcid0                                                <L 470>
                var_83 = (var_efcid == var_efcid0);
                // fc = _eval_elliptic_middle(N, T, D0, mu, ufrictionj, is_normal)            <L 471>
                var_84 = _eval_elliptic_middle_0(var_43, var_48, var_D0, var_mu, var_ufrictionj, var_83);
                // return wp.vec3(fc[0], float(types.ConstraintState.CONE.value), fc[1])       <L 472>
                var_86 = wp::extract(var_84, var_85);
                var_89 = wp::float(var_88);
                var_91 = wp::extract(var_84, var_90);
                var_92 = wp::vec_t<3, wp::float32>(var_86, var_89, var_91);
                return var_92;
            }
        }
    }
    // if jaref >= 0.0:                                                                       <L 474>
    var_94 = (var_jaref >= var_93);
    if (var_94) {
        // return wp.vec3(0.0, float(types.ConstraintState.SATISFIED.value), 0.0)             <L 475>
        var_98 = wp::float(var_97);
        var_100 = wp::vec_t<3, wp::float32>(var_95, var_98, var_99);
        return var_100;
    }
    if (!var_94) {
        // return wp.vec3(-D * jaref, float(types.ConstraintState.QUADRATIC.value), 0.5 * D * jaref * jaref)       <L 477>
        var_101 = wp::neg(var_D);
        var_102 = wp::mul(var_101, var_jaref);
        var_105 = wp::float(var_104);
        var_107 = wp::mul(var_106, var_D);
        var_108 = wp::mul(var_107, var_jaref);
        var_109 = wp::mul(var_108, var_jaref);
        var_110 = wp::vec_t<3, wp::float32>(var_102, var_105, var_109);
        return var_110;
    }
    return {};
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


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:405
static CUDA_CALLABLE void adj__eval_elliptic_middle_0(
    wp::float32 var_N,
    wp::float32 var_T,
    wp::float32 var_D0,
    wp::float32 var_mu,
    wp::float32 var_ufrictionj,
    bool var_is_normal,
    wp::float32 & adj_N,
    wp::float32 & adj_T,
    wp::float32 & adj_D0,
    wp::float32 & adj_mu,
    wp::float32 & adj_ufrictionj,
    bool & adj_is_normal,
    wp::vec_t<2, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py:424
static CUDA_CALLABLE void adj__eval_constraint_0(
    bool var_is_equality,
    bool var_is_friction,
    bool var_is_elliptic,
    wp::float32 var_jaref,
    wp::float32 var_D,
    wp::float32 var_frictionloss,
    wp::int32 var_efcid,
    wp::int32 var_efcid0,
    wp::float32 var_jaref0,
    wp::float32 var_D0,
    wp::float32 var_mu,
    wp::float32 var_ufrictionj,
    wp::float32 var_TT,
    bool & adj_is_equality,
    bool & adj_is_friction,
    bool & adj_is_elliptic,
    wp::float32 & adj_jaref,
    wp::float32 & adj_D,
    wp::float32 & adj_frictionloss,
    wp::int32 & adj_efcid,
    wp::int32 & adj_efcid0,
    wp::float32 & adj_jaref0,
    wp::float32 & adj_D0,
    wp::float32 & adj_mu,
    wp::float32 & adj_ufrictionj,
    wp::float32 & adj_TT,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void _update_constraint_efc__locals__kernel_d88e782a_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_impratio_invsqrt,
    wp::array_t<wp::int32> var_ne_in,
    wp::array_t<wp::int32> var_nf_in,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_efc_type_in,
    wp::array_t<wp::int32> var_efc_id_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::float32> var_efc_frictionloss_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::float32> var_ctx_Jaref_in,
    wp::array_t<bool> var_ctx_ls_exhausted_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_efc_force_out,
    wp::array_t<wp::int32> var_efc_state_out,
    wp::array_t<wp::int32> var_quad_changed_ids_out,
    wp::array_t<wp::int32> var_quad_changed_count_out,
    wp::array_t<wp::int32> var_state_changed_count_out)
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
        const bool var_5 = false;
        wp::int32* var_6;
        bool var_7;
        wp::int32 var_8;
        const bool var_9 = false;
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::int32* var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        bool var_22;
        bool var_23;
        bool var_24;
        wp::int32 var_25;
        bool var_26;
        wp::int32* var_27;
        const wp::int32 var_28 = 7;
        bool var_29;
        wp::int32 var_30;
        wp::float32* var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        const wp::float32 var_34 = 0.0;
        wp::float32 var_35;
        const wp::int32 var_36 = -1;
        const wp::float32 var_37 = 0.0;
        wp::float32 var_38;
        const wp::float32 var_39 = 0.0;
        wp::float32 var_40;
        const wp::float32 var_41 = 0.0;
        wp::float32 var_42;
        const wp::float32 var_43 = 0.0;
        wp::float32 var_44;
        const wp::float32 var_45 = 0.0;
        wp::float32 var_46;
        wp::int32* var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        const wp::int32 var_50 = 0;
        wp::int32* var_51;
        bool var_52;
        wp::int32 var_53;
        const wp::int32 var_54 = 0;
        wp::int32* var_55;
        wp::int32 var_56;
        wp::int32 var_57;
        const wp::int32 var_58 = 0;
        bool var_59;
        wp::int32* var_60;
        wp::int32 var_61;
        wp::int32 var_62;
        wp::vec_t<5, wp::float32>* var_63;
        wp::vec_t<5, wp::float32> var_64;
        wp::vec_t<5, wp::float32> var_65;
        const wp::int32 var_66 = 0;
        wp::float32 var_67;
        wp::shape_t* var_68;
        const wp::int32 var_69 = 0;
        wp::int32 var_70;
        wp::shape_t var_71;
        wp::int32 var_72;
        wp::float32* var_73;
        wp::float32 var_74;
        wp::float32 var_75;
        wp::float32* var_76;
        wp::float32 var_77;
        wp::float32 var_78;
        wp::float32* var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        const wp::int32 var_82 = 1;
        wp::range_t var_83;
        wp::int32 var_84;
        wp::int32* var_85;
        wp::int32 var_86;
        wp::int32 var_87;
        const wp::int32 var_88 = 0;
        bool var_89;
        const wp::int32 var_90 = 1;
        wp::int32 var_91;
        wp::float32 var_92;
        wp::float32* var_93;
        wp::float32 var_94;
        wp::float32 var_95;
        wp::float32 var_96;
        wp::float32 var_97;
        bool var_98;
        wp::float32 var_99;
        wp::float32 var_100;
        wp::int32 var_101;
        wp::float32 var_102;
        wp::float32 var_103;
        wp::float32 var_104;
        wp::vec_t<3, wp::float32> var_105;
        const wp::int32 var_106 = 1;
        wp::float32 var_107;
        wp::int32 var_108;
        const wp::int32 var_109 = 0;
        wp::float32 var_110;
        const bool var_111 = false;
        //---------
        // forward
        // def kernel(                                                                            <L 1703>
        // worldid, efcid = wp.tid()                                                              <L 1730>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 1732>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 1733>
            return;
        }
        var_4 = wp::load(var_2);
        // if wp.static(TRACK_CHANGES):                                                           <L 1737>
        // if efcid >= nefc_in[worldid]:                                                          <L 1741>
        var_6 = wp::address(var_nefc_in, var_0);
        var_8 = wp::load(var_6);
        var_7 = (var_1 >= var_8);
        if (var_7) {
            // return                                                                             <L 1742>
            return;
        }
        // if wp.static(TRACK_CHANGES):                                                           <L 1745>
        // efc_D = efc_D_in[worldid, efcid]                                                       <L 1748>
        var_10 = wp::address(var_efc_D_in, var_0, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // Jaref = ctx_Jaref_in[worldid, efcid]                                                   <L 1749>
        var_13 = wp::address(var_ctx_Jaref_in, var_0, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // ne = ne_in[worldid]                                                                    <L 1751>
        var_16 = wp::address(var_ne_in, var_0);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // nf = nf_in[worldid]                                                                    <L 1752>
        var_19 = wp::address(var_nf_in, var_0);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // is_equality = efcid < ne                                                               <L 1754>
        var_22 = (var_1 < var_17);
        // is_friction = (not is_equality) and (efcid < ne + nf)                                  <L 1755>
        var_24 = wp::unot(var_22);
        var_23 = var_24;
        if (var_23) {
            var_25 = wp::add(var_17, var_20);
            var_26 = (var_1 < var_25);
            var_23 = var_23 && var_26;
        }
        // is_elliptic = efc_type_in[worldid, efcid] == types.ConstraintType.CONTACT_ELLIPTIC       <L 1756>
        var_27 = wp::address(var_efc_type_in, var_0, var_1);
        var_30 = wp::load(var_27);
        var_29 = (var_30 == var_28);
        // frictionloss = efc_frictionloss_in[worldid, efcid] if is_friction else 0.0             <L 1758>
        if (var_23) {
            var_31 = wp::address(var_efc_frictionloss_in, var_0, var_1);
            var_33 = wp::load(var_31);
            var_32 = wp::copy(var_33);
        }
        if (!var_23) {
        }
        var_35 = wp::where(var_23, var_32, var_34);
        // efcid0 = -1                                                                            <L 1760>
        // jaref0 = float(0.0)                                                                    <L 1761>
        var_38 = wp::float(var_37);
        // D0 = float(0.0)                                                                        <L 1762>
        var_40 = wp::float(var_39);
        // mu = float(0.0)                                                                        <L 1763>
        var_42 = wp::float(var_41);
        // ufrictionj = float(0.0)                                                                <L 1764>
        var_44 = wp::float(var_43);
        // TT = float(0.0)                                                                        <L 1765>
        var_46 = wp::float(var_45);
        // if is_elliptic:                                                                        <L 1767>
        if (var_29) {
            // conid = efc_id_in[worldid, efcid]                                                  <L 1768>
            var_47 = wp::address(var_efc_id_in, var_0, var_1);
            var_49 = wp::load(var_47);
            var_48 = wp::copy(var_49);
            // if conid >= nacon_in[0]:                                                           <L 1769>
            var_51 = wp::address(var_nacon_in, var_50);
            var_53 = wp::load(var_51);
            var_52 = (var_48 >= var_53);
            if (var_52) {
                // return                                                                         <L 1770>
                return;
            }
            // efcid0 = contact_efc_address_in[conid, 0]                                          <L 1771>
            var_55 = wp::address(var_contact_efc_address_in, var_48, var_54);
            var_57 = wp::load(var_55);
            var_56 = wp::copy(var_57);
            // if efcid0 < 0:                                                                     <L 1772>
            var_59 = (var_56 < var_58);
            if (var_59) {
                // return                                                                         <L 1773>
                return;
            }
            // dim = contact_dim_in[conid]                                                        <L 1775>
            var_60 = wp::address(var_contact_dim_in, var_48);
            var_62 = wp::load(var_60);
            var_61 = wp::copy(var_62);
            // friction = contact_friction_in[conid]                                              <L 1776>
            var_63 = wp::address(var_contact_friction_in, var_48);
            var_65 = wp::load(var_63);
            var_64 = wp::copy(var_65);
            // mu = friction[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]       <L 1777>
            var_67 = wp::extract(var_64, var_66);
            var_68 = &(var_opt_impratio_invsqrt.shape);
            var_71 = wp::load(var_68);
            var_70 = wp::extract(var_71, var_69);
            var_72 = wp::mod(var_0, var_70);
            var_73 = wp::address(var_opt_impratio_invsqrt, var_72);
            var_75 = wp::load(var_73);
            var_74 = wp::mul(var_67, var_75);
            // jaref0 = ctx_Jaref_in[worldid, efcid0]                                             <L 1778>
            var_76 = wp::address(var_ctx_Jaref_in, var_0, var_56);
            var_78 = wp::load(var_76);
            var_77 = wp::copy(var_78);
            // D0 = efc_D_in[worldid, efcid0]                                                     <L 1779>
            var_79 = wp::address(var_efc_D_in, var_0, var_56);
            var_81 = wp::load(var_79);
            var_80 = wp::copy(var_81);
            // for j in range(1, dim):                                                            <L 1781>
            var_83 = wp::range(var_82, var_61);
            start_for_4:;
                if (iter_cmp(var_83) == 0) goto end_for_4;
                var_84 = wp::iter_next(var_83);
                // efcidj = contact_efc_address_in[conid, j]                                      <L 1782>
                var_85 = wp::address(var_contact_efc_address_in, var_48, var_84);
                var_87 = wp::load(var_85);
                var_86 = wp::copy(var_87);
                // if efcidj < 0:                                                                 <L 1783>
                var_89 = (var_86 < var_88);
                if (var_89) {
                    // return                                                                     <L 1784>
                    return;
                }
                // frictionj = friction[j - 1]                                                    <L 1785>
                var_91 = wp::sub(var_84, var_90);
                var_92 = wp::extract(var_64, var_91);
                // uj = ctx_Jaref_in[worldid, efcidj] * frictionj                                 <L 1786>
                var_93 = wp::address(var_ctx_Jaref_in, var_0, var_86);
                var_95 = wp::load(var_93);
                var_94 = wp::mul(var_95, var_92);
                // TT += uj * uj                                                                  <L 1787>
                var_96 = wp::mul(var_94, var_94);
                var_97 = wp::add(var_46, var_96);
                // if efcid == efcidj:                                                            <L 1788>
                var_98 = (var_1 == var_86);
                if (var_98) {
                    // ufrictionj = uj * frictionj                                                <L 1789>
                    var_99 = wp::mul(var_94, var_92);
                }
                var_100 = wp::where(var_98, var_99, var_44);
                wp::assign(var_44, var_100);
                wp::assign(var_46, var_97);
                goto start_for_4;
            end_for_4:;
        }
        var_101 = wp::where(var_29, var_56, var_36);
        var_102 = wp::where(var_29, var_77, var_38);
        var_103 = wp::where(var_29, var_80, var_40);
        var_104 = wp::where(var_29, var_74, var_42);
        // res = _eval_constraint(                                                                <L 1791>
        // is_equality,                                                                           <L 1792>
        // is_friction,                                                                           <L 1793>
        // is_elliptic,                                                                           <L 1794>
        // Jaref,                                                                                 <L 1795>
        // efc_D,                                                                                 <L 1796>
        // frictionloss,                                                                          <L 1797>
        // efcid,                                                                                 <L 1798>
        // efcid0,                                                                                <L 1799>
        // jaref0,                                                                                <L 1800>
        // D0,                                                                                    <L 1801>
        // mu,                                                                                    <L 1802>
        // ufrictionj,                                                                            <L 1803>
        // TT,                                                                                    <L 1804>
        var_105 = _eval_constraint_0(var_22, var_23, var_29, var_14, var_11, var_35, var_1, var_101, var_102, var_103, var_104, var_44, var_46);
        // new_state = int(res[1])                                                                <L 1807>
        var_107 = wp::extract(var_105, var_106);
        var_108 = wp::int(var_107);
        // efc_force_out[worldid, efcid] = res[0]                                                 <L 1808>
        var_110 = wp::extract(var_105, var_109);
        wp::array_store(var_efc_force_out, var_0, var_1, var_110);
        // efc_state_out[worldid, efcid] = new_state                                              <L 1809>
        wp::array_store(var_efc_state_out, var_0, var_1, var_108);
        // if wp.static(TRACK_CHANGES):                                                           <L 1811>
}

