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


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/math.py:52
static CUDA_CALLABLE wp::quat_t<wp::float32> axis_angle_to_quat_0(
    wp::vec_t<3, wp::float32> var_axis,
    wp::float32 var_angle)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.5;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::float32 var_3 = 0.5;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    const wp::int32 var_7 = 0;
    wp::float32 var_8;
    const wp::int32 var_9 = 1;
    wp::float32 var_10;
    const wp::int32 var_11 = 2;
    wp::float32 var_12;
    wp::quat_t<wp::float32> var_13;
    //---------
    // forward
    // def axis_angle_to_quat(axis: wp.vec3, angle: float) -> wp.quat:                        <L 53>
    // s, c = wp.sin(angle * 0.5), wp.cos(angle * 0.5)                                        <L 54>
    var_1 = wp::mul(var_angle, var_0);
    var_2 = wp::sin(var_1);
    var_4 = wp::mul(var_angle, var_3);
    var_5 = wp::cos(var_4);
    // axis = axis * s                                                                        <L 55>
    var_6 = wp::mul(var_axis, var_2);
    // return wp.quat(c, axis[0], axis[1], axis[2])                                           <L 56>
    var_8 = wp::extract(var_6, var_7);
    var_10 = wp::extract(var_6, var_9);
    var_12 = wp::extract(var_6, var_11);
    var_13 = wp::quat_t<wp::float32>(var_5, var_8, var_10, var_12);
    return var_13;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/math.py:23
static CUDA_CALLABLE wp::quat_t<wp::float32> mul_quat_0(
    wp::quat_t<wp::float32> var_u,
    wp::quat_t<wp::float32> var_v)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 1;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::int32 var_11 = 2;
    wp::float32 var_12;
    const wp::int32 var_13 = 2;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 3;
    wp::float32 var_18;
    const wp::int32 var_19 = 3;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::int32 var_23 = 0;
    wp::float32 var_24;
    const wp::int32 var_25 = 1;
    wp::float32 var_26;
    wp::float32 var_27;
    const wp::int32 var_28 = 1;
    wp::float32 var_29;
    const wp::int32 var_30 = 0;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::int32 var_34 = 2;
    wp::float32 var_35;
    const wp::int32 var_36 = 3;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::int32 var_40 = 3;
    wp::float32 var_41;
    const wp::int32 var_42 = 2;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    const wp::int32 var_46 = 0;
    wp::float32 var_47;
    const wp::int32 var_48 = 2;
    wp::float32 var_49;
    wp::float32 var_50;
    const wp::int32 var_51 = 1;
    wp::float32 var_52;
    const wp::int32 var_53 = 3;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    const wp::int32 var_57 = 2;
    wp::float32 var_58;
    const wp::int32 var_59 = 0;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    const wp::int32 var_63 = 3;
    wp::float32 var_64;
    const wp::int32 var_65 = 1;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    const wp::int32 var_69 = 0;
    wp::float32 var_70;
    const wp::int32 var_71 = 3;
    wp::float32 var_72;
    wp::float32 var_73;
    const wp::int32 var_74 = 1;
    wp::float32 var_75;
    const wp::int32 var_76 = 2;
    wp::float32 var_77;
    wp::float32 var_78;
    wp::float32 var_79;
    const wp::int32 var_80 = 2;
    wp::float32 var_81;
    const wp::int32 var_82 = 1;
    wp::float32 var_83;
    wp::float32 var_84;
    wp::float32 var_85;
    const wp::int32 var_86 = 3;
    wp::float32 var_87;
    const wp::int32 var_88 = 0;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    wp::quat_t<wp::float32> var_92;
    //---------
    // forward
    // def mul_quat(u: wp.quat, v: wp.quat) -> wp.quat:                                       <L 24>
    // return wp.quat(                                                                        <L 25>
    // u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],                                 <L 26>
    var_1 = wp::extract(var_u, var_0);
    var_3 = wp::extract(var_v, var_2);
    var_4 = wp::mul(var_1, var_3);
    var_6 = wp::extract(var_u, var_5);
    var_8 = wp::extract(var_v, var_7);
    var_9 = wp::mul(var_6, var_8);
    var_10 = wp::sub(var_4, var_9);
    var_12 = wp::extract(var_u, var_11);
    var_14 = wp::extract(var_v, var_13);
    var_15 = wp::mul(var_12, var_14);
    var_16 = wp::sub(var_10, var_15);
    var_18 = wp::extract(var_u, var_17);
    var_20 = wp::extract(var_v, var_19);
    var_21 = wp::mul(var_18, var_20);
    var_22 = wp::sub(var_16, var_21);
    // u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],                                 <L 27>
    var_24 = wp::extract(var_u, var_23);
    var_26 = wp::extract(var_v, var_25);
    var_27 = wp::mul(var_24, var_26);
    var_29 = wp::extract(var_u, var_28);
    var_31 = wp::extract(var_v, var_30);
    var_32 = wp::mul(var_29, var_31);
    var_33 = wp::add(var_27, var_32);
    var_35 = wp::extract(var_u, var_34);
    var_37 = wp::extract(var_v, var_36);
    var_38 = wp::mul(var_35, var_37);
    var_39 = wp::add(var_33, var_38);
    var_41 = wp::extract(var_u, var_40);
    var_43 = wp::extract(var_v, var_42);
    var_44 = wp::mul(var_41, var_43);
    var_45 = wp::sub(var_39, var_44);
    // u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],                                 <L 28>
    var_47 = wp::extract(var_u, var_46);
    var_49 = wp::extract(var_v, var_48);
    var_50 = wp::mul(var_47, var_49);
    var_52 = wp::extract(var_u, var_51);
    var_54 = wp::extract(var_v, var_53);
    var_55 = wp::mul(var_52, var_54);
    var_56 = wp::sub(var_50, var_55);
    var_58 = wp::extract(var_u, var_57);
    var_60 = wp::extract(var_v, var_59);
    var_61 = wp::mul(var_58, var_60);
    var_62 = wp::add(var_56, var_61);
    var_64 = wp::extract(var_u, var_63);
    var_66 = wp::extract(var_v, var_65);
    var_67 = wp::mul(var_64, var_66);
    var_68 = wp::add(var_62, var_67);
    // u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],                                 <L 29>
    var_70 = wp::extract(var_u, var_69);
    var_72 = wp::extract(var_v, var_71);
    var_73 = wp::mul(var_70, var_72);
    var_75 = wp::extract(var_u, var_74);
    var_77 = wp::extract(var_v, var_76);
    var_78 = wp::mul(var_75, var_77);
    var_79 = wp::add(var_73, var_78);
    var_81 = wp::extract(var_u, var_80);
    var_83 = wp::extract(var_v, var_82);
    var_84 = wp::mul(var_81, var_83);
    var_85 = wp::sub(var_79, var_84);
    var_87 = wp::extract(var_u, var_86);
    var_89 = wp::extract(var_v, var_88);
    var_90 = wp::mul(var_87, var_89);
    var_91 = wp::add(var_85, var_90);
    var_92 = wp::quat_t<wp::float32>(var_22, var_45, var_68, var_91);
    return var_92;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/math.py:188
static CUDA_CALLABLE wp::quat_t<wp::float32> quat_integrate_0(
    wp::quat_t<wp::float32> var_q,
    wp::vec_t<3, wp::float32> var_v,
    wp::float32 var_dt)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::float32 var_2;
    wp::quat_t<wp::float32> var_3;
    wp::quat_t<wp::float32> var_4;
    wp::quat_t<wp::float32> var_5;
    wp::quat_t<wp::float32> var_6;
    //---------
    // forward
    // def quat_integrate(q: wp.quat, v: wp.vec3, dt: float) -> wp.quat:                      <L 189>
    // norm_ = wp.length(v)                                                                   <L 191>
    var_0 = wp::length(var_v);
    // v = wp.normalize(v)  # does that need proper zero gradient handling?                   <L 192>
    var_1 = wp::normalize(var_v);
    // angle = dt * norm_                                                                     <L 193>
    var_2 = wp::mul(var_dt, var_0);
    // q_res = axis_angle_to_quat(v, angle)                                                   <L 195>
    var_3 = axis_angle_to_quat_0(var_1, var_2);
    // q = wp.normalize(q)                                                                    <L 196>
    var_4 = wp::normalize(var_q);
    // q_res = mul_quat(q, q_res)                                                             <L 197>
    var_5 = mul_quat_0(var_4, var_3);
    // return wp.normalize(q_res)                                                             <L 199>
    var_6 = wp::normalize(var_5);
    return var_6;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:554
static CUDA_CALLABLE wp::float32 _sigmoid_0(
    wp::float32 var_x)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 1.0;
    bool var_4;
    const wp::float32 var_5 = 1.0;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::float32 var_8 = 3.0;
    wp::float32 var_9;
    const wp::float32 var_10 = 2.0;
    wp::float32 var_11;
    const wp::float32 var_12 = 5.0;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::float32 var_15 = 10.0;
    wp::float32 var_16;
    wp::float32 var_17;
    //---------
    // forward
    // def _sigmoid(x: float) -> float:                                                       <L 555>
    // if x <= 0.0:                                                                           <L 557>
    var_1 = (var_x <= var_0);
    if (var_1) {
        // return 0.0                                                                         <L 558>
        return var_2;
    }
    // if x >= 1.0:                                                                           <L 560>
    var_4 = (var_x >= var_3);
    if (var_4) {
        // return 1.0                                                                         <L 561>
        return var_5;
    }
    // return x * x * x * (3.0 * x * (2.0 * x - 5.0) + 10.0)                                  <L 565>
    var_6 = wp::mul(var_x, var_x);
    var_7 = wp::mul(var_6, var_x);
    var_9 = wp::mul(var_8, var_x);
    var_11 = wp::mul(var_10, var_x);
    var_13 = wp::sub(var_11, var_12);
    var_14 = wp::mul(var_9, var_13);
    var_16 = wp::add(var_14, var_15);
    var_17 = wp::mul(var_7, var_16);
    return var_17;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:568
static CUDA_CALLABLE wp::float32 muscle_dynamics_timescale_0(
    wp::float32 var_dctrl,
    wp::float32 var_tau_act,
    wp::float32 var_tau_deact,
    wp::float32 var_smooth_width)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 1e-15;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    bool var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::float32 var_6 = 0.5;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    //---------
    // forward
    // def muscle_dynamics_timescale(dctrl: float, tau_act: float, tau_deact: float, smooth_width: float) -> float:       <L 569>
    // if smooth_width < MJ_MINVAL:                                                           <L 572>
    var_1 = (var_smooth_width < var_0);
    if (var_1) {
        // if dctrl > 0.0:                                                                    <L 573>
        var_3 = (var_dctrl > var_2);
        if (var_3) {
            // return tau_act                                                                 <L 574>
            return var_tau_act;
        }
        if (!var_3) {
            // return tau_deact                                                               <L 576>
            return var_tau_deact;
        }
    }
    if (!var_1) {
        // return tau_deact + (tau_act - tau_deact) * _sigmoid(dctrl / smooth_width + 0.5)       <L 579>
        var_4 = wp::sub(var_tau_act, var_tau_deact);
        var_5 = wp::div(var_dctrl, var_smooth_width);
        var_7 = wp::add(var_5, var_6);
        var_8 = _sigmoid_0(var_7);
        var_9 = wp::mul(var_4, var_8);
        var_10 = wp::add(var_tau_deact, var_9);
        return var_10;
    }
    return {};
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:582
static CUDA_CALLABLE wp::float32 muscle_dynamics_0(
    wp::float32 var_control,
    wp::float32 var_activation,
    wp::vec_t<10, wp::float32> var_prm)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 1.0;
    wp::float32 var_2;
    const wp::float32 var_3 = 0.0;
    const wp::float32 var_4 = 1.0;
    wp::float32 var_5;
    const wp::int32 var_6 = 0;
    wp::float32 var_7;
    const wp::float32 var_8 = 0.5;
    const wp::float32 var_9 = 1.5;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    const wp::int32 var_13 = 1;
    wp::float32 var_14;
    const wp::float32 var_15 = 0.5;
    const wp::float32 var_16 = 1.5;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    const wp::int32 var_20 = 2;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    const wp::float32 var_24 = 1e-15;
    wp::float32 var_25;
    wp::float32 var_26;
    //---------
    // forward
    // def muscle_dynamics(control: float, activation: float, prm: vec10) -> float:           <L 583>
    // ctrlclamp = wp.clamp(control, 0.0, 1.0)                                                <L 586>
    var_2 = wp::clamp(var_control, var_0, var_1);
    // actclamp = wp.clamp(activation, 0.0, 1.0)                                              <L 589>
    var_5 = wp::clamp(var_activation, var_3, var_4);
    // tau_act = prm[0] * (0.5 + 1.5 * actclamp)  # activation timescale                      <L 592>
    var_7 = wp::extract(var_prm, var_6);
    var_10 = wp::mul(var_9, var_5);
    var_11 = wp::add(var_8, var_10);
    var_12 = wp::mul(var_7, var_11);
    // tau_deact = prm[1] / (0.5 + 1.5 * actclamp)  # deactivation timescale                  <L 593>
    var_14 = wp::extract(var_prm, var_13);
    var_17 = wp::mul(var_16, var_5);
    var_18 = wp::add(var_15, var_17);
    var_19 = wp::div(var_14, var_18);
    // smooth_width = prm[2]  # width of smoothing sigmoid                                    <L 594>
    var_21 = wp::extract(var_prm, var_20);
    // dctrl = ctrlclamp - activation  # excess excitation                                    <L 595>
    var_22 = wp::sub(var_2, var_activation);
    // tau = muscle_dynamics_timescale(dctrl, tau_act, tau_deact, smooth_width)               <L 597>
    var_23 = muscle_dynamics_timescale_0(var_22, var_12, var_19, var_21);
    // return dctrl / wp.max(MJ_MINVAL, tau)                                                  <L 600>
    var_25 = wp::max(var_24, var_23);
    var_26 = wp::div(var_22, var_25);
    return var_26;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:603
static CUDA_CALLABLE wp::vec_t<6, wp::int32> dcmotor_slots_0(
    wp::vec_t<10, wp::float32> var_dynprm,
    wp::vec_t<10, wp::float32> var_gainprm)
{
    //---------
    // primal vars
    const wp::int32 var_0 = -1;
    const wp::int32 var_1 = -1;
    const wp::int32 var_2 = -1;
    const wp::int32 var_3 = -1;
    const wp::int32 var_4 = -1;
    const wp::int32 var_5 = 0;
    wp::vec_t<6, wp::int32> var_6;
    const wp::int32 var_7 = 0;
    const wp::int32 var_8 = 7;
    wp::float32 var_9;
    const wp::float32 var_10 = 0.0;
    bool var_11;
    const wp::int32 var_12 = 0;
    const wp::int32 var_13 = 1;
    wp::int32 var_14;
    wp::int32 var_15;
    const wp::int32 var_16 = 5;
    wp::float32 var_17;
    const wp::float32 var_18 = 0.0;
    bool var_19;
    const wp::int32 var_20 = 1;
    const wp::int32 var_21 = 1;
    wp::int32 var_22;
    wp::int32 var_23;
    const wp::int32 var_24 = 2;
    wp::float32 var_25;
    const wp::float32 var_26 = 0.0;
    bool var_27;
    const wp::int32 var_28 = 2;
    const wp::int32 var_29 = 1;
    wp::int32 var_30;
    wp::int32 var_31;
    const wp::int32 var_32 = 5;
    wp::float32 var_33;
    const wp::float32 var_34 = 0.0;
    bool var_35;
    const wp::int32 var_36 = 3;
    const wp::int32 var_37 = 1;
    wp::int32 var_38;
    wp::int32 var_39;
    const wp::int32 var_40 = 0;
    wp::float32 var_41;
    const wp::float32 var_42 = 0.0;
    bool var_43;
    const wp::int32 var_44 = 4;
    const wp::int32 var_45 = 1;
    wp::int32 var_46;
    wp::int32 var_47;
    const wp::int32 var_48 = 5;
    //---------
    // forward
    // def dcmotor_slots(dynprm: types.vec10, gainprm: types.vec10) -> types.vec6i:           <L 604>
    // s = types.vec6i(-1, -1, -1, -1, -1, 0)                                                 <L 622>
    var_6 = wp::vec_t<6, wp::int32>({var_0, var_1, var_2, var_3, var_4, var_5});
    // num_slots = 0                                                                          <L 623>
    // if dynprm[7] > 0.0:                                                                    <L 624>
    var_9 = wp::extract(var_dynprm, var_8);
    var_11 = (var_9 > var_10);
    if (var_11) {
        // s[0] = num_slots                                                                   <L 625>
        wp::assign_inplace(var_6, var_12, var_7);
        // num_slots += 1                                                                     <L 626>
        var_14 = wp::add(var_7, var_13);
    }
    var_15 = wp::where(var_11, var_14, var_7);
    // if gainprm[5] > 0.0:                                                                   <L 627>
    var_17 = wp::extract(var_gainprm, var_16);
    var_19 = (var_17 > var_18);
    if (var_19) {
        // s[1] = num_slots                                                                   <L 628>
        wp::assign_inplace(var_6, var_20, var_15);
        // num_slots += 1                                                                     <L 629>
        var_22 = wp::add(var_15, var_21);
    }
    var_23 = wp::where(var_19, var_22, var_15);
    // if dynprm[2] > 0.0:                                                                    <L 630>
    var_25 = wp::extract(var_dynprm, var_24);
    var_27 = (var_25 > var_26);
    if (var_27) {
        // s[2] = num_slots                                                                   <L 631>
        wp::assign_inplace(var_6, var_28, var_23);
        // num_slots += 1                                                                     <L 632>
        var_30 = wp::add(var_23, var_29);
    }
    var_31 = wp::where(var_27, var_30, var_23);
    // if dynprm[5] > 0.0:                                                                    <L 633>
    var_33 = wp::extract(var_dynprm, var_32);
    var_35 = (var_33 > var_34);
    if (var_35) {
        // s[3] = num_slots                                                                   <L 634>
        wp::assign_inplace(var_6, var_36, var_31);
        // num_slots += 1                                                                     <L 635>
        var_38 = wp::add(var_31, var_37);
    }
    var_39 = wp::where(var_35, var_38, var_31);
    // if dynprm[0] > 0.0:                                                                    <L 636>
    var_41 = wp::extract(var_dynprm, var_40);
    var_43 = (var_41 > var_42);
    if (var_43) {
        // s[4] = num_slots                                                                   <L 637>
        wp::assign_inplace(var_6, var_44, var_39);
        // num_slots += 1                                                                     <L 638>
        var_46 = wp::add(var_39, var_45);
    }
    var_47 = wp::where(var_43, var_46, var_39);
    // s[5] = num_slots                                                                       <L 639>
    wp::assign_inplace(var_6, var_48, var_47);
    // return s                                                                               <L 640>
    return var_6;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:649
static CUDA_CALLABLE wp::float32 dcmotor_voltage_0(
    wp::float32 var_u,
    wp::float32 var_length,
    wp::float32 var_velocity,
    wp::float32 var_x_I,
    wp::vec_t<10, wp::float32> var_gainprm)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 8;
    wp::float32 var_1;
    wp::int32 var_2;
    const wp::int32 var_3 = 7;
    wp::float32 var_4;
    const wp::float32 var_5 = 0.0;
    const wp::int32 var_6 = 0;
    bool var_7;
    const wp::int32 var_8 = 4;
    wp::float32 var_9;
    const wp::int32 var_10 = 5;
    wp::float32 var_11;
    const wp::int32 var_12 = 6;
    wp::float32 var_13;
    const wp::int32 var_14 = 1;
    bool var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    const wp::float32 var_30 = 0.0;
    bool var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    //---------
    // forward
    // def dcmotor_voltage(u: float, length: float, velocity: float, x_I: float, gainprm: types.vec10) -> float:       <L 650>
    // input_mode = int(gainprm[8])                                                           <L 651>
    var_1 = wp::extract(var_gainprm, var_0);
    var_2 = wp::int(var_1);
    // Vmax = gainprm[7]                                                                      <L 652>
    var_4 = wp::extract(var_gainprm, var_3);
    // voltage = 0.0                                                                          <L 653>
    // if input_mode > 0:                                                                     <L 655>
    var_7 = (var_2 > var_6);
    if (var_7) {
        // kp = gainprm[4]                                                                    <L 656>
        var_9 = wp::extract(var_gainprm, var_8);
        // ki = gainprm[5]                                                                    <L 657>
        var_11 = wp::extract(var_gainprm, var_10);
        // kd = gainprm[6]                                                                    <L 658>
        var_13 = wp::extract(var_gainprm, var_12);
        // if input_mode == 1:                                                                <L 660>
        var_15 = (var_2 == var_14);
        if (var_15) {
            // voltage = kp * (u - length) + ki * x_I - kd * velocity                         <L 662>
            var_16 = wp::sub(var_u, var_length);
            var_17 = wp::mul(var_9, var_16);
            var_18 = wp::mul(var_11, var_x_I);
            var_19 = wp::add(var_17, var_18);
            var_20 = wp::mul(var_13, var_velocity);
            var_21 = wp::sub(var_19, var_20);
        }
        if (!var_15) {
            // voltage = kp * (u - velocity) + ki * (x_I - length)                            <L 665>
            var_22 = wp::sub(var_u, var_velocity);
            var_23 = wp::mul(var_9, var_22);
            var_24 = wp::sub(var_x_I, var_length);
            var_25 = wp::mul(var_11, var_24);
            var_26 = wp::add(var_23, var_25);
        }
        var_27 = wp::where(var_15, var_21, var_26);
    }
    if (!var_7) {
        // voltage = u                                                                        <L 667>
        var_28 = wp::copy(var_u);
    }
    var_29 = wp::where(var_7, var_27, var_28);
    // if Vmax > 0.0:                                                                         <L 669>
    var_31 = (var_4 > var_30);
    if (var_31) {
        // voltage = wp.clamp(voltage, -Vmax, Vmax)                                           <L 670>
        var_32 = wp::neg(var_4);
        var_33 = wp::clamp(var_29, var_32, var_4);
    }
    var_34 = wp::where(var_31, var_33, var_29);
    // return voltage                                                                         <L 672>
    return var_34;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:643
static CUDA_CALLABLE wp::float32 lugre_stribeck_0(
    wp::float32 var_velocity,
    wp::float32 var_F_C,
    wp::float32 var_F_S,
    wp::float32 var_v_S)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 1e-15;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    //---------
    // forward
    // def lugre_stribeck(velocity: float, F_C: float, F_S: float, v_S: float) -> float:       <L 644>
    // ratio = velocity / wp.max(MJ_MINVAL, v_S)                                              <L 645>
    var_1 = wp::max(var_0, var_v_S);
    var_2 = wp::div(var_velocity, var_1);
    // return F_C + (F_S - F_C) * wp.exp(-ratio * ratio)                                      <L 646>
    var_3 = wp::sub(var_F_S, var_F_C);
    var_4 = wp::neg(var_2);
    var_5 = wp::mul(var_4, var_2);
    var_6 = wp::exp(var_5);
    var_7 = wp::mul(var_3, var_6);
    var_8 = wp::add(var_F_C, var_7);
    return var_8;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/support.py:37
static CUDA_CALLABLE wp::float32 next_act_0(
    wp::float32 var_opt_timestep,
    wp::int32 var_actuator_dyntype,
    wp::vec_t<10, wp::float32> var_actuator_dynprm,
    wp::vec_t<2, wp::float32> var_actuator_actrange,
    wp::float32 var_act_in,
    wp::float32 var_act_dot_in,
    wp::float32 var_act_dot_scale,
    bool var_clamp)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 3;
    bool var_1;
    const wp::float32 var_2 = 1e-15;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::float32 var_8 = 1.0;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::int32 var_15 = 7;
    bool var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    const wp::int32 var_21 = 0;
    wp::float32 var_22;
    const wp::int32 var_23 = 1;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    //---------
    // forward
    // def next_act(                                                                          <L 38>
    // if actuator_dyntype == DynType.FILTEREXACT:                                            <L 52>
    var_1 = (var_actuator_dyntype == var_0);
    if (var_1) {
        // tau = wp.max(MJ_MINVAL, actuator_dynprm[0])                                        <L 53>
        var_4 = wp::extract(var_actuator_dynprm, var_3);
        var_5 = wp::max(var_2, var_4);
        // act = act_in + act_dot_scale * act_dot_in * tau * (1.0 - wp.exp(-opt_timestep / tau))       <L 54>
        var_6 = wp::mul(var_act_dot_scale, var_act_dot_in);
        var_7 = wp::mul(var_6, var_5);
        var_9 = wp::neg(var_opt_timestep);
        var_10 = wp::div(var_9, var_5);
        var_11 = wp::exp(var_10);
        var_12 = wp::sub(var_8, var_11);
        var_13 = wp::mul(var_7, var_12);
        var_14 = wp::add(var_act_in, var_13);
    }
    if (!var_1) {
        // elif actuator_dyntype == DynType.USER:                                             <L 55>
        var_16 = (var_actuator_dyntype == var_15);
        if (var_16) {
            // return act_in                                                                  <L 56>
            return var_act_in;
        }
        if (!var_16) {
            // act = act_in + act_dot_scale * act_dot_in * opt_timestep                       <L 58>
            var_17 = wp::mul(var_act_dot_scale, var_act_dot_in);
            var_18 = wp::mul(var_17, var_opt_timestep);
            var_19 = wp::add(var_act_in, var_18);
        }
    }
    var_20 = wp::where(var_1, var_14, var_19);
    // if clamp:                                                                              <L 61>
    if (var_clamp) {
        // act = wp.clamp(act, actuator_actrange[0], actuator_actrange[1])                    <L 62>
        var_22 = wp::extract(var_actuator_actrange, var_21);
        var_24 = wp::extract(var_actuator_actrange, var_23);
        var_25 = wp::clamp(var_20, var_22, var_24);
    }
    var_26 = wp::where(var_clamp, var_25, var_20);
    // return act                                                                             <L 64>
    return var_26;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:454
static CUDA_CALLABLE wp::float32 muscle_gain_length_0(
    wp::float32 var_length,
    wp::float32 var_lmin,
    wp::float32 var_lmax)
{
    //---------
    // primal vars
    bool var_0;
    bool var_1;
    bool var_2;
    const wp::float32 var_3 = 0.0;
    const wp::float32 var_4 = 0.5;
    const wp::float32 var_5 = 1.0;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::float32 var_8 = 0.5;
    const wp::float32 var_9 = 1.0;
    wp::float32 var_10;
    wp::float32 var_11;
    bool var_12;
    wp::float32 var_13;
    const wp::float32 var_14 = 1e-15;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    const wp::float32 var_18 = 0.5;
    wp::float32 var_19;
    wp::float32 var_20;
    const wp::float32 var_21 = 1.0;
    bool var_22;
    const wp::float32 var_23 = 1.0;
    wp::float32 var_24;
    const wp::float32 var_25 = 1e-15;
    const wp::float32 var_26 = 1.0;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    const wp::float32 var_30 = 1.0;
    const wp::float32 var_31 = 0.5;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    bool var_35;
    const wp::float32 var_36 = 1.0;
    wp::float32 var_37;
    const wp::float32 var_38 = 1e-15;
    const wp::float32 var_39 = 1.0;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    const wp::float32 var_43 = 1.0;
    const wp::float32 var_44 = 0.5;
    wp::float32 var_45;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    const wp::float32 var_49 = 1e-15;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    const wp::float32 var_53 = 0.5;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    //---------
    // forward
    // def muscle_gain_length(length: float, lmin: float, lmax: float) -> float:              <L 455>
    // if (lmin > length) or (length > lmax):                                                 <L 457>
    var_1 = (var_lmin > var_length);
    var_0 = var_1;
    if (!var_0) {
        var_2 = (var_length > var_lmax);
        var_0 = var_0 || var_2;
    }
    if (var_0) {
        // return 0.0                                                                         <L 458>
        return var_3;
    }
    // a = 0.5 * (lmin + 1.0)                                                                 <L 461>
    var_6 = wp::add(var_lmin, var_5);
    var_7 = wp::mul(var_4, var_6);
    // b = 0.5 * (1.0 + lmax)                                                                 <L 462>
    var_10 = wp::add(var_9, var_lmax);
    var_11 = wp::mul(var_8, var_10);
    // if length <= a:                                                                        <L 464>
    var_12 = (var_length <= var_7);
    if (var_12) {
        // x = (length - lmin) / wp.max(MJ_MINVAL, a - lmin)                                  <L 465>
        var_13 = wp::sub(var_length, var_lmin);
        var_15 = wp::sub(var_7, var_lmin);
        var_16 = wp::max(var_14, var_15);
        var_17 = wp::div(var_13, var_16);
        // return 0.5 * x * x                                                                 <L 466>
        var_19 = wp::mul(var_18, var_17);
        var_20 = wp::mul(var_19, var_17);
        return var_20;
    }
    if (!var_12) {
        // elif length <= 1.0:                                                                <L 467>
        var_22 = (var_length <= var_21);
        if (var_22) {
            // x = (1.0 - length) / wp.max(MJ_MINVAL, 1.0 - a)                                <L 468>
            var_24 = wp::sub(var_23, var_length);
            var_27 = wp::sub(var_26, var_7);
            var_28 = wp::max(var_25, var_27);
            var_29 = wp::div(var_24, var_28);
            // return 1.0 - 0.5 * x * x                                                       <L 469>
            var_32 = wp::mul(var_31, var_29);
            var_33 = wp::mul(var_32, var_29);
            var_34 = wp::sub(var_30, var_33);
            return var_34;
        }
        if (!var_22) {
            // elif length <= b:                                                              <L 470>
            var_35 = (var_length <= var_11);
            if (var_35) {
                // x = (length - 1.0) / wp.max(MJ_MINVAL, b - 1.0)                            <L 471>
                var_37 = wp::sub(var_length, var_36);
                var_40 = wp::sub(var_11, var_39);
                var_41 = wp::max(var_38, var_40);
                var_42 = wp::div(var_37, var_41);
                // return 1.0 - 0.5 * x * x                                                   <L 472>
                var_45 = wp::mul(var_44, var_42);
                var_46 = wp::mul(var_45, var_42);
                var_47 = wp::sub(var_43, var_46);
                return var_47;
            }
            if (!var_35) {
                // x = (lmax - length) / wp.max(MJ_MINVAL, lmax - b)                          <L 474>
                var_48 = wp::sub(var_lmax, var_length);
                var_50 = wp::sub(var_lmax, var_11);
                var_51 = wp::max(var_49, var_50);
                var_52 = wp::div(var_48, var_51);
                // return 0.5 * x * x                                                         <L 475>
                var_54 = wp::mul(var_53, var_52);
                var_55 = wp::mul(var_54, var_52);
                return var_55;
            }
            var_56 = wp::where(var_35, var_38, var_49);
            var_57 = wp::where(var_35, var_42, var_52);
        }
        var_58 = wp::where(var_22, var_25, var_56);
        var_59 = wp::where(var_22, var_29, var_57);
    }
    var_60 = wp::where(var_12, var_14, var_58);
    var_61 = wp::where(var_12, var_17, var_59);
    return {};
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:478
static CUDA_CALLABLE wp::float32 muscle_gain_0(
    wp::float32 var_len,
    wp::float32 var_vel,
    wp::vec_t<2, wp::float32> var_lengthrange,
    wp::float32 var_acc0,
    wp::vec_t<10, wp::float32> var_prm)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    wp::vec_t<2, wp::float32> var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    const wp::int32 var_7 = 3;
    wp::float32 var_8;
    const wp::int32 var_9 = 4;
    wp::float32 var_10;
    const wp::int32 var_11 = 5;
    wp::float32 var_12;
    const wp::int32 var_13 = 6;
    wp::float32 var_14;
    const wp::int32 var_15 = 8;
    wp::float32 var_16;
    const wp::float32 var_17 = 0.0;
    bool var_18;
    const wp::float32 var_19 = 1e-15;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::int32 var_23 = 1;
    wp::float32 var_24;
    const wp::int32 var_25 = 0;
    wp::float32 var_26;
    wp::float32 var_27;
    const wp::int32 var_28 = 1;
    wp::float32 var_29;
    const wp::int32 var_30 = 0;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    const wp::int32 var_35 = 0;
    wp::float32 var_36;
    const wp::int32 var_37 = 0;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    const wp::float32 var_47 = 1.0;
    wp::float32 var_48;
    const wp::float32 var_49 = -1.0;
    bool var_50;
    const wp::float32 var_51 = 0.0;
    const wp::float32 var_52 = 0.0;
    bool var_53;
    const wp::float32 var_54 = 1.0;
    wp::float32 var_55;
    const wp::float32 var_56 = 1.0;
    wp::float32 var_57;
    wp::float32 var_58;
    bool var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    //---------
    // forward
    // def muscle_gain(len: float, vel: float, lengthrange: wp.vec2, acc0: float, prm: vec10) -> float:       <L 479>
    // range_ = wp.vec2(prm[0], prm[1])                                                       <L 482>
    var_1 = wp::extract(var_prm, var_0);
    var_3 = wp::extract(var_prm, var_2);
    var_4 = wp::vec_t<2, wp::float32>(var_1, var_3);
    // force = prm[2]                                                                         <L 483>
    var_6 = wp::extract(var_prm, var_5);
    // scale = prm[3]                                                                         <L 484>
    var_8 = wp::extract(var_prm, var_7);
    // lmin = prm[4]                                                                          <L 485>
    var_10 = wp::extract(var_prm, var_9);
    // lmax = prm[5]                                                                          <L 486>
    var_12 = wp::extract(var_prm, var_11);
    // vmax = prm[6]                                                                          <L 487>
    var_14 = wp::extract(var_prm, var_13);
    // fvmax = prm[8]                                                                         <L 488>
    var_16 = wp::extract(var_prm, var_15);
    // if force < 0.0:                                                                        <L 491>
    var_18 = (var_6 < var_17);
    if (var_18) {
        // force = scale / wp.max(MJ_MINVAL, acc0)                                            <L 492>
        var_20 = wp::max(var_19, var_acc0);
        var_21 = wp::div(var_8, var_20);
    }
    var_22 = wp::where(var_18, var_21, var_6);
    // L0 = (lengthrange[1] - lengthrange[0]) / wp.max(MJ_MINVAL, range_[1] - range_[0])       <L 495>
    var_24 = wp::extract(var_lengthrange, var_23);
    var_26 = wp::extract(var_lengthrange, var_25);
    var_27 = wp::sub(var_24, var_26);
    var_29 = wp::extract(var_4, var_28);
    var_31 = wp::extract(var_4, var_30);
    var_32 = wp::sub(var_29, var_31);
    var_33 = wp::max(var_19, var_32);
    var_34 = wp::div(var_27, var_33);
    // L = range_[0] + (len - lengthrange[0]) / wp.max(MJ_MINVAL, L0)                         <L 498>
    var_36 = wp::extract(var_4, var_35);
    var_38 = wp::extract(var_lengthrange, var_37);
    var_39 = wp::sub(var_len, var_38);
    var_40 = wp::max(var_19, var_34);
    var_41 = wp::div(var_39, var_40);
    var_42 = wp::add(var_36, var_41);
    // V = vel / wp.max(MJ_MINVAL, L0 * vmax)                                                 <L 499>
    var_43 = wp::mul(var_34, var_14);
    var_44 = wp::max(var_19, var_43);
    var_45 = wp::div(var_vel, var_44);
    // FL = muscle_gain_length(L, lmin, lmax)                                                 <L 502>
    var_46 = muscle_gain_length_0(var_42, var_10, var_12);
    // y = fvmax - 1.0                                                                        <L 505>
    var_48 = wp::sub(var_16, var_47);
    // if V <= -1.0:                                                                          <L 506>
    var_50 = (var_45 <= var_49);
    if (var_50) {
        // FV = 0.0                                                                           <L 507>
    }
    if (!var_50) {
        // elif V <= 0.0:                                                                     <L 508>
        var_53 = (var_45 <= var_52);
        if (var_53) {
            // FV = (V + 1.0) * (V + 1.0)                                                     <L 509>
            var_55 = wp::add(var_45, var_54);
            var_57 = wp::add(var_45, var_56);
            var_58 = wp::mul(var_55, var_57);
        }
        if (!var_53) {
            // elif V <= y:                                                                   <L 510>
            var_59 = (var_45 <= var_48);
            if (var_59) {
                // FV = fvmax - (y - V) * (y - V) / wp.max(MJ_MINVAL, y)                      <L 511>
                var_60 = wp::sub(var_48, var_45);
                var_61 = wp::sub(var_48, var_45);
                var_62 = wp::mul(var_60, var_61);
                var_63 = wp::max(var_19, var_48);
                var_64 = wp::div(var_62, var_63);
                var_65 = wp::sub(var_16, var_64);
            }
            if (!var_59) {
                // FV = fvmax                                                                 <L 513>
                var_66 = wp::copy(var_16);
            }
            var_67 = wp::where(var_59, var_65, var_66);
        }
        var_68 = wp::where(var_53, var_58, var_67);
    }
    var_69 = wp::where(var_50, var_51, var_68);
    // return -force * FL * FV                                                                <L 516>
    var_70 = wp::neg(var_22);
    var_71 = wp::mul(var_70, var_46);
    var_72 = wp::mul(var_71, var_69);
    return var_72;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:519
static CUDA_CALLABLE wp::float32 muscle_bias_0(
    wp::float32 var_len,
    wp::vec_t<2, wp::float32> var_lengthrange,
    wp::float32 var_acc0,
    wp::vec_t<10, wp::float32> var_prm)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    wp::vec_t<2, wp::float32> var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    const wp::int32 var_7 = 3;
    wp::float32 var_8;
    const wp::int32 var_9 = 5;
    wp::float32 var_10;
    const wp::int32 var_11 = 7;
    wp::float32 var_12;
    const wp::float32 var_13 = 0.0;
    bool var_14;
    const wp::float32 var_15 = 1e-15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    const wp::int32 var_19 = 1;
    wp::float32 var_20;
    const wp::int32 var_21 = 0;
    wp::float32 var_22;
    wp::float32 var_23;
    const wp::int32 var_24 = 1;
    wp::float32 var_25;
    const wp::int32 var_26 = 0;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    const wp::int32 var_31 = 0;
    wp::float32 var_32;
    const wp::int32 var_33 = 0;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::float32 var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    const wp::float32 var_39 = 0.5;
    const wp::float32 var_40 = 1.0;
    wp::float32 var_41;
    wp::float32 var_42;
    const wp::float32 var_43 = 1.0;
    bool var_44;
    const wp::float32 var_45 = 0.0;
    bool var_46;
    const wp::float32 var_47 = 1.0;
    wp::float32 var_48;
    const wp::float32 var_49 = 1.0;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    const wp::float32 var_55 = 0.5;
    wp::float32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    const wp::float32 var_60 = 1.0;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    const wp::float32 var_66 = 0.5;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    //---------
    // forward
    // def muscle_bias(len: float, lengthrange: wp.vec2, acc0: float, prm: vec10) -> float:       <L 520>
    // range_ = wp.vec2(prm[0], prm[1])                                                       <L 526>
    var_1 = wp::extract(var_prm, var_0);
    var_3 = wp::extract(var_prm, var_2);
    var_4 = wp::vec_t<2, wp::float32>(var_1, var_3);
    // force = prm[2]                                                                         <L 527>
    var_6 = wp::extract(var_prm, var_5);
    // scale = prm[3]                                                                         <L 528>
    var_8 = wp::extract(var_prm, var_7);
    // lmax = prm[5]                                                                          <L 529>
    var_10 = wp::extract(var_prm, var_9);
    // fpmax = prm[7]                                                                         <L 530>
    var_12 = wp::extract(var_prm, var_11);
    // if force < 0.0:                                                                        <L 533>
    var_14 = (var_6 < var_13);
    if (var_14) {
        // force = scale / wp.max(MJ_MINVAL, acc0)                                            <L 534>
        var_16 = wp::max(var_15, var_acc0);
        var_17 = wp::div(var_8, var_16);
    }
    var_18 = wp::where(var_14, var_17, var_6);
    // L0 = (lengthrange[1] - lengthrange[0]) / wp.max(MJ_MINVAL, range_[1] - range_[0])       <L 537>
    var_20 = wp::extract(var_lengthrange, var_19);
    var_22 = wp::extract(var_lengthrange, var_21);
    var_23 = wp::sub(var_20, var_22);
    var_25 = wp::extract(var_4, var_24);
    var_27 = wp::extract(var_4, var_26);
    var_28 = wp::sub(var_25, var_27);
    var_29 = wp::max(var_15, var_28);
    var_30 = wp::div(var_23, var_29);
    // L = range_[0] + (len - lengthrange[0]) / wp.max(MJ_MINVAL, L0)                         <L 540>
    var_32 = wp::extract(var_4, var_31);
    var_34 = wp::extract(var_lengthrange, var_33);
    var_35 = wp::sub(var_len, var_34);
    var_36 = wp::max(var_15, var_30);
    var_37 = wp::div(var_35, var_36);
    var_38 = wp::add(var_32, var_37);
    // b = 0.5 * (1.0 + lmax)                                                                 <L 543>
    var_41 = wp::add(var_40, var_10);
    var_42 = wp::mul(var_39, var_41);
    // if L <= 1.0:                                                                           <L 544>
    var_44 = (var_38 <= var_43);
    if (var_44) {
        // return 0.0                                                                         <L 545>
        return var_45;
    }
    if (!var_44) {
        // elif L <= b:                                                                       <L 546>
        var_46 = (var_38 <= var_42);
        if (var_46) {
            // x = (L - 1.0) / wp.max(MJ_MINVAL, b - 1.0)                                     <L 547>
            var_48 = wp::sub(var_38, var_47);
            var_50 = wp::sub(var_42, var_49);
            var_51 = wp::max(var_15, var_50);
            var_52 = wp::div(var_48, var_51);
            // return -force * fpmax * 0.5 * x * x                                            <L 548>
            var_53 = wp::neg(var_18);
            var_54 = wp::mul(var_53, var_12);
            var_56 = wp::mul(var_54, var_55);
            var_57 = wp::mul(var_56, var_52);
            var_58 = wp::mul(var_57, var_52);
            return var_58;
        }
        if (!var_46) {
            // x = (L - b) / wp.max(MJ_MINVAL, b - 1.0)                                       <L 550>
            var_59 = wp::sub(var_38, var_42);
            var_61 = wp::sub(var_42, var_60);
            var_62 = wp::max(var_15, var_61);
            var_63 = wp::div(var_59, var_62);
            // return -force * fpmax * (0.5 + x)                                              <L 551>
            var_64 = wp::neg(var_18);
            var_65 = wp::mul(var_64, var_12);
            var_67 = wp::add(var_66, var_63);
            var_68 = wp::mul(var_65, var_67);
            return var_68;
        }
        var_69 = wp::where(var_46, var_52, var_63);
    }
    return {};
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:717
static CUDA_CALLABLE wp::float32 _poly_force_deriv_0(
    wp::float32 var_linear,
    wp::vec_t<2, wp::float32> var_poly,
    wp::float32 var_x,
    wp::int32 var_flg_odd)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    bool var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::float32 var_5 = 2.0;
    const wp::int32 var_6 = 0;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::float32 var_11 = 3.0;
    const wp::int32 var_12 = 1;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    //---------
    // forward
    // def _poly_force_deriv(linear: float, poly: wp.vec2, x: float, flg_odd: int) -> float:       <L 718>
    // x_val = wp.where(flg_odd == 1, wp.abs(x), x)                                           <L 719>
    var_1 = (var_flg_odd == var_0);
    var_2 = wp::abs(var_x);
    var_3 = wp::where(var_1, var_2, var_x);
    // res = linear                                                                           <L 720>
    var_4 = wp::copy(var_linear);
    // res += 2.0 * poly[0] * x_val                                                           <L 721>
    var_7 = wp::extract(var_poly, var_6);
    var_8 = wp::mul(var_5, var_7);
    var_9 = wp::mul(var_8, var_3);
    var_10 = wp::add(var_4, var_9);
    // res += 3.0 * poly[1] * x_val * x_val                                                   <L 722>
    var_13 = wp::extract(var_poly, var_12);
    var_14 = wp::mul(var_11, var_13);
    var_15 = wp::mul(var_14, var_3);
    var_16 = wp::mul(var_15, var_3);
    var_17 = wp::add(var_10, var_16);
    // return res                                                                             <L 723>
    return var_17;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/math.py:52
static CUDA_CALLABLE void adj_axis_angle_to_quat_0(
    wp::vec_t<3, wp::float32> var_axis,
    wp::float32 var_angle,
    wp::vec_t<3, wp::float32> & adj_axis,
    wp::float32 & adj_angle,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/math.py:23
static CUDA_CALLABLE void adj_mul_quat_0(
    wp::quat_t<wp::float32> var_u,
    wp::quat_t<wp::float32> var_v,
    wp::quat_t<wp::float32> & adj_u,
    wp::quat_t<wp::float32> & adj_v,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/math.py:188
static CUDA_CALLABLE void adj_quat_integrate_0(
    wp::quat_t<wp::float32> var_q,
    wp::vec_t<3, wp::float32> var_v,
    wp::float32 var_dt,
    wp::quat_t<wp::float32> & adj_q,
    wp::vec_t<3, wp::float32> & adj_v,
    wp::float32 & adj_dt,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:554
static CUDA_CALLABLE void adj__sigmoid_0(
    wp::float32 var_x,
    wp::float32 & adj_x,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:568
static CUDA_CALLABLE void adj_muscle_dynamics_timescale_0(
    wp::float32 var_dctrl,
    wp::float32 var_tau_act,
    wp::float32 var_tau_deact,
    wp::float32 var_smooth_width,
    wp::float32 & adj_dctrl,
    wp::float32 & adj_tau_act,
    wp::float32 & adj_tau_deact,
    wp::float32 & adj_smooth_width,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:582
static CUDA_CALLABLE void adj_muscle_dynamics_0(
    wp::float32 var_control,
    wp::float32 var_activation,
    wp::vec_t<10, wp::float32> var_prm,
    wp::float32 & adj_control,
    wp::float32 & adj_activation,
    wp::vec_t<10, wp::float32> & adj_prm,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:603
static CUDA_CALLABLE void adj_dcmotor_slots_0(
    wp::vec_t<10, wp::float32> var_dynprm,
    wp::vec_t<10, wp::float32> var_gainprm,
    wp::vec_t<10, wp::float32> & adj_dynprm,
    wp::vec_t<10, wp::float32> & adj_gainprm,
    wp::vec_t<6, wp::int32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:649
static CUDA_CALLABLE void adj_dcmotor_voltage_0(
    wp::float32 var_u,
    wp::float32 var_length,
    wp::float32 var_velocity,
    wp::float32 var_x_I,
    wp::vec_t<10, wp::float32> var_gainprm,
    wp::float32 & adj_u,
    wp::float32 & adj_length,
    wp::float32 & adj_velocity,
    wp::float32 & adj_x_I,
    wp::vec_t<10, wp::float32> & adj_gainprm,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:643
static CUDA_CALLABLE void adj_lugre_stribeck_0(
    wp::float32 var_velocity,
    wp::float32 var_F_C,
    wp::float32 var_F_S,
    wp::float32 var_v_S,
    wp::float32 & adj_velocity,
    wp::float32 & adj_F_C,
    wp::float32 & adj_F_S,
    wp::float32 & adj_v_S,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/support.py:37
static CUDA_CALLABLE void adj_next_act_0(
    wp::float32 var_opt_timestep,
    wp::int32 var_actuator_dyntype,
    wp::vec_t<10, wp::float32> var_actuator_dynprm,
    wp::vec_t<2, wp::float32> var_actuator_actrange,
    wp::float32 var_act_in,
    wp::float32 var_act_dot_in,
    wp::float32 var_act_dot_scale,
    bool var_clamp,
    wp::float32 & adj_opt_timestep,
    wp::int32 & adj_actuator_dyntype,
    wp::vec_t<10, wp::float32> & adj_actuator_dynprm,
    wp::vec_t<2, wp::float32> & adj_actuator_actrange,
    wp::float32 & adj_act_in,
    wp::float32 & adj_act_dot_in,
    wp::float32 & adj_act_dot_scale,
    bool & adj_clamp,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:454
static CUDA_CALLABLE void adj_muscle_gain_length_0(
    wp::float32 var_length,
    wp::float32 var_lmin,
    wp::float32 var_lmax,
    wp::float32 & adj_length,
    wp::float32 & adj_lmin,
    wp::float32 & adj_lmax,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:478
static CUDA_CALLABLE void adj_muscle_gain_0(
    wp::float32 var_len,
    wp::float32 var_vel,
    wp::vec_t<2, wp::float32> var_lengthrange,
    wp::float32 var_acc0,
    wp::vec_t<10, wp::float32> var_prm,
    wp::float32 & adj_len,
    wp::float32 & adj_vel,
    wp::vec_t<2, wp::float32> & adj_lengthrange,
    wp::float32 & adj_acc0,
    wp::vec_t<10, wp::float32> & adj_prm,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:519
static CUDA_CALLABLE void adj_muscle_bias_0(
    wp::float32 var_len,
    wp::vec_t<2, wp::float32> var_lengthrange,
    wp::float32 var_acc0,
    wp::vec_t<10, wp::float32> var_prm,
    wp::float32 & adj_len,
    wp::vec_t<2, wp::float32> & adj_lengthrange,
    wp::float32 & adj_acc0,
    wp::vec_t<10, wp::float32> & adj_prm,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:717
static CUDA_CALLABLE void adj__poly_force_deriv_0(
    wp::float32 var_linear,
    wp::vec_t<2, wp::float32> var_poly,
    wp::float32 var_x,
    wp::int32 var_flg_odd,
    wp::float32 & adj_linear,
    wp::vec_t<2, wp::float32> & adj_poly,
    wp::float32 & adj_x,
    wp::int32 & adj_flg_odd,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void _actuator_velocity_00a052f5_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::int32> var_moment_rownnz_in,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::int32> var_moment_colind_in,
    wp::array_t<wp::float32> var_actuator_moment_in,
    wp::array_t<wp::float32> var_actuator_velocity_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        const wp::float32 var_8 = 0.0;
        wp::float32 var_9;
        wp::range_t var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::float32* var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        //---------
        // forward
        // def _actuator_velocity(                                                                <L 681>
        // worldid, actid = wp.tid()                                                              <L 691>
        builtin_tid2d(var_0, var_1);
        // rownnz = moment_rownnz_in[worldid, actid]                                              <L 693>
        var_2 = wp::address(var_moment_rownnz_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // rowadr = moment_rowadr_in[worldid, actid]                                              <L 694>
        var_5 = wp::address(var_moment_rowadr_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // vel = float(0.0)                                                                       <L 696>
        var_9 = wp::float(var_8);
        // for i in range(rownnz):                                                                <L 697>
        var_10 = wp::range(var_3);
        start_for_0:;
            if (iter_cmp(var_10) == 0) goto end_for_0;
            var_11 = wp::iter_next(var_10);
            // sparseid = rowadr + i                                                              <L 698>
            var_12 = wp::add(var_6, var_11);
            // colind = moment_colind_in[worldid, sparseid]                                       <L 699>
            var_13 = wp::address(var_moment_colind_in, var_0, var_12);
            var_15 = wp::load(var_13);
            var_14 = wp::copy(var_15);
            // vel += actuator_moment_in[worldid, sparseid] * qvel_in[worldid, colind]            <L 700>
            var_16 = wp::address(var_actuator_moment_in, var_0, var_12);
            var_17 = wp::address(var_qvel_in, var_0, var_14);
            var_19 = wp::load(var_16);
            var_20 = wp::load(var_17);
            var_18 = wp::mul(var_19, var_20);
            var_21 = wp::add(var_9, var_18);
            wp::assign(var_9, var_21);
            goto start_for_0;
        end_for_0:;
        // actuator_velocity_out[worldid, actid] = vel                                            <L 702>
        wp::array_store(var_actuator_velocity_out, var_0, var_1, var_9);
    }
}



extern "C" __global__ void _next_position_1c101e8d_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::float32> var_qpos_in,
    wp::array_t<wp::float32> var_qvel_in,
    wp::float32 var_qvel_scale_in,
    wp::array_t<wp::float32> var_qpos_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::array_t<wp::float32> var_19;
        wp::array_t<wp::float32> var_20;
        wp::array_t<wp::float32> var_21;
        const wp::int32 var_22 = 0;
        bool var_23;
        wp::float32* var_24;
        const wp::int32 var_25 = 1;
        wp::int32 var_26;
        wp::float32* var_27;
        const wp::int32 var_28 = 2;
        wp::int32 var_29;
        wp::float32* var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32* var_35;
        const wp::int32 var_36 = 1;
        wp::int32 var_37;
        wp::float32* var_38;
        const wp::int32 var_39 = 2;
        wp::int32 var_40;
        wp::float32* var_41;
        wp::vec_t<3, wp::float32> var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32> var_48;
        const wp::int32 var_49 = 3;
        wp::int32 var_50;
        wp::float32* var_51;
        const wp::int32 var_52 = 4;
        wp::int32 var_53;
        wp::float32* var_54;
        const wp::int32 var_55 = 5;
        wp::int32 var_56;
        wp::float32* var_57;
        const wp::int32 var_58 = 6;
        wp::int32 var_59;
        wp::float32* var_60;
        wp::quat_t<wp::float32> var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::float32 var_64;
        wp::float32 var_65;
        const wp::int32 var_66 = 3;
        wp::int32 var_67;
        wp::float32* var_68;
        const wp::int32 var_69 = 4;
        wp::int32 var_70;
        wp::float32* var_71;
        const wp::int32 var_72 = 5;
        wp::int32 var_73;
        wp::float32* var_74;
        wp::vec_t<3, wp::float32> var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::float32 var_78;
        wp::vec_t<3, wp::float32> var_79;
        wp::quat_t<wp::float32> var_80;
        const wp::int32 var_81 = 0;
        wp::float32 var_82;
        const wp::int32 var_83 = 0;
        wp::int32 var_84;
        const wp::int32 var_85 = 1;
        wp::float32 var_86;
        const wp::int32 var_87 = 1;
        wp::int32 var_88;
        const wp::int32 var_89 = 2;
        wp::float32 var_90;
        const wp::int32 var_91 = 2;
        wp::int32 var_92;
        const wp::int32 var_93 = 0;
        wp::float32 var_94;
        const wp::int32 var_95 = 3;
        wp::int32 var_96;
        const wp::int32 var_97 = 1;
        wp::float32 var_98;
        const wp::int32 var_99 = 4;
        wp::int32 var_100;
        const wp::int32 var_101 = 2;
        wp::float32 var_102;
        const wp::int32 var_103 = 5;
        wp::int32 var_104;
        const wp::int32 var_105 = 3;
        wp::float32 var_106;
        const wp::int32 var_107 = 6;
        wp::int32 var_108;
        const wp::int32 var_109 = 1;
        bool var_110;
        const wp::int32 var_111 = 0;
        wp::int32 var_112;
        wp::float32* var_113;
        const wp::int32 var_114 = 1;
        wp::int32 var_115;
        wp::float32* var_116;
        const wp::int32 var_117 = 2;
        wp::int32 var_118;
        wp::float32* var_119;
        const wp::int32 var_120 = 3;
        wp::int32 var_121;
        wp::float32* var_122;
        wp::quat_t<wp::float32> var_123;
        wp::float32 var_124;
        wp::float32 var_125;
        wp::float32 var_126;
        wp::float32 var_127;
        wp::float32* var_128;
        const wp::int32 var_129 = 1;
        wp::int32 var_130;
        wp::float32* var_131;
        const wp::int32 var_132 = 2;
        wp::int32 var_133;
        wp::float32* var_134;
        wp::vec_t<3, wp::float32> var_135;
        wp::float32 var_136;
        wp::float32 var_137;
        wp::float32 var_138;
        wp::vec_t<3, wp::float32> var_139;
        wp::quat_t<wp::float32> var_140;
        const wp::int32 var_141 = 0;
        wp::float32 var_142;
        const wp::int32 var_143 = 0;
        wp::int32 var_144;
        const wp::int32 var_145 = 1;
        wp::float32 var_146;
        const wp::int32 var_147 = 1;
        wp::int32 var_148;
        const wp::int32 var_149 = 2;
        wp::float32 var_150;
        const wp::int32 var_151 = 2;
        wp::int32 var_152;
        const wp::int32 var_153 = 3;
        wp::float32 var_154;
        const wp::int32 var_155 = 3;
        wp::int32 var_156;
        wp::float32* var_157;
        wp::float32* var_158;
        wp::float32 var_159;
        wp::float32 var_160;
        wp::float32 var_161;
        wp::float32 var_162;
        wp::float32 var_163;
        wp::quat_t<wp::float32> var_164;
        wp::vec_t<3, wp::float32> var_165;
        wp::quat_t<wp::float32> var_166;
        //---------
        // forward
        // def _next_position(                                                                    <L 54>
        // worldid, jntid = wp.tid()                                                              <L 68>
        builtin_tid2d(var_0, var_1);
        // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                               <L 69>
        var_2 = &(var_opt_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_opt_timestep, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // jnttype = jnt_type[jntid]                                                              <L 71>
        var_10 = wp::address(var_jnt_type, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // qpos_adr = jnt_qposadr[jntid]                                                          <L 72>
        var_13 = wp::address(var_jnt_qposadr, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // dof_adr = jnt_dofadr[jntid]                                                            <L 73>
        var_16 = wp::address(var_jnt_dofadr, var_1);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // qpos = qpos_in[worldid]                                                                <L 74>
        var_19 = wp::view(var_qpos_in, var_0);
        // qpos_next = qpos_out[worldid]                                                          <L 75>
        var_20 = wp::view(var_qpos_out, var_0);
        // qvel = qvel_in[worldid]                                                                <L 76>
        var_21 = wp::view(var_qvel_in, var_0);
        // if jnttype == JointType.FREE:                                                          <L 78>
        var_23 = (var_11 == var_22);
        if (var_23) {
            // qpos_pos = wp.vec3(qpos[qpos_adr], qpos[qpos_adr + 1], qpos[qpos_adr + 2])         <L 79>
            var_24 = wp::address(var_19, var_14);
            var_26 = wp::add(var_14, var_25);
            var_27 = wp::address(var_19, var_26);
            var_29 = wp::add(var_14, var_28);
            var_30 = wp::address(var_19, var_29);
            var_32 = wp::load(var_24);
            var_33 = wp::load(var_27);
            var_34 = wp::load(var_30);
            var_31 = wp::vec_t<3, wp::float32>(var_32, var_33, var_34);
            // qvel_lin = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2]) * qvel_scale_in       <L 80>
            var_35 = wp::address(var_21, var_17);
            var_37 = wp::add(var_17, var_36);
            var_38 = wp::address(var_21, var_37);
            var_40 = wp::add(var_17, var_39);
            var_41 = wp::address(var_21, var_40);
            var_43 = wp::load(var_35);
            var_44 = wp::load(var_38);
            var_45 = wp::load(var_41);
            var_42 = wp::vec_t<3, wp::float32>(var_43, var_44, var_45);
            var_46 = wp::mul(var_42, var_qvel_scale_in);
            // qpos_new = qpos_pos + timestep * qvel_lin                                          <L 82>
            var_47 = wp::mul(var_8, var_46);
            var_48 = wp::add(var_31, var_47);
            // qpos_quat = wp.quat(                                                               <L 84>
            // qpos[qpos_adr + 3],                                                                <L 85>
            var_50 = wp::add(var_14, var_49);
            var_51 = wp::address(var_19, var_50);
            // qpos[qpos_adr + 4],                                                                <L 86>
            var_53 = wp::add(var_14, var_52);
            var_54 = wp::address(var_19, var_53);
            // qpos[qpos_adr + 5],                                                                <L 87>
            var_56 = wp::add(var_14, var_55);
            var_57 = wp::address(var_19, var_56);
            // qpos[qpos_adr + 6],                                                                <L 88>
            var_59 = wp::add(var_14, var_58);
            var_60 = wp::address(var_19, var_59);
            var_62 = wp::load(var_51);
            var_63 = wp::load(var_54);
            var_64 = wp::load(var_57);
            var_65 = wp::load(var_60);
            var_61 = wp::quat_t<wp::float32>(var_62, var_63, var_64, var_65);
            // qvel_ang = wp.vec3(qvel[dof_adr + 3], qvel[dof_adr + 4], qvel[dof_adr + 5]) * qvel_scale_in       <L 90>
            var_67 = wp::add(var_17, var_66);
            var_68 = wp::address(var_21, var_67);
            var_70 = wp::add(var_17, var_69);
            var_71 = wp::address(var_21, var_70);
            var_73 = wp::add(var_17, var_72);
            var_74 = wp::address(var_21, var_73);
            var_76 = wp::load(var_68);
            var_77 = wp::load(var_71);
            var_78 = wp::load(var_74);
            var_75 = wp::vec_t<3, wp::float32>(var_76, var_77, var_78);
            var_79 = wp::mul(var_75, var_qvel_scale_in);
            // qpos_quat_new = math.quat_integrate(qpos_quat, qvel_ang, timestep)                 <L 92>
            var_80 = quat_integrate_0(var_61, var_79, var_8);
            // qpos_next[qpos_adr + 0] = qpos_new[0]                                              <L 94>
            var_82 = wp::extract(var_48, var_81);
            var_84 = wp::add(var_14, var_83);
            wp::array_store(var_20, var_84, var_82);
            // qpos_next[qpos_adr + 1] = qpos_new[1]                                              <L 95>
            var_86 = wp::extract(var_48, var_85);
            var_88 = wp::add(var_14, var_87);
            wp::array_store(var_20, var_88, var_86);
            // qpos_next[qpos_adr + 2] = qpos_new[2]                                              <L 96>
            var_90 = wp::extract(var_48, var_89);
            var_92 = wp::add(var_14, var_91);
            wp::array_store(var_20, var_92, var_90);
            // qpos_next[qpos_adr + 3] = qpos_quat_new[0]                                         <L 97>
            var_94 = wp::extract(var_80, var_93);
            var_96 = wp::add(var_14, var_95);
            wp::array_store(var_20, var_96, var_94);
            // qpos_next[qpos_adr + 4] = qpos_quat_new[1]                                         <L 98>
            var_98 = wp::extract(var_80, var_97);
            var_100 = wp::add(var_14, var_99);
            wp::array_store(var_20, var_100, var_98);
            // qpos_next[qpos_adr + 5] = qpos_quat_new[2]                                         <L 99>
            var_102 = wp::extract(var_80, var_101);
            var_104 = wp::add(var_14, var_103);
            wp::array_store(var_20, var_104, var_102);
            // qpos_next[qpos_adr + 6] = qpos_quat_new[3]                                         <L 100>
            var_106 = wp::extract(var_80, var_105);
            var_108 = wp::add(var_14, var_107);
            wp::array_store(var_20, var_108, var_106);
        }
        if (!var_23) {
            // elif jnttype == JointType.BALL:                                                    <L 102>
            var_110 = (var_11 == var_109);
            if (var_110) {
                // qpos_quat = wp.quat(qpos[qpos_adr + 0], qpos[qpos_adr + 1], qpos[qpos_adr + 2], qpos[qpos_adr + 3])       <L 103>
                var_112 = wp::add(var_14, var_111);
                var_113 = wp::address(var_19, var_112);
                var_115 = wp::add(var_14, var_114);
                var_116 = wp::address(var_19, var_115);
                var_118 = wp::add(var_14, var_117);
                var_119 = wp::address(var_19, var_118);
                var_121 = wp::add(var_14, var_120);
                var_122 = wp::address(var_19, var_121);
                var_124 = wp::load(var_113);
                var_125 = wp::load(var_116);
                var_126 = wp::load(var_119);
                var_127 = wp::load(var_122);
                var_123 = wp::quat_t<wp::float32>(var_124, var_125, var_126, var_127);
                // qvel_ang = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2]) * qvel_scale_in       <L 104>
                var_128 = wp::address(var_21, var_17);
                var_130 = wp::add(var_17, var_129);
                var_131 = wp::address(var_21, var_130);
                var_133 = wp::add(var_17, var_132);
                var_134 = wp::address(var_21, var_133);
                var_136 = wp::load(var_128);
                var_137 = wp::load(var_131);
                var_138 = wp::load(var_134);
                var_135 = wp::vec_t<3, wp::float32>(var_136, var_137, var_138);
                var_139 = wp::mul(var_135, var_qvel_scale_in);
                // qpos_quat_new = math.quat_integrate(qpos_quat, qvel_ang, timestep)             <L 106>
                var_140 = quat_integrate_0(var_123, var_139, var_8);
                // qpos_next[qpos_adr + 0] = qpos_quat_new[0]                                     <L 108>
                var_142 = wp::extract(var_140, var_141);
                var_144 = wp::add(var_14, var_143);
                wp::array_store(var_20, var_144, var_142);
                // qpos_next[qpos_adr + 1] = qpos_quat_new[1]                                     <L 109>
                var_146 = wp::extract(var_140, var_145);
                var_148 = wp::add(var_14, var_147);
                wp::array_store(var_20, var_148, var_146);
                // qpos_next[qpos_adr + 2] = qpos_quat_new[2]                                     <L 110>
                var_150 = wp::extract(var_140, var_149);
                var_152 = wp::add(var_14, var_151);
                wp::array_store(var_20, var_152, var_150);
                // qpos_next[qpos_adr + 3] = qpos_quat_new[3]                                     <L 111>
                var_154 = wp::extract(var_140, var_153);
                var_156 = wp::add(var_14, var_155);
                wp::array_store(var_20, var_156, var_154);
            }
            if (!var_110) {
                // qpos_next[qpos_adr] = qpos[qpos_adr] + timestep * qvel[dof_adr] * qvel_scale_in       <L 114>
                var_157 = wp::address(var_19, var_14);
                var_158 = wp::address(var_21, var_17);
                var_160 = wp::load(var_158);
                var_159 = wp::mul(var_8, var_160);
                var_161 = wp::mul(var_159, var_qvel_scale_in);
                var_163 = wp::load(var_157);
                var_162 = wp::add(var_163, var_161);
                wp::array_store(var_20, var_14, var_162);
            }
        }
        var_164 = wp::where(var_23, var_61, var_123);
        var_165 = wp::where(var_23, var_79, var_139);
        var_166 = wp::where(var_23, var_80, var_140);
    }
}



extern "C" __global__ void _map_m2d_273b5a0a_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_mapM2D,
    wp::array_t<wp::float32> var_qH_M,
    wp::array_t<wp::float32> var_qLU_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        const wp::float32 var_9 = 0.0;
        //---------
        // forward
        // def _map_m2d(                                                                          <L 561>
        // worldid, elemid = wp.tid()                                                             <L 570>
        builtin_tid2d(var_0, var_1);
        // m_idx = mapM2D[elemid]                                                                 <L 571>
        var_2 = wp::address(var_mapM2D, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if m_idx >= 0:                                                                         <L 572>
        var_6 = (var_3 >= var_5);
        if (var_6) {
            // qLU_out[worldid, elemid] = qH_M[worldid, m_idx]                                    <L 573>
            var_7 = wp::address(var_qH_M, var_0, var_3);
            var_8 = wp::load(var_7);
            wp::array_store(var_qLU_out, var_0, var_1, var_8);
        }
        if (!var_6) {
            // qLU_out[worldid, elemid] = 0.0                                                     <L 575>
            wp::array_store(var_qLU_out, var_0, var_1, var_9);
        }
    }
}



extern "C" __global__ void _tendon_actuator_force_2da6552b_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_actuator_trntype,
    wp::array_t<wp::vec_t<2, wp::int32>> var_actuator_trnid,
    wp::array_t<wp::float32> var_actuator_force_in,
    wp::array_t<wp::float32> var_ten_actfrc_out)
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
        wp::int32* var_2;
        const wp::int32 var_3 = 3;
        bool var_4;
        wp::int32 var_5;
        wp::vec_t<2, wp::int32>* var_6;
        const wp::int32 var_7 = 0;
        wp::int32 var_8;
        wp::vec_t<2, wp::int32> var_9;
        wp::array_t<wp::float32> var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        //---------
        // forward
        // def _tendon_actuator_force(                                                            <L 1054>
        // worldid, actid = wp.tid()                                                              <L 1063>
        builtin_tid2d(var_0, var_1);
        // if actuator_trntype[actid] == TrnType.TENDON:                                          <L 1065>
        var_2 = wp::address(var_actuator_trntype, var_1);
        var_5 = wp::load(var_2);
        var_4 = (var_5 == var_3);
        if (var_4) {
            // tenid = actuator_trnid[actid][0]                                                   <L 1066>
            var_6 = wp::address(var_actuator_trnid, var_1);
            var_9 = wp::load(var_6);
            var_8 = wp::extract(var_9, var_7);
            // wp.atomic_add(ten_actfrc_out[worldid], tenid, actuator_force_in[worldid, actid])       <L 1068>
            var_10 = wp::view(var_ten_actfrc_out, var_0);
            var_11 = wp::address(var_actuator_force_in, var_0, var_1);
            var_13 = wp::load(var_11);
            var_12 = wp::atomic_add(var_10, var_8, var_13);
        }
    }
}



extern "C" __global__ void _rk_accumulate_activation_velocity_5935ab3f_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_act_dot_in,
    wp::float32 var_scale,
    wp::array_t<wp::float32> var_act_dot_out)
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
        wp::float32 var_5;
        //---------
        // forward
        // def _rk_accumulate_activation_velocity(                                                <L 486>
        // worldid, actid = wp.tid()                                                              <L 494>
        builtin_tid2d(var_0, var_1);
        // act_dot_out[worldid, actid] += scale * act_dot_in[worldid, actid]                      <L 495>
        var_2 = wp::address(var_act_dot_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::mul(var_scale, var_4);
        var_5 = wp::atomic_add(var_act_dot_out, var_0, var_1, var_3);
    }
}



extern "C" __global__ void _qfrc_actuator_6de5c51c_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_moment_rownnz_in,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::int32> var_moment_colind_in,
    wp::array_t<wp::float32> var_actuator_moment_in,
    wp::array_t<wp::float32> var_actuator_force_in,
    wp::array_t<wp::float32> var_qfrc_actuator_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::range_t var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::float32* var_14;
        wp::float32* var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::array_t<wp::float32> var_19;
        wp::float32 var_20;
        //---------
        // forward
        // def _qfrc_actuator(                                                                    <L 1098>
        // worldid, actid = wp.tid()                                                              <L 1108>
        builtin_tid2d(var_0, var_1);
        // rownnz = moment_rownnz_in[worldid, actid]                                              <L 1110>
        var_2 = wp::address(var_moment_rownnz_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // rowadr = moment_rowadr_in[worldid, actid]                                              <L 1111>
        var_5 = wp::address(var_moment_rowadr_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // for i in range(rownnz):                                                                <L 1113>
        var_8 = wp::range(var_3);
        start_for_0:;
            if (iter_cmp(var_8) == 0) goto end_for_0;
            var_9 = wp::iter_next(var_8);
            // sparseid = rowadr + i                                                              <L 1114>
            var_10 = wp::add(var_6, var_9);
            // colind = moment_colind_in[worldid, sparseid]                                       <L 1115>
            var_11 = wp::address(var_moment_colind_in, var_0, var_10);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
            // qfrc = actuator_moment_in[worldid, sparseid] * actuator_force_in[worldid, actid]       <L 1116>
            var_14 = wp::address(var_actuator_moment_in, var_0, var_10);
            var_15 = wp::address(var_actuator_force_in, var_0, var_1);
            var_17 = wp::load(var_14);
            var_18 = wp::load(var_15);
            var_16 = wp::mul(var_17, var_18);
            // wp.atomic_add(qfrc_actuator_out[worldid], colind, qfrc)                            <L 1117>
            var_19 = wp::view(var_qfrc_actuator_out, var_0);
            var_20 = wp::atomic_add(var_19, var_12, var_16);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void _rk_accumulate_velocity_acceleration_829bb7cc_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_qacc_in,
    wp::float32 var_scale,
    wp::array_t<wp::float32> var_qvel_out,
    wp::array_t<wp::float32> var_qacc_out)
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
        wp::float32 var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        //---------
        // forward
        // def _rk_accumulate_velocity_acceleration(                                              <L 470>
        // worldid, dofid = wp.tid()                                                              <L 480>
        builtin_tid2d(var_0, var_1);
        // qvel_out[worldid, dofid] += scale * qvel_in[worldid, dofid]                            <L 481>
        var_2 = wp::address(var_qvel_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::mul(var_scale, var_4);
        var_5 = wp::atomic_add(var_qvel_out, var_0, var_1, var_3);
        // qacc_out[worldid, dofid] += scale * qacc_in[worldid, dofid]                            <L 482>
        var_6 = wp::address(var_qacc_in, var_0, var_1);
        var_8 = wp::load(var_6);
        var_7 = wp::mul(var_scale, var_8);
        var_9 = wp::atomic_add(var_qacc_out, var_0, var_1, var_7);
    }
}



extern "C" __global__ void _actuator_force_a442448b_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_na,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::int32> var_actuator_dyntype,
    wp::array_t<wp::int32> var_actuator_gaintype,
    wp::array_t<wp::int32> var_actuator_biastype,
    wp::array_t<wp::int32> var_actuator_actadr,
    wp::array_t<wp::int32> var_actuator_actnum,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_dynprm,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_gainprm,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_biasprm,
    wp::array_t<bool> var_actuator_actlimited,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_actrange,
    wp::array_t<bool> var_actuator_actearly,
    wp::array_t<bool> var_actuator_forcelimited,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_forcerange,
    wp::array_t<bool> var_actuator_ctrllimited,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_ctrlrange,
    wp::array_t<wp::float32> var_actuator_acc0,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_lengthrange,
    wp::array_t<wp::float32> var_act_in,
    wp::array_t<wp::float32> var_ctrl_in,
    wp::array_t<wp::float32> var_actuator_length_in,
    wp::array_t<wp::float32> var_actuator_velocity_in,
    wp::int32 var_dsbl_clampctrl,
    wp::array_t<wp::float32> var_act_dot_out,
    wp::array_t<wp::float32> var_actuator_force_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        bool var_10;
        bool* var_11;
        bool var_12;
        bool var_13;
        wp::vec_t<2, wp::float32>* var_14;
        wp::vec_t<2, wp::float32> var_15;
        wp::vec_t<2, wp::float32> var_16;
        const wp::int32 var_17 = 0;
        wp::float32 var_18;
        const wp::int32 var_19 = 1;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        bool var_27;
        const wp::int32 var_28 = 0;
        bool var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        const wp::int32 var_33 = 1;
        wp::int32 var_34;
        wp::int32* var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        wp::shape_t* var_38;
        const wp::int32 var_39 = 0;
        wp::int32 var_40;
        wp::shape_t var_41;
        wp::int32 var_42;
        wp::vec_t<10, wp::float32>* var_43;
        wp::vec_t<10, wp::float32> var_44;
        wp::vec_t<10, wp::float32> var_45;
        const wp::int32 var_46 = 1;
        bool var_47;
        wp::float32 var_48;
        bool var_49;
        const wp::int32 var_50 = 2;
        bool var_51;
        const wp::int32 var_52 = 3;
        bool var_53;
        wp::float32* var_54;
        wp::float32 var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        const wp::int32 var_58 = 0;
        wp::float32 var_59;
        const wp::float32 var_60 = 1e-15;
        wp::float32 var_61;
        wp::float32 var_62;
        const wp::int32 var_63 = 4;
        bool var_64;
        wp::shape_t* var_65;
        const wp::int32 var_66 = 0;
        wp::int32 var_67;
        wp::shape_t var_68;
        wp::int32 var_69;
        wp::vec_t<10, wp::float32>* var_70;
        wp::vec_t<10, wp::float32> var_71;
        wp::vec_t<10, wp::float32> var_72;
        wp::float32* var_73;
        wp::float32 var_74;
        wp::float32 var_75;
        wp::float32 var_76;
        const wp::int32 var_77 = 5;
        bool var_78;
        wp::shape_t* var_79;
        const wp::int32 var_80 = 0;
        wp::int32 var_81;
        wp::shape_t var_82;
        wp::int32 var_83;
        wp::vec_t<10, wp::float32>* var_84;
        wp::vec_t<10, wp::float32> var_85;
        wp::vec_t<10, wp::float32> var_86;
        wp::vec_t<6, wp::int32> var_87;
        wp::int32 var_88;
        const wp::float32 var_89 = 0.0;
        const wp::int32 var_90 = 0;
        wp::int32 var_91;
        const wp::int32 var_92 = 0;
        bool var_93;
        wp::float32* var_94;
        wp::float32 var_95;
        wp::float32 var_96;
        const wp::int32 var_97 = 7;
        wp::float32 var_98;
        wp::shape_t* var_99;
        const wp::int32 var_100 = 0;
        wp::int32 var_101;
        wp::shape_t var_102;
        wp::int32 var_103;
        wp::float32* var_104;
        wp::float32 var_105;
        wp::float32 var_106;
        wp::float32 var_107;
        wp::float32 var_108;
        wp::float32 var_109;
        wp::float32 var_110;
        wp::shape_t* var_111;
        const wp::int32 var_112 = 0;
        wp::int32 var_113;
        wp::shape_t var_114;
        wp::int32 var_115;
        wp::float32* var_116;
        wp::float32 var_117;
        wp::float32 var_118;
        wp::float32 var_119;
        const wp::int32 var_120 = 1;
        wp::int32 var_121;
        wp::float32 var_122;
        wp::int32 var_123;
        wp::float32 var_124;
        const wp::float32 var_125 = 0.0;
        const wp::int32 var_126 = 1;
        wp::int32 var_127;
        const wp::int32 var_128 = 0;
        bool var_129;
        wp::float32* var_130;
        wp::float32 var_131;
        wp::float32 var_132;
        const wp::int32 var_133 = 8;
        wp::float32 var_134;
        wp::int32 var_135;
        const wp::int32 var_136 = 8;
        wp::float32 var_137;
        wp::float32 var_138;
        const wp::int32 var_139 = 1;
        bool var_140;
        wp::float32* var_141;
        wp::float32 var_142;
        wp::float32 var_143;
        wp::float32 var_144;
        const wp::float32 var_145 = 0.0;
        bool var_146;
        bool var_147;
        const wp::float32 var_148 = 0.0;
        wp::float32 var_149;
        wp::float32 var_150;
        bool var_151;
        const wp::float32 var_152 = 0.0;
        wp::float32 var_153;
        wp::float32 var_154;
        wp::float32 var_155;
        wp::float32 var_156;
        const wp::int32 var_157 = 1;
        wp::int32 var_158;
        wp::int32 var_159;
        wp::float32 var_160;
        wp::float32 var_161;
        wp::float32* var_162;
        wp::float32* var_163;
        wp::float32 var_164;
        wp::float32 var_165;
        wp::float32 var_166;
        const wp::int32 var_167 = 0;
        wp::float32 var_168;
        const wp::int32 var_169 = 1;
        wp::float32 var_170;
        const wp::float32 var_171 = 1e-15;
        const wp::int32 var_172 = 0;
        wp::float32 var_173;
        wp::float32 var_174;
        const wp::int32 var_175 = 2;
        wp::int32 var_176;
        const wp::int32 var_177 = 0;
        bool var_178;
        const wp::int32 var_179 = 2;
        wp::float32 var_180;
        const wp::int32 var_181 = 3;
        wp::float32 var_182;
        const wp::int32 var_183 = 4;
        wp::float32 var_184;
        const wp::int32 var_185 = 2;
        wp::float32 var_186;
        const wp::int32 var_187 = 3;
        wp::float32 var_188;
        wp::float32* var_189;
        wp::float32 var_190;
        wp::float32 var_191;
        const wp::float32 var_192 = 1.0;
        wp::float32 var_193;
        wp::float32 var_194;
        wp::float32 var_195;
        wp::float32 var_196;
        wp::float32 var_197;
        wp::float32* var_198;
        wp::float32 var_199;
        wp::float32 var_200;
        wp::float32 var_201;
        wp::float32 var_202;
        const wp::int32 var_203 = 4;
        wp::int32 var_204;
        const wp::int32 var_205 = 0;
        bool var_206;
        wp::float32* var_207;
        wp::float32 var_208;
        wp::float32 var_209;
        wp::float32 var_210;
        wp::float32 var_211;
        wp::float32 var_212;
        wp::float32 var_213;
        wp::float32 var_214;
        wp::float32 var_215;
        const wp::int32 var_216 = 1;
        wp::int32 var_217;
        wp::float32 var_218;
        wp::int32 var_219;
        wp::float32 var_220;
        wp::float32 var_221;
        const wp::int32 var_222 = 3;
        wp::int32 var_223;
        const wp::int32 var_224 = 0;
        bool var_225;
        const wp::int32 var_226 = 5;
        wp::float32 var_227;
        wp::shape_t* var_228;
        const wp::int32 var_229 = 0;
        wp::int32 var_230;
        wp::shape_t var_231;
        wp::int32 var_232;
        wp::vec_t<10, wp::float32>* var_233;
        wp::vec_t<10, wp::float32> var_234;
        wp::vec_t<10, wp::float32> var_235;
        const wp::int32 var_236 = 3;
        wp::float32 var_237;
        const wp::int32 var_238 = 4;
        wp::float32 var_239;
        const wp::int32 var_240 = 5;
        wp::float32 var_241;
        wp::float32* var_242;
        wp::float32 var_243;
        wp::float32 var_244;
        wp::float32* var_245;
        wp::float32 var_246;
        wp::float32 var_247;
        wp::float32 var_248;
        wp::float32* var_249;
        wp::float32 var_250;
        wp::float32 var_251;
        wp::float32 var_252;
        wp::float32 var_253;
        wp::float32 var_254;
        wp::float32 var_255;
        wp::float32* var_256;
        wp::float32 var_257;
        wp::float32 var_258;
        const wp::int32 var_259 = 1;
        wp::int32 var_260;
        wp::int32 var_261;
        wp::float32 var_262;
        const wp::int32 var_263 = 4;
        wp::int32 var_264;
        const wp::int32 var_265 = 0;
        bool var_266;
        const wp::int32 var_267 = 1;
        wp::float32 var_268;
        wp::float32 var_269;
        wp::float32 var_270;
        wp::float32* var_271;
        wp::float32 var_272;
        wp::float32 var_273;
        wp::float32 var_274;
        wp::float32* var_275;
        wp::float32 var_276;
        wp::float32 var_277;
        wp::float32 var_278;
        const wp::float32 var_279 = 0.0;
        bool var_280;
        wp::float32 var_281;
        wp::float32 var_282;
        wp::float32 var_283;
        wp::float32 var_284;
        const wp::int32 var_285 = 7;
        bool var_286;
        const wp::float32 var_287 = 0.0;
        const wp::float32 var_288 = 0.0;
        wp::float32 var_289;
        wp::float32 var_290;
        wp::float32 var_291;
        wp::float32 var_292;
        wp::vec_t<10, wp::float32> var_293;
        wp::float32 var_294;
        wp::float32 var_295;
        wp::vec_t<10, wp::float32> var_296;
        wp::float32 var_297;
        wp::float32 var_298;
        wp::float32 var_299;
        wp::float32 var_300;
        wp::vec_t<10, wp::float32> var_301;
        wp::float32 var_302;
        bool* var_303;
        bool var_304;
        bool var_305;
        const wp::int32 var_306 = 1;
        bool var_307;
        const wp::int32 var_308 = 0;
        bool var_309;
        const wp::int32 var_310 = 5;
        bool var_311;
        wp::float32* var_312;
        wp::float32 var_313;
        wp::float32 var_314;
        wp::float32 var_315;
        const wp::int32 var_316 = 5;
        bool var_317;
        wp::shape_t* var_318;
        const wp::int32 var_319 = 0;
        wp::int32 var_320;
        wp::shape_t var_321;
        wp::int32 var_322;
        wp::vec_t<10, wp::float32>* var_323;
        wp::vec_t<10, wp::float32> var_324;
        wp::vec_t<10, wp::float32> var_325;
        wp::vec_t<6, wp::int32> var_326;
        wp::int32* var_327;
        const wp::int32 var_328 = 1;
        wp::int32 var_329;
        wp::int32 var_330;
        const wp::int32 var_331 = 4;
        wp::int32 var_332;
        bool var_333;
        const wp::int32 var_334 = 0;
        wp::float32 var_335;
        wp::float32 var_336;
        wp::float32 var_337;
        const wp::float32 var_338 = 1.0;
        wp::shape_t* var_339;
        const wp::int32 var_340 = 0;
        wp::int32 var_341;
        wp::shape_t var_342;
        wp::int32 var_343;
        wp::float32* var_344;
        wp::float32 var_345;
        wp::float32 var_346;
        wp::float32 var_347;
        wp::float32 var_348;
        wp::float32 var_349;
        wp::float32 var_350;
        wp::float32 var_351;
        const wp::int32 var_352 = 3;
        wp::int32 var_353;
        bool var_354;
        const wp::int32 var_355 = 5;
        wp::float32 var_356;
        wp::shape_t* var_357;
        const wp::int32 var_358 = 0;
        wp::int32 var_359;
        wp::shape_t var_360;
        wp::int32 var_361;
        wp::vec_t<10, wp::float32>* var_362;
        wp::vec_t<10, wp::float32> var_363;
        wp::vec_t<10, wp::float32> var_364;
        const wp::int32 var_365 = 3;
        wp::float32 var_366;
        const wp::int32 var_367 = 4;
        wp::float32 var_368;
        const wp::int32 var_369 = 5;
        wp::float32 var_370;
        wp::float32* var_371;
        wp::float32 var_372;
        wp::float32 var_373;
        wp::float32 var_374;
        wp::float32 var_375;
        wp::float32 var_376;
        wp::float32 var_377;
        wp::float32 var_378;
        wp::float32 var_379;
        wp::shape_t* var_380;
        const wp::int32 var_381 = 0;
        wp::int32 var_382;
        wp::shape_t var_383;
        wp::int32 var_384;
        wp::float32* var_385;
        wp::float32 var_386;
        wp::float32 var_387;
        wp::float32 var_388;
        wp::float32 var_389;
        wp::float32 var_390;
        wp::float32 var_391;
        bool var_392;
        const wp::float32 var_393 = 1.0;
        wp::float32 var_394;
        wp::float32 var_395;
        wp::float32 var_396;
        wp::float32 var_397;
        wp::float32 var_398;
        wp::float32 var_399;
        const wp::int32 var_400 = 1;
        wp::int32 var_401;
        bool var_402;
        wp::shape_t* var_403;
        const wp::int32 var_404 = 0;
        wp::int32 var_405;
        wp::shape_t var_406;
        wp::int32 var_407;
        wp::float32* var_408;
        wp::float32 var_409;
        wp::float32 var_410;
        wp::float32 var_411;
        const wp::int32 var_412 = 8;
        wp::float32 var_413;
        const wp::float32 var_414 = 0.0;
        bool var_415;
        wp::float32 var_416;
        wp::float32 var_417;
        wp::float32 var_418;
        wp::shape_t* var_419;
        const wp::int32 var_420 = 0;
        wp::int32 var_421;
        wp::shape_t var_422;
        wp::int32 var_423;
        wp::float32* var_424;
        wp::float32 var_425;
        wp::float32 var_426;
        wp::float32 var_427;
        wp::float32 var_428;
        wp::float32 var_429;
        wp::float32 var_430;
        wp::float32 var_431;
        wp::float32 var_432;
        wp::vec_t<10, wp::float32> var_433;
        wp::float32 var_434;
        wp::float32 var_435;
        wp::float32 var_436;
        wp::float32 var_437;
        wp::float32 var_438;
        wp::float32 var_439;
        wp::float32 var_440;
        wp::float32 var_441;
        wp::float32 var_442;
        wp::vec_t<10, wp::float32> var_443;
        wp::float32 var_444;
        wp::float32 var_445;
        wp::float32 var_446;
        wp::float32 var_447;
        wp::float32 var_448;
        bool* var_449;
        bool var_450;
        wp::shape_t* var_451;
        const wp::int32 var_452 = 0;
        wp::int32 var_453;
        wp::shape_t var_454;
        wp::int32 var_455;
        wp::vec_t<2, wp::float32>* var_456;
        wp::vec_t<2, wp::float32> var_457;
        wp::vec_t<2, wp::float32> var_458;
        const wp::int32 var_459 = 0;
        wp::float32 var_460;
        const wp::int32 var_461 = 1;
        wp::float32 var_462;
        wp::float32 var_463;
        bool var_464;
        wp::float32 var_465;
        bool var_466;
        wp::shape_t* var_467;
        const wp::int32 var_468 = 0;
        wp::int32 var_469;
        wp::shape_t var_470;
        wp::int32 var_471;
        wp::float32* var_472;
        wp::shape_t* var_473;
        const wp::int32 var_474 = 0;
        wp::int32 var_475;
        wp::shape_t var_476;
        wp::int32 var_477;
        wp::vec_t<2, wp::float32>* var_478;
        const wp::float32 var_479 = 1.0;
        bool* var_480;
        wp::float32 var_481;
        wp::float32 var_482;
        wp::vec_t<2, wp::float32> var_483;
        bool var_484;
        wp::float32 var_485;
        wp::vec_t<10, wp::float32> var_486;
        wp::vec_t<6, wp::int32> var_487;
        wp::float32 var_488;
        wp::float32 var_489;
        wp::float32 var_490;
        wp::vec_t<10, wp::float32> var_491;
        wp::float32 var_492;
        wp::float32 var_493;
        wp::float32 var_494;
        wp::float32 var_495;
        wp::float32 var_496;
        bool var_497;
        bool var_498;
        wp::float32* var_499;
        wp::float32 var_500;
        wp::float32 var_501;
        bool var_502;
        wp::float32 var_503;
        bool var_504;
        wp::float32 var_505;
        bool var_506;
        wp::vec_t<10, wp::float32> var_507;
        bool var_508;
        wp::vec_t<6, wp::int32> var_509;
        bool var_510;
        wp::float32 var_511;
        bool var_512;
        wp::float32 var_513;
        bool var_514;
        wp::float32 var_515;
        bool var_516;
        wp::vec_t<10, wp::float32> var_517;
        bool var_518;
        wp::float32 var_519;
        bool var_520;
        wp::float32 var_521;
        bool var_522;
        wp::float32 var_523;
        bool var_524;
        wp::float32 var_525;
        bool var_526;
        wp::float32 var_527;
        bool var_528;
        wp::float32 var_529;
        wp::float32 var_530;
        wp::float32* var_531;
        wp::float32 var_532;
        wp::float32 var_533;
        wp::float32* var_534;
        wp::float32 var_535;
        wp::float32 var_536;
        wp::int32* var_537;
        wp::int32 var_538;
        wp::int32 var_539;
        wp::shape_t* var_540;
        const wp::int32 var_541 = 0;
        wp::int32 var_542;
        wp::shape_t var_543;
        wp::int32 var_544;
        wp::vec_t<10, wp::float32>* var_545;
        wp::vec_t<10, wp::float32> var_546;
        wp::vec_t<10, wp::float32> var_547;
        const wp::float32 var_548 = 0.0;
        const wp::int32 var_549 = 0;
        bool var_550;
        const wp::int32 var_551 = 0;
        wp::float32 var_552;
        const wp::int32 var_553 = 1;
        bool var_554;
        const wp::int32 var_555 = 0;
        wp::float32 var_556;
        const wp::int32 var_557 = 1;
        wp::float32 var_558;
        wp::float32 var_559;
        wp::float32 var_560;
        const wp::int32 var_561 = 2;
        wp::float32 var_562;
        wp::float32 var_563;
        wp::float32 var_564;
        const wp::int32 var_565 = 2;
        bool var_566;
        wp::shape_t* var_567;
        const wp::int32 var_568 = 0;
        wp::int32 var_569;
        wp::shape_t var_570;
        wp::int32 var_571;
        wp::float32* var_572;
        wp::float32 var_573;
        wp::float32 var_574;
        wp::shape_t* var_575;
        const wp::int32 var_576 = 0;
        wp::int32 var_577;
        wp::shape_t var_578;
        wp::int32 var_579;
        wp::vec_t<2, wp::float32>* var_580;
        wp::vec_t<2, wp::float32> var_581;
        wp::vec_t<2, wp::float32> var_582;
        wp::float32 var_583;
        const wp::int32 var_584 = 3;
        bool var_585;
        const wp::int32 var_586 = 0;
        wp::float32 var_587;
        const wp::int32 var_588 = 1;
        wp::float32 var_589;
        const wp::int32 var_590 = 0;
        wp::float32 var_591;
        wp::vec_t<6, wp::int32> var_592;
        wp::int32 var_593;
        const wp::int32 var_594 = 2;
        wp::int32 var_595;
        const wp::int32 var_596 = 0;
        bool var_597;
        const wp::int32 var_598 = 2;
        wp::int32 var_599;
        wp::int32 var_600;
        wp::float32* var_601;
        wp::float32 var_602;
        wp::float32 var_603;
        const wp::int32 var_604 = 2;
        wp::float32 var_605;
        const wp::int32 var_606 = 3;
        wp::float32 var_607;
        const wp::int32 var_608 = 4;
        wp::float32 var_609;
        const wp::float32 var_610 = 1.0;
        wp::float32 var_611;
        wp::float32 var_612;
        wp::float32 var_613;
        wp::float32 var_614;
        wp::float32 var_615;
        wp::float32 var_616;
        wp::float32 var_617;
        wp::float32 var_618;
        wp::float32 var_619;
        wp::float32 var_620;
        const wp::float32 var_621 = 0.0;
        bool var_622;
        wp::float32 var_623;
        wp::float32 var_624;
        wp::float32 var_625;
        const wp::float32 var_626 = 0.0;
        bool var_627;
        const wp::int32 var_628 = 8;
        wp::float32 var_629;
        wp::int32 var_630;
        const wp::int32 var_631 = 0;
        bool var_632;
        const wp::float32 var_633 = 0.0;
        const wp::int32 var_634 = 1;
        wp::int32 var_635;
        const wp::int32 var_636 = 0;
        bool var_637;
        const wp::int32 var_638 = 1;
        wp::int32 var_639;
        wp::int32 var_640;
        wp::float32* var_641;
        wp::float32 var_642;
        wp::float32 var_643;
        wp::float32 var_644;
        wp::float32 var_645;
        wp::float32 var_646;
        wp::float32 var_647;
        wp::float32 var_648;
        wp::float32 var_649;
        wp::float32 var_650;
        wp::int32 var_651;
        wp::float32 var_652;
        wp::vec_t<6, wp::int32> var_653;
        wp::int32 var_654;
        wp::float32 var_655;
        wp::int32 var_656;
        wp::float32 var_657;
        wp::float32 var_658;
        wp::float32 var_659;
        wp::float32 var_660;
        wp::float32 var_661;
        wp::float32 var_662;
        wp::float32 var_663;
        wp::float32 var_664;
        wp::float32 var_665;
        wp::vec_t<6, wp::int32> var_666;
        wp::int32 var_667;
        wp::float32 var_668;
        wp::int32 var_669;
        wp::float32 var_670;
        wp::float32 var_671;
        wp::float32 var_672;
        wp::float32 var_673;
        wp::float32 var_674;
        wp::float32 var_675;
        wp::float32 var_676;
        wp::float32 var_677;
        wp::float32 var_678;
        wp::vec_t<6, wp::int32> var_679;
        wp::int32 var_680;
        wp::float32 var_681;
        wp::int32 var_682;
        wp::float32 var_683;
        wp::float32 var_684;
        wp::float32 var_685;
        wp::float32 var_686;
        wp::float32 var_687;
        wp::float32 var_688;
        wp::float32 var_689;
        wp::float32 var_690;
        wp::float32 var_691;
        wp::vec_t<6, wp::int32> var_692;
        wp::int32 var_693;
        wp::float32 var_694;
        wp::int32 var_695;
        wp::float32 var_696;
        wp::float32 var_697;
        wp::float32 var_698;
        wp::float32 var_699;
        wp::float32 var_700;
        wp::float32 var_701;
        wp::float32 var_702;
        wp::float32 var_703;
        wp::int32* var_704;
        wp::int32 var_705;
        wp::int32 var_706;
        wp::shape_t* var_707;
        const wp::int32 var_708 = 0;
        wp::int32 var_709;
        wp::shape_t var_710;
        wp::int32 var_711;
        wp::vec_t<10, wp::float32>* var_712;
        wp::vec_t<10, wp::float32> var_713;
        wp::vec_t<10, wp::float32> var_714;
        const wp::float32 var_715 = 0.0;
        const wp::int32 var_716 = 1;
        bool var_717;
        const wp::int32 var_718 = 0;
        wp::float32 var_719;
        const wp::int32 var_720 = 1;
        wp::float32 var_721;
        wp::float32 var_722;
        wp::float32 var_723;
        const wp::int32 var_724 = 2;
        wp::float32 var_725;
        wp::float32 var_726;
        wp::float32 var_727;
        const wp::int32 var_728 = 2;
        bool var_729;
        wp::shape_t* var_730;
        const wp::int32 var_731 = 0;
        wp::int32 var_732;
        wp::shape_t var_733;
        wp::int32 var_734;
        wp::float32* var_735;
        wp::float32 var_736;
        wp::float32 var_737;
        wp::shape_t* var_738;
        const wp::int32 var_739 = 0;
        wp::int32 var_740;
        wp::shape_t var_741;
        wp::int32 var_742;
        wp::vec_t<2, wp::float32>* var_743;
        wp::vec_t<2, wp::float32> var_744;
        wp::vec_t<2, wp::float32> var_745;
        wp::float32 var_746;
        const wp::int32 var_747 = 3;
        bool var_748;
        const wp::int32 var_749 = 0;
        wp::float32 var_750;
        const wp::float32 var_751 = 0.0;
        bool var_752;
        const wp::int32 var_753 = 1;
        wp::float32 var_754;
        wp::float32 var_755;
        wp::float32 var_756;
        wp::float32 var_757;
        wp::float32 var_758;
        wp::float32 var_759;
        wp::float32 var_760;
        wp::float32 var_761;
        wp::float32 var_762;
        wp::float32 var_763;
        wp::vec_t<2, wp::float32> var_764;
        wp::float32 var_765;
        wp::float32 var_766;
        wp::float32 var_767;
        wp::vec_t<2, wp::float32> var_768;
        wp::float32 var_769;
        wp::float32 var_770;
        wp::float32 var_771;
        bool* var_772;
        bool var_773;
        wp::shape_t* var_774;
        const wp::int32 var_775 = 0;
        wp::int32 var_776;
        wp::shape_t var_777;
        wp::int32 var_778;
        wp::vec_t<2, wp::float32>* var_779;
        wp::vec_t<2, wp::float32> var_780;
        wp::vec_t<2, wp::float32> var_781;
        const wp::int32 var_782 = 0;
        wp::float32 var_783;
        const wp::int32 var_784 = 1;
        wp::float32 var_785;
        wp::float32 var_786;
        bool var_787;
        wp::float32 var_788;
        bool var_789;
        const wp::int32 var_790 = 3;
        bool var_791;
        const wp::int32 var_792 = 0;
        wp::float32 var_793;
        const wp::float32 var_794 = 0.0;
        bool var_795;
        const wp::int32 var_796 = 1;
        wp::float32 var_797;
        const wp::int32 var_798 = 2;
        wp::float32 var_799;
        wp::float32 var_800;
        wp::float32 var_801;
        wp::float32 var_802;
        wp::float32 var_803;
        wp::float32 var_804;
        wp::float32 var_805;
        const wp::int32 var_806 = 5;
        wp::float32 var_807;
        const wp::float32 var_808 = 0.0;
        bool var_809;
        const wp::int32 var_810 = 6;
        wp::float32 var_811;
        wp::vec_t<6, wp::int32> var_812;
        const wp::int32 var_813 = 3;
        wp::int32 var_814;
        wp::int32 var_815;
        wp::float32* var_816;
        wp::float32 var_817;
        wp::float32 var_818;
        wp::float32* var_819;
        wp::float32 var_820;
        wp::float32 var_821;
        wp::float32 var_822;
        wp::float32 var_823;
        wp::float32 var_824;
        wp::float32 var_825;
        wp::vec_t<6, wp::int32> var_826;
        wp::int32 var_827;
        wp::float32 var_828;
        wp::float32 var_829;
        wp::vec_t<6, wp::int32> var_830;
        wp::int32 var_831;
        wp::float32 var_832;
        wp::float32 var_833;
        wp::float32 var_834;
        //---------
        // forward
        // def _actuator_force(                                                                   <L 757>
        // worldid, uid = wp.tid()                                                                <L 789>
        builtin_tid2d(var_0, var_1);
        // actuator_ctrlrange_id = worldid % actuator_ctrlrange.shape[0]                          <L 791>
        var_2 = &(var_actuator_ctrlrange.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // ctrl = ctrl_in[worldid, uid]                                                           <L 793>
        var_7 = wp::address(var_ctrl_in, var_0, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // if actuator_ctrllimited[uid] and not dsbl_clampctrl:                                   <L 795>
        var_11 = wp::address(var_actuator_ctrllimited, var_1);
        var_12 = wp::load(var_11);
        var_10 = var_12;
        if (var_10) {
            var_13 = wp::unot(var_dsbl_clampctrl);
            var_10 = var_10 && var_13;
        }
        if (var_10) {
            // ctrlrange = actuator_ctrlrange[actuator_ctrlrange_id, uid]                         <L 796>
            var_14 = wp::address(var_actuator_ctrlrange, var_6, var_1);
            var_16 = wp::load(var_14);
            var_15 = wp::copy(var_16);
            // ctrl = wp.clamp(ctrl, ctrlrange[0], ctrlrange[1])                                  <L 797>
            var_18 = wp::extract(var_15, var_17);
            var_20 = wp::extract(var_15, var_19);
            var_21 = wp::clamp(var_8, var_18, var_20);
        }
        var_22 = wp::where(var_10, var_21, var_8);
        // ctrl_act = ctrl                                                                        <L 798>
        var_23 = wp::copy(var_22);
        // act_first = actuator_actadr[uid]                                                       <L 800>
        var_24 = wp::address(var_actuator_actadr, var_1);
        var_26 = wp::load(var_24);
        var_25 = wp::copy(var_26);
        // if na and act_first >= 0:                                                              <L 801>
        var_27 = var_na;
        if (var_27) {
            var_29 = (var_25 >= var_28);
            var_27 = var_27 && var_29;
        }
        if (var_27) {
            // act_last = act_first + actuator_actnum[uid] - 1                                    <L 802>
            var_30 = wp::address(var_actuator_actnum, var_1);
            var_32 = wp::load(var_30);
            var_31 = wp::add(var_25, var_32);
            var_34 = wp::sub(var_31, var_33);
            // dyntype = actuator_dyntype[uid]                                                    <L 803>
            var_35 = wp::address(var_actuator_dyntype, var_1);
            var_37 = wp::load(var_35);
            var_36 = wp::copy(var_37);
            // dynprm = actuator_dynprm[worldid % actuator_dynprm.shape[0], uid]                  <L 804>
            var_38 = &(var_actuator_dynprm.shape);
            var_41 = wp::load(var_38);
            var_40 = wp::extract(var_41, var_39);
            var_42 = wp::mod(var_0, var_40);
            var_43 = wp::address(var_actuator_dynprm, var_42, var_1);
            var_45 = wp::load(var_43);
            var_44 = wp::copy(var_45);
            // if dyntype == DynType.INTEGRATOR:                                                  <L 806>
            var_47 = (var_36 == var_46);
            if (var_47) {
                // act_dot = ctrl                                                                 <L 807>
                var_48 = wp::copy(var_22);
            }
            if (!var_47) {
                // elif dyntype == DynType.FILTER or dyntype == DynType.FILTEREXACT:              <L 808>
                var_51 = (var_36 == var_50);
                var_49 = var_51;
                if (!var_49) {
                    var_53 = (var_36 == var_52);
                    var_49 = var_49 || var_53;
                }
                if (var_49) {
                    // act = act_in[worldid, act_last]                                            <L 809>
                    var_54 = wp::address(var_act_in, var_0, var_34);
                    var_56 = wp::load(var_54);
                    var_55 = wp::copy(var_56);
                    // act_dot = (ctrl - act) / wp.max(dynprm[0], MJ_MINVAL)                      <L 810>
                    var_57 = wp::sub(var_22, var_55);
                    var_59 = wp::extract(var_44, var_58);
                    var_61 = wp::max(var_59, var_60);
                    var_62 = wp::div(var_57, var_61);
                }
                if (!var_49) {
                    // elif dyntype == DynType.MUSCLE:                                            <L 811>
                    var_64 = (var_36 == var_63);
                    if (var_64) {
                        // dynprm = actuator_dynprm[worldid % actuator_dynprm.shape[0], uid]       <L 812>
                        var_65 = &(var_actuator_dynprm.shape);
                        var_68 = wp::load(var_65);
                        var_67 = wp::extract(var_68, var_66);
                        var_69 = wp::mod(var_0, var_67);
                        var_70 = wp::address(var_actuator_dynprm, var_69, var_1);
                        var_72 = wp::load(var_70);
                        var_71 = wp::copy(var_72);
                        // act = act_in[worldid, act_last]                                        <L 813>
                        var_73 = wp::address(var_act_in, var_0, var_34);
                        var_75 = wp::load(var_73);
                        var_74 = wp::copy(var_75);
                        // act_dot = util_misc.muscle_dynamics(ctrl, act, dynprm)                 <L 814>
                        var_76 = muscle_dynamics_0(var_22, var_74, var_71);
                    }
                    if (!var_64) {
                        // elif dyntype == DynType.DCMOTOR:                                       <L 815>
                        var_78 = (var_36 == var_77);
                        if (var_78) {
                            // gainprm = actuator_gainprm[worldid % actuator_gainprm.shape[0], uid]       <L 816>
                            var_79 = &(var_actuator_gainprm.shape);
                            var_82 = wp::load(var_79);
                            var_81 = wp::extract(var_82, var_80);
                            var_83 = wp::mod(var_0, var_81);
                            var_84 = wp::address(var_actuator_gainprm, var_83, var_1);
                            var_86 = wp::load(var_84);
                            var_85 = wp::copy(var_86);
                            // slots = util_misc.dcmotor_slots(dynprm, gainprm)                   <L 817>
                            var_87 = dcmotor_slots_0(var_44, var_85);
                            // adr = act_first                                                    <L 818>
                            var_88 = wp::copy(var_25);
                            // act_dot = 0.0                                                      <L 820>
                            // if slots[0] >= 0:                                                  <L 823>
                            var_91 = wp::extract(var_87, var_90);
                            var_93 = (var_91 >= var_92);
                            if (var_93) {
                                // u_prev = act_in[worldid, adr]                                  <L 824>
                                var_94 = wp::address(var_act_in, var_0, var_88);
                                var_96 = wp::load(var_94);
                                var_95 = wp::copy(var_96);
                                // slew_s = dynprm[7]                                             <L 825>
                                var_98 = wp::extract(var_44, var_97);
                                // slew = slew_s * opt_timestep[worldid % opt_timestep.shape[0]]       <L 826>
                                var_99 = &(var_opt_timestep.shape);
                                var_102 = wp::load(var_99);
                                var_101 = wp::extract(var_102, var_100);
                                var_103 = wp::mod(var_0, var_101);
                                var_104 = wp::address(var_opt_timestep, var_103);
                                var_106 = wp::load(var_104);
                                var_105 = wp::mul(var_98, var_106);
                                // u_eff = wp.clamp(ctrl, u_prev - slew, u_prev + slew)           <L 827>
                                var_107 = wp::sub(var_95, var_105);
                                var_108 = wp::add(var_95, var_105);
                                var_109 = wp::clamp(var_22, var_107, var_108);
                                // act_dot = (u_eff - u_prev) / opt_timestep[worldid % opt_timestep.shape[0]]       <L 828>
                                var_110 = wp::sub(var_109, var_95);
                                var_111 = &(var_opt_timestep.shape);
                                var_114 = wp::load(var_111);
                                var_113 = wp::extract(var_114, var_112);
                                var_115 = wp::mod(var_0, var_113);
                                var_116 = wp::address(var_opt_timestep, var_115);
                                var_118 = wp::load(var_116);
                                var_117 = wp::div(var_110, var_118);
                                // act_dot_out[worldid, adr] = act_dot                            <L 829>
                                wp::array_store(var_act_dot_out, var_0, var_88, var_117);
                                // ctrl = u_eff                                                   <L 830>
                                var_119 = wp::copy(var_109);
                                // adr += 1                                                       <L 831>
                                var_121 = wp::add(var_88, var_120);
                            }
                            var_122 = wp::where(var_93, var_119, var_22);
                            var_123 = wp::where(var_93, var_121, var_88);
                            var_124 = wp::where(var_93, var_117, var_89);
                            // x_I = 0.0                                                          <L 834>
                            // if slots[1] >= 0:                                                  <L 835>
                            var_127 = wp::extract(var_87, var_126);
                            var_129 = (var_127 >= var_128);
                            if (var_129) {
                                // x_I = act_in[worldid, adr]                                     <L 836>
                                var_130 = wp::address(var_act_in, var_0, var_123);
                                var_132 = wp::load(var_130);
                                var_131 = wp::copy(var_132);
                                // input_mode = int(gainprm[8])                                   <L 837>
                                var_134 = wp::extract(var_85, var_133);
                                var_135 = wp::int(var_134);
                                // Imax = dynprm[8]                                               <L 838>
                                var_137 = wp::extract(var_44, var_136);
                                // act_dot = ctrl                                                 <L 839>
                                var_138 = wp::copy(var_122);
                                // if input_mode == 1:                                            <L 840>
                                var_140 = (var_135 == var_139);
                                if (var_140) {
                                    // act_dot = ctrl - actuator_length_in[worldid, uid]          <L 841>
                                    var_141 = wp::address(var_actuator_length_in, var_0, var_1);
                                    var_143 = wp::load(var_141);
                                    var_142 = wp::sub(var_122, var_143);
                                }
                                var_144 = wp::where(var_140, var_142, var_138);
                                // if Imax > 0.0:                                                 <L 843>
                                var_146 = (var_137 > var_145);
                                if (var_146) {
                                    // if x_I >= Imax:                                            <L 844>
                                    var_147 = (var_131 >= var_137);
                                    if (var_147) {
                                        // act_dot = wp.min(act_dot, 0.0)                         <L 845>
                                        var_149 = wp::min(var_144, var_148);
                                    }
                                    if (!var_147) {
                                        // elif x_I <= -Imax:                                     <L 846>
                                        var_150 = wp::neg(var_137);
                                        var_151 = (var_131 <= var_150);
                                        if (var_151) {
                                            // act_dot = wp.max(act_dot, 0.0)                     <L 847>
                                            var_153 = wp::max(var_144, var_152);
                                        }
                                        var_154 = wp::where(var_151, var_153, var_144);
                                    }
                                    var_155 = wp::where(var_147, var_149, var_154);
                                }
                                var_156 = wp::where(var_146, var_155, var_144);
                                // act_dot_out[worldid, adr] = act_dot                            <L 849>
                                wp::array_store(var_act_dot_out, var_0, var_123, var_156);
                                // adr += 1                                                       <L 850>
                                var_158 = wp::add(var_123, var_157);
                            }
                            var_159 = wp::where(var_129, var_158, var_123);
                            var_160 = wp::where(var_129, var_156, var_124);
                            var_161 = wp::where(var_129, var_131, var_125);
                            // V = util_misc.dcmotor_voltage(                                     <L 853>
                            // ctrl,                                                              <L 854>
                            // actuator_length_in[worldid, uid],                                  <L 855>
                            var_162 = wp::address(var_actuator_length_in, var_0, var_1);
                            // actuator_velocity_in[worldid, uid],                                <L 856>
                            var_163 = wp::address(var_actuator_velocity_in, var_0, var_1);
                            // x_I,                                                               <L 857>
                            // gainprm,                                                           <L 858>
                            var_165 = wp::load(var_162);
                            var_166 = wp::load(var_163);
                            var_164 = dcmotor_voltage_0(var_122, var_165, var_166, var_161, var_85);
                            // R = gainprm[0]                                                     <L 862>
                            var_168 = wp::extract(var_85, var_167);
                            // K = gainprm[1]                                                     <L 863>
                            var_170 = wp::extract(var_85, var_169);
                            // te = wp.max(MJ_MINVAL, dynprm[0])                                  <L 864>
                            var_173 = wp::extract(var_44, var_172);
                            var_174 = wp::max(var_171, var_173);
                            // if slots[2] >= 0:                                                  <L 866>
                            var_176 = wp::extract(var_87, var_175);
                            var_178 = (var_176 >= var_177);
                            if (var_178) {
                                // RT = dynprm[2]                                                 <L 867>
                                var_180 = wp::extract(var_44, var_179);
                                // C = dynprm[3]                                                  <L 868>
                                var_182 = wp::extract(var_44, var_181);
                                // Ta = dynprm[4]                                                 <L 869>
                                var_184 = wp::extract(var_44, var_183);
                                // alpha = gainprm[2]                                             <L 870>
                                var_186 = wp::extract(var_85, var_185);
                                // T0 = gainprm[3]                                                <L 871>
                                var_188 = wp::extract(var_85, var_187);
                                // T = act_in[worldid, adr]                                       <L 872>
                                var_189 = wp::address(var_act_in, var_0, var_159);
                                var_191 = wp::load(var_189);
                                var_190 = wp::copy(var_191);
                                // R_eff = R * (1.0 + alpha * (T + Ta - T0))                      <L 873>
                                var_193 = wp::add(var_190, var_184);
                                var_194 = wp::sub(var_193, var_188);
                                var_195 = wp::mul(var_186, var_194);
                                var_196 = wp::add(var_192, var_195);
                                var_197 = wp::mul(var_168, var_196);
                                // current = (V - K * actuator_velocity_in[worldid, uid]) / R_eff       <L 875>
                                var_198 = wp::address(var_actuator_velocity_in, var_0, var_1);
                                var_200 = wp::load(var_198);
                                var_199 = wp::mul(var_170, var_200);
                                var_201 = wp::sub(var_164, var_199);
                                var_202 = wp::div(var_201, var_197);
                                // if slots[4] >= 0:                                              <L 876>
                                var_204 = wp::extract(var_87, var_203);
                                var_206 = (var_204 >= var_205);
                                if (var_206) {
                                    // current = act_in[worldid, act_last]                        <L 877>
                                    var_207 = wp::address(var_act_in, var_0, var_34);
                                    var_209 = wp::load(var_207);
                                    var_208 = wp::copy(var_209);
                                }
                                var_210 = wp::where(var_206, var_208, var_202);
                                // act_dot = (R_eff * current * current - T / RT) / C             <L 879>
                                var_211 = wp::mul(var_197, var_210);
                                var_212 = wp::mul(var_211, var_210);
                                var_213 = wp::div(var_190, var_180);
                                var_214 = wp::sub(var_212, var_213);
                                var_215 = wp::div(var_214, var_182);
                                // act_dot_out[worldid, adr] = act_dot                            <L 880>
                                wp::array_store(var_act_dot_out, var_0, var_159, var_215);
                                // adr += 1                                                       <L 881>
                                var_217 = wp::add(var_159, var_216);
                                // R = R_eff                                                      <L 882>
                                var_218 = wp::copy(var_197);
                            }
                            var_219 = wp::where(var_178, var_217, var_159);
                            var_220 = wp::where(var_178, var_215, var_160);
                            var_221 = wp::where(var_178, var_218, var_168);
                            // if slots[3] >= 0:                                                  <L 885>
                            var_223 = wp::extract(var_87, var_222);
                            var_225 = (var_223 >= var_224);
                            if (var_225) {
                                // sigma0 = dynprm[5]                                             <L 886>
                                var_227 = wp::extract(var_44, var_226);
                                // biasprm = actuator_biasprm[worldid % actuator_biasprm.shape[0], uid]       <L 887>
                                var_228 = &(var_actuator_biasprm.shape);
                                var_231 = wp::load(var_228);
                                var_230 = wp::extract(var_231, var_229);
                                var_232 = wp::mod(var_0, var_230);
                                var_233 = wp::address(var_actuator_biasprm, var_232, var_1);
                                var_235 = wp::load(var_233);
                                var_234 = wp::copy(var_235);
                                // F_C = biasprm[3]                                               <L 888>
                                var_237 = wp::extract(var_234, var_236);
                                // F_S = biasprm[4]                                               <L 889>
                                var_239 = wp::extract(var_234, var_238);
                                // v_S = biasprm[5]                                               <L 890>
                                var_241 = wp::extract(var_234, var_240);
                                // z = act_in[worldid, adr]                                       <L 891>
                                var_242 = wp::address(var_act_in, var_0, var_219);
                                var_244 = wp::load(var_242);
                                var_243 = wp::copy(var_244);
                                // g = util_misc.lugre_stribeck(actuator_velocity_in[worldid, uid], F_C, F_S, v_S)       <L 892>
                                var_245 = wp::address(var_actuator_velocity_in, var_0, var_1);
                                var_247 = wp::load(var_245);
                                var_246 = lugre_stribeck_0(var_247, var_237, var_239, var_241);
                                // a = -sigma0 * wp.abs(actuator_velocity_in[worldid, uid]) / wp.max(MJ_MINVAL, g)       <L 893>
                                var_248 = wp::neg(var_227);
                                var_249 = wp::address(var_actuator_velocity_in, var_0, var_1);
                                var_251 = wp::load(var_249);
                                var_250 = wp::abs(var_251);
                                var_252 = wp::mul(var_248, var_250);
                                var_253 = wp::max(var_171, var_246);
                                var_254 = wp::div(var_252, var_253);
                                // act_dot = a * z + actuator_velocity_in[worldid, uid]           <L 894>
                                var_255 = wp::mul(var_254, var_243);
                                var_256 = wp::address(var_actuator_velocity_in, var_0, var_1);
                                var_258 = wp::load(var_256);
                                var_257 = wp::add(var_255, var_258);
                                // act_dot_out[worldid, adr] = act_dot                            <L 895>
                                wp::array_store(var_act_dot_out, var_0, var_219, var_257);
                                // adr += 1                                                       <L 896>
                                var_260 = wp::add(var_219, var_259);
                            }
                            var_261 = wp::where(var_225, var_260, var_219);
                            var_262 = wp::where(var_225, var_257, var_220);
                            // if slots[4] >= 0:                                                  <L 899>
                            var_264 = wp::extract(var_87, var_263);
                            var_266 = (var_264 >= var_265);
                            if (var_266) {
                                // dimax = dynprm[1]                                              <L 900>
                                var_268 = wp::extract(var_44, var_267);
                                // act_dot = (V / R - K / R * actuator_velocity_in[worldid, uid] - act_in[worldid, act_last]) / te       <L 901>
                                var_269 = wp::div(var_164, var_221);
                                var_270 = wp::div(var_170, var_221);
                                var_271 = wp::address(var_actuator_velocity_in, var_0, var_1);
                                var_273 = wp::load(var_271);
                                var_272 = wp::mul(var_270, var_273);
                                var_274 = wp::sub(var_269, var_272);
                                var_275 = wp::address(var_act_in, var_0, var_34);
                                var_277 = wp::load(var_275);
                                var_276 = wp::sub(var_274, var_277);
                                var_278 = wp::div(var_276, var_174);
                                // if dimax > 0.0:                                                <L 902>
                                var_280 = (var_268 > var_279);
                                if (var_280) {
                                    // act_dot = wp.clamp(act_dot, -dimax, dimax)                 <L 903>
                                    var_281 = wp::neg(var_268);
                                    var_282 = wp::clamp(var_278, var_281, var_268);
                                }
                                var_283 = wp::where(var_280, var_282, var_278);
                                // act_dot_out[worldid, act_last] = act_dot                       <L 904>
                                wp::array_store(var_act_dot_out, var_0, var_34, var_283);
                            }
                            var_284 = wp::where(var_266, var_283, var_262);
                        }
                        if (!var_78) {
                            // elif dyntype == DynType.USER:                                      <L 906>
                            var_286 = (var_36 == var_285);
                            if (var_286) {
                                // act_dot = 0.0  # set by act_dyn_callback                       <L 907>
                            }
                            if (!var_286) {
                                // act_dot = 0.0                                                  <L 909>
                            }
                            var_289 = wp::where(var_286, var_287, var_288);
                        }
                        var_290 = wp::where(var_78, var_122, var_22);
                        var_291 = wp::where(var_78, var_284, var_289);
                    }
                    var_292 = wp::where(var_64, var_22, var_290);
                    var_293 = wp::where(var_64, var_71, var_44);
                    var_294 = wp::where(var_64, var_76, var_291);
                }
                var_295 = wp::where(var_49, var_22, var_292);
                var_296 = wp::where(var_49, var_44, var_293);
                var_297 = wp::where(var_49, var_55, var_74);
                var_298 = wp::where(var_49, var_60, var_171);
                var_299 = wp::where(var_49, var_62, var_294);
            }
            var_300 = wp::where(var_47, var_22, var_295);
            var_301 = wp::where(var_47, var_44, var_296);
            var_302 = wp::where(var_47, var_48, var_299);
            // act_dot_out[worldid, act_last] = act_dot                                           <L 911>
            wp::array_store(var_act_dot_out, var_0, var_34, var_302);
            // if actuator_actearly[uid]:                                                         <L 913>
            var_303 = wp::address(var_actuator_actearly, var_1);
            var_304 = wp::load(var_303);
            if (var_304) {
                // if dyntype == DynType.INTEGRATOR or dyntype == DynType.NONE or dyntype == DynType.DCMOTOR:       <L 914>
                var_307 = (var_36 == var_306);
                var_305 = var_307;
                if (!var_305) {
                    var_309 = (var_36 == var_308);
                    var_305 = var_305 || var_309;
                }
                if (!var_305) {
                    var_311 = (var_36 == var_310);
                    var_305 = var_305 || var_311;
                }
                if (var_305) {
                    // act = act_in[worldid, act_last]                                            <L 915>
                    var_312 = wp::address(var_act_in, var_0, var_34);
                    var_314 = wp::load(var_312);
                    var_313 = wp::copy(var_314);
                }
                var_315 = wp::where(var_305, var_313, var_297);
                // if dyntype == DynType.DCMOTOR:                                                 <L 917>
                var_317 = (var_36 == var_316);
                if (var_317) {
                    // gainprm = actuator_gainprm[worldid % actuator_gainprm.shape[0], uid]       <L 918>
                    var_318 = &(var_actuator_gainprm.shape);
                    var_321 = wp::load(var_318);
                    var_320 = wp::extract(var_321, var_319);
                    var_322 = wp::mod(var_0, var_320);
                    var_323 = wp::address(var_actuator_gainprm, var_322, var_1);
                    var_325 = wp::load(var_323);
                    var_324 = wp::copy(var_325);
                    // slots = util_misc.dcmotor_slots(dynprm, gainprm)                           <L 919>
                    var_326 = dcmotor_slots_0(var_301, var_324);
                    // offset = actuator_actnum[uid] - 1                                          <L 920>
                    var_327 = wp::address(var_actuator_actnum, var_1);
                    var_330 = wp::load(var_327);
                    var_329 = wp::sub(var_330, var_328);
                    // if offset == slots[4]:  # current                                          <L 922>
                    var_332 = wp::extract(var_326, var_331);
                    var_333 = (var_329 == var_332);
                    if (var_333) {
                        // te = wp.max(MJ_MINVAL, dynprm[0])                                      <L 923>
                        var_335 = wp::extract(var_301, var_334);
                        var_336 = wp::max(var_298, var_335);
                        // ctrl_act = act + act_dot * te * (1.0 - wp.exp(-opt_timestep[worldid % opt_timestep.shape[0]] / te))       <L 924>
                        var_337 = wp::mul(var_302, var_336);
                        var_339 = &(var_opt_timestep.shape);
                        var_342 = wp::load(var_339);
                        var_341 = wp::extract(var_342, var_340);
                        var_343 = wp::mod(var_0, var_341);
                        var_344 = wp::address(var_opt_timestep, var_343);
                        var_346 = wp::load(var_344);
                        var_345 = wp::neg(var_346);
                        var_347 = wp::div(var_345, var_336);
                        var_348 = wp::exp(var_347);
                        var_349 = wp::sub(var_338, var_348);
                        var_350 = wp::mul(var_337, var_349);
                        var_351 = wp::add(var_315, var_350);
                    }
                    if (!var_333) {
                        // elif offset == slots[3]:  # bristle                                    <L 925>
                        var_353 = wp::extract(var_326, var_352);
                        var_354 = (var_329 == var_353);
                        if (var_354) {
                            // sigma0 = dynprm[5]                                                 <L 926>
                            var_356 = wp::extract(var_301, var_355);
                            // biasprm = actuator_biasprm[worldid % actuator_biasprm.shape[0], uid]       <L 927>
                            var_357 = &(var_actuator_biasprm.shape);
                            var_360 = wp::load(var_357);
                            var_359 = wp::extract(var_360, var_358);
                            var_361 = wp::mod(var_0, var_359);
                            var_362 = wp::address(var_actuator_biasprm, var_361, var_1);
                            var_364 = wp::load(var_362);
                            var_363 = wp::copy(var_364);
                            // F_C = biasprm[3]                                                   <L 928>
                            var_366 = wp::extract(var_363, var_365);
                            // F_S = biasprm[4]                                                   <L 929>
                            var_368 = wp::extract(var_363, var_367);
                            // v_S = biasprm[5]                                                   <L 930>
                            var_370 = wp::extract(var_363, var_369);
                            // velocity = actuator_velocity_in[worldid, uid]                      <L 931>
                            var_371 = wp::address(var_actuator_velocity_in, var_0, var_1);
                            var_373 = wp::load(var_371);
                            var_372 = wp::copy(var_373);
                            // g = util_misc.lugre_stribeck(velocity, F_C, F_S, v_S)              <L 932>
                            var_374 = lugre_stribeck_0(var_372, var_366, var_368, var_370);
                            // a = -sigma0 * wp.abs(velocity) / wp.max(MJ_MINVAL, g)              <L 933>
                            var_375 = wp::neg(var_356);
                            var_376 = wp::abs(var_372);
                            var_377 = wp::mul(var_375, var_376);
                            var_378 = wp::max(var_298, var_374);
                            var_379 = wp::div(var_377, var_378);
                            // h = opt_timestep[worldid % opt_timestep.shape[0]]                  <L 934>
                            var_380 = &(var_opt_timestep.shape);
                            var_383 = wp::load(var_380);
                            var_382 = wp::extract(var_383, var_381);
                            var_384 = wp::mod(var_0, var_382);
                            var_385 = wp::address(var_opt_timestep, var_384);
                            var_387 = wp::load(var_385);
                            var_386 = wp::copy(var_387);
                            // exp_ah = wp.exp(a * h)                                             <L 935>
                            var_388 = wp::mul(var_379, var_386);
                            var_389 = wp::exp(var_388);
                            // int_h = h                                                          <L 936>
                            var_390 = wp::copy(var_386);
                            // if wp.abs(a) > MJ_MINVAL:                                          <L 937>
                            var_391 = wp::abs(var_379);
                            var_392 = (var_391 > var_298);
                            if (var_392) {
                                // int_h = (exp_ah - 1.0) / a                                     <L 938>
                                var_394 = wp::sub(var_389, var_393);
                                var_395 = wp::div(var_394, var_379);
                            }
                            var_396 = wp::where(var_392, var_395, var_390);
                            // ctrl_act = exp_ah * act + int_h * velocity                         <L 939>
                            var_397 = wp::mul(var_389, var_315);
                            var_398 = wp::mul(var_396, var_372);
                            var_399 = wp::add(var_397, var_398);
                        }
                        if (!var_354) {
                            // elif offset == slots[1]:  # integral                               <L 940>
                            var_401 = wp::extract(var_326, var_400);
                            var_402 = (var_329 == var_401);
                            if (var_402) {
                                // ctrl_act = act + act_dot * opt_timestep[worldid % opt_timestep.shape[0]]       <L 941>
                                var_403 = &(var_opt_timestep.shape);
                                var_406 = wp::load(var_403);
                                var_405 = wp::extract(var_406, var_404);
                                var_407 = wp::mod(var_0, var_405);
                                var_408 = wp::address(var_opt_timestep, var_407);
                                var_410 = wp::load(var_408);
                                var_409 = wp::mul(var_302, var_410);
                                var_411 = wp::add(var_315, var_409);
                                // Imax = dynprm[8]                                               <L 942>
                                var_413 = wp::extract(var_301, var_412);
                                // if Imax > 0.0:                                                 <L 943>
                                var_415 = (var_413 > var_414);
                                if (var_415) {
                                    // ctrl_act = wp.clamp(ctrl_act, -Imax, Imax)                 <L 944>
                                    var_416 = wp::neg(var_413);
                                    var_417 = wp::clamp(var_411, var_416, var_413);
                                }
                                var_418 = wp::where(var_415, var_417, var_411);
                            }
                            if (!var_402) {
                                // ctrl_act = act + act_dot * opt_timestep[worldid % opt_timestep.shape[0]]       <L 946>
                                var_419 = &(var_opt_timestep.shape);
                                var_422 = wp::load(var_419);
                                var_421 = wp::extract(var_422, var_420);
                                var_423 = wp::mod(var_0, var_421);
                                var_424 = wp::address(var_opt_timestep, var_423);
                                var_426 = wp::load(var_424);
                                var_425 = wp::mul(var_302, var_426);
                                var_427 = wp::add(var_315, var_425);
                            }
                            var_428 = wp::where(var_402, var_418, var_427);
                            var_429 = wp::where(var_402, var_413, var_137);
                        }
                        var_430 = wp::where(var_354, var_399, var_428);
                        var_431 = wp::where(var_354, var_137, var_429);
                        var_432 = wp::where(var_354, var_356, var_227);
                        var_433 = wp::where(var_354, var_363, var_234);
                        var_434 = wp::where(var_354, var_366, var_237);
                        var_435 = wp::where(var_354, var_368, var_239);
                        var_436 = wp::where(var_354, var_370, var_241);
                        var_437 = wp::where(var_354, var_374, var_246);
                        var_438 = wp::where(var_354, var_379, var_254);
                    }
                    var_439 = wp::where(var_333, var_351, var_430);
                    var_440 = wp::where(var_333, var_137, var_431);
                    var_441 = wp::where(var_333, var_336, var_174);
                    var_442 = wp::where(var_333, var_227, var_432);
                    var_443 = wp::where(var_333, var_234, var_433);
                    var_444 = wp::where(var_333, var_237, var_434);
                    var_445 = wp::where(var_333, var_239, var_435);
                    var_446 = wp::where(var_333, var_241, var_436);
                    var_447 = wp::where(var_333, var_246, var_437);
                    var_448 = wp::where(var_333, var_254, var_438);
                    // if actuator_actlimited[uid]:                                               <L 948>
                    var_449 = wp::address(var_actuator_actlimited, var_1);
                    var_450 = wp::load(var_449);
                    if (var_450) {
                        // actrange = actuator_actrange[worldid % actuator_actrange.shape[0], uid]       <L 949>
                        var_451 = &(var_actuator_actrange.shape);
                        var_454 = wp::load(var_451);
                        var_453 = wp::extract(var_454, var_452);
                        var_455 = wp::mod(var_0, var_453);
                        var_456 = wp::address(var_actuator_actrange, var_455, var_1);
                        var_458 = wp::load(var_456);
                        var_457 = wp::copy(var_458);
                        // ctrl_act = wp.clamp(ctrl_act, actrange[0], actrange[1])                <L 950>
                        var_460 = wp::extract(var_457, var_459);
                        var_462 = wp::extract(var_457, var_461);
                        var_463 = wp::clamp(var_439, var_460, var_462);
                    }
                    var_464 = wp::load(var_449);
                    var_466 = wp::load(var_449);
                    var_465 = wp::where(var_466, var_463, var_439);
                }
                if (!var_317) {
                    // ctrl_act = next_act(                                                       <L 952>
                    // opt_timestep[worldid % opt_timestep.shape[0]],                             <L 953>
                    var_467 = &(var_opt_timestep.shape);
                    var_470 = wp::load(var_467);
                    var_469 = wp::extract(var_470, var_468);
                    var_471 = wp::mod(var_0, var_469);
                    var_472 = wp::address(var_opt_timestep, var_471);
                    // dyntype,                                                                   <L 954>
                    // dynprm,                                                                    <L 955>
                    // actuator_actrange[worldid % actuator_actrange.shape[0], uid],              <L 956>
                    var_473 = &(var_actuator_actrange.shape);
                    var_476 = wp::load(var_473);
                    var_475 = wp::extract(var_476, var_474);
                    var_477 = wp::mod(var_0, var_475);
                    var_478 = wp::address(var_actuator_actrange, var_477, var_1);
                    // act,                                                                       <L 957>
                    // act_dot,                                                                   <L 958>
                    // 1.0,                                                                       <L 959>
                    // actuator_actlimited[uid],                                                  <L 960>
                    var_480 = wp::address(var_actuator_actlimited, var_1);
                    var_482 = wp::load(var_472);
                    var_483 = wp::load(var_478);
                    var_484 = wp::load(var_480);
                    var_481 = next_act_0(var_482, var_36, var_301, var_483, var_315, var_302, var_479, var_484);
                }
                var_485 = wp::where(var_317, var_465, var_481);
                var_486 = wp::where(var_317, var_324, var_85);
                var_487 = wp::where(var_317, var_326, var_87);
                var_488 = wp::where(var_317, var_440, var_137);
                var_489 = wp::where(var_317, var_441, var_174);
                var_490 = wp::where(var_317, var_442, var_227);
                var_491 = wp::where(var_317, var_443, var_234);
                var_492 = wp::where(var_317, var_444, var_237);
                var_493 = wp::where(var_317, var_445, var_239);
                var_494 = wp::where(var_317, var_446, var_241);
                var_495 = wp::where(var_317, var_447, var_246);
                var_496 = wp::where(var_317, var_448, var_254);
            }
            var_497 = wp::load(var_303);
            var_498 = wp::load(var_303);
            if (!var_498) {
                // ctrl_act = act_in[worldid, act_last]                                           <L 963>
                var_499 = wp::address(var_act_in, var_0, var_34);
                var_501 = wp::load(var_499);
                var_500 = wp::copy(var_501);
            }
            var_502 = wp::load(var_303);
            var_504 = wp::load(var_303);
            var_503 = wp::where(var_504, var_485, var_500);
            var_506 = wp::load(var_303);
            var_505 = wp::where(var_506, var_315, var_297);
            var_508 = wp::load(var_303);
            var_507 = wp::where(var_508, var_486, var_85);
            var_510 = wp::load(var_303);
            var_509 = wp::where(var_510, var_487, var_87);
            var_512 = wp::load(var_303);
            var_511 = wp::where(var_512, var_488, var_137);
            var_514 = wp::load(var_303);
            var_513 = wp::where(var_514, var_489, var_174);
            var_516 = wp::load(var_303);
            var_515 = wp::where(var_516, var_490, var_227);
            var_518 = wp::load(var_303);
            var_517 = wp::where(var_518, var_491, var_234);
            var_520 = wp::load(var_303);
            var_519 = wp::where(var_520, var_492, var_237);
            var_522 = wp::load(var_303);
            var_521 = wp::where(var_522, var_493, var_239);
            var_524 = wp::load(var_303);
            var_523 = wp::where(var_524, var_494, var_241);
            var_526 = wp::load(var_303);
            var_525 = wp::where(var_526, var_495, var_246);
            var_528 = wp::load(var_303);
            var_527 = wp::where(var_528, var_496, var_254);
        }
        var_529 = wp::where(var_27, var_300, var_22);
        var_530 = wp::where(var_27, var_503, var_23);
        // length = actuator_length_in[worldid, uid]                                              <L 965>
        var_531 = wp::address(var_actuator_length_in, var_0, var_1);
        var_533 = wp::load(var_531);
        var_532 = wp::copy(var_533);
        // velocity = actuator_velocity_in[worldid, uid]                                          <L 966>
        var_534 = wp::address(var_actuator_velocity_in, var_0, var_1);
        var_536 = wp::load(var_534);
        var_535 = wp::copy(var_536);
        // gaintype = actuator_gaintype[uid]                                                      <L 969>
        var_537 = wp::address(var_actuator_gaintype, var_1);
        var_539 = wp::load(var_537);
        var_538 = wp::copy(var_539);
        // gainprm = actuator_gainprm[worldid % actuator_gainprm.shape[0], uid]                   <L 970>
        var_540 = &(var_actuator_gainprm.shape);
        var_543 = wp::load(var_540);
        var_542 = wp::extract(var_543, var_541);
        var_544 = wp::mod(var_0, var_542);
        var_545 = wp::address(var_actuator_gainprm, var_544, var_1);
        var_547 = wp::load(var_545);
        var_546 = wp::copy(var_547);
        // gain = 0.0                                                                             <L 972>
        // if gaintype == GainType.FIXED:                                                         <L 973>
        var_550 = (var_538 == var_549);
        if (var_550) {
            // gain = gainprm[0]                                                                  <L 974>
            var_552 = wp::extract(var_546, var_551);
        }
        if (!var_550) {
            // elif gaintype == GainType.AFFINE:                                                  <L 975>
            var_554 = (var_538 == var_553);
            if (var_554) {
                // gain = gainprm[0] + gainprm[1] * length + gainprm[2] * velocity                <L 976>
                var_556 = wp::extract(var_546, var_555);
                var_558 = wp::extract(var_546, var_557);
                var_559 = wp::mul(var_558, var_532);
                var_560 = wp::add(var_556, var_559);
                var_562 = wp::extract(var_546, var_561);
                var_563 = wp::mul(var_562, var_535);
                var_564 = wp::add(var_560, var_563);
            }
            if (!var_554) {
                // elif gaintype == GainType.MUSCLE:                                              <L 977>
                var_566 = (var_538 == var_565);
                if (var_566) {
                    // acc0 = actuator_acc0[worldid % actuator_acc0.shape[0], uid]                <L 978>
                    var_567 = &(var_actuator_acc0.shape);
                    var_570 = wp::load(var_567);
                    var_569 = wp::extract(var_570, var_568);
                    var_571 = wp::mod(var_0, var_569);
                    var_572 = wp::address(var_actuator_acc0, var_571, var_1);
                    var_574 = wp::load(var_572);
                    var_573 = wp::copy(var_574);
                    // lengthrange = actuator_lengthrange[worldid % actuator_lengthrange.shape[0], uid]       <L 979>
                    var_575 = &(var_actuator_lengthrange.shape);
                    var_578 = wp::load(var_575);
                    var_577 = wp::extract(var_578, var_576);
                    var_579 = wp::mod(var_0, var_577);
                    var_580 = wp::address(var_actuator_lengthrange, var_579, var_1);
                    var_582 = wp::load(var_580);
                    var_581 = wp::copy(var_582);
                    // gain = util_misc.muscle_gain(length, velocity, lengthrange, acc0, gainprm)       <L 980>
                    var_583 = muscle_gain_0(var_532, var_535, var_581, var_573, var_546);
                }
                if (!var_566) {
                    // elif gaintype == GainType.DCMOTOR:                                         <L 981>
                    var_585 = (var_538 == var_584);
                    if (var_585) {
                        // R = gainprm[0]                                                         <L 982>
                        var_587 = wp::extract(var_546, var_586);
                        // K = gainprm[1]                                                         <L 983>
                        var_589 = wp::extract(var_546, var_588);
                        // te = dynprm[0]                                                         <L 984>
                        var_591 = wp::extract(var_301, var_590);
                        // slots = util_misc.dcmotor_slots(dynprm, gainprm)                       <L 986>
                        var_592 = dcmotor_slots_0(var_301, var_546);
                        // adr = act_first                                                        <L 987>
                        var_593 = wp::copy(var_25);
                        // if slots[2] >= 0:                                                      <L 989>
                        var_595 = wp::extract(var_592, var_594);
                        var_597 = (var_595 >= var_596);
                        if (var_597) {
                            // T = act_in[worldid, adr + slots[2]]                                <L 990>
                            var_599 = wp::extract(var_592, var_598);
                            var_600 = wp::add(var_593, var_599);
                            var_601 = wp::address(var_act_in, var_0, var_600);
                            var_603 = wp::load(var_601);
                            var_602 = wp::copy(var_603);
                            // alpha = gainprm[2]                                                 <L 991>
                            var_605 = wp::extract(var_546, var_604);
                            // T0 = gainprm[3]                                                    <L 992>
                            var_607 = wp::extract(var_546, var_606);
                            // Ta = dynprm[4]                                                     <L 993>
                            var_609 = wp::extract(var_301, var_608);
                            // R *= 1.0 + alpha * (T + Ta - T0)                                   <L 994>
                            var_611 = wp::add(var_602, var_609);
                            var_612 = wp::sub(var_611, var_607);
                            var_613 = wp::mul(var_605, var_612);
                            var_614 = wp::add(var_610, var_613);
                            var_615 = wp::mul(var_587, var_614);
                        }
                        var_616 = wp::where(var_597, var_615, var_587);
                        var_617 = wp::where(var_597, var_609, var_184);
                        var_618 = wp::where(var_597, var_605, var_186);
                        var_619 = wp::where(var_597, var_607, var_188);
                        var_620 = wp::where(var_597, var_602, var_190);
                        // gain = K if te > 0.0 else K / wp.max(MJ_MINVAL, R)                     <L 996>
                        var_622 = (var_591 > var_621);
                        if (var_622) {
                        }
                        if (!var_622) {
                            var_623 = wp::max(var_298, var_616);
                            var_624 = wp::div(var_589, var_623);
                        }
                        var_625 = wp::where(var_622, var_589, var_624);
                        // if te <= 0.0:                                                          <L 998>
                        var_627 = (var_591 <= var_626);
                        if (var_627) {
                            // input_mode = int(gainprm[8])                                       <L 999>
                            var_629 = wp::extract(var_546, var_628);
                            var_630 = wp::int(var_629);
                            // if input_mode > 0:                                                 <L 1000>
                            var_632 = (var_630 > var_631);
                            if (var_632) {
                                // x_I = 0.0                                                      <L 1001>
                                // if slots[1] >= 0:                                              <L 1002>
                                var_635 = wp::extract(var_592, var_634);
                                var_637 = (var_635 >= var_636);
                                if (var_637) {
                                    // x_I = act_in[worldid, adr + slots[1]]                      <L 1003>
                                    var_639 = wp::extract(var_592, var_638);
                                    var_640 = wp::add(var_593, var_639);
                                    var_641 = wp::address(var_act_in, var_0, var_640);
                                    var_643 = wp::load(var_641);
                                    var_642 = wp::copy(var_643);
                                }
                                var_644 = wp::where(var_637, var_642, var_633);
                                // ctrl_act = util_misc.dcmotor_voltage(ctrl, length, velocity, x_I, gainprm)       <L 1004>
                                var_645 = dcmotor_voltage_0(var_529, var_532, var_535, var_644, var_546);
                            }
                            if (!var_632) {
                                // ctrl_act = ctrl                                                <L 1006>
                                var_646 = wp::copy(var_529);
                            }
                            var_647 = wp::where(var_632, var_645, var_646);
                            var_648 = wp::where(var_632, var_644, var_161);
                        }
                        var_649 = wp::where(var_627, var_647, var_530);
                        var_650 = wp::where(var_627, var_648, var_161);
                        var_651 = wp::where(var_627, var_630, var_135);
                    }
                    var_652 = wp::where(var_585, var_649, var_530);
                    var_653 = wp::where(var_585, var_592, var_509);
                    var_654 = wp::where(var_585, var_593, var_261);
                    var_655 = wp::where(var_585, var_650, var_161);
                    var_656 = wp::where(var_585, var_651, var_135);
                    var_657 = wp::where(var_585, var_616, var_221);
                    var_658 = wp::where(var_585, var_589, var_170);
                    var_659 = wp::where(var_585, var_591, var_513);
                    var_660 = wp::where(var_585, var_617, var_184);
                    var_661 = wp::where(var_585, var_618, var_186);
                    var_662 = wp::where(var_585, var_619, var_188);
                    var_663 = wp::where(var_585, var_620, var_190);
                    var_664 = wp::where(var_585, var_625, var_548);
                }
                var_665 = wp::where(var_566, var_530, var_652);
                var_666 = wp::where(var_566, var_509, var_653);
                var_667 = wp::where(var_566, var_261, var_654);
                var_668 = wp::where(var_566, var_161, var_655);
                var_669 = wp::where(var_566, var_135, var_656);
                var_670 = wp::where(var_566, var_221, var_657);
                var_671 = wp::where(var_566, var_170, var_658);
                var_672 = wp::where(var_566, var_513, var_659);
                var_673 = wp::where(var_566, var_184, var_660);
                var_674 = wp::where(var_566, var_186, var_661);
                var_675 = wp::where(var_566, var_188, var_662);
                var_676 = wp::where(var_566, var_190, var_663);
                var_677 = wp::where(var_566, var_583, var_664);
            }
            var_678 = wp::where(var_554, var_530, var_665);
            var_679 = wp::where(var_554, var_509, var_666);
            var_680 = wp::where(var_554, var_261, var_667);
            var_681 = wp::where(var_554, var_161, var_668);
            var_682 = wp::where(var_554, var_135, var_669);
            var_683 = wp::where(var_554, var_221, var_670);
            var_684 = wp::where(var_554, var_170, var_671);
            var_685 = wp::where(var_554, var_513, var_672);
            var_686 = wp::where(var_554, var_184, var_673);
            var_687 = wp::where(var_554, var_186, var_674);
            var_688 = wp::where(var_554, var_188, var_675);
            var_689 = wp::where(var_554, var_190, var_676);
            var_690 = wp::where(var_554, var_564, var_677);
        }
        var_691 = wp::where(var_550, var_530, var_678);
        var_692 = wp::where(var_550, var_509, var_679);
        var_693 = wp::where(var_550, var_261, var_680);
        var_694 = wp::where(var_550, var_161, var_681);
        var_695 = wp::where(var_550, var_135, var_682);
        var_696 = wp::where(var_550, var_221, var_683);
        var_697 = wp::where(var_550, var_170, var_684);
        var_698 = wp::where(var_550, var_513, var_685);
        var_699 = wp::where(var_550, var_184, var_686);
        var_700 = wp::where(var_550, var_186, var_687);
        var_701 = wp::where(var_550, var_188, var_688);
        var_702 = wp::where(var_550, var_190, var_689);
        var_703 = wp::where(var_550, var_552, var_690);
        // biastype = actuator_biastype[uid]                                                      <L 1010>
        var_704 = wp::address(var_actuator_biastype, var_1);
        var_706 = wp::load(var_704);
        var_705 = wp::copy(var_706);
        // biasprm = actuator_biasprm[worldid % actuator_biasprm.shape[0], uid]                   <L 1011>
        var_707 = &(var_actuator_biasprm.shape);
        var_710 = wp::load(var_707);
        var_709 = wp::extract(var_710, var_708);
        var_711 = wp::mod(var_0, var_709);
        var_712 = wp::address(var_actuator_biasprm, var_711, var_1);
        var_714 = wp::load(var_712);
        var_713 = wp::copy(var_714);
        // bias = 0.0  # BiasType.NONE or BiasType.USER (modified by act_bias_callback)           <L 1013>
        // if biastype == BiasType.AFFINE:                                                        <L 1014>
        var_717 = (var_705 == var_716);
        if (var_717) {
            // bias = biasprm[0] + biasprm[1] * length + biasprm[2] * velocity                    <L 1015>
            var_719 = wp::extract(var_713, var_718);
            var_721 = wp::extract(var_713, var_720);
            var_722 = wp::mul(var_721, var_532);
            var_723 = wp::add(var_719, var_722);
            var_725 = wp::extract(var_713, var_724);
            var_726 = wp::mul(var_725, var_535);
            var_727 = wp::add(var_723, var_726);
        }
        if (!var_717) {
            // elif biastype == BiasType.MUSCLE:                                                  <L 1016>
            var_729 = (var_705 == var_728);
            if (var_729) {
                // acc0 = actuator_acc0[worldid % actuator_acc0.shape[0], uid]                    <L 1017>
                var_730 = &(var_actuator_acc0.shape);
                var_733 = wp::load(var_730);
                var_732 = wp::extract(var_733, var_731);
                var_734 = wp::mod(var_0, var_732);
                var_735 = wp::address(var_actuator_acc0, var_734, var_1);
                var_737 = wp::load(var_735);
                var_736 = wp::copy(var_737);
                // lengthrange = actuator_lengthrange[worldid % actuator_lengthrange.shape[0], uid]       <L 1018>
                var_738 = &(var_actuator_lengthrange.shape);
                var_741 = wp::load(var_738);
                var_740 = wp::extract(var_741, var_739);
                var_742 = wp::mod(var_0, var_740);
                var_743 = wp::address(var_actuator_lengthrange, var_742, var_1);
                var_745 = wp::load(var_743);
                var_744 = wp::copy(var_745);
                // bias = util_misc.muscle_bias(length, lengthrange, acc0, biasprm)               <L 1019>
                var_746 = muscle_bias_0(var_532, var_744, var_736, var_713);
            }
            if (!var_729) {
                // elif biastype == BiasType.DCMOTOR:                                             <L 1020>
                var_748 = (var_705 == var_747);
                if (var_748) {
                    // if dynprm[0] <= 0.0:                                                       <L 1021>
                    var_750 = wp::extract(var_301, var_749);
                    var_752 = (var_750 <= var_751);
                    if (var_752) {
                        // K = gainprm[1]                                                         <L 1022>
                        var_754 = wp::extract(var_546, var_753);
                        // bias -= gain * K * velocity                                            <L 1023>
                        var_755 = wp::mul(var_703, var_754);
                        var_756 = wp::mul(var_755, var_535);
                        var_757 = wp::sub(var_715, var_756);
                    }
                    var_758 = wp::where(var_752, var_754, var_697);
                    var_759 = wp::where(var_752, var_757, var_715);
                }
                var_760 = wp::where(var_748, var_758, var_697);
                var_761 = wp::where(var_748, var_759, var_715);
            }
            var_762 = wp::where(var_729, var_697, var_760);
            var_763 = wp::where(var_729, var_736, var_573);
            var_764 = wp::where(var_729, var_744, var_581);
            var_765 = wp::where(var_729, var_746, var_761);
        }
        var_766 = wp::where(var_717, var_697, var_762);
        var_767 = wp::where(var_717, var_573, var_763);
        var_768 = wp::where(var_717, var_581, var_764);
        var_769 = wp::where(var_717, var_727, var_765);
        // force = gain * ctrl_act + bias                                                         <L 1025>
        var_770 = wp::mul(var_703, var_691);
        var_771 = wp::add(var_770, var_769);
        // if actuator_forcelimited[uid]:                                                         <L 1027>
        var_772 = wp::address(var_actuator_forcelimited, var_1);
        var_773 = wp::load(var_772);
        if (var_773) {
            // forcerange = actuator_forcerange[worldid % actuator_forcerange.shape[0], uid]       <L 1028>
            var_774 = &(var_actuator_forcerange.shape);
            var_777 = wp::load(var_774);
            var_776 = wp::extract(var_777, var_775);
            var_778 = wp::mod(var_0, var_776);
            var_779 = wp::address(var_actuator_forcerange, var_778, var_1);
            var_781 = wp::load(var_779);
            var_780 = wp::copy(var_781);
            // force = wp.clamp(force, forcerange[0], forcerange[1])                              <L 1029>
            var_783 = wp::extract(var_780, var_782);
            var_785 = wp::extract(var_780, var_784);
            var_786 = wp::clamp(var_771, var_783, var_785);
        }
        var_787 = wp::load(var_772);
        var_789 = wp::load(var_772);
        var_788 = wp::where(var_789, var_786, var_771);
        // if biastype == BiasType.DCMOTOR:                                                       <L 1032>
        var_791 = (var_705 == var_790);
        if (var_791) {
            // A = biasprm[0]                                                                     <L 1034>
            var_793 = wp::extract(var_713, var_792);
            // if A != 0.0:                                                                       <L 1035>
            var_795 = (var_793 != var_794);
            if (var_795) {
                // Np = biasprm[1]                                                                <L 1036>
                var_797 = wp::extract(var_713, var_796);
                // phi = biasprm[2]                                                               <L 1037>
                var_799 = wp::extract(var_713, var_798);
                // force += A * wp.sin(Np * length + phi)                                         <L 1038>
                var_800 = wp::mul(var_797, var_532);
                var_801 = wp::add(var_800, var_799);
                var_802 = wp::sin(var_801);
                var_803 = wp::mul(var_793, var_802);
                var_804 = wp::add(var_788, var_803);
            }
            var_805 = wp::where(var_795, var_804, var_788);
            // sigma0 = dynprm[5]                                                                 <L 1041>
            var_807 = wp::extract(var_301, var_806);
            // if sigma0 > 0.0:                                                                   <L 1042>
            var_809 = (var_807 > var_808);
            if (var_809) {
                // sigma1 = dynprm[6]                                                             <L 1043>
                var_811 = wp::extract(var_301, var_810);
                // slots = util_misc.dcmotor_slots(dynprm, gainprm)                               <L 1044>
                var_812 = dcmotor_slots_0(var_301, var_546);
                // adr = act_first + slots[3]  # slots[3] is bristle                              <L 1045>
                var_814 = wp::extract(var_812, var_813);
                var_815 = wp::add(var_25, var_814);
                // z = act_in[worldid, adr]                                                       <L 1046>
                var_816 = wp::address(var_act_in, var_0, var_815);
                var_818 = wp::load(var_816);
                var_817 = wp::copy(var_818);
                // z_dot = act_dot_out[worldid, adr]                                              <L 1047>
                var_819 = wp::address(var_act_dot_out, var_0, var_815);
                var_821 = wp::load(var_819);
                var_820 = wp::copy(var_821);
                // force -= sigma0 * z + sigma1 * z_dot                                           <L 1048>
                var_822 = wp::mul(var_807, var_817);
                var_823 = wp::mul(var_811, var_820);
                var_824 = wp::add(var_822, var_823);
                var_825 = wp::sub(var_805, var_824);
            }
            var_826 = wp::where(var_809, var_812, var_692);
            var_827 = wp::where(var_809, var_815, var_693);
            var_828 = wp::where(var_809, var_817, var_243);
            var_829 = wp::where(var_809, var_825, var_805);
        }
        var_830 = wp::where(var_791, var_826, var_692);
        var_831 = wp::where(var_791, var_827, var_693);
        var_832 = wp::where(var_791, var_807, var_515);
        var_833 = wp::where(var_791, var_828, var_243);
        var_834 = wp::where(var_791, var_829, var_788);
        // actuator_force_out[worldid, uid] = force                                               <L 1050>
        wp::array_store(var_actuator_force_out, var_0, var_1, var_834);
    }
}



extern "C" __global__ void _qfrc_actuator_gravcomp_limits_a91eb6a0_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<bool> var_jnt_actfrclimited,
    wp::array_t<wp::int32> var_jnt_actgravcomp,
    wp::array_t<wp::vec_t<2, wp::float32>> var_jnt_actfrcrange,
    wp::array_t<wp::int32> var_dof_jntid,
    wp::array_t<wp::float32> var_qfrc_gravcomp_in,
    wp::array_t<wp::float32> var_qfrc_actuator_in,
    bool var_gravity_enabled,
    wp::array_t<wp::float32> var_qfrc_actuator_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::float32* var_5;
        wp::float32 var_6;
        wp::float32 var_7;
        bool var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        bool* var_15;
        bool var_16;
        wp::shape_t* var_17;
        const wp::int32 var_18 = 0;
        wp::int32 var_19;
        wp::shape_t var_20;
        wp::int32 var_21;
        wp::vec_t<2, wp::float32>* var_22;
        wp::vec_t<2, wp::float32> var_23;
        wp::vec_t<2, wp::float32> var_24;
        const wp::int32 var_25 = 0;
        wp::float32 var_26;
        const wp::int32 var_27 = 1;
        wp::float32 var_28;
        wp::float32 var_29;
        bool var_30;
        wp::float32 var_31;
        bool var_32;
        //---------
        // forward
        // def _qfrc_actuator_gravcomp_limits(                                                    <L 1121>
        // worldid, dofid = wp.tid()                                                              <L 1135>
        builtin_tid2d(var_0, var_1);
        // jntid = dof_jntid[dofid]                                                               <L 1136>
        var_2 = wp::address(var_dof_jntid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // qfrc = qfrc_actuator_in[worldid, dofid]                                                <L 1138>
        var_5 = wp::address(var_qfrc_actuator_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if gravity_enabled and jnt_actgravcomp[jntid]:                                         <L 1141>
        var_8 = var_gravity_enabled;
        if (var_8) {
            var_9 = wp::address(var_jnt_actgravcomp, var_3);
            var_10 = wp::load(var_9);
            var_8 = var_8 && var_10;
        }
        if (var_8) {
            // qfrc += qfrc_gravcomp_in[worldid, dofid]                                           <L 1142>
            var_11 = wp::address(var_qfrc_gravcomp_in, var_0, var_1);
            var_13 = wp::load(var_11);
            var_12 = wp::add(var_6, var_13);
        }
        var_14 = wp::where(var_8, var_12, var_6);
        // if jnt_actfrclimited[jntid]:                                                           <L 1145>
        var_15 = wp::address(var_jnt_actfrclimited, var_3);
        var_16 = wp::load(var_15);
        if (var_16) {
            // frcrange = jnt_actfrcrange[worldid % jnt_actfrcrange.shape[0], jntid]              <L 1146>
            var_17 = &(var_jnt_actfrcrange.shape);
            var_20 = wp::load(var_17);
            var_19 = wp::extract(var_20, var_18);
            var_21 = wp::mod(var_0, var_19);
            var_22 = wp::address(var_jnt_actfrcrange, var_21, var_3);
            var_24 = wp::load(var_22);
            var_23 = wp::copy(var_24);
            // qfrc = wp.clamp(qfrc, frcrange[0], frcrange[1])                                    <L 1147>
            var_26 = wp::extract(var_23, var_25);
            var_28 = wp::extract(var_23, var_27);
            var_29 = wp::clamp(var_14, var_26, var_28);
        }
        var_30 = wp::load(var_15);
        var_32 = wp::load(var_15);
        var_31 = wp::where(var_32, var_29, var_14);
        // qfrc_actuator_out[worldid, dofid] = qfrc                                               <L 1149>
        wp::array_store(var_qfrc_actuator_out, var_0, var_1, var_31);
    }
}



extern "C" __global__ void _next_activation_becb463e_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::int32> var_actuator_dyntype,
    wp::array_t<wp::int32> var_actuator_actadr,
    wp::array_t<wp::int32> var_actuator_actnum,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_dynprm,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_gainprm,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_biasprm,
    wp::array_t<bool> var_actuator_actlimited,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_actrange,
    wp::array_t<wp::float32> var_act_in,
    wp::array_t<wp::float32> var_act_dot_in,
    wp::array_t<wp::float32> var_actuator_velocity_in,
    wp::float32 var_act_dot_scale,
    bool var_limit,
    wp::array_t<wp::float32> var_act_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::shape_t* var_7;
        const wp::int32 var_8 = 0;
        wp::int32 var_9;
        wp::shape_t var_10;
        wp::int32 var_11;
        wp::shape_t* var_12;
        const wp::int32 var_13 = 0;
        wp::int32 var_14;
        wp::shape_t var_15;
        wp::int32 var_16;
        wp::shape_t* var_17;
        const wp::int32 var_18 = 0;
        wp::int32 var_19;
        wp::shape_t var_20;
        wp::int32 var_21;
        wp::shape_t* var_22;
        const wp::int32 var_23 = 0;
        wp::int32 var_24;
        wp::shape_t var_25;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        wp::int32* var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        const wp::int32 var_36 = 5;
        bool var_37;
        wp::vec_t<10, wp::float32>* var_38;
        wp::vec_t<10, wp::float32> var_39;
        wp::vec_t<10, wp::float32> var_40;
        wp::vec_t<10, wp::float32>* var_41;
        wp::vec_t<10, wp::float32> var_42;
        wp::vec_t<10, wp::float32> var_43;
        wp::vec_t<10, wp::float32>* var_44;
        wp::vec_t<10, wp::float32> var_45;
        wp::vec_t<10, wp::float32> var_46;
        wp::vec_t<6, wp::int32> var_47;
        wp::int32 var_48;
        wp::range_t var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        wp::float32* var_52;
        wp::float32 var_53;
        wp::float32 var_54;
        wp::float32* var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        const wp::int32 var_58 = 4;
        wp::int32 var_59;
        bool var_60;
        const wp::int32 var_61 = 0;
        wp::float32 var_62;
        const wp::float32 var_63 = 1e-15;
        const wp::int32 var_64 = 0;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        const wp::float32 var_68 = 1.0;
        wp::float32* var_69;
        wp::float32 var_70;
        wp::float32 var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        wp::float32 var_75;
        wp::float32 var_76;
        const wp::int32 var_77 = 3;
        wp::int32 var_78;
        bool var_79;
        const wp::int32 var_80 = 3;
        wp::float32 var_81;
        const wp::int32 var_82 = 4;
        wp::float32 var_83;
        const wp::int32 var_84 = 5;
        wp::float32 var_85;
        const wp::int32 var_86 = 5;
        wp::float32 var_87;
        wp::float32* var_88;
        wp::float32 var_89;
        wp::float32 var_90;
        wp::float32 var_91;
        wp::float32 var_92;
        wp::float32 var_93;
        wp::float32 var_94;
        const wp::float32 var_95 = 1e-15;
        wp::float32 var_96;
        wp::float32 var_97;
        wp::float32* var_98;
        wp::float32 var_99;
        wp::float32 var_100;
        wp::float32 var_101;
        wp::float32 var_102;
        wp::float32 var_103;
        wp::float32 var_104;
        bool var_105;
        const wp::float32 var_106 = 1.0;
        wp::float32 var_107;
        wp::float32 var_108;
        wp::float32 var_109;
        wp::float32 var_110;
        wp::float32 var_111;
        wp::float32 var_112;
        const wp::int32 var_113 = 1;
        wp::int32 var_114;
        bool var_115;
        wp::float32* var_116;
        wp::float32 var_117;
        wp::float32 var_118;
        wp::float32 var_119;
        const wp::int32 var_120 = 8;
        wp::float32 var_121;
        const wp::float32 var_122 = 0.0;
        bool var_123;
        wp::float32 var_124;
        wp::float32 var_125;
        wp::float32 var_126;
        wp::float32* var_127;
        wp::float32 var_128;
        wp::float32 var_129;
        wp::float32 var_130;
        wp::float32 var_131;
        wp::float32 var_132;
        wp::float32 var_133;
        wp::float32 var_134;
        wp::int32 var_135;
        wp::range_t var_136;
        wp::int32 var_137;
        wp::float32* var_138;
        wp::vec_t<10, wp::float32>* var_139;
        wp::vec_t<2, wp::float32>* var_140;
        wp::float32* var_141;
        wp::float32* var_142;
        bool var_143;
        bool* var_144;
        bool var_145;
        wp::float32 var_146;
        wp::float32 var_147;
        wp::vec_t<10, wp::float32> var_148;
        wp::vec_t<2, wp::float32> var_149;
        wp::float32 var_150;
        wp::float32 var_151;
        wp::int32 var_152;
        wp::float32 var_153;
        //---------
        // forward
        // def _next_activation(                                                                  <L 135>
        // worldid, uid = wp.tid()                                                                <L 156>
        builtin_tid2d(var_0, var_1);
        // opt_timestep_id = worldid % opt_timestep.shape[0]                                      <L 157>
        var_2 = &(var_opt_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // actuator_dynprm_id = worldid % actuator_dynprm.shape[0]                                <L 158>
        var_7 = &(var_actuator_dynprm.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // actuator_actrange_id = worldid % actuator_actrange.shape[0]                            <L 159>
        var_12 = &(var_actuator_actrange.shape);
        var_15 = wp::load(var_12);
        var_14 = wp::extract(var_15, var_13);
        var_16 = wp::mod(var_0, var_14);
        // actuator_gainprm_id = worldid % actuator_gainprm.shape[0]                              <L 160>
        var_17 = &(var_actuator_gainprm.shape);
        var_20 = wp::load(var_17);
        var_19 = wp::extract(var_20, var_18);
        var_21 = wp::mod(var_0, var_19);
        // actuator_biasprm_id = worldid % actuator_biasprm.shape[0]                              <L 161>
        var_22 = &(var_actuator_biasprm.shape);
        var_25 = wp::load(var_22);
        var_24 = wp::extract(var_25, var_23);
        var_26 = wp::mod(var_0, var_24);
        // actadr = actuator_actadr[uid]                                                          <L 163>
        var_27 = wp::address(var_actuator_actadr, var_1);
        var_29 = wp::load(var_27);
        var_28 = wp::copy(var_29);
        // actnum = actuator_actnum[uid]                                                          <L 164>
        var_30 = wp::address(var_actuator_actnum, var_1);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // dyntype = actuator_dyntype[uid]                                                        <L 165>
        var_33 = wp::address(var_actuator_dyntype, var_1);
        var_35 = wp::load(var_33);
        var_34 = wp::copy(var_35);
        // if dyntype == DynType.DCMOTOR:                                                         <L 167>
        var_37 = (var_34 == var_36);
        if (var_37) {
            // dynprm = actuator_dynprm[actuator_dynprm_id, uid]                                  <L 168>
            var_38 = wp::address(var_actuator_dynprm, var_11, var_1);
            var_40 = wp::load(var_38);
            var_39 = wp::copy(var_40);
            // gainprm = actuator_gainprm[actuator_gainprm_id, uid]                               <L 169>
            var_41 = wp::address(var_actuator_gainprm, var_21, var_1);
            var_43 = wp::load(var_41);
            var_42 = wp::copy(var_43);
            // biasprm = actuator_biasprm[actuator_biasprm_id, uid]                               <L 170>
            var_44 = wp::address(var_actuator_biasprm, var_26, var_1);
            var_46 = wp::load(var_44);
            var_45 = wp::copy(var_46);
            // slots = util_misc.dcmotor_slots(dynprm, gainprm)                                   <L 171>
            var_47 = dcmotor_slots_0(var_39, var_42);
            // for j in range(actadr, actadr + actnum):                                           <L 173>
            var_48 = wp::add(var_28, var_31);
            var_49 = wp::range(var_28, var_48);
            start_for_0:;
                if (iter_cmp(var_49) == 0) goto end_for_0;
                var_50 = wp::iter_next(var_49);
                // offset = j - actadr                                                            <L 174>
                var_51 = wp::sub(var_50, var_28);
                // act = act_in[worldid, j]                                                       <L 175>
                var_52 = wp::address(var_act_in, var_0, var_50);
                var_54 = wp::load(var_52);
                var_53 = wp::copy(var_54);
                // act_dot = act_dot_in[worldid, j]                                               <L 176>
                var_55 = wp::address(var_act_dot_in, var_0, var_50);
                var_57 = wp::load(var_55);
                var_56 = wp::copy(var_57);
                // if offset == slots[4]:  # current                                              <L 178>
                var_59 = wp::extract(var_47, var_58);
                var_60 = (var_51 == var_59);
                if (var_60) {
                    // R = gainprm[0]                                                             <L 179>
                    var_62 = wp::extract(var_42, var_61);
                    // te = wp.max(MJ_MINVAL, dynprm[0])                                          <L 180>
                    var_65 = wp::extract(var_39, var_64);
                    var_66 = wp::max(var_63, var_65);
                    // act = act + act_dot * te * (1.0 - wp.exp(-opt_timestep[opt_timestep_id] / te))       <L 181>
                    var_67 = wp::mul(var_56, var_66);
                    var_69 = wp::address(var_opt_timestep, var_6);
                    var_71 = wp::load(var_69);
                    var_70 = wp::neg(var_71);
                    var_72 = wp::div(var_70, var_66);
                    var_73 = wp::exp(var_72);
                    var_74 = wp::sub(var_68, var_73);
                    var_75 = wp::mul(var_67, var_74);
                    var_76 = wp::add(var_53, var_75);
                }
                if (!var_60) {
                    // elif offset == slots[3]:  # bristle                                        <L 182>
                    var_78 = wp::extract(var_47, var_77);
                    var_79 = (var_51 == var_78);
                    if (var_79) {
                        // F_C = biasprm[3]                                                       <L 183>
                        var_81 = wp::extract(var_45, var_80);
                        // F_S = biasprm[4]                                                       <L 184>
                        var_83 = wp::extract(var_45, var_82);
                        // v_S = biasprm[5]                                                       <L 185>
                        var_85 = wp::extract(var_45, var_84);
                        // sigma0 = dynprm[5]                                                     <L 186>
                        var_87 = wp::extract(var_39, var_86);
                        // velocity = actuator_velocity_in[worldid, uid]                          <L 187>
                        var_88 = wp::address(var_actuator_velocity_in, var_0, var_1);
                        var_90 = wp::load(var_88);
                        var_89 = wp::copy(var_90);
                        // g = util_misc.lugre_stribeck(velocity, F_C, F_S, v_S)                  <L 188>
                        var_91 = lugre_stribeck_0(var_89, var_81, var_83, var_85);
                        // a = -sigma0 * wp.abs(velocity) / wp.max(MJ_MINVAL, g)                  <L 190>
                        var_92 = wp::neg(var_87);
                        var_93 = wp::abs(var_89);
                        var_94 = wp::mul(var_92, var_93);
                        var_96 = wp::max(var_95, var_91);
                        var_97 = wp::div(var_94, var_96);
                        // h = opt_timestep[opt_timestep_id]                                      <L 191>
                        var_98 = wp::address(var_opt_timestep, var_6);
                        var_100 = wp::load(var_98);
                        var_99 = wp::copy(var_100);
                        // exp_ah = wp.exp(a * h)                                                 <L 192>
                        var_101 = wp::mul(var_97, var_99);
                        var_102 = wp::exp(var_101);
                        // int_h = h                                                              <L 193>
                        var_103 = wp::copy(var_99);
                        // if wp.abs(a) > MJ_MINVAL:                                              <L 194>
                        var_104 = wp::abs(var_97);
                        var_105 = (var_104 > var_95);
                        if (var_105) {
                            // int_h = (exp_ah - 1.0) / a                                         <L 195>
                            var_107 = wp::sub(var_102, var_106);
                            var_108 = wp::div(var_107, var_97);
                        }
                        var_109 = wp::where(var_105, var_108, var_103);
                        // act = exp_ah * act + int_h * velocity                                  <L 196>
                        var_110 = wp::mul(var_102, var_53);
                        var_111 = wp::mul(var_109, var_89);
                        var_112 = wp::add(var_110, var_111);
                    }
                    if (!var_79) {
                        // elif offset == slots[1]:  # integral                                   <L 197>
                        var_114 = wp::extract(var_47, var_113);
                        var_115 = (var_51 == var_114);
                        if (var_115) {
                            // act = act + act_dot * opt_timestep[opt_timestep_id]                <L 198>
                            var_116 = wp::address(var_opt_timestep, var_6);
                            var_118 = wp::load(var_116);
                            var_117 = wp::mul(var_56, var_118);
                            var_119 = wp::add(var_53, var_117);
                            // Imax = dynprm[8]                                                   <L 199>
                            var_121 = wp::extract(var_39, var_120);
                            // if Imax > 0.0:                                                     <L 200>
                            var_123 = (var_121 > var_122);
                            if (var_123) {
                                // act = wp.clamp(act, -Imax, Imax)                               <L 201>
                                var_124 = wp::neg(var_121);
                                var_125 = wp::clamp(var_119, var_124, var_121);
                            }
                            var_126 = wp::where(var_123, var_125, var_119);
                        }
                        if (!var_115) {
                            // act = act + act_dot * opt_timestep[opt_timestep_id]                <L 203>
                            var_127 = wp::address(var_opt_timestep, var_6);
                            var_129 = wp::load(var_127);
                            var_128 = wp::mul(var_56, var_129);
                            var_130 = wp::add(var_53, var_128);
                        }
                        var_131 = wp::where(var_115, var_126, var_130);
                    }
                    var_132 = wp::where(var_79, var_112, var_131);
                }
                var_133 = wp::where(var_60, var_76, var_132);
                var_134 = wp::where(var_60, var_63, var_95);
                // act_out[worldid, j] = act                                                      <L 205>
                wp::array_store(var_act_out, var_0, var_50, var_133);
                goto start_for_0;
            end_for_0:;
        }
        if (!var_37) {
            // for j in range(actadr, actadr + actnum):                                           <L 207>
            var_135 = wp::add(var_28, var_31);
            var_136 = wp::range(var_28, var_135);
            start_for_2:;
                if (iter_cmp(var_136) == 0) goto end_for_2;
                var_137 = wp::iter_next(var_136);
                // act = next_act(                                                                <L 208>
                // opt_timestep[opt_timestep_id],                                                 <L 209>
                var_138 = wp::address(var_opt_timestep, var_6);
                // dyntype,                                                                       <L 210>
                // actuator_dynprm[actuator_dynprm_id, uid],                                      <L 211>
                var_139 = wp::address(var_actuator_dynprm, var_11, var_1);
                // actuator_actrange[actuator_actrange_id, uid],                                  <L 212>
                var_140 = wp::address(var_actuator_actrange, var_16, var_1);
                // act_in[worldid, j],                                                            <L 213>
                var_141 = wp::address(var_act_in, var_0, var_137);
                // act_dot_in[worldid, j],                                                        <L 214>
                var_142 = wp::address(var_act_dot_in, var_0, var_137);
                // act_dot_scale,                                                                 <L 215>
                // limit and actuator_actlimited[uid],                                            <L 216>
                var_143 = var_limit;
                if (var_143) {
                    var_144 = wp::address(var_actuator_actlimited, var_1);
                    var_145 = wp::load(var_144);
                    var_143 = var_143 && var_145;
                }
                var_147 = wp::load(var_138);
                var_148 = wp::load(var_139);
                var_149 = wp::load(var_140);
                var_150 = wp::load(var_141);
                var_151 = wp::load(var_142);
                var_146 = next_act_0(var_147, var_34, var_148, var_149, var_150, var_151, var_act_dot_scale, var_143);
                // act_out[worldid, j] = act                                                      <L 218>
                wp::array_store(var_act_out, var_0, var_137, var_146);
                goto start_for_2;
            end_for_2:;
        }
        var_152 = wp::where(var_37, var_50, var_137);
        var_153 = wp::where(var_37, var_133, var_146);
    }
}



extern "C" __global__ void _tendon_velocity_ca816a63_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_ten_velocity_out)
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
        wp::int32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::range_t var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        const wp::float32 var_16 = 0.0;
        bool var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        //---------
        // forward
        // def _tendon_velocity(                                                                  <L 706>
        // worldid, tenid = wp.tid()                                                              <L 717>
        builtin_tid2d(var_0, var_1);
        // velocity = float(0.0)                                                                  <L 719>
        var_3 = wp::float(var_2);
        // rownnz = ten_J_rownnz[tenid]                                                           <L 720>
        var_4 = wp::address(var_ten_J_rownnz, var_1);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // rowadr = ten_J_rowadr[tenid]                                                           <L 721>
        var_7 = wp::address(var_ten_J_rowadr, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // for i in range(rownnz):                                                                <L 722>
        var_10 = wp::range(var_5);
        start_for_0:;
            if (iter_cmp(var_10) == 0) goto end_for_0;
            var_11 = wp::iter_next(var_10);
            // sparseid = rowadr + i                                                              <L 723>
            var_12 = wp::add(var_8, var_11);
            // J = ten_J_in[worldid, sparseid]                                                    <L 724>
            var_13 = wp::address(var_ten_J_in, var_0, var_12);
            var_15 = wp::load(var_13);
            var_14 = wp::copy(var_15);
            // if J != 0.0:                                                                       <L 725>
            var_17 = (var_14 != var_16);
            if (var_17) {
                // colind = ten_J_colind[sparseid]                                                <L 726>
                var_18 = wp::address(var_ten_J_colind, var_12);
                var_20 = wp::load(var_18);
                var_19 = wp::copy(var_20);
                // velocity += J * qvel_in[worldid, colind]                                       <L 727>
                var_21 = wp::address(var_qvel_in, var_0, var_19);
                var_23 = wp::load(var_21);
                var_22 = wp::mul(var_14, var_23);
                var_24 = wp::add(var_3, var_22);
            }
            var_25 = wp::where(var_17, var_24, var_3);
            wp::assign(var_3, var_25);
            goto start_for_0;
        end_for_0:;
        // ten_velocity_out[worldid, tenid] = velocity                                            <L 729>
        wp::array_store(var_ten_velocity_out, var_0, var_1, var_3);
    }
}



extern "C" __global__ void _compute_damping_deriv_cfbf3e5a_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_dof_damping,
    wp::array_t<wp::vec_t<2, wp::float32>> var_dof_dampingpoly,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_deriv_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::shape_t* var_10;
        const wp::int32 var_11 = 0;
        wp::int32 var_12;
        wp::shape_t var_13;
        wp::int32 var_14;
        wp::vec_t<2, wp::float32>* var_15;
        wp::vec_t<2, wp::float32> var_16;
        wp::vec_t<2, wp::float32> var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        const wp::int32 var_21 = 1;
        wp::float32 var_22;
        //---------
        // forward
        // def _compute_damping_deriv(                                                            <L 353>
        // worldid, tid = wp.tid()                                                                <L 362>
        builtin_tid2d(var_0, var_1);
        // damping = dof_damping[worldid % dof_damping.shape[0], tid]                             <L 363>
        var_2 = &(var_dof_damping.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_dof_damping, var_6, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // dpoly = dof_dampingpoly[worldid % dof_dampingpoly.shape[0], tid]                       <L 364>
        var_10 = &(var_dof_dampingpoly.shape);
        var_13 = wp::load(var_10);
        var_12 = wp::extract(var_13, var_11);
        var_14 = wp::mod(var_0, var_12);
        var_15 = wp::address(var_dof_dampingpoly, var_14, var_1);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // v = qvel_in[worldid, tid]                                                              <L 365>
        var_18 = wp::address(var_qvel_in, var_0, var_1);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // deriv_out[worldid, tid] = util_misc._poly_force_deriv(damping, dpoly, v, 1)            <L 366>
        var_22 = _poly_force_deriv_0(var_8, var_16, var_19, var_21);
        wp::array_store(var_deriv_out, var_0, var_1, var_22);
    }
}



extern "C" __global__ void _next_velocity_d66c2f53_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_qacc_in,
    wp::float32 var_qacc_scale_in,
    wp::array_t<wp::float32> var_qvel_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::float32* var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        //---------
        // forward
        // def _next_velocity(                                                                    <L 118>
        // worldid, dofid = wp.tid()                                                              <L 129>
        builtin_tid2d(var_0, var_1);
        // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                               <L 130>
        var_2 = &(var_opt_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_opt_timestep, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // qvel_out[worldid, dofid] = qvel_in[worldid, dofid] + qacc_scale_in * qacc_in[worldid, dofid] * timestep       <L 131>
        var_10 = wp::address(var_qvel_in, var_0, var_1);
        var_11 = wp::address(var_qacc_in, var_0, var_1);
        var_13 = wp::load(var_11);
        var_12 = wp::mul(var_qacc_scale_in, var_13);
        var_14 = wp::mul(var_12, var_8);
        var_16 = wp::load(var_10);
        var_15 = wp::add(var_16, var_14);
        wp::array_store(var_qvel_out, var_0, var_1, var_15);
    }
}



extern "C" __global__ void _euler_damp_qfrc_de139d07_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::int32> var_M_rownnz,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::float32> var_damp_deriv,
    wp::array_t<wp::float32> var_M_integration_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::int32* var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 1;
        wp::int32 var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        //---------
        // forward
        // def _euler_damp_qfrc(                                                                  <L 370>
        // worldid, tid = wp.tid()                                                                <L 380>
        builtin_tid2d(var_0, var_1);
        // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                               <L 381>
        var_2 = &(var_opt_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_opt_timestep, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // adr = M_rowadr[tid] + M_rownnz[tid] - 1                                                <L 383>
        var_10 = wp::address(var_M_rowadr, var_1);
        var_11 = wp::address(var_M_rownnz, var_1);
        var_13 = wp::load(var_10);
        var_14 = wp::load(var_11);
        var_12 = wp::add(var_13, var_14);
        var_16 = wp::sub(var_12, var_15);
        // M_integration_out[worldid, adr] += timestep * damp_deriv[worldid, tid]                 <L 384>
        var_17 = wp::address(var_damp_deriv, var_0, var_1);
        var_19 = wp::load(var_17);
        var_18 = wp::mul(var_8, var_19);
        var_20 = wp::atomic_add(var_M_integration_out, var_0, var_16, var_18);
    }
}



extern "C" __global__ void _tendon_actuator_force_clamp_fd85b13d_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<bool> var_tendon_actfrclimited,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_actfrcrange,
    wp::array_t<wp::int32> var_actuator_trntype,
    wp::array_t<wp::vec_t<2, wp::int32>> var_actuator_trnid,
    wp::array_t<wp::float32> var_ten_actfrc_in,
    wp::array_t<wp::float32> var_actuator_force_out)
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
        wp::int32* var_2;
        const wp::int32 var_3 = 3;
        bool var_4;
        wp::int32 var_5;
        wp::vec_t<2, wp::int32>* var_6;
        const wp::int32 var_7 = 0;
        wp::int32 var_8;
        wp::vec_t<2, wp::int32> var_9;
        bool* var_10;
        bool var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::shape_t* var_15;
        const wp::int32 var_16 = 0;
        wp::int32 var_17;
        wp::shape_t var_18;
        wp::int32 var_19;
        wp::vec_t<2, wp::float32>* var_20;
        wp::vec_t<2, wp::float32> var_21;
        wp::vec_t<2, wp::float32> var_22;
        const wp::int32 var_23 = 0;
        wp::float32 var_24;
        bool var_25;
        const wp::int32 var_26 = 0;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::float32* var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        const wp::int32 var_32 = 1;
        wp::float32 var_33;
        bool var_34;
        const wp::int32 var_35 = 1;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32* var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        bool var_41;
        //---------
        // forward
        // def _tendon_actuator_force_clamp(                                                      <L 1072>
        // worldid, actid = wp.tid()                                                              <L 1083>
        builtin_tid2d(var_0, var_1);
        // if actuator_trntype[actid] == TrnType.TENDON:                                          <L 1085>
        var_2 = wp::address(var_actuator_trntype, var_1);
        var_5 = wp::load(var_2);
        var_4 = (var_5 == var_3);
        if (var_4) {
            // tenid = actuator_trnid[actid][0]                                                   <L 1086>
            var_6 = wp::address(var_actuator_trnid, var_1);
            var_9 = wp::load(var_6);
            var_8 = wp::extract(var_9, var_7);
            // if tendon_actfrclimited[tenid]:                                                    <L 1087>
            var_10 = wp::address(var_tendon_actfrclimited, var_8);
            var_11 = wp::load(var_10);
            if (var_11) {
                // ten_actfrc = ten_actfrc_in[worldid, tenid]                                     <L 1088>
                var_12 = wp::address(var_ten_actfrc_in, var_0, var_8);
                var_14 = wp::load(var_12);
                var_13 = wp::copy(var_14);
                // actfrcrange = tendon_actfrcrange[worldid % tendon_actfrcrange.shape[0], tenid]       <L 1089>
                var_15 = &(var_tendon_actfrcrange.shape);
                var_18 = wp::load(var_15);
                var_17 = wp::extract(var_18, var_16);
                var_19 = wp::mod(var_0, var_17);
                var_20 = wp::address(var_tendon_actfrcrange, var_19, var_8);
                var_22 = wp::load(var_20);
                var_21 = wp::copy(var_22);
                // if ten_actfrc < actfrcrange[0]:                                                <L 1091>
                var_24 = wp::extract(var_21, var_23);
                var_25 = (var_13 < var_24);
                if (var_25) {
                    // actuator_force_out[worldid, actid] *= actfrcrange[0] / ten_actfrc          <L 1092>
                    var_27 = wp::extract(var_21, var_26);
                    var_28 = wp::div(var_27, var_13);
                    var_29 = wp::address(var_actuator_force_out, var_0, var_1);
                    var_31 = wp::load(var_29);
                    var_30 = wp::mul(var_31, var_28);
                    wp::array_store(var_actuator_force_out, var_0, var_1, var_30);
                }
                if (!var_25) {
                    // elif ten_actfrc > actfrcrange[1]:                                          <L 1093>
                    var_33 = wp::extract(var_21, var_32);
                    var_34 = (var_13 > var_33);
                    if (var_34) {
                        // actuator_force_out[worldid, actid] *= actfrcrange[1] / ten_actfrc       <L 1094>
                        var_36 = wp::extract(var_21, var_35);
                        var_37 = wp::div(var_36, var_13);
                        var_38 = wp::address(var_actuator_force_out, var_0, var_1);
                        var_40 = wp::load(var_38);
                        var_39 = wp::mul(var_40, var_37);
                        wp::array_store(var_actuator_force_out, var_0, var_1, var_39);
                    }
                }
            }
            var_41 = wp::load(var_10);
        }
    }
}

