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


// /workspace/isaaclab/source/isaaclab_newton/isaaclab_newton/assets/kernels.py:176
static CUDA_CALLABLE wp::vec_t<6, wp::float32> get_link_velocity_in_com_frame_func_0(
    wp::vec_t<6, wp::float32> var_link_velocity_w,
    wp::transform_t<wp::float32> var_link_pose_w,
    wp::vec_t<3, wp::float32> var_body_com_pos_b)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::quat_t<wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<6, wp::float32> var_7;
    //---------
    // forward
    // def get_link_velocity_in_com_frame_func(                                               <L 177>
    // return wp.spatial_vector(                                                              <L 195>
    // wp.spatial_top(link_velocity_w)                                                        <L 196>
    var_0 = wp::spatial_top(var_link_velocity_w);
    // + wp.cross(                                                                            <L 197>
    // wp.spatial_bottom(link_velocity_w),                                                    <L 198>
    var_1 = wp::spatial_bottom(var_link_velocity_w);
    // wp.quat_rotate(wp.transform_get_rotation(link_pose_w), body_com_pos_b),                <L 199>
    var_2 = wp::transform_get_rotation(var_link_pose_w);
    var_3 = wp::quat_rotate(var_2, var_body_com_pos_b);
    var_4 = wp::cross(var_1, var_3);
    var_5 = wp::add(var_0, var_4);
    // wp.spatial_bottom(link_velocity_w),                                                    <L 201>
    var_6 = wp::spatial_bottom(var_link_velocity_w);
    var_7 = wp::vec_t<6, wp::float32>(var_5, var_6);
    return var_7;
}


// /workspace/isaaclab/source/isaaclab_newton/isaaclab_newton/assets/kernels.py:103
static CUDA_CALLABLE wp::float32 compute_heading_w_func_0(
    wp::vec_t<3, wp::float32> var_forward_vec,
    wp::quat_t<wp::float32> var_quat)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    const wp::int32 var_1 = 1;
    wp::float32 var_2;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    wp::float32 var_5;
    //---------
    // forward
    // def compute_heading_w_func(                                                            <L 104>
    // forward_w = wp.quat_rotate(quat, forward_vec)                                          <L 119>
    var_0 = wp::quat_rotate(var_quat, var_forward_vec);
    // return wp.atan2(forward_w[1], forward_w[0])                                            <L 120>
    var_2 = wp::extract(var_0, var_1);
    var_4 = wp::extract(var_0, var_3);
    var_5 = wp::atan2(var_2, var_4);
    return var_5;
}


// /workspace/isaaclab/source/isaaclab_newton/isaaclab_newton/assets/kernels.py:1059
static CUDA_CALLABLE void _update_body_inertial_properties_inverse_0(
    wp::int32 var_env_id,
    wp::int32 var_body_id,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::float32> var_body_inertia,
    bool var_update_inv_mass,
    wp::array_t<wp::float32> var_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_body_inv_inertia)
{
    //---------
    // primal vars
    wp::float32* var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::float32 var_3 = 0.0;
    bool var_4;
    const wp::float32 var_5 = 1.0;
    wp::float32 var_6;
    const wp::int32 var_7 = 0;
    wp::float32* var_8;
    const wp::int32 var_9 = 1;
    wp::float32* var_10;
    const wp::int32 var_11 = 2;
    wp::float32* var_12;
    const wp::int32 var_13 = 3;
    wp::float32* var_14;
    const wp::int32 var_15 = 4;
    wp::float32* var_16;
    const wp::int32 var_17 = 5;
    wp::float32* var_18;
    const wp::int32 var_19 = 6;
    wp::float32* var_20;
    const wp::int32 var_21 = 7;
    wp::float32* var_22;
    const wp::int32 var_23 = 8;
    wp::float32* var_24;
    wp::mat_t<3, 3, wp::float32> var_25;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    wp::mat_t<3, 3, wp::float32> var_35;
    const wp::float32 var_36 = 0.0;
    const wp::float32 var_37 = 0.0;
    wp::mat_t<3, 3, wp::float32> var_38;
    //---------
    // forward
    // def _update_body_inertial_properties_inverse(                                          <L 1060>
    // mass = body_mass[env_id, body_id]                                                      <L 1070>
    var_0 = wp::address(var_body_mass, var_env_id, var_body_id);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // if mass > 0.0:                                                                         <L 1071>
    var_4 = (var_1 > var_3);
    if (var_4) {
        // if update_inv_mass:                                                                <L 1072>
        if (var_update_inv_mass) {
            // body_inv_mass[env_id, body_id] = 1.0 / mass                                    <L 1073>
            var_6 = wp::div(var_5, var_1);
            wp::array_store(var_body_inv_mass, var_env_id, var_body_id, var_6);
        }
        // inertia = wp.mat33f(                                                               <L 1074>
        // body_inertia[env_id, body_id, 0],                                                  <L 1075>
        var_8 = wp::address(var_body_inertia, var_env_id, var_body_id, var_7);
        // body_inertia[env_id, body_id, 1],                                                  <L 1076>
        var_10 = wp::address(var_body_inertia, var_env_id, var_body_id, var_9);
        // body_inertia[env_id, body_id, 2],                                                  <L 1077>
        var_12 = wp::address(var_body_inertia, var_env_id, var_body_id, var_11);
        // body_inertia[env_id, body_id, 3],                                                  <L 1078>
        var_14 = wp::address(var_body_inertia, var_env_id, var_body_id, var_13);
        // body_inertia[env_id, body_id, 4],                                                  <L 1079>
        var_16 = wp::address(var_body_inertia, var_env_id, var_body_id, var_15);
        // body_inertia[env_id, body_id, 5],                                                  <L 1080>
        var_18 = wp::address(var_body_inertia, var_env_id, var_body_id, var_17);
        // body_inertia[env_id, body_id, 6],                                                  <L 1081>
        var_20 = wp::address(var_body_inertia, var_env_id, var_body_id, var_19);
        // body_inertia[env_id, body_id, 7],                                                  <L 1082>
        var_22 = wp::address(var_body_inertia, var_env_id, var_body_id, var_21);
        // body_inertia[env_id, body_id, 8],                                                  <L 1083>
        var_24 = wp::address(var_body_inertia, var_env_id, var_body_id, var_23);
        var_26 = wp::load(var_8);
        var_27 = wp::load(var_10);
        var_28 = wp::load(var_12);
        var_29 = wp::load(var_14);
        var_30 = wp::load(var_16);
        var_31 = wp::load(var_18);
        var_32 = wp::load(var_20);
        var_33 = wp::load(var_22);
        var_34 = wp::load(var_24);
        var_25 = wp::mat_t<3, 3, wp::float32>(var_26, var_27, var_28, var_29, var_30, var_31, var_32, var_33, var_34);
        // body_inv_inertia[env_id, body_id] = wp.inverse(inertia)                            <L 1085>
        var_35 = wp::inverse(var_25);
        wp::array_store(var_body_inv_inertia, var_env_id, var_body_id, var_35);
    }
    if (!var_4) {
        // if update_inv_mass:                                                                <L 1087>
        if (var_update_inv_mass) {
            // body_inv_mass[env_id, body_id] = 0.0                                           <L 1088>
            wp::array_store(var_body_inv_mass, var_env_id, var_body_id, var_36);
        }
        // body_inv_inertia[env_id, body_id] = wp.mat33f(0.0)                                 <L 1089>
        var_38 = wp::mat_t<3, 3, wp::float32>(var_37);
        wp::array_store(var_body_inv_inertia, var_env_id, var_body_id, var_38);
    }
}


// /workspace/isaaclab/source/isaaclab_newton/isaaclab_newton/assets/kernels.py:205
static CUDA_CALLABLE wp::transform_t<wp::float32> get_com_pose_in_link_frame_func_0(
    wp::transform_t<wp::float32> var_com_pose_w,
    wp::vec_t<3, wp::float32> var_com_pos_b)
{
    //---------
    // primal vars
    wp::quat_t<wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::transform_t<wp::float32> var_4;
    //---------
    // forward
    // def get_com_pose_in_link_frame_func(                                                   <L 206>
    // com_rot_w = wp.transform_get_rotation(com_pose_w)                                      <L 223>
    var_0 = wp::transform_get_rotation(var_com_pose_w);
    // link_pos_w = wp.transform_get_translation(com_pose_w) - wp.quat_rotate(com_rot_w, com_pos_b)       <L 224>
    var_1 = wp::transform_get_translation(var_com_pose_w);
    var_2 = wp::quat_rotate(var_0, var_com_pos_b);
    var_3 = wp::sub(var_1, var_2);
    // return wp.transform(link_pos_w, com_rot_w)                                             <L 225>
    var_4 = wp::transform_t<wp::float32>(var_3, var_0);
    return var_4;
}


// /workspace/isaaclab/source/isaaclab_newton/isaaclab_newton/assets/kernels.py:82
static CUDA_CALLABLE wp::vec_t<13, wp::float32> concat_pose_and_vel_to_state_func_0(
    wp::transform_t<wp::float32> var_pose,
    wp::vec_t<6, wp::float32> var_vel)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 2;
    wp::float32 var_5;
    const wp::int32 var_6 = 3;
    wp::float32 var_7;
    const wp::int32 var_8 = 4;
    wp::float32 var_9;
    const wp::int32 var_10 = 5;
    wp::float32 var_11;
    const wp::int32 var_12 = 6;
    wp::float32 var_13;
    const wp::int32 var_14 = 0;
    wp::float32 var_15;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    const wp::int32 var_18 = 2;
    wp::float32 var_19;
    const wp::int32 var_20 = 3;
    wp::float32 var_21;
    const wp::int32 var_22 = 4;
    wp::float32 var_23;
    const wp::int32 var_24 = 5;
    wp::float32 var_25;
    wp::vec_t<13, wp::float32> var_26;
    //---------
    // forward
    // def concat_pose_and_vel_to_state_func(                                                 <L 83>
    // return vec13f(                                                                         <L 98>
    // pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6], vel[0], vel[1], vel[2], vel[3], vel[4], vel[5]       <L 99>
    var_1 = wp::extract(var_pose, var_0);
    var_3 = wp::extract(var_pose, var_2);
    var_5 = wp::extract(var_pose, var_4);
    var_7 = wp::extract(var_pose, var_6);
    var_9 = wp::extract(var_pose, var_8);
    var_11 = wp::extract(var_pose, var_10);
    var_13 = wp::extract(var_pose, var_12);
    var_15 = wp::extract(var_vel, var_14);
    var_17 = wp::extract(var_vel, var_16);
    var_19 = wp::extract(var_vel, var_18);
    var_21 = wp::extract(var_vel, var_20);
    var_23 = wp::extract(var_vel, var_22);
    var_25 = wp::extract(var_vel, var_24);
    var_26 = wp::vec_t<13, wp::float32>({var_1, var_3, var_5, var_7, var_9, var_11, var_13, var_15, var_17, var_19, var_21, var_23, var_25});
    return var_26;
}


// /workspace/isaaclab/source/isaaclab_newton/isaaclab_newton/assets/kernels.py:39
static CUDA_CALLABLE wp::vec_t<6, wp::float32> get_link_vel_from_root_com_vel_func_0(
    wp::vec_t<6, wp::float32> var_com_vel,
    wp::transform_t<wp::float32> var_link_pose,
    wp::vec_t<3, wp::float32> var_body_com_pos_b)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::quat_t<wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<6, wp::float32> var_8;
    //---------
    // forward
    // def get_link_vel_from_root_com_vel_func(                                               <L 40>
    // projected_vel = wp.cross(                                                              <L 58>
    // wp.spatial_bottom(com_vel),                                                            <L 59>
    var_0 = wp::spatial_bottom(var_com_vel);
    // wp.quat_rotate(wp.transform_get_rotation(link_pose), -body_com_pos_b),                 <L 60>
    var_1 = wp::transform_get_rotation(var_link_pose);
    var_2 = wp::neg(var_body_com_pos_b);
    var_3 = wp::quat_rotate(var_1, var_2);
    var_4 = wp::cross(var_0, var_3);
    // return wp.spatial_vector(wp.spatial_top(com_vel) + projected_vel, wp.spatial_bottom(com_vel))       <L 62>
    var_5 = wp::spatial_top(var_com_vel);
    var_6 = wp::add(var_5, var_4);
    var_7 = wp::spatial_bottom(var_com_vel);
    var_8 = wp::vec_t<6, wp::float32>(var_6, var_7);
    return var_8;
}


// /workspace/isaaclab/source/isaaclab_newton/isaaclab_newton/assets/kernels.py:65
static CUDA_CALLABLE wp::transform_t<wp::float32> get_com_pose_from_link_pose_func_0(
    wp::transform_t<wp::float32> var_link_pose,
    wp::vec_t<3, wp::float32> var_body_com_pos_b)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 1.0;
    wp::quat_t<wp::float32> var_4;
    wp::transform_t<wp::float32> var_5;
    wp::transform_t<wp::float32> var_6;
    //---------
    // forward
    // def get_com_pose_from_link_pose_func(                                                  <L 66>
    // return link_pose * wp.transformf(body_com_pos_b, wp.quatf(0.0, 0.0, 0.0, 1.0))         <L 79>
    var_4 = wp::quat_t<wp::float32>(var_0, var_1, var_2, var_3);
    var_5 = wp::transform_t<wp::float32>(var_body_com_pos_b, var_4);
    var_6 = wp::mul(var_link_pose, var_5);
    return var_6;
}


// /workspace/isaaclab/source/isaaclab_newton/isaaclab_newton/assets/kernels.py:25
static CUDA_CALLABLE wp::vec_t<6, wp::float32> pack_body_wrench_to_world_0(
    wp::vec_t<3, wp::float32> var_force_b,
    wp::vec_t<3, wp::float32> var_torque_b,
    wp::quat_t<wp::float32> var_body_rot_w)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<6, wp::float32> var_2;
    //---------
    // forward
    // def pack_body_wrench_to_world(                                                         <L 26>
    // return wp.spatial_vector(                                                              <L 32>
    // wp.quat_rotate(body_rot_w, force_b),                                                   <L 33>
    var_0 = wp::quat_rotate(var_body_rot_w, var_force_b);
    // wp.quat_rotate(body_rot_w, torque_b),                                                  <L 34>
    var_1 = wp::quat_rotate(var_body_rot_w, var_torque_b);
    // wp.float32,                                                                            <L 35>
    var_2 = wp::vec_t<6, wp::float32>(var_0, var_1);
    return var_2;
}


// /workspace/isaaclab/source/isaaclab_newton/isaaclab_newton/assets/kernels.py:176
static CUDA_CALLABLE void adj_get_link_velocity_in_com_frame_func_0(
    wp::vec_t<6, wp::float32> var_link_velocity_w,
    wp::transform_t<wp::float32> var_link_pose_w,
    wp::vec_t<3, wp::float32> var_body_com_pos_b,
    wp::vec_t<6, wp::float32> & adj_link_velocity_w,
    wp::transform_t<wp::float32> & adj_link_pose_w,
    wp::vec_t<3, wp::float32> & adj_body_com_pos_b,
    wp::vec_t<6, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::quat_t<wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<6, wp::float32> var_7;
    //---------
    // dual vars
    wp::vec_t<3, wp::float32> adj_0 = {};
    wp::vec_t<3, wp::float32> adj_1 = {};
    wp::quat_t<wp::float32> adj_2 = {};
    wp::vec_t<3, wp::float32> adj_3 = {};
    wp::vec_t<3, wp::float32> adj_4 = {};
    wp::vec_t<3, wp::float32> adj_5 = {};
    wp::vec_t<3, wp::float32> adj_6 = {};
    wp::vec_t<6, wp::float32> adj_7 = {};
    //---------
    // forward
    // def get_link_velocity_in_com_frame_func(                                               <L 177>
    // return wp.spatial_vector(                                                              <L 195>
    // wp.spatial_top(link_velocity_w)                                                        <L 196>
    var_0 = wp::spatial_top(var_link_velocity_w);
    // + wp.cross(                                                                            <L 197>
    // wp.spatial_bottom(link_velocity_w),                                                    <L 198>
    var_1 = wp::spatial_bottom(var_link_velocity_w);
    // wp.quat_rotate(wp.transform_get_rotation(link_pose_w), body_com_pos_b),                <L 199>
    var_2 = wp::transform_get_rotation(var_link_pose_w);
    var_3 = wp::quat_rotate(var_2, var_body_com_pos_b);
    var_4 = wp::cross(var_1, var_3);
    var_5 = wp::add(var_0, var_4);
    // wp.spatial_bottom(link_velocity_w),                                                    <L 201>
    var_6 = wp::spatial_bottom(var_link_velocity_w);
    var_7 = wp::vec_t<6, wp::float32>(var_5, var_6);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_7 += adj_ret;
    wp::adj_vec_t(var_5, var_6, adj_5, adj_6, adj_7);
    wp::adj_spatial_bottom(var_link_velocity_w, adj_link_velocity_w, adj_6);
    // adj: wp.spatial_bottom(link_velocity_w),                                               <L 201>
    wp::adj_add(var_0, var_4, adj_0, adj_4, adj_5);
    wp::adj_cross(var_1, var_3, adj_1, adj_3, adj_4);
    wp::adj_quat_rotate(var_2, var_body_com_pos_b, adj_2, adj_body_com_pos_b, adj_3);
    wp::adj_transform_get_rotation(var_link_pose_w, adj_link_pose_w, adj_2);
    // adj: wp.quat_rotate(wp.transform_get_rotation(link_pose_w), body_com_pos_b),           <L 199>
    wp::adj_spatial_bottom(var_link_velocity_w, adj_link_velocity_w, adj_1);
    // adj: wp.spatial_bottom(link_velocity_w),                                               <L 198>
    // adj: + wp.cross(                                                                       <L 197>
    wp::adj_spatial_top(var_link_velocity_w, adj_link_velocity_w, adj_0);
    // adj: wp.spatial_top(link_velocity_w)                                                   <L 196>
    // adj: return wp.spatial_vector(                                                         <L 195>
    // adj: def get_link_velocity_in_com_frame_func(                                          <L 177>
    return;
}


// /workspace/isaaclab/source/isaaclab_newton/isaaclab_newton/assets/kernels.py:103
static CUDA_CALLABLE void adj_compute_heading_w_func_0(
    wp::vec_t<3, wp::float32> var_forward_vec,
    wp::quat_t<wp::float32> var_quat,
    wp::vec_t<3, wp::float32> & adj_forward_vec,
    wp::quat_t<wp::float32> & adj_quat,
    wp::float32 & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    const wp::int32 var_1 = 1;
    wp::float32 var_2;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    wp::float32 var_5;
    //---------
    // dual vars
    wp::vec_t<3, wp::float32> adj_0 = {};
    wp::int32 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::int32 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::float32 adj_5 = {};
    //---------
    // forward
    // def compute_heading_w_func(                                                            <L 104>
    // forward_w = wp.quat_rotate(quat, forward_vec)                                          <L 119>
    var_0 = wp::quat_rotate(var_quat, var_forward_vec);
    // return wp.atan2(forward_w[1], forward_w[0])                                            <L 120>
    var_2 = wp::extract(var_0, var_1);
    var_4 = wp::extract(var_0, var_3);
    var_5 = wp::atan2(var_2, var_4);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_5 += adj_ret;
    wp::adj_atan2(var_2, var_4, adj_2, adj_4, adj_5);
    wp::adj_extract(var_0, var_3, adj_0, adj_3, adj_4);
    wp::adj_extract(var_0, var_1, adj_0, adj_1, adj_2);
    // adj: return wp.atan2(forward_w[1], forward_w[0])                                       <L 120>
    wp::adj_quat_rotate(var_quat, var_forward_vec, adj_quat, adj_forward_vec, adj_0);
    // adj: forward_w = wp.quat_rotate(quat, forward_vec)                                     <L 119>
    // adj: def compute_heading_w_func(                                                       <L 104>
    return;
}


// /workspace/isaaclab/source/isaaclab_newton/isaaclab_newton/assets/kernels.py:1059
static CUDA_CALLABLE void adj__update_body_inertial_properties_inverse_0(
    wp::int32 var_env_id,
    wp::int32 var_body_id,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::float32> var_body_inertia,
    bool var_update_inv_mass,
    wp::array_t<wp::float32> var_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_body_inv_inertia,
    wp::int32 & adj_env_id,
    wp::int32 & adj_body_id,
    wp::array_t<wp::float32> & adj_body_mass,
    wp::array_t<wp::float32> & adj_body_inertia,
    bool & adj_update_inv_mass,
    wp::array_t<wp::float32> & adj_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_body_inv_inertia)
{
    //---------
    // primal vars
    wp::float32* var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::float32 var_3 = 0.0;
    bool var_4;
    const wp::float32 var_5 = 1.0;
    wp::float32 var_6;
    const wp::int32 var_7 = 0;
    wp::float32* var_8;
    const wp::int32 var_9 = 1;
    wp::float32* var_10;
    const wp::int32 var_11 = 2;
    wp::float32* var_12;
    const wp::int32 var_13 = 3;
    wp::float32* var_14;
    const wp::int32 var_15 = 4;
    wp::float32* var_16;
    const wp::int32 var_17 = 5;
    wp::float32* var_18;
    const wp::int32 var_19 = 6;
    wp::float32* var_20;
    const wp::int32 var_21 = 7;
    wp::float32* var_22;
    const wp::int32 var_23 = 8;
    wp::float32* var_24;
    wp::mat_t<3, 3, wp::float32> var_25;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    wp::mat_t<3, 3, wp::float32> var_35;
    const wp::float32 var_36 = 0.0;
    const wp::float32 var_37 = 0.0;
    wp::mat_t<3, 3, wp::float32> var_38;
    //---------
    // dual vars
    wp::float32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float32 adj_3 = {};
    bool adj_4 = {};
    wp::float32 adj_5 = {};
    wp::float32 adj_6 = {};
    wp::int32 adj_7 = {};
    wp::float32 adj_8 = {};
    wp::int32 adj_9 = {};
    wp::float32 adj_10 = {};
    wp::int32 adj_11 = {};
    wp::float32 adj_12 = {};
    wp::int32 adj_13 = {};
    wp::float32 adj_14 = {};
    wp::int32 adj_15 = {};
    wp::float32 adj_16 = {};
    wp::int32 adj_17 = {};
    wp::float32 adj_18 = {};
    wp::int32 adj_19 = {};
    wp::float32 adj_20 = {};
    wp::int32 adj_21 = {};
    wp::float32 adj_22 = {};
    wp::int32 adj_23 = {};
    wp::float32 adj_24 = {};
    wp::mat_t<3, 3, wp::float32> adj_25 = {};
    wp::float32 adj_26 = {};
    wp::float32 adj_27 = {};
    wp::float32 adj_28 = {};
    wp::float32 adj_29 = {};
    wp::float32 adj_30 = {};
    wp::float32 adj_31 = {};
    wp::float32 adj_32 = {};
    wp::float32 adj_33 = {};
    wp::float32 adj_34 = {};
    wp::mat_t<3, 3, wp::float32> adj_35 = {};
    wp::float32 adj_36 = {};
    wp::float32 adj_37 = {};
    wp::mat_t<3, 3, wp::float32> adj_38 = {};
    //---------
    // forward
    // def _update_body_inertial_properties_inverse(                                          <L 1060>
    // mass = body_mass[env_id, body_id]                                                      <L 1070>
    var_0 = wp::address(var_body_mass, var_env_id, var_body_id);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // if mass > 0.0:                                                                         <L 1071>
    var_4 = (var_1 > var_3);
    if (var_4) {
        // if update_inv_mass:                                                                <L 1072>
        if (var_update_inv_mass) {
            // body_inv_mass[env_id, body_id] = 1.0 / mass                                    <L 1073>
            var_6 = wp::div(var_5, var_1);
            // wp::array_store(var_body_inv_mass, var_env_id, var_body_id, var_6);
        }
        // inertia = wp.mat33f(                                                               <L 1074>
        // body_inertia[env_id, body_id, 0],                                                  <L 1075>
        var_8 = wp::address(var_body_inertia, var_env_id, var_body_id, var_7);
        // body_inertia[env_id, body_id, 1],                                                  <L 1076>
        var_10 = wp::address(var_body_inertia, var_env_id, var_body_id, var_9);
        // body_inertia[env_id, body_id, 2],                                                  <L 1077>
        var_12 = wp::address(var_body_inertia, var_env_id, var_body_id, var_11);
        // body_inertia[env_id, body_id, 3],                                                  <L 1078>
        var_14 = wp::address(var_body_inertia, var_env_id, var_body_id, var_13);
        // body_inertia[env_id, body_id, 4],                                                  <L 1079>
        var_16 = wp::address(var_body_inertia, var_env_id, var_body_id, var_15);
        // body_inertia[env_id, body_id, 5],                                                  <L 1080>
        var_18 = wp::address(var_body_inertia, var_env_id, var_body_id, var_17);
        // body_inertia[env_id, body_id, 6],                                                  <L 1081>
        var_20 = wp::address(var_body_inertia, var_env_id, var_body_id, var_19);
        // body_inertia[env_id, body_id, 7],                                                  <L 1082>
        var_22 = wp::address(var_body_inertia, var_env_id, var_body_id, var_21);
        // body_inertia[env_id, body_id, 8],                                                  <L 1083>
        var_24 = wp::address(var_body_inertia, var_env_id, var_body_id, var_23);
        var_26 = wp::load(var_8);
        var_27 = wp::load(var_10);
        var_28 = wp::load(var_12);
        var_29 = wp::load(var_14);
        var_30 = wp::load(var_16);
        var_31 = wp::load(var_18);
        var_32 = wp::load(var_20);
        var_33 = wp::load(var_22);
        var_34 = wp::load(var_24);
        var_25 = wp::mat_t<3, 3, wp::float32>(var_26, var_27, var_28, var_29, var_30, var_31, var_32, var_33, var_34);
        // body_inv_inertia[env_id, body_id] = wp.inverse(inertia)                            <L 1085>
        var_35 = wp::inverse(var_25);
        // wp::array_store(var_body_inv_inertia, var_env_id, var_body_id, var_35);
    }
    if (!var_4) {
        // if update_inv_mass:                                                                <L 1087>
        if (var_update_inv_mass) {
            // body_inv_mass[env_id, body_id] = 0.0                                           <L 1088>
            // wp::array_store(var_body_inv_mass, var_env_id, var_body_id, var_36);
        }
        // body_inv_inertia[env_id, body_id] = wp.mat33f(0.0)                                 <L 1089>
        var_38 = wp::mat_t<3, 3, wp::float32>(var_37);
        // wp::array_store(var_body_inv_inertia, var_env_id, var_body_id, var_38);
    }
    //---------
    // reverse
    if (!var_4) {
        wp::adj_array_store(var_body_inv_inertia, var_env_id, var_body_id, var_38, adj_body_inv_inertia, adj_env_id, adj_body_id, adj_38);
        wp::adj_mat_t(var_37, adj_37, adj_38);
        // adj: body_inv_inertia[env_id, body_id] = wp.mat33f(0.0)                            <L 1089>
        if (var_update_inv_mass) {
            wp::adj_array_store(var_body_inv_mass, var_env_id, var_body_id, var_36, adj_body_inv_mass, adj_env_id, adj_body_id, adj_36);
            // adj: body_inv_mass[env_id, body_id] = 0.0                                      <L 1088>
        }
        // adj: if update_inv_mass:                                                           <L 1087>
    }
    if (var_4) {
        wp::adj_array_store(var_body_inv_inertia, var_env_id, var_body_id, var_35, adj_body_inv_inertia, adj_env_id, adj_body_id, adj_35);
        wp::adj_inverse(var_25, var_35, adj_25, adj_35);
        // adj: body_inv_inertia[env_id, body_id] = wp.inverse(inertia)                       <L 1085>
        wp::adj_mat_t(var_26, var_27, var_28, var_29, var_30, var_31, var_32, var_33, var_34, adj_8, adj_10, adj_12, adj_14, adj_16, adj_18, adj_20, adj_22, adj_24, adj_25);
        wp::adj_address(var_body_inertia, var_env_id, var_body_id, var_23, adj_body_inertia, adj_env_id, adj_body_id, adj_23, adj_24);
        // adj: body_inertia[env_id, body_id, 8],                                             <L 1083>
        wp::adj_address(var_body_inertia, var_env_id, var_body_id, var_21, adj_body_inertia, adj_env_id, adj_body_id, adj_21, adj_22);
        // adj: body_inertia[env_id, body_id, 7],                                             <L 1082>
        wp::adj_address(var_body_inertia, var_env_id, var_body_id, var_19, adj_body_inertia, adj_env_id, adj_body_id, adj_19, adj_20);
        // adj: body_inertia[env_id, body_id, 6],                                             <L 1081>
        wp::adj_address(var_body_inertia, var_env_id, var_body_id, var_17, adj_body_inertia, adj_env_id, adj_body_id, adj_17, adj_18);
        // adj: body_inertia[env_id, body_id, 5],                                             <L 1080>
        wp::adj_address(var_body_inertia, var_env_id, var_body_id, var_15, adj_body_inertia, adj_env_id, adj_body_id, adj_15, adj_16);
        // adj: body_inertia[env_id, body_id, 4],                                             <L 1079>
        wp::adj_address(var_body_inertia, var_env_id, var_body_id, var_13, adj_body_inertia, adj_env_id, adj_body_id, adj_13, adj_14);
        // adj: body_inertia[env_id, body_id, 3],                                             <L 1078>
        wp::adj_address(var_body_inertia, var_env_id, var_body_id, var_11, adj_body_inertia, adj_env_id, adj_body_id, adj_11, adj_12);
        // adj: body_inertia[env_id, body_id, 2],                                             <L 1077>
        wp::adj_address(var_body_inertia, var_env_id, var_body_id, var_9, adj_body_inertia, adj_env_id, adj_body_id, adj_9, adj_10);
        // adj: body_inertia[env_id, body_id, 1],                                             <L 1076>
        wp::adj_address(var_body_inertia, var_env_id, var_body_id, var_7, adj_body_inertia, adj_env_id, adj_body_id, adj_7, adj_8);
        // adj: body_inertia[env_id, body_id, 0],                                             <L 1075>
        // adj: inertia = wp.mat33f(                                                          <L 1074>
        if (var_update_inv_mass) {
            wp::adj_array_store(var_body_inv_mass, var_env_id, var_body_id, var_6, adj_body_inv_mass, adj_env_id, adj_body_id, adj_6);
            wp::adj_div(var_5, var_1, var_6, adj_5, adj_1, adj_6);
            // adj: body_inv_mass[env_id, body_id] = 1.0 / mass                               <L 1073>
        }
        // adj: if update_inv_mass:                                                           <L 1072>
    }
    // adj: if mass > 0.0:                                                                    <L 1071>
    wp::adj_copy(var_2, adj_0, adj_1);
    wp::adj_address(var_body_mass, var_env_id, var_body_id, adj_body_mass, adj_env_id, adj_body_id, adj_0);
    // adj: mass = body_mass[env_id, body_id]                                                 <L 1070>
    // adj: def _update_body_inertial_properties_inverse(                                     <L 1060>
    return;
}


// /workspace/isaaclab/source/isaaclab_newton/isaaclab_newton/assets/kernels.py:205
static CUDA_CALLABLE void adj_get_com_pose_in_link_frame_func_0(
    wp::transform_t<wp::float32> var_com_pose_w,
    wp::vec_t<3, wp::float32> var_com_pos_b,
    wp::transform_t<wp::float32> & adj_com_pose_w,
    wp::vec_t<3, wp::float32> & adj_com_pos_b,
    wp::transform_t<wp::float32> & adj_ret)
{
    //---------
    // primal vars
    wp::quat_t<wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::transform_t<wp::float32> var_4;
    //---------
    // dual vars
    wp::quat_t<wp::float32> adj_0 = {};
    wp::vec_t<3, wp::float32> adj_1 = {};
    wp::vec_t<3, wp::float32> adj_2 = {};
    wp::vec_t<3, wp::float32> adj_3 = {};
    wp::transform_t<wp::float32> adj_4 = {};
    //---------
    // forward
    // def get_com_pose_in_link_frame_func(                                                   <L 206>
    // com_rot_w = wp.transform_get_rotation(com_pose_w)                                      <L 223>
    var_0 = wp::transform_get_rotation(var_com_pose_w);
    // link_pos_w = wp.transform_get_translation(com_pose_w) - wp.quat_rotate(com_rot_w, com_pos_b)       <L 224>
    var_1 = wp::transform_get_translation(var_com_pose_w);
    var_2 = wp::quat_rotate(var_0, var_com_pos_b);
    var_3 = wp::sub(var_1, var_2);
    // return wp.transform(link_pos_w, com_rot_w)                                             <L 225>
    var_4 = wp::transform_t<wp::float32>(var_3, var_0);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_4 += adj_ret;
    wp::adj_transform_t(var_3, var_0, adj_3, adj_0, adj_4);
    // adj: return wp.transform(link_pos_w, com_rot_w)                                        <L 225>
    wp::adj_sub(var_1, var_2, adj_1, adj_2, adj_3);
    wp::adj_quat_rotate(var_0, var_com_pos_b, adj_0, adj_com_pos_b, adj_2);
    wp::adj_transform_get_translation(var_com_pose_w, adj_com_pose_w, adj_1);
    // adj: link_pos_w = wp.transform_get_translation(com_pose_w) - wp.quat_rotate(com_rot_w, com_pos_b)  <L 224>
    wp::adj_transform_get_rotation(var_com_pose_w, adj_com_pose_w, adj_0);
    // adj: com_rot_w = wp.transform_get_rotation(com_pose_w)                                 <L 223>
    // adj: def get_com_pose_in_link_frame_func(                                              <L 206>
    return;
}


// /workspace/isaaclab/source/isaaclab_newton/isaaclab_newton/assets/kernels.py:82
static CUDA_CALLABLE void adj_concat_pose_and_vel_to_state_func_0(
    wp::transform_t<wp::float32> var_pose,
    wp::vec_t<6, wp::float32> var_vel,
    wp::transform_t<wp::float32> & adj_pose,
    wp::vec_t<6, wp::float32> & adj_vel,
    wp::vec_t<13, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 2;
    wp::float32 var_5;
    const wp::int32 var_6 = 3;
    wp::float32 var_7;
    const wp::int32 var_8 = 4;
    wp::float32 var_9;
    const wp::int32 var_10 = 5;
    wp::float32 var_11;
    const wp::int32 var_12 = 6;
    wp::float32 var_13;
    const wp::int32 var_14 = 0;
    wp::float32 var_15;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    const wp::int32 var_18 = 2;
    wp::float32 var_19;
    const wp::int32 var_20 = 3;
    wp::float32 var_21;
    const wp::int32 var_22 = 4;
    wp::float32 var_23;
    const wp::int32 var_24 = 5;
    wp::float32 var_25;
    wp::vec_t<13, wp::float32> var_26;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::int32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::int32 adj_4 = {};
    wp::float32 adj_5 = {};
    wp::int32 adj_6 = {};
    wp::float32 adj_7 = {};
    wp::int32 adj_8 = {};
    wp::float32 adj_9 = {};
    wp::int32 adj_10 = {};
    wp::float32 adj_11 = {};
    wp::int32 adj_12 = {};
    wp::float32 adj_13 = {};
    wp::int32 adj_14 = {};
    wp::float32 adj_15 = {};
    wp::int32 adj_16 = {};
    wp::float32 adj_17 = {};
    wp::int32 adj_18 = {};
    wp::float32 adj_19 = {};
    wp::int32 adj_20 = {};
    wp::float32 adj_21 = {};
    wp::int32 adj_22 = {};
    wp::float32 adj_23 = {};
    wp::int32 adj_24 = {};
    wp::float32 adj_25 = {};
    wp::vec_t<13, wp::float32> adj_26 = {};
    //---------
    // forward
    // def concat_pose_and_vel_to_state_func(                                                 <L 83>
    // return vec13f(                                                                         <L 98>
    // pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6], vel[0], vel[1], vel[2], vel[3], vel[4], vel[5]       <L 99>
    var_1 = wp::extract(var_pose, var_0);
    var_3 = wp::extract(var_pose, var_2);
    var_5 = wp::extract(var_pose, var_4);
    var_7 = wp::extract(var_pose, var_6);
    var_9 = wp::extract(var_pose, var_8);
    var_11 = wp::extract(var_pose, var_10);
    var_13 = wp::extract(var_pose, var_12);
    var_15 = wp::extract(var_vel, var_14);
    var_17 = wp::extract(var_vel, var_16);
    var_19 = wp::extract(var_vel, var_18);
    var_21 = wp::extract(var_vel, var_20);
    var_23 = wp::extract(var_vel, var_22);
    var_25 = wp::extract(var_vel, var_24);
    var_26 = wp::vec_t<13, wp::float32>({var_1, var_3, var_5, var_7, var_9, var_11, var_13, var_15, var_17, var_19, var_21, var_23, var_25});
    goto label0;
    //---------
    // reverse
    label0:;
    adj_26 += adj_ret;
    wp::adj_vec_t({var_1, var_3, var_5, var_7, var_9, var_11, var_13, var_15, var_17, var_19, var_21, var_23, var_25}, {&adj_1, &adj_3, &adj_5, &adj_7, &adj_9, &adj_11, &adj_13, &adj_15, &adj_17, &adj_19, &adj_21, &adj_23, &adj_25}, adj_26);
    wp::adj_extract(var_vel, var_24, adj_vel, adj_24, adj_25);
    wp::adj_extract(var_vel, var_22, adj_vel, adj_22, adj_23);
    wp::adj_extract(var_vel, var_20, adj_vel, adj_20, adj_21);
    wp::adj_extract(var_vel, var_18, adj_vel, adj_18, adj_19);
    wp::adj_extract(var_vel, var_16, adj_vel, adj_16, adj_17);
    wp::adj_extract(var_vel, var_14, adj_vel, adj_14, adj_15);
    wp::adj_extract(var_pose, var_12, adj_pose, adj_12, adj_13);
    wp::adj_extract(var_pose, var_10, adj_pose, adj_10, adj_11);
    wp::adj_extract(var_pose, var_8, adj_pose, adj_8, adj_9);
    wp::adj_extract(var_pose, var_6, adj_pose, adj_6, adj_7);
    wp::adj_extract(var_pose, var_4, adj_pose, adj_4, adj_5);
    wp::adj_extract(var_pose, var_2, adj_pose, adj_2, adj_3);
    wp::adj_extract(var_pose, var_0, adj_pose, adj_0, adj_1);
    // adj: pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6], vel[0], vel[1], vel[2], vel[3], vel[4], vel[5]  <L 99>
    // adj: return vec13f(                                                                    <L 98>
    // adj: def concat_pose_and_vel_to_state_func(                                            <L 83>
    return;
}


// /workspace/isaaclab/source/isaaclab_newton/isaaclab_newton/assets/kernels.py:39
static CUDA_CALLABLE void adj_get_link_vel_from_root_com_vel_func_0(
    wp::vec_t<6, wp::float32> var_com_vel,
    wp::transform_t<wp::float32> var_link_pose,
    wp::vec_t<3, wp::float32> var_body_com_pos_b,
    wp::vec_t<6, wp::float32> & adj_com_vel,
    wp::transform_t<wp::float32> & adj_link_pose,
    wp::vec_t<3, wp::float32> & adj_body_com_pos_b,
    wp::vec_t<6, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::quat_t<wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<6, wp::float32> var_8;
    //---------
    // dual vars
    wp::vec_t<3, wp::float32> adj_0 = {};
    wp::quat_t<wp::float32> adj_1 = {};
    wp::vec_t<3, wp::float32> adj_2 = {};
    wp::vec_t<3, wp::float32> adj_3 = {};
    wp::vec_t<3, wp::float32> adj_4 = {};
    wp::vec_t<3, wp::float32> adj_5 = {};
    wp::vec_t<3, wp::float32> adj_6 = {};
    wp::vec_t<3, wp::float32> adj_7 = {};
    wp::vec_t<6, wp::float32> adj_8 = {};
    //---------
    // forward
    // def get_link_vel_from_root_com_vel_func(                                               <L 40>
    // projected_vel = wp.cross(                                                              <L 58>
    // wp.spatial_bottom(com_vel),                                                            <L 59>
    var_0 = wp::spatial_bottom(var_com_vel);
    // wp.quat_rotate(wp.transform_get_rotation(link_pose), -body_com_pos_b),                 <L 60>
    var_1 = wp::transform_get_rotation(var_link_pose);
    var_2 = wp::neg(var_body_com_pos_b);
    var_3 = wp::quat_rotate(var_1, var_2);
    var_4 = wp::cross(var_0, var_3);
    // return wp.spatial_vector(wp.spatial_top(com_vel) + projected_vel, wp.spatial_bottom(com_vel))       <L 62>
    var_5 = wp::spatial_top(var_com_vel);
    var_6 = wp::add(var_5, var_4);
    var_7 = wp::spatial_bottom(var_com_vel);
    var_8 = wp::vec_t<6, wp::float32>(var_6, var_7);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_8 += adj_ret;
    wp::adj_vec_t(var_6, var_7, adj_6, adj_7, adj_8);
    wp::adj_spatial_bottom(var_com_vel, adj_com_vel, adj_7);
    wp::adj_add(var_5, var_4, adj_5, adj_4, adj_6);
    wp::adj_spatial_top(var_com_vel, adj_com_vel, adj_5);
    // adj: return wp.spatial_vector(wp.spatial_top(com_vel) + projected_vel, wp.spatial_bottom(com_vel))  <L 62>
    wp::adj_cross(var_0, var_3, adj_0, adj_3, adj_4);
    wp::adj_quat_rotate(var_1, var_2, adj_1, adj_2, adj_3);
    wp::adj_neg(var_body_com_pos_b, adj_body_com_pos_b, adj_2);
    wp::adj_transform_get_rotation(var_link_pose, adj_link_pose, adj_1);
    // adj: wp.quat_rotate(wp.transform_get_rotation(link_pose), -body_com_pos_b),            <L 60>
    wp::adj_spatial_bottom(var_com_vel, adj_com_vel, adj_0);
    // adj: wp.spatial_bottom(com_vel),                                                       <L 59>
    // adj: projected_vel = wp.cross(                                                         <L 58>
    // adj: def get_link_vel_from_root_com_vel_func(                                          <L 40>
    return;
}


// /workspace/isaaclab/source/isaaclab_newton/isaaclab_newton/assets/kernels.py:65
static CUDA_CALLABLE void adj_get_com_pose_from_link_pose_func_0(
    wp::transform_t<wp::float32> var_link_pose,
    wp::vec_t<3, wp::float32> var_body_com_pos_b,
    wp::transform_t<wp::float32> & adj_link_pose,
    wp::vec_t<3, wp::float32> & adj_body_com_pos_b,
    wp::transform_t<wp::float32> & adj_ret)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 1.0;
    wp::quat_t<wp::float32> var_4;
    wp::transform_t<wp::float32> var_5;
    wp::transform_t<wp::float32> var_6;
    //---------
    // dual vars
    wp::float32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::quat_t<wp::float32> adj_4 = {};
    wp::transform_t<wp::float32> adj_5 = {};
    wp::transform_t<wp::float32> adj_6 = {};
    //---------
    // forward
    // def get_com_pose_from_link_pose_func(                                                  <L 66>
    // return link_pose * wp.transformf(body_com_pos_b, wp.quatf(0.0, 0.0, 0.0, 1.0))         <L 79>
    var_4 = wp::quat_t<wp::float32>(var_0, var_1, var_2, var_3);
    var_5 = wp::transform_t<wp::float32>(var_body_com_pos_b, var_4);
    var_6 = wp::mul(var_link_pose, var_5);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_6 += adj_ret;
    wp::adj_mul(var_link_pose, var_5, adj_link_pose, adj_5, adj_6);
    wp::adj_transform_t(var_body_com_pos_b, var_4, adj_body_com_pos_b, adj_4, adj_5);
    wp::adj_quat_t(var_0, var_1, var_2, var_3, adj_0, adj_1, adj_2, adj_3, adj_4);
    // adj: return link_pose * wp.transformf(body_com_pos_b, wp.quatf(0.0, 0.0, 0.0, 1.0))    <L 79>
    // adj: def get_com_pose_from_link_pose_func(                                             <L 66>
    return;
}


// /workspace/isaaclab/source/isaaclab_newton/isaaclab_newton/assets/kernels.py:25
static CUDA_CALLABLE void adj_pack_body_wrench_to_world_0(
    wp::vec_t<3, wp::float32> var_force_b,
    wp::vec_t<3, wp::float32> var_torque_b,
    wp::quat_t<wp::float32> var_body_rot_w,
    wp::vec_t<3, wp::float32> & adj_force_b,
    wp::vec_t<3, wp::float32> & adj_torque_b,
    wp::quat_t<wp::float32> & adj_body_rot_w,
    wp::vec_t<6, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<6, wp::float32> var_2;
    //---------
    // dual vars
    wp::vec_t<3, wp::float32> adj_0 = {};
    wp::vec_t<3, wp::float32> adj_1 = {};
    wp::vec_t<6, wp::float32> adj_2 = {};
    //---------
    // forward
    // def pack_body_wrench_to_world(                                                         <L 26>
    // return wp.spatial_vector(                                                              <L 32>
    // wp.quat_rotate(body_rot_w, force_b),                                                   <L 33>
    var_0 = wp::quat_rotate(var_body_rot_w, var_force_b);
    // wp.quat_rotate(body_rot_w, torque_b),                                                  <L 34>
    var_1 = wp::quat_rotate(var_body_rot_w, var_torque_b);
    // wp.float32,                                                                            <L 35>
    var_2 = wp::vec_t<6, wp::float32>(var_0, var_1);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_2 += adj_ret;
    wp::adj_vec_t(var_0, var_1, adj_0, adj_1, adj_2);
    // adj: wp.float32,                                                                       <L 35>
    wp::adj_quat_rotate(var_body_rot_w, var_torque_b, adj_body_rot_w, adj_torque_b, adj_1);
    // adj: wp.quat_rotate(body_rot_w, torque_b),                                             <L 34>
    wp::adj_quat_rotate(var_body_rot_w, var_force_b, adj_body_rot_w, adj_force_b, adj_0);
    // adj: wp.quat_rotate(body_rot_w, force_b),                                              <L 33>
    // adj: return wp.spatial_vector(                                                         <L 32>
    // adj: def pack_body_wrench_to_world(                                                    <L 26>
    return;
}



extern "C" __global__ void set_root_link_velocity_to_sim_index_01cbe378_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> var_link_pose_w,
    wp::array_t<wp::int32> var_env_ids,
    wp::int32 var_num_bodies,
    wp::array_t<wp::vec_t<6, wp::float32>> var_root_link_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_root_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w)
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
        wp::vec_t<6, wp::float32>* var_4;
        wp::vec_t<6, wp::float32> var_5;
        wp::vec_t<6, wp::float32>* var_6;
        wp::transform_t<wp::float32>* var_7;
        const wp::int32 var_8 = 0;
        wp::vec_t<3, wp::float32>* var_9;
        wp::vec_t<6, wp::float32> var_10;
        wp::vec_t<6, wp::float32> var_11;
        wp::transform_t<wp::float32> var_12;
        wp::vec_t<3, wp::float32> var_13;
        wp::range_t var_14;
        wp::int32 var_15;
        const wp::float32 var_16 = 0.0;
        const wp::float32 var_17 = 0.0;
        const wp::float32 var_18 = 0.0;
        const wp::float32 var_19 = 0.0;
        const wp::float32 var_20 = 0.0;
        const wp::float32 var_21 = 0.0;
        wp::vec_t<6, wp::float32> var_22;
        //---------
        // forward
        // def set_root_link_velocity_to_sim_index(                                               <L 1>
        // i = wp.tid()                                                                           <L 31>
        var_0 = builtin_tid1d();
        // env_id = wp.int32(env_ids[i])                                                          <L 32>
        var_1 = wp::address(var_env_ids, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::int32(var_3);
        // root_link_velocity_w[env_id] = data[i]                                                 <L 33>
        var_4 = wp::address(var_data, var_0);
        var_5 = wp::load(var_4);
        wp::array_store(var_root_link_velocity_w, var_2, var_5);
        // root_com_velocity_w[env_id] = get_link_velocity_in_com_frame_func(                     <L 35>
        // root_link_velocity_w[env_id], link_pose_w[env_id], body_com_pos_b[env_id, 0]           <L 36>
        var_6 = wp::address(var_root_link_velocity_w, var_2);
        var_7 = wp::address(var_link_pose_w, var_2);
        var_9 = wp::address(var_body_com_pos_b, var_2, var_8);
        var_11 = wp::load(var_6);
        var_12 = wp::load(var_7);
        var_13 = wp::load(var_9);
        var_10 = get_link_velocity_in_com_frame_func_0(var_11, var_12, var_13);
        // root_com_velocity_w[env_id] = get_link_velocity_in_com_frame_func(                     <L 35>
        wp::array_store(var_root_com_velocity_w, var_2, var_10);
        // for j in range(num_bodies):                                                            <L 39>
        var_14 = wp::range(var_num_bodies);
        start_for_0:;
            if (iter_cmp(var_14) == 0) goto end_for_0;
            var_15 = wp::iter_next(var_14);
            // body_acc_w[env_id, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)           <L 40>
            var_22 = wp::vec_t<6, wp::float32>({var_16, var_17, var_18, var_19, var_20, var_21});
            wp::array_store(var_body_acc_w, var_2, var_15, var_22);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void set_root_link_velocity_to_sim_index_01cbe378_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> var_link_pose_w,
    wp::array_t<wp::int32> var_env_ids,
    wp::int32 var_num_bodies,
    wp::array_t<wp::vec_t<6, wp::float32>> var_root_link_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_root_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_data,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> adj_link_pose_w,
    wp::array_t<wp::int32> adj_env_ids,
    wp::int32 adj_num_bodies,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_root_link_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_root_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_acc_w)
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
        wp::vec_t<6, wp::float32>* var_4;
        wp::vec_t<6, wp::float32> var_5;
        wp::vec_t<6, wp::float32>* var_6;
        wp::transform_t<wp::float32>* var_7;
        const wp::int32 var_8 = 0;
        wp::vec_t<3, wp::float32>* var_9;
        wp::vec_t<6, wp::float32> var_10;
        wp::vec_t<6, wp::float32> var_11;
        wp::transform_t<wp::float32> var_12;
        wp::vec_t<3, wp::float32> var_13;
        wp::range_t var_14;
        wp::int32 var_15;
        const wp::float32 var_16 = 0.0;
        const wp::float32 var_17 = 0.0;
        const wp::float32 var_18 = 0.0;
        const wp::float32 var_19 = 0.0;
        const wp::float32 var_20 = 0.0;
        const wp::float32 var_21 = 0.0;
        wp::vec_t<6, wp::float32> var_22;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::vec_t<6, wp::float32> adj_4 = {};
        wp::vec_t<6, wp::float32> adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        wp::transform_t<wp::float32> adj_7 = {};
        wp::int32 adj_8 = {};
        wp::vec_t<3, wp::float32> adj_9 = {};
        wp::vec_t<6, wp::float32> adj_10 = {};
        wp::vec_t<6, wp::float32> adj_11 = {};
        wp::transform_t<wp::float32> adj_12 = {};
        wp::vec_t<3, wp::float32> adj_13 = {};
        wp::range_t adj_14 = {};
        wp::int32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::vec_t<6, wp::float32> adj_22 = {};
        //---------
        // forward
        // def set_root_link_velocity_to_sim_index(                                               <L 1>
        // i = wp.tid()                                                                           <L 31>
        var_0 = builtin_tid1d();
        // env_id = wp.int32(env_ids[i])                                                          <L 32>
        var_1 = wp::address(var_env_ids, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::int32(var_3);
        // root_link_velocity_w[env_id] = data[i]                                                 <L 33>
        var_4 = wp::address(var_data, var_0);
        var_5 = wp::load(var_4);
        // wp::array_store(var_root_link_velocity_w, var_2, var_5);
        // root_com_velocity_w[env_id] = get_link_velocity_in_com_frame_func(                     <L 35>
        // root_link_velocity_w[env_id], link_pose_w[env_id], body_com_pos_b[env_id, 0]           <L 36>
        var_6 = wp::address(var_root_link_velocity_w, var_2);
        var_7 = wp::address(var_link_pose_w, var_2);
        var_9 = wp::address(var_body_com_pos_b, var_2, var_8);
        var_11 = wp::load(var_6);
        var_12 = wp::load(var_7);
        var_13 = wp::load(var_9);
        var_10 = get_link_velocity_in_com_frame_func_0(var_11, var_12, var_13);
        // root_com_velocity_w[env_id] = get_link_velocity_in_com_frame_func(                     <L 35>
        // wp::array_store(var_root_com_velocity_w, var_2, var_10);
        // for j in range(num_bodies):                                                            <L 39>
        var_14 = wp::range(var_num_bodies);
        //---------
        // reverse
        var_14 = wp::iter_reverse(var_14);
        start_for_0:;
            if (iter_cmp(var_14) == 0) goto end_for_0;
            var_15 = wp::iter_next(var_14);
        	adj_16 = {};
        	adj_17 = {};
        	adj_18 = {};
        	adj_19 = {};
        	adj_20 = {};
        	adj_21 = {};
        	adj_22 = {};
            // body_acc_w[env_id, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)           <L 40>
            var_22 = wp::vec_t<6, wp::float32>({var_16, var_17, var_18, var_19, var_20, var_21});
            // wp::array_store(var_body_acc_w, var_2, var_15, var_22);
            wp::adj_array_store(var_body_acc_w, var_2, var_15, var_22, adj_body_acc_w, adj_2, adj_15, adj_22);
            wp::adj_vec_t({var_16, var_17, var_18, var_19, var_20, var_21}, {&adj_16, &adj_17, &adj_18, &adj_19, &adj_20, &adj_21}, adj_22);
            // adj: body_acc_w[env_id, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)      <L 40>
        	goto start_for_0;
        end_for_0:;
        // adj: for j in range(num_bodies):                                                       <L 39>
        wp::adj_array_store(var_root_com_velocity_w, var_2, var_10, adj_root_com_velocity_w, adj_2, adj_10);
        // adj: root_com_velocity_w[env_id] = get_link_velocity_in_com_frame_func(                <L 35>
        adj_get_link_velocity_in_com_frame_func_0(var_11, var_12, var_13, adj_6, adj_7, adj_9, adj_10);
        wp::adj_address(var_body_com_pos_b, var_2, var_8, adj_body_com_pos_b, adj_2, adj_8, adj_9);
        wp::adj_address(var_link_pose_w, var_2, adj_link_pose_w, adj_2, adj_7);
        wp::adj_address(var_root_link_velocity_w, var_2, adj_root_link_velocity_w, adj_2, adj_6);
        // adj: root_link_velocity_w[env_id], link_pose_w[env_id], body_com_pos_b[env_id, 0]      <L 36>
        // adj: root_com_velocity_w[env_id] = get_link_velocity_in_com_frame_func(                <L 35>
        wp::adj_array_store(var_root_link_velocity_w, var_2, var_5, adj_root_link_velocity_w, adj_2, adj_4);
        wp::adj_address(var_data, var_0, adj_data, adj_0, adj_4);
        // adj: root_link_velocity_w[env_id] = data[i]                                            <L 33>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_1);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 32>
        // adj: i = wp.tid()                                                                      <L 31>
        // adj: def set_root_link_velocity_to_sim_index(                                          <L 1>
        continue;
    }
}



extern "C" __global__ void body_heading_w_077e1fb3_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_forward_vec,
    wp::array_t<wp::quat_t<wp::float32>> var_quat,
    wp::array_t<wp::float32> var_heading_w)
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
        wp::vec_t<3, wp::float32>* var_2;
        wp::quat_t<wp::float32>* var_3;
        wp::float32 var_4;
        wp::vec_t<3, wp::float32> var_5;
        wp::quat_t<wp::float32> var_6;
        //---------
        // forward
        // def body_heading_w(                                                                    <L 501>
        // i, j = wp.tid()                                                                        <L 517>
        builtin_tid2d(var_0, var_1);
        // heading_w[i, j] = compute_heading_w_func(forward_vec[i, j], quat[i, j])                <L 518>
        var_2 = wp::address(var_forward_vec, var_0, var_1);
        var_3 = wp::address(var_quat, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = compute_heading_w_func_0(var_5, var_6);
        wp::array_store(var_heading_w, var_0, var_1, var_4);
    }
}



extern "C" __global__ void body_heading_w_077e1fb3_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_forward_vec,
    wp::array_t<wp::quat_t<wp::float32>> var_quat,
    wp::array_t<wp::float32> var_heading_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_forward_vec,
    wp::array_t<wp::quat_t<wp::float32>> adj_quat,
    wp::array_t<wp::float32> adj_heading_w)
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
        wp::vec_t<3, wp::float32>* var_2;
        wp::quat_t<wp::float32>* var_3;
        wp::float32 var_4;
        wp::vec_t<3, wp::float32> var_5;
        wp::quat_t<wp::float32> var_6;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::vec_t<3, wp::float32> adj_2 = {};
        wp::quat_t<wp::float32> adj_3 = {};
        wp::float32 adj_4 = {};
        wp::vec_t<3, wp::float32> adj_5 = {};
        wp::quat_t<wp::float32> adj_6 = {};
        //---------
        // forward
        // def body_heading_w(                                                                    <L 501>
        // i, j = wp.tid()                                                                        <L 517>
        builtin_tid2d(var_0, var_1);
        // heading_w[i, j] = compute_heading_w_func(forward_vec[i, j], quat[i, j])                <L 518>
        var_2 = wp::address(var_forward_vec, var_0, var_1);
        var_3 = wp::address(var_quat, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = compute_heading_w_func_0(var_5, var_6);
        // wp::array_store(var_heading_w, var_0, var_1, var_4);
        //---------
        // reverse
        wp::adj_array_store(var_heading_w, var_0, var_1, var_4, adj_heading_w, adj_0, adj_1, adj_4);
        adj_compute_heading_w_func_0(var_5, var_6, adj_2, adj_3, adj_4);
        wp::adj_address(var_quat, var_0, var_1, adj_quat, adj_0, adj_1, adj_3);
        wp::adj_address(var_forward_vec, var_0, var_1, adj_forward_vec, adj_0, adj_1, adj_2);
        // adj: heading_w[i, j] = compute_heading_w_func(forward_vec[i, j], quat[i, j])           <L 518>
        // adj: i, j = wp.tid()                                                                   <L 517>
        // adj: def body_heading_w(                                                               <L 501>
        continue;
    }
}



extern "C" __global__ void write_2d_data_to_buffer_with_indices_08316243_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::array_t<wp::float32> var_out_data)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        //---------
        // forward
        // def write_2d_data_to_buffer_with_indices(                                              <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 19>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 20>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // out_data[env_id, joint_id] = in_data[i, j]                                             <L 21>
        var_8 = wp::address(var_in_data, var_0, var_1);
        var_9 = wp::load(var_8);
        wp::array_store(var_out_data, var_3, var_6, var_9);
    }
}



extern "C" __global__ void write_2d_data_to_buffer_with_indices_08316243_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::array_t<wp::float32> var_out_data,
    wp::array_t<wp::float32> adj_in_data,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int32> adj_joint_ids,
    wp::array_t<wp::float32> adj_out_data)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int64 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int64 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        //---------
        // forward
        // def write_2d_data_to_buffer_with_indices(                                              <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 19>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 20>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // out_data[env_id, joint_id] = in_data[i, j]                                             <L 21>
        var_8 = wp::address(var_in_data, var_0, var_1);
        var_9 = wp::load(var_8);
        // wp::array_store(var_out_data, var_3, var_6, var_9);
        //---------
        // reverse
        wp::adj_array_store(var_out_data, var_3, var_6, var_9, adj_out_data, adj_3, adj_6, adj_8);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_8);
        // adj: out_data[env_id, joint_id] = in_data[i, j]                                        <L 21>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 20>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 19>
        // adj: i, j = wp.tid()                                                                   <L 18>
        // adj: def write_2d_data_to_buffer_with_indices(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void write_body_inertia_and_inverse_index_09f584a1_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_inertias,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    wp::array_t<wp::int32> var_body_user_to_backend,
    bool var_has_body_ordering,
    wp::array_t<wp::float32> var_body_mass_backend,
    wp::array_t<wp::float32> var_body_inertia_user,
    wp::array_t<wp::float32> var_body_inertia_backend,
    wp::array_t<wp::float32> var_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_body_inv_inertia)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 0;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        const wp::int32 var_17 = 1;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        const wp::int32 var_21 = 2;
        wp::float32* var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        const wp::int32 var_25 = 3;
        wp::float32* var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        const wp::int32 var_29 = 4;
        wp::float32* var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        const wp::int32 var_33 = 5;
        wp::float32* var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        const wp::int32 var_37 = 6;
        wp::float32* var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        const wp::int32 var_41 = 7;
        wp::float32* var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        const wp::int32 var_45 = 8;
        wp::float32* var_46;
        wp::float32 var_47;
        wp::float32 var_48;
        const bool var_49 = false;
        //---------
        // forward
        // def write_body_inertia_and_inverse_index(                                              <L 1>
        // i, j = wp.tid()                                                                        <L 14>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 15>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_body_id = wp.int32(body_ids[j])                                                   <L 16>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // backend_body_id = user_body_id                                                         <L 17>
        var_8 = wp::copy(var_6);
        // if has_body_ordering:                                                                  <L 18>
        if (var_has_body_ordering) {
            // backend_body_id = body_user_to_backend[user_body_id]                               <L 19>
            var_9 = wp::address(var_body_user_to_backend, var_6);
            var_11 = wp::load(var_9);
            var_10 = wp::copy(var_11);
        }
        var_12 = wp::where(var_has_body_ordering, var_10, var_8);
        // for k in range(9):                                                                     <L 20>
        // value = inertias[i, j, k]                                                              <L 21>
        var_14 = wp::address(var_inertias, var_0, var_1, var_13);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_13, var_15);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_13, var_15);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_18 = wp::address(var_inertias, var_0, var_1, var_17);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_17, var_19);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_17, var_19);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_22 = wp::address(var_inertias, var_0, var_1, var_21);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_21, var_23);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_21, var_23);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_26 = wp::address(var_inertias, var_0, var_1, var_25);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_25, var_27);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_25, var_27);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_30 = wp::address(var_inertias, var_0, var_1, var_29);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_29, var_31);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_29, var_31);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_34 = wp::address(var_inertias, var_0, var_1, var_33);
        var_36 = wp::load(var_34);
        var_35 = wp::copy(var_36);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_33, var_35);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_33, var_35);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_38 = wp::address(var_inertias, var_0, var_1, var_37);
        var_40 = wp::load(var_38);
        var_39 = wp::copy(var_40);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_37, var_39);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_37, var_39);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_42 = wp::address(var_inertias, var_0, var_1, var_41);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_41, var_43);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_41, var_43);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_46 = wp::address(var_inertias, var_0, var_1, var_45);
        var_48 = wp::load(var_46);
        var_47 = wp::copy(var_48);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_45, var_47);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_45, var_47);
        }
        // _update_body_inertial_properties_inverse(                                              <L 25>
        // env_id, backend_body_id, body_mass_backend, body_inertia_backend, False, body_inv_mass, body_inv_inertia       <L 26>
        _update_body_inertial_properties_inverse_0(var_3, var_12, var_body_mass_backend, var_body_inertia_backend, var_49, var_body_inv_mass, var_body_inv_inertia);
    }
}



extern "C" __global__ void write_body_inertia_and_inverse_index_09f584a1_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_inertias,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    wp::array_t<wp::int32> var_body_user_to_backend,
    bool var_has_body_ordering,
    wp::array_t<wp::float32> var_body_mass_backend,
    wp::array_t<wp::float32> var_body_inertia_user,
    wp::array_t<wp::float32> var_body_inertia_backend,
    wp::array_t<wp::float32> var_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_body_inv_inertia,
    wp::array_t<wp::float32> adj_inertias,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int32> adj_body_ids,
    wp::array_t<wp::int32> adj_body_user_to_backend,
    bool adj_has_body_ordering,
    wp::array_t<wp::float32> adj_body_mass_backend,
    wp::array_t<wp::float32> adj_body_inertia_user,
    wp::array_t<wp::float32> adj_body_inertia_backend,
    wp::array_t<wp::float32> adj_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> adj_body_inv_inertia)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 0;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        const wp::int32 var_17 = 1;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        const wp::int32 var_21 = 2;
        wp::float32* var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        const wp::int32 var_25 = 3;
        wp::float32* var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        const wp::int32 var_29 = 4;
        wp::float32* var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        const wp::int32 var_33 = 5;
        wp::float32* var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        const wp::int32 var_37 = 6;
        wp::float32* var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        const wp::int32 var_41 = 7;
        wp::float32* var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        const wp::int32 var_45 = 8;
        wp::float32* var_46;
        wp::float32 var_47;
        wp::float32 var_48;
        const bool var_49 = false;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int64 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int64 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::float32 adj_39 = {};
        wp::float32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::float32 adj_42 = {};
        wp::float32 adj_43 = {};
        wp::float32 adj_44 = {};
        wp::int32 adj_45 = {};
        wp::float32 adj_46 = {};
        wp::float32 adj_47 = {};
        wp::float32 adj_48 = {};
        bool adj_49 = {};
        //---------
        // forward
        // def write_body_inertia_and_inverse_index(                                              <L 1>
        // i, j = wp.tid()                                                                        <L 14>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 15>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_body_id = wp.int32(body_ids[j])                                                   <L 16>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // backend_body_id = user_body_id                                                         <L 17>
        var_8 = wp::copy(var_6);
        // if has_body_ordering:                                                                  <L 18>
        if (var_has_body_ordering) {
            // backend_body_id = body_user_to_backend[user_body_id]                               <L 19>
            var_9 = wp::address(var_body_user_to_backend, var_6);
            var_11 = wp::load(var_9);
            var_10 = wp::copy(var_11);
        }
        var_12 = wp::where(var_has_body_ordering, var_10, var_8);
        // for k in range(9):                                                                     <L 20>
        // value = inertias[i, j, k]                                                              <L 21>
        var_14 = wp::address(var_inertias, var_0, var_1, var_13);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_13, var_15);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_13, var_15);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_18 = wp::address(var_inertias, var_0, var_1, var_17);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_17, var_19);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_17, var_19);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_22 = wp::address(var_inertias, var_0, var_1, var_21);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_21, var_23);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_21, var_23);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_26 = wp::address(var_inertias, var_0, var_1, var_25);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_25, var_27);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_25, var_27);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_30 = wp::address(var_inertias, var_0, var_1, var_29);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_29, var_31);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_29, var_31);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_34 = wp::address(var_inertias, var_0, var_1, var_33);
        var_36 = wp::load(var_34);
        var_35 = wp::copy(var_36);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_33, var_35);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_33, var_35);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_38 = wp::address(var_inertias, var_0, var_1, var_37);
        var_40 = wp::load(var_38);
        var_39 = wp::copy(var_40);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_37, var_39);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_37, var_39);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_42 = wp::address(var_inertias, var_0, var_1, var_41);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_41, var_43);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_41, var_43);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_46 = wp::address(var_inertias, var_0, var_1, var_45);
        var_48 = wp::load(var_46);
        var_47 = wp::copy(var_48);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_45, var_47);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_45, var_47);
        }
        // _update_body_inertial_properties_inverse(                                              <L 25>
        // env_id, backend_body_id, body_mass_backend, body_inertia_backend, False, body_inv_mass, body_inv_inertia       <L 26>
        _update_body_inertial_properties_inverse_0(var_3, var_12, var_body_mass_backend, var_body_inertia_backend, var_49, var_body_inv_mass, var_body_inv_inertia);
        //---------
        // reverse
        adj__update_body_inertial_properties_inverse_0(var_3, var_12, var_body_mass_backend, var_body_inertia_backend, var_49, var_body_inv_mass, var_body_inv_inertia, adj_3, adj_12, adj_body_mass_backend, adj_body_inertia_backend, adj_49, adj_body_inv_mass, adj_body_inv_inertia);
        // adj: env_id, backend_body_id, body_mass_backend, body_inertia_backend, False, body_inv_mass, body_inv_inertia  <L 26>
        // adj: _update_body_inertial_properties_inverse(                                         <L 25>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_45, var_47, adj_body_inertia_backend, adj_3, adj_12, adj_45, adj_47);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_45, var_47, adj_body_inertia_user, adj_3, adj_6, adj_45, adj_47);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_48, adj_46, adj_47);
        wp::adj_address(var_inertias, var_0, var_1, var_45, adj_inertias, adj_0, adj_1, adj_45, adj_46);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_41, var_43, adj_body_inertia_backend, adj_3, adj_12, adj_41, adj_43);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_41, var_43, adj_body_inertia_user, adj_3, adj_6, adj_41, adj_43);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_44, adj_42, adj_43);
        wp::adj_address(var_inertias, var_0, var_1, var_41, adj_inertias, adj_0, adj_1, adj_41, adj_42);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_37, var_39, adj_body_inertia_backend, adj_3, adj_12, adj_37, adj_39);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_37, var_39, adj_body_inertia_user, adj_3, adj_6, adj_37, adj_39);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_40, adj_38, adj_39);
        wp::adj_address(var_inertias, var_0, var_1, var_37, adj_inertias, adj_0, adj_1, adj_37, adj_38);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_33, var_35, adj_body_inertia_backend, adj_3, adj_12, adj_33, adj_35);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_33, var_35, adj_body_inertia_user, adj_3, adj_6, adj_33, adj_35);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_36, adj_34, adj_35);
        wp::adj_address(var_inertias, var_0, var_1, var_33, adj_inertias, adj_0, adj_1, adj_33, adj_34);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_29, var_31, adj_body_inertia_backend, adj_3, adj_12, adj_29, adj_31);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_29, var_31, adj_body_inertia_user, adj_3, adj_6, adj_29, adj_31);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_32, adj_30, adj_31);
        wp::adj_address(var_inertias, var_0, var_1, var_29, adj_inertias, adj_0, adj_1, adj_29, adj_30);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_25, var_27, adj_body_inertia_backend, adj_3, adj_12, adj_25, adj_27);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_25, var_27, adj_body_inertia_user, adj_3, adj_6, adj_25, adj_27);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_28, adj_26, adj_27);
        wp::adj_address(var_inertias, var_0, var_1, var_25, adj_inertias, adj_0, adj_1, adj_25, adj_26);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_21, var_23, adj_body_inertia_backend, adj_3, adj_12, adj_21, adj_23);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_21, var_23, adj_body_inertia_user, adj_3, adj_6, adj_21, adj_23);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_24, adj_22, adj_23);
        wp::adj_address(var_inertias, var_0, var_1, var_21, adj_inertias, adj_0, adj_1, adj_21, adj_22);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_17, var_19, adj_body_inertia_backend, adj_3, adj_12, adj_17, adj_19);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_17, var_19, adj_body_inertia_user, adj_3, adj_6, adj_17, adj_19);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_20, adj_18, adj_19);
        wp::adj_address(var_inertias, var_0, var_1, var_17, adj_inertias, adj_0, adj_1, adj_17, adj_18);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_13, var_15, adj_body_inertia_backend, adj_3, adj_12, adj_13, adj_15);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_13, var_15, adj_body_inertia_user, adj_3, adj_6, adj_13, adj_15);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_16, adj_14, adj_15);
        wp::adj_address(var_inertias, var_0, var_1, var_13, adj_inertias, adj_0, adj_1, adj_13, adj_14);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        // adj: for k in range(9):                                                                <L 20>
        wp::adj_where(var_has_body_ordering, var_10, var_8, adj_has_body_ordering, adj_10, adj_8, adj_12);
        if (var_has_body_ordering) {
            wp::adj_copy(var_11, adj_9, adj_10);
            wp::adj_address(var_body_user_to_backend, var_6, adj_body_user_to_backend, adj_6, adj_9);
            // adj: backend_body_id = body_user_to_backend[user_body_id]                          <L 19>
        }
        // adj: if has_body_ordering:                                                             <L 18>
        wp::adj_copy(var_6, adj_6, adj_8);
        // adj: backend_body_id = user_body_id                                                    <L 17>
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_5);
        // adj: user_body_id = wp.int32(body_ids[j])                                              <L 16>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 15>
        // adj: i, j = wp.tid()                                                                   <L 14>
        // adj: def write_body_inertia_and_inverse_index(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_body_com_pose_to_sim_1749a69d_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::transform_t<wp::float32>* var_8;
        wp::transform_t<wp::float32> var_9;
        wp::transform_t<wp::float32>* var_10;
        wp::transform_t<wp::float32> var_11;
        wp::transform_t<wp::float32>* var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::transform_t<wp::float32> var_14;
        wp::transform_t<wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        //---------
        // forward
        // def set_body_com_pose_to_sim(                                                          <L 1>
        // i, j = wp.tid()                                                                        <L 29>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 30>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 31>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 32>
        if (var_from_mask) {
            // body_com_pose_w[env_id, body_id] = data[env_id, body_id]                           <L 33>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_body_com_pose_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_com_pose_w[env_id, body_id] = data[i, j]                                      <L 35>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            wp::array_store(var_body_com_pose_w, var_3, var_6, var_11);
        }
        // body_link_pose_w[env_id, body_id] = get_com_pose_in_link_frame_func(                   <L 37>
        // body_com_pose_w[env_id, body_id], body_com_pos_b[env_id, body_id]                      <L 38>
        var_12 = wp::address(var_body_com_pose_w, var_3, var_6);
        var_13 = wp::address(var_body_com_pos_b, var_3, var_6);
        var_15 = wp::load(var_12);
        var_16 = wp::load(var_13);
        var_14 = get_com_pose_in_link_frame_func_0(var_15, var_16);
        // body_link_pose_w[env_id, body_id] = get_com_pose_in_link_frame_func(                   <L 37>
        wp::array_store(var_body_link_pose_w, var_3, var_6, var_14);
    }
}



extern "C" __global__ void set_body_com_pose_to_sim_1749a69d_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_data,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_body_com_pos_b,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int32> adj_body_ids,
    bool adj_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_link_pose_w)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::transform_t<wp::float32>* var_8;
        wp::transform_t<wp::float32> var_9;
        wp::transform_t<wp::float32>* var_10;
        wp::transform_t<wp::float32> var_11;
        wp::transform_t<wp::float32>* var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::transform_t<wp::float32> var_14;
        wp::transform_t<wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int64 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int64 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::transform_t<wp::float32> adj_8 = {};
        wp::transform_t<wp::float32> adj_9 = {};
        wp::transform_t<wp::float32> adj_10 = {};
        wp::transform_t<wp::float32> adj_11 = {};
        wp::transform_t<wp::float32> adj_12 = {};
        wp::vec_t<3, wp::float32> adj_13 = {};
        wp::transform_t<wp::float32> adj_14 = {};
        wp::transform_t<wp::float32> adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        //---------
        // forward
        // def set_body_com_pose_to_sim(                                                          <L 1>
        // i, j = wp.tid()                                                                        <L 29>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 30>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 31>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 32>
        if (var_from_mask) {
            // body_com_pose_w[env_id, body_id] = data[env_id, body_id]                           <L 33>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            // wp::array_store(var_body_com_pose_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_com_pose_w[env_id, body_id] = data[i, j]                                      <L 35>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            // wp::array_store(var_body_com_pose_w, var_3, var_6, var_11);
        }
        // body_link_pose_w[env_id, body_id] = get_com_pose_in_link_frame_func(                   <L 37>
        // body_com_pose_w[env_id, body_id], body_com_pos_b[env_id, body_id]                      <L 38>
        var_12 = wp::address(var_body_com_pose_w, var_3, var_6);
        var_13 = wp::address(var_body_com_pos_b, var_3, var_6);
        var_15 = wp::load(var_12);
        var_16 = wp::load(var_13);
        var_14 = get_com_pose_in_link_frame_func_0(var_15, var_16);
        // body_link_pose_w[env_id, body_id] = get_com_pose_in_link_frame_func(                   <L 37>
        // wp::array_store(var_body_link_pose_w, var_3, var_6, var_14);
        //---------
        // reverse
        wp::adj_array_store(var_body_link_pose_w, var_3, var_6, var_14, adj_body_link_pose_w, adj_3, adj_6, adj_14);
        // adj: body_link_pose_w[env_id, body_id] = get_com_pose_in_link_frame_func(              <L 37>
        adj_get_com_pose_in_link_frame_func_0(var_15, var_16, adj_12, adj_13, adj_14);
        wp::adj_address(var_body_com_pos_b, var_3, var_6, adj_body_com_pos_b, adj_3, adj_6, adj_13);
        wp::adj_address(var_body_com_pose_w, var_3, var_6, adj_body_com_pose_w, adj_3, adj_6, adj_12);
        // adj: body_com_pose_w[env_id, body_id], body_com_pos_b[env_id, body_id]                 <L 38>
        // adj: body_link_pose_w[env_id, body_id] = get_com_pose_in_link_frame_func(              <L 37>
        if (!var_from_mask) {
            wp::adj_array_store(var_body_com_pose_w, var_3, var_6, var_11, adj_body_com_pose_w, adj_3, adj_6, adj_10);
            wp::adj_address(var_data, var_0, var_1, adj_data, adj_0, adj_1, adj_10);
            // adj: body_com_pose_w[env_id, body_id] = data[i, j]                                 <L 35>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_body_com_pose_w, var_3, var_6, var_9, adj_body_com_pose_w, adj_3, adj_6, adj_8);
            wp::adj_address(var_data, var_3, var_6, adj_data, adj_3, adj_6, adj_8);
            // adj: body_com_pose_w[env_id, body_id] = data[env_id, body_id]                      <L 33>
        }
        // adj: if from_mask:                                                                     <L 32>
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_5);
        // adj: body_id = wp.int32(body_ids[j])                                                   <L 31>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 30>
        // adj: i, j = wp.tid()                                                                   <L 29>
        // adj: def set_body_com_pose_to_sim(                                                     <L 1>
        continue;
    }
}



extern "C" __global__ void set_body_link_pose_to_sim_197cd7a8_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::transform_t<wp::float32>* var_8;
        wp::transform_t<wp::float32> var_9;
        wp::transform_t<wp::float32>* var_10;
        wp::transform_t<wp::float32> var_11;
        //---------
        // forward
        // def set_body_link_pose_to_sim(                                                         <L 1>
        // i, j = wp.tid()                                                                        <L 22>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 23>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 24>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 25>
        if (var_from_mask) {
            // body_link_pose_w[env_id, body_id] = data[env_id, body_id]                          <L 26>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_body_link_pose_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_link_pose_w[env_id, body_id] = data[i, j]                                     <L 28>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            wp::array_store(var_body_link_pose_w, var_3, var_6, var_11);
        }
    }
}



extern "C" __global__ void set_body_link_pose_to_sim_197cd7a8_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int64> adj_body_ids,
    bool adj_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_link_pose_w)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::transform_t<wp::float32>* var_8;
        wp::transform_t<wp::float32> var_9;
        wp::transform_t<wp::float32>* var_10;
        wp::transform_t<wp::float32> var_11;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int64 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int64 adj_7 = {};
        wp::transform_t<wp::float32> adj_8 = {};
        wp::transform_t<wp::float32> adj_9 = {};
        wp::transform_t<wp::float32> adj_10 = {};
        wp::transform_t<wp::float32> adj_11 = {};
        //---------
        // forward
        // def set_body_link_pose_to_sim(                                                         <L 1>
        // i, j = wp.tid()                                                                        <L 22>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 23>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 24>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 25>
        if (var_from_mask) {
            // body_link_pose_w[env_id, body_id] = data[env_id, body_id]                          <L 26>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            // wp::array_store(var_body_link_pose_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_link_pose_w[env_id, body_id] = data[i, j]                                     <L 28>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            // wp::array_store(var_body_link_pose_w, var_3, var_6, var_11);
        }
        //---------
        // reverse
        if (!var_from_mask) {
            wp::adj_array_store(var_body_link_pose_w, var_3, var_6, var_11, adj_body_link_pose_w, adj_3, adj_6, adj_10);
            wp::adj_address(var_data, var_0, var_1, adj_data, adj_0, adj_1, adj_10);
            // adj: body_link_pose_w[env_id, body_id] = data[i, j]                                <L 28>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_body_link_pose_w, var_3, var_6, var_9, adj_body_link_pose_w, adj_3, adj_6, adj_8);
            wp::adj_address(var_data, var_3, var_6, adj_data, adj_3, adj_6, adj_8);
            // adj: body_link_pose_w[env_id, body_id] = data[env_id, body_id]                     <L 26>
        }
        // adj: if from_mask:                                                                     <L 25>
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_5);
        // adj: body_id = wp.int32(body_ids[j])                                                   <L 24>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 23>
        // adj: i, j = wp.tid()                                                                   <L 22>
        // adj: def set_body_link_pose_to_sim(                                                    <L 1>
        continue;
    }
}



extern "C" __global__ void concat_body_pose_and_vel_to_state_1de6a656_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_pose,
    wp::array_t<wp::vec_t<6, wp::float32>> var_vel,
    wp::array_t<wp::vec_t<13, wp::float32>> var_state)
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
        wp::transform_t<wp::float32>* var_2;
        wp::vec_t<6, wp::float32>* var_3;
        wp::vec_t<13, wp::float32> var_4;
        wp::transform_t<wp::float32> var_5;
        wp::vec_t<6, wp::float32> var_6;
        //---------
        // forward
        // def concat_body_pose_and_vel_to_state(                                                 <L 374>
        // i, j = wp.tid()                                                                        <L 390>
        builtin_tid2d(var_0, var_1);
        // state[i, j] = concat_pose_and_vel_to_state_func(pose[i, j], vel[i, j])                 <L 391>
        var_2 = wp::address(var_pose, var_0, var_1);
        var_3 = wp::address(var_vel, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = concat_pose_and_vel_to_state_func_0(var_5, var_6);
        wp::array_store(var_state, var_0, var_1, var_4);
    }
}



extern "C" __global__ void concat_body_pose_and_vel_to_state_1de6a656_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_pose,
    wp::array_t<wp::vec_t<6, wp::float32>> var_vel,
    wp::array_t<wp::vec_t<13, wp::float32>> var_state,
    wp::array_t<wp::transform_t<wp::float32>> adj_pose,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_vel,
    wp::array_t<wp::vec_t<13, wp::float32>> adj_state)
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
        wp::transform_t<wp::float32>* var_2;
        wp::vec_t<6, wp::float32>* var_3;
        wp::vec_t<13, wp::float32> var_4;
        wp::transform_t<wp::float32> var_5;
        wp::vec_t<6, wp::float32> var_6;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::transform_t<wp::float32> adj_2 = {};
        wp::vec_t<6, wp::float32> adj_3 = {};
        wp::vec_t<13, wp::float32> adj_4 = {};
        wp::transform_t<wp::float32> adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        //---------
        // forward
        // def concat_body_pose_and_vel_to_state(                                                 <L 374>
        // i, j = wp.tid()                                                                        <L 390>
        builtin_tid2d(var_0, var_1);
        // state[i, j] = concat_pose_and_vel_to_state_func(pose[i, j], vel[i, j])                 <L 391>
        var_2 = wp::address(var_pose, var_0, var_1);
        var_3 = wp::address(var_vel, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = concat_pose_and_vel_to_state_func_0(var_5, var_6);
        // wp::array_store(var_state, var_0, var_1, var_4);
        //---------
        // reverse
        wp::adj_array_store(var_state, var_0, var_1, var_4, adj_state, adj_0, adj_1, adj_4);
        adj_concat_pose_and_vel_to_state_func_0(var_5, var_6, adj_2, adj_3, adj_4);
        wp::adj_address(var_vel, var_0, var_1, adj_vel, adj_0, adj_1, adj_3);
        wp::adj_address(var_pose, var_0, var_1, adj_pose, adj_0, adj_1, adj_2);
        // adj: state[i, j] = concat_pose_and_vel_to_state_func(pose[i, j], vel[i, j])            <L 391>
        // adj: i, j = wp.tid()                                                                   <L 390>
        // adj: def concat_body_pose_and_vel_to_state(                                            <L 374>
        continue;
    }
}



extern "C" __global__ void get_body_link_vel_from_body_com_vel_226e289c_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_vel,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_link_vel)
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
        wp::vec_t<6, wp::float32>* var_2;
        wp::transform_t<wp::float32>* var_3;
        wp::vec_t<3, wp::float32>* var_4;
        wp::vec_t<6, wp::float32> var_5;
        wp::vec_t<6, wp::float32> var_6;
        wp::transform_t<wp::float32> var_7;
        wp::vec_t<3, wp::float32> var_8;
        //---------
        // forward
        // def get_body_link_vel_from_body_com_vel(                                               <L 330>
        // i, j = wp.tid()                                                                        <L 347>
        builtin_tid2d(var_0, var_1);
        // body_link_vel[i, j] = get_link_vel_from_root_com_vel_func(                             <L 348>
        // body_com_vel[i, j], body_link_pose[i, j], body_com_pos_b[i, j]                         <L 349>
        var_2 = wp::address(var_body_com_vel, var_0, var_1);
        var_3 = wp::address(var_body_link_pose, var_0, var_1);
        var_4 = wp::address(var_body_com_pos_b, var_0, var_1);
        var_6 = wp::load(var_2);
        var_7 = wp::load(var_3);
        var_8 = wp::load(var_4);
        var_5 = get_link_vel_from_root_com_vel_func_0(var_6, var_7, var_8);
        // body_link_vel[i, j] = get_link_vel_from_root_com_vel_func(                             <L 348>
        wp::array_store(var_body_link_vel, var_0, var_1, var_5);
    }
}



extern "C" __global__ void get_body_link_vel_from_body_com_vel_226e289c_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_vel,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_link_vel,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_com_vel,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_link_pose,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_body_com_pos_b,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_link_vel)
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
        wp::vec_t<6, wp::float32>* var_2;
        wp::transform_t<wp::float32>* var_3;
        wp::vec_t<3, wp::float32>* var_4;
        wp::vec_t<6, wp::float32> var_5;
        wp::vec_t<6, wp::float32> var_6;
        wp::transform_t<wp::float32> var_7;
        wp::vec_t<3, wp::float32> var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::vec_t<6, wp::float32> adj_2 = {};
        wp::transform_t<wp::float32> adj_3 = {};
        wp::vec_t<3, wp::float32> adj_4 = {};
        wp::vec_t<6, wp::float32> adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        wp::transform_t<wp::float32> adj_7 = {};
        wp::vec_t<3, wp::float32> adj_8 = {};
        //---------
        // forward
        // def get_body_link_vel_from_body_com_vel(                                               <L 330>
        // i, j = wp.tid()                                                                        <L 347>
        builtin_tid2d(var_0, var_1);
        // body_link_vel[i, j] = get_link_vel_from_root_com_vel_func(                             <L 348>
        // body_com_vel[i, j], body_link_pose[i, j], body_com_pos_b[i, j]                         <L 349>
        var_2 = wp::address(var_body_com_vel, var_0, var_1);
        var_3 = wp::address(var_body_link_pose, var_0, var_1);
        var_4 = wp::address(var_body_com_pos_b, var_0, var_1);
        var_6 = wp::load(var_2);
        var_7 = wp::load(var_3);
        var_8 = wp::load(var_4);
        var_5 = get_link_vel_from_root_com_vel_func_0(var_6, var_7, var_8);
        // body_link_vel[i, j] = get_link_vel_from_root_com_vel_func(                             <L 348>
        // wp::array_store(var_body_link_vel, var_0, var_1, var_5);
        //---------
        // reverse
        wp::adj_array_store(var_body_link_vel, var_0, var_1, var_5, adj_body_link_vel, adj_0, adj_1, adj_5);
        // adj: body_link_vel[i, j] = get_link_vel_from_root_com_vel_func(                        <L 348>
        adj_get_link_vel_from_root_com_vel_func_0(var_6, var_7, var_8, adj_2, adj_3, adj_4, adj_5);
        wp::adj_address(var_body_com_pos_b, var_0, var_1, adj_body_com_pos_b, adj_0, adj_1, adj_4);
        wp::adj_address(var_body_link_pose, var_0, var_1, adj_body_link_pose, adj_0, adj_1, adj_3);
        wp::adj_address(var_body_com_vel, var_0, var_1, adj_body_com_vel, adj_0, adj_1, adj_2);
        // adj: body_com_vel[i, j], body_link_pose[i, j], body_com_pos_b[i, j]                    <L 349>
        // adj: body_link_vel[i, j] = get_link_vel_from_root_com_vel_func(                        <L 348>
        // adj: i, j = wp.tid()                                                                   <L 347>
        // adj: def get_body_link_vel_from_body_com_vel(                                          <L 330>
        continue;
    }
}



extern "C" __global__ void set_body_com_velocity_to_sim_2274b78c_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::vec_t<6, wp::float32>* var_8;
        wp::vec_t<6, wp::float32> var_9;
        wp::vec_t<6, wp::float32>* var_10;
        wp::vec_t<6, wp::float32> var_11;
        const wp::float32 var_12 = 0.0;
        const wp::float32 var_13 = 0.0;
        const wp::float32 var_14 = 0.0;
        const wp::float32 var_15 = 0.0;
        const wp::float32 var_16 = 0.0;
        const wp::float32 var_17 = 0.0;
        wp::vec_t<6, wp::float32> var_18;
        //---------
        // forward
        // def set_body_com_velocity_to_sim(                                                      <L 1>
        // i, j = wp.tid()                                                                        <L 26>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 27>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 28>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 29>
        if (var_from_mask) {
            // body_com_velocity_w[env_id, body_id] = data[env_id, body_id]                       <L 30>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_body_com_velocity_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_com_velocity_w[env_id, body_id] = data[i, j]                                  <L 32>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            wp::array_store(var_body_com_velocity_w, var_3, var_6, var_11);
        }
        // body_acc_w[env_id, body_id] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)         <L 34>
        var_18 = wp::vec_t<6, wp::float32>({var_12, var_13, var_14, var_15, var_16, var_17});
        wp::array_store(var_body_acc_w, var_3, var_6, var_18);
    }
}



extern "C" __global__ void set_body_com_velocity_to_sim_2274b78c_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int64> adj_body_ids,
    bool adj_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_acc_w)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::vec_t<6, wp::float32>* var_8;
        wp::vec_t<6, wp::float32> var_9;
        wp::vec_t<6, wp::float32>* var_10;
        wp::vec_t<6, wp::float32> var_11;
        const wp::float32 var_12 = 0.0;
        const wp::float32 var_13 = 0.0;
        const wp::float32 var_14 = 0.0;
        const wp::float32 var_15 = 0.0;
        const wp::float32 var_16 = 0.0;
        const wp::float32 var_17 = 0.0;
        wp::vec_t<6, wp::float32> var_18;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int64 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int64 adj_7 = {};
        wp::vec_t<6, wp::float32> adj_8 = {};
        wp::vec_t<6, wp::float32> adj_9 = {};
        wp::vec_t<6, wp::float32> adj_10 = {};
        wp::vec_t<6, wp::float32> adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::vec_t<6, wp::float32> adj_18 = {};
        //---------
        // forward
        // def set_body_com_velocity_to_sim(                                                      <L 1>
        // i, j = wp.tid()                                                                        <L 26>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 27>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 28>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 29>
        if (var_from_mask) {
            // body_com_velocity_w[env_id, body_id] = data[env_id, body_id]                       <L 30>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            // wp::array_store(var_body_com_velocity_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_com_velocity_w[env_id, body_id] = data[i, j]                                  <L 32>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            // wp::array_store(var_body_com_velocity_w, var_3, var_6, var_11);
        }
        // body_acc_w[env_id, body_id] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)         <L 34>
        var_18 = wp::vec_t<6, wp::float32>({var_12, var_13, var_14, var_15, var_16, var_17});
        // wp::array_store(var_body_acc_w, var_3, var_6, var_18);
        //---------
        // reverse
        wp::adj_array_store(var_body_acc_w, var_3, var_6, var_18, adj_body_acc_w, adj_3, adj_6, adj_18);
        wp::adj_vec_t({var_12, var_13, var_14, var_15, var_16, var_17}, {&adj_12, &adj_13, &adj_14, &adj_15, &adj_16, &adj_17}, adj_18);
        // adj: body_acc_w[env_id, body_id] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)    <L 34>
        if (!var_from_mask) {
            wp::adj_array_store(var_body_com_velocity_w, var_3, var_6, var_11, adj_body_com_velocity_w, adj_3, adj_6, adj_10);
            wp::adj_address(var_data, var_0, var_1, adj_data, adj_0, adj_1, adj_10);
            // adj: body_com_velocity_w[env_id, body_id] = data[i, j]                             <L 32>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_body_com_velocity_w, var_3, var_6, var_9, adj_body_com_velocity_w, adj_3, adj_6, adj_8);
            wp::adj_address(var_data, var_3, var_6, adj_data, adj_3, adj_6, adj_8);
            // adj: body_com_velocity_w[env_id, body_id] = data[env_id, body_id]                  <L 30>
        }
        // adj: if from_mask:                                                                     <L 29>
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_5);
        // adj: body_id = wp.int32(body_ids[j])                                                   <L 28>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 27>
        // adj: i, j = wp.tid()                                                                   <L 26>
        // adj: def set_body_com_velocity_to_sim(                                                 <L 1>
        continue;
    }
}



extern "C" __global__ void write_body_mass_and_inverse_mask_290c2bef_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_masses,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_body_mask,
    wp::array_t<wp::int32> var_body_user_to_backend,
    bool var_has_body_ordering,
    wp::array_t<wp::float32> var_body_inertia_backend,
    wp::array_t<wp::float32> var_body_mass_user,
    wp::array_t<wp::float32> var_body_mass_backend,
    wp::array_t<wp::float32> var_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_body_inv_inertia)
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
        bool var_2;
        bool* var_3;
        bool var_4;
        bool var_5;
        bool* var_6;
        bool var_7;
        bool var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        const bool var_17 = true;
        //---------
        // forward
        // def write_body_mass_and_inverse_mask(                                                  <L 1135>
        // env_id, user_body_id = wp.tid()                                                        <L 1161>
        builtin_tid2d(var_0, var_1);
        // if not env_mask[env_id] or not body_mask[user_body_id]:                                <L 1162>
        var_3 = wp::address(var_env_mask, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::unot(var_5);
        var_2 = var_4;
        if (!var_2) {
            var_6 = wp::address(var_body_mask, var_1);
            var_8 = wp::load(var_6);
            var_7 = wp::unot(var_8);
            var_2 = var_2 || var_7;
        }
        if (var_2) {
            // return                                                                             <L 1163>
            continue;
        }
        // backend_body_id = user_body_id                                                         <L 1164>
        var_9 = wp::copy(var_1);
        // if has_body_ordering:                                                                  <L 1165>
        if (var_has_body_ordering) {
            // backend_body_id = body_user_to_backend[user_body_id]                               <L 1166>
            var_10 = wp::address(var_body_user_to_backend, var_1);
            var_12 = wp::load(var_10);
            var_11 = wp::copy(var_12);
        }
        var_13 = wp::where(var_has_body_ordering, var_11, var_9);
        // mass = masses[env_id, user_body_id]                                                    <L 1167>
        var_14 = wp::address(var_masses, var_0, var_1);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // body_mass_user[env_id, user_body_id] = mass                                            <L 1168>
        wp::array_store(var_body_mass_user, var_0, var_1, var_15);
        // if has_body_ordering:                                                                  <L 1169>
        if (var_has_body_ordering) {
            // body_mass_backend[env_id, backend_body_id] = mass                                  <L 1170>
            wp::array_store(var_body_mass_backend, var_0, var_13, var_15);
        }
        // _update_body_inertial_properties_inverse(                                              <L 1171>
        // env_id, backend_body_id, body_mass_backend, body_inertia_backend, True, body_inv_mass, body_inv_inertia       <L 1172>
        _update_body_inertial_properties_inverse_0(var_0, var_13, var_body_mass_backend, var_body_inertia_backend, var_17, var_body_inv_mass, var_body_inv_inertia);
    }
}



extern "C" __global__ void write_body_mass_and_inverse_mask_290c2bef_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_masses,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_body_mask,
    wp::array_t<wp::int32> var_body_user_to_backend,
    bool var_has_body_ordering,
    wp::array_t<wp::float32> var_body_inertia_backend,
    wp::array_t<wp::float32> var_body_mass_user,
    wp::array_t<wp::float32> var_body_mass_backend,
    wp::array_t<wp::float32> var_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_body_inv_inertia,
    wp::array_t<wp::float32> adj_masses,
    wp::array_t<bool> adj_env_mask,
    wp::array_t<bool> adj_body_mask,
    wp::array_t<wp::int32> adj_body_user_to_backend,
    bool adj_has_body_ordering,
    wp::array_t<wp::float32> adj_body_inertia_backend,
    wp::array_t<wp::float32> adj_body_mass_user,
    wp::array_t<wp::float32> adj_body_mass_backend,
    wp::array_t<wp::float32> adj_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> adj_body_inv_inertia)
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
        bool var_2;
        bool* var_3;
        bool var_4;
        bool var_5;
        bool* var_6;
        bool var_7;
        bool var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        const bool var_17 = true;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        bool adj_6 = {};
        bool adj_7 = {};
        bool adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float32 adj_16 = {};
        bool adj_17 = {};
        //---------
        // forward
        // def write_body_mass_and_inverse_mask(                                                  <L 1135>
        // env_id, user_body_id = wp.tid()                                                        <L 1161>
        builtin_tid2d(var_0, var_1);
        // if not env_mask[env_id] or not body_mask[user_body_id]:                                <L 1162>
        var_3 = wp::address(var_env_mask, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::unot(var_5);
        var_2 = var_4;
        if (!var_2) {
            var_6 = wp::address(var_body_mask, var_1);
            var_8 = wp::load(var_6);
            var_7 = wp::unot(var_8);
            var_2 = var_2 || var_7;
        }
        if (var_2) {
            // return                                                                             <L 1163>
            goto label0;
        }
        // backend_body_id = user_body_id                                                         <L 1164>
        var_9 = wp::copy(var_1);
        // if has_body_ordering:                                                                  <L 1165>
        if (var_has_body_ordering) {
            // backend_body_id = body_user_to_backend[user_body_id]                               <L 1166>
            var_10 = wp::address(var_body_user_to_backend, var_1);
            var_12 = wp::load(var_10);
            var_11 = wp::copy(var_12);
        }
        var_13 = wp::where(var_has_body_ordering, var_11, var_9);
        // mass = masses[env_id, user_body_id]                                                    <L 1167>
        var_14 = wp::address(var_masses, var_0, var_1);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // body_mass_user[env_id, user_body_id] = mass                                            <L 1168>
        // wp::array_store(var_body_mass_user, var_0, var_1, var_15);
        // if has_body_ordering:                                                                  <L 1169>
        if (var_has_body_ordering) {
            // body_mass_backend[env_id, backend_body_id] = mass                                  <L 1170>
            // wp::array_store(var_body_mass_backend, var_0, var_13, var_15);
        }
        // _update_body_inertial_properties_inverse(                                              <L 1171>
        // env_id, backend_body_id, body_mass_backend, body_inertia_backend, True, body_inv_mass, body_inv_inertia       <L 1172>
        _update_body_inertial_properties_inverse_0(var_0, var_13, var_body_mass_backend, var_body_inertia_backend, var_17, var_body_inv_mass, var_body_inv_inertia);
        //---------
        // reverse
        adj__update_body_inertial_properties_inverse_0(var_0, var_13, var_body_mass_backend, var_body_inertia_backend, var_17, var_body_inv_mass, var_body_inv_inertia, adj_0, adj_13, adj_body_mass_backend, adj_body_inertia_backend, adj_17, adj_body_inv_mass, adj_body_inv_inertia);
        // adj: env_id, backend_body_id, body_mass_backend, body_inertia_backend, True, body_inv_mass, body_inv_inertia  <L 1172>
        // adj: _update_body_inertial_properties_inverse(                                         <L 1171>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_mass_backend, var_0, var_13, var_15, adj_body_mass_backend, adj_0, adj_13, adj_15);
            // adj: body_mass_backend[env_id, backend_body_id] = mass                             <L 1170>
        }
        // adj: if has_body_ordering:                                                             <L 1169>
        wp::adj_array_store(var_body_mass_user, var_0, var_1, var_15, adj_body_mass_user, adj_0, adj_1, adj_15);
        // adj: body_mass_user[env_id, user_body_id] = mass                                       <L 1168>
        wp::adj_copy(var_16, adj_14, adj_15);
        wp::adj_address(var_masses, var_0, var_1, adj_masses, adj_0, adj_1, adj_14);
        // adj: mass = masses[env_id, user_body_id]                                               <L 1167>
        wp::adj_where(var_has_body_ordering, var_11, var_9, adj_has_body_ordering, adj_11, adj_9, adj_13);
        if (var_has_body_ordering) {
            wp::adj_copy(var_12, adj_10, adj_11);
            wp::adj_address(var_body_user_to_backend, var_1, adj_body_user_to_backend, adj_1, adj_10);
            // adj: backend_body_id = body_user_to_backend[user_body_id]                          <L 1166>
        }
        // adj: if has_body_ordering:                                                             <L 1165>
        wp::adj_copy(var_1, adj_1, adj_9);
        // adj: backend_body_id = user_body_id                                                    <L 1164>
        if (var_2) {
            label0:;
            // adj: return                                                                        <L 1163>
        }
        if (!var_2) {
            wp::adj_address(var_body_mask, var_1, adj_body_mask, adj_1, adj_6);
        }
        wp::adj_address(var_env_mask, var_0, adj_env_mask, adj_0, adj_3);
        // adj: if not env_mask[env_id] or not body_mask[user_body_id]:                           <L 1162>
        // adj: env_id, user_body_id = wp.tid()                                                   <L 1161>
        // adj: def write_body_mass_and_inverse_mask(                                             <L 1135>
        continue;
    }
}



extern "C" __global__ void set_body_com_velocity_to_sim_2d9b3f32_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::vec_t<6, wp::float32>* var_8;
        wp::vec_t<6, wp::float32> var_9;
        wp::vec_t<6, wp::float32>* var_10;
        wp::vec_t<6, wp::float32> var_11;
        const wp::float32 var_12 = 0.0;
        const wp::float32 var_13 = 0.0;
        const wp::float32 var_14 = 0.0;
        const wp::float32 var_15 = 0.0;
        const wp::float32 var_16 = 0.0;
        const wp::float32 var_17 = 0.0;
        wp::vec_t<6, wp::float32> var_18;
        //---------
        // forward
        // def set_body_com_velocity_to_sim(                                                      <L 1>
        // i, j = wp.tid()                                                                        <L 26>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 27>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 28>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 29>
        if (var_from_mask) {
            // body_com_velocity_w[env_id, body_id] = data[env_id, body_id]                       <L 30>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_body_com_velocity_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_com_velocity_w[env_id, body_id] = data[i, j]                                  <L 32>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            wp::array_store(var_body_com_velocity_w, var_3, var_6, var_11);
        }
        // body_acc_w[env_id, body_id] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)         <L 34>
        var_18 = wp::vec_t<6, wp::float32>({var_12, var_13, var_14, var_15, var_16, var_17});
        wp::array_store(var_body_acc_w, var_3, var_6, var_18);
    }
}



extern "C" __global__ void set_body_com_velocity_to_sim_2d9b3f32_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_data,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int64> adj_body_ids,
    bool adj_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_acc_w)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::vec_t<6, wp::float32>* var_8;
        wp::vec_t<6, wp::float32> var_9;
        wp::vec_t<6, wp::float32>* var_10;
        wp::vec_t<6, wp::float32> var_11;
        const wp::float32 var_12 = 0.0;
        const wp::float32 var_13 = 0.0;
        const wp::float32 var_14 = 0.0;
        const wp::float32 var_15 = 0.0;
        const wp::float32 var_16 = 0.0;
        const wp::float32 var_17 = 0.0;
        wp::vec_t<6, wp::float32> var_18;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int64 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int64 adj_4 = {};
        wp::int64 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int64 adj_7 = {};
        wp::vec_t<6, wp::float32> adj_8 = {};
        wp::vec_t<6, wp::float32> adj_9 = {};
        wp::vec_t<6, wp::float32> adj_10 = {};
        wp::vec_t<6, wp::float32> adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::vec_t<6, wp::float32> adj_18 = {};
        //---------
        // forward
        // def set_body_com_velocity_to_sim(                                                      <L 1>
        // i, j = wp.tid()                                                                        <L 26>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 27>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 28>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 29>
        if (var_from_mask) {
            // body_com_velocity_w[env_id, body_id] = data[env_id, body_id]                       <L 30>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            // wp::array_store(var_body_com_velocity_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_com_velocity_w[env_id, body_id] = data[i, j]                                  <L 32>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            // wp::array_store(var_body_com_velocity_w, var_3, var_6, var_11);
        }
        // body_acc_w[env_id, body_id] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)         <L 34>
        var_18 = wp::vec_t<6, wp::float32>({var_12, var_13, var_14, var_15, var_16, var_17});
        // wp::array_store(var_body_acc_w, var_3, var_6, var_18);
        //---------
        // reverse
        wp::adj_array_store(var_body_acc_w, var_3, var_6, var_18, adj_body_acc_w, adj_3, adj_6, adj_18);
        wp::adj_vec_t({var_12, var_13, var_14, var_15, var_16, var_17}, {&adj_12, &adj_13, &adj_14, &adj_15, &adj_16, &adj_17}, adj_18);
        // adj: body_acc_w[env_id, body_id] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)    <L 34>
        if (!var_from_mask) {
            wp::adj_array_store(var_body_com_velocity_w, var_3, var_6, var_11, adj_body_com_velocity_w, adj_3, adj_6, adj_10);
            wp::adj_address(var_data, var_0, var_1, adj_data, adj_0, adj_1, adj_10);
            // adj: body_com_velocity_w[env_id, body_id] = data[i, j]                             <L 32>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_body_com_velocity_w, var_3, var_6, var_9, adj_body_com_velocity_w, adj_3, adj_6, adj_8);
            wp::adj_address(var_data, var_3, var_6, adj_data, adj_3, adj_6, adj_8);
            // adj: body_com_velocity_w[env_id, body_id] = data[env_id, body_id]                  <L 30>
        }
        // adj: if from_mask:                                                                     <L 29>
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_5);
        // adj: body_id = wp.int32(body_ids[j])                                                   <L 28>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 27>
        // adj: i, j = wp.tid()                                                                   <L 26>
        // adj: def set_body_com_velocity_to_sim(                                                 <L 1>
        continue;
    }
}



extern "C" __global__ void set_body_com_velocity_to_sim_392ae9a8_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w)
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
        wp::vec_t<6, wp::float32>* var_8;
        wp::vec_t<6, wp::float32> var_9;
        wp::vec_t<6, wp::float32>* var_10;
        wp::vec_t<6, wp::float32> var_11;
        const wp::float32 var_12 = 0.0;
        const wp::float32 var_13 = 0.0;
        const wp::float32 var_14 = 0.0;
        const wp::float32 var_15 = 0.0;
        const wp::float32 var_16 = 0.0;
        const wp::float32 var_17 = 0.0;
        wp::vec_t<6, wp::float32> var_18;
        //---------
        // forward
        // def set_body_com_velocity_to_sim(                                                      <L 1>
        // i, j = wp.tid()                                                                        <L 26>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 27>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 28>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 29>
        if (var_from_mask) {
            // body_com_velocity_w[env_id, body_id] = data[env_id, body_id]                       <L 30>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_body_com_velocity_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_com_velocity_w[env_id, body_id] = data[i, j]                                  <L 32>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            wp::array_store(var_body_com_velocity_w, var_3, var_6, var_11);
        }
        // body_acc_w[env_id, body_id] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)         <L 34>
        var_18 = wp::vec_t<6, wp::float32>({var_12, var_13, var_14, var_15, var_16, var_17});
        wp::array_store(var_body_acc_w, var_3, var_6, var_18);
    }
}



extern "C" __global__ void set_body_com_velocity_to_sim_392ae9a8_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_body_ids,
    bool adj_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_acc_w)
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
        wp::vec_t<6, wp::float32>* var_8;
        wp::vec_t<6, wp::float32> var_9;
        wp::vec_t<6, wp::float32>* var_10;
        wp::vec_t<6, wp::float32> var_11;
        const wp::float32 var_12 = 0.0;
        const wp::float32 var_13 = 0.0;
        const wp::float32 var_14 = 0.0;
        const wp::float32 var_15 = 0.0;
        const wp::float32 var_16 = 0.0;
        const wp::float32 var_17 = 0.0;
        wp::vec_t<6, wp::float32> var_18;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::vec_t<6, wp::float32> adj_8 = {};
        wp::vec_t<6, wp::float32> adj_9 = {};
        wp::vec_t<6, wp::float32> adj_10 = {};
        wp::vec_t<6, wp::float32> adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::vec_t<6, wp::float32> adj_18 = {};
        //---------
        // forward
        // def set_body_com_velocity_to_sim(                                                      <L 1>
        // i, j = wp.tid()                                                                        <L 26>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 27>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 28>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 29>
        if (var_from_mask) {
            // body_com_velocity_w[env_id, body_id] = data[env_id, body_id]                       <L 30>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            // wp::array_store(var_body_com_velocity_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_com_velocity_w[env_id, body_id] = data[i, j]                                  <L 32>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            // wp::array_store(var_body_com_velocity_w, var_3, var_6, var_11);
        }
        // body_acc_w[env_id, body_id] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)         <L 34>
        var_18 = wp::vec_t<6, wp::float32>({var_12, var_13, var_14, var_15, var_16, var_17});
        // wp::array_store(var_body_acc_w, var_3, var_6, var_18);
        //---------
        // reverse
        wp::adj_array_store(var_body_acc_w, var_3, var_6, var_18, adj_body_acc_w, adj_3, adj_6, adj_18);
        wp::adj_vec_t({var_12, var_13, var_14, var_15, var_16, var_17}, {&adj_12, &adj_13, &adj_14, &adj_15, &adj_16, &adj_17}, adj_18);
        // adj: body_acc_w[env_id, body_id] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)    <L 34>
        if (!var_from_mask) {
            wp::adj_array_store(var_body_com_velocity_w, var_3, var_6, var_11, adj_body_com_velocity_w, adj_3, adj_6, adj_10);
            wp::adj_address(var_data, var_0, var_1, adj_data, adj_0, adj_1, adj_10);
            // adj: body_com_velocity_w[env_id, body_id] = data[i, j]                             <L 32>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_body_com_velocity_w, var_3, var_6, var_9, adj_body_com_velocity_w, adj_3, adj_6, adj_8);
            wp::adj_address(var_data, var_3, var_6, adj_data, adj_3, adj_6, adj_8);
            // adj: body_com_velocity_w[env_id, body_id] = data[env_id, body_id]                  <L 30>
        }
        // adj: if from_mask:                                                                     <L 29>
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_5);
        // adj: body_id = wp.int32(body_ids[j])                                                   <L 28>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 27>
        // adj: i, j = wp.tid()                                                                   <L 26>
        // adj: def set_body_com_velocity_to_sim(                                                 <L 1>
        continue;
    }
}



extern "C" __global__ void set_root_link_velocity_to_sim_mask_3cfbbb23_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> var_link_pose_w,
    wp::array_t<bool> var_env_mask,
    wp::int32 var_num_bodies,
    wp::array_t<wp::vec_t<6, wp::float32>> var_root_link_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_root_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w)
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
        bool* var_1;
        bool var_2;
        wp::vec_t<6, wp::float32>* var_3;
        wp::vec_t<6, wp::float32> var_4;
        wp::vec_t<6, wp::float32>* var_5;
        wp::transform_t<wp::float32>* var_6;
        const wp::int32 var_7 = 0;
        wp::vec_t<3, wp::float32>* var_8;
        wp::vec_t<6, wp::float32> var_9;
        wp::vec_t<6, wp::float32> var_10;
        wp::transform_t<wp::float32> var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::range_t var_13;
        wp::int32 var_14;
        const wp::float32 var_15 = 0.0;
        const wp::float32 var_16 = 0.0;
        const wp::float32 var_17 = 0.0;
        const wp::float32 var_18 = 0.0;
        const wp::float32 var_19 = 0.0;
        const wp::float32 var_20 = 0.0;
        wp::vec_t<6, wp::float32> var_21;
        bool var_22;
        //---------
        // forward
        // def set_root_link_velocity_to_sim_mask(                                                <L 726>
        // i = wp.tid()                                                                           <L 756>
        var_0 = builtin_tid1d();
        // if env_mask[i]:                                                                        <L 757>
        var_1 = wp::address(var_env_mask, var_0);
        var_2 = wp::load(var_1);
        if (var_2) {
            // root_link_velocity_w[i] = data[i]                                                  <L 758>
            var_3 = wp::address(var_data, var_0);
            var_4 = wp::load(var_3);
            wp::array_store(var_root_link_velocity_w, var_0, var_4);
            // root_com_velocity_w[i] = get_link_velocity_in_com_frame_func(                      <L 760>
            // root_link_velocity_w[i], link_pose_w[i], body_com_pos_b[i, 0]                      <L 761>
            var_5 = wp::address(var_root_link_velocity_w, var_0);
            var_6 = wp::address(var_link_pose_w, var_0);
            var_8 = wp::address(var_body_com_pos_b, var_0, var_7);
            var_10 = wp::load(var_5);
            var_11 = wp::load(var_6);
            var_12 = wp::load(var_8);
            var_9 = get_link_velocity_in_com_frame_func_0(var_10, var_11, var_12);
            // root_com_velocity_w[i] = get_link_velocity_in_com_frame_func(                      <L 760>
            wp::array_store(var_root_com_velocity_w, var_0, var_9);
            // for j in range(num_bodies):                                                        <L 764>
            var_13 = wp::range(var_num_bodies);
            start_for_0:;
                if (iter_cmp(var_13) == 0) goto end_for_0;
                var_14 = wp::iter_next(var_13);
                // body_acc_w[i, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)            <L 765>
                var_21 = wp::vec_t<6, wp::float32>({var_15, var_16, var_17, var_18, var_19, var_20});
                wp::array_store(var_body_acc_w, var_0, var_14, var_21);
                goto start_for_0;
            end_for_0:;
        }
        var_22 = wp::load(var_1);
    }
}



extern "C" __global__ void set_root_link_velocity_to_sim_mask_3cfbbb23_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> var_link_pose_w,
    wp::array_t<bool> var_env_mask,
    wp::int32 var_num_bodies,
    wp::array_t<wp::vec_t<6, wp::float32>> var_root_link_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_root_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_data,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> adj_link_pose_w,
    wp::array_t<bool> adj_env_mask,
    wp::int32 adj_num_bodies,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_root_link_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_root_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_acc_w)
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
        bool* var_1;
        bool var_2;
        wp::vec_t<6, wp::float32>* var_3;
        wp::vec_t<6, wp::float32> var_4;
        wp::vec_t<6, wp::float32>* var_5;
        wp::transform_t<wp::float32>* var_6;
        const wp::int32 var_7 = 0;
        wp::vec_t<3, wp::float32>* var_8;
        wp::vec_t<6, wp::float32> var_9;
        wp::vec_t<6, wp::float32> var_10;
        wp::transform_t<wp::float32> var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::range_t var_13;
        wp::int32 var_14;
        const wp::float32 var_15 = 0.0;
        const wp::float32 var_16 = 0.0;
        const wp::float32 var_17 = 0.0;
        const wp::float32 var_18 = 0.0;
        const wp::float32 var_19 = 0.0;
        const wp::float32 var_20 = 0.0;
        wp::vec_t<6, wp::float32> var_21;
        bool var_22;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        bool adj_2 = {};
        wp::vec_t<6, wp::float32> adj_3 = {};
        wp::vec_t<6, wp::float32> adj_4 = {};
        wp::vec_t<6, wp::float32> adj_5 = {};
        wp::transform_t<wp::float32> adj_6 = {};
        wp::int32 adj_7 = {};
        wp::vec_t<3, wp::float32> adj_8 = {};
        wp::vec_t<6, wp::float32> adj_9 = {};
        wp::vec_t<6, wp::float32> adj_10 = {};
        wp::transform_t<wp::float32> adj_11 = {};
        wp::vec_t<3, wp::float32> adj_12 = {};
        wp::range_t adj_13 = {};
        wp::int32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::vec_t<6, wp::float32> adj_21 = {};
        bool adj_22 = {};
        //---------
        // forward
        // def set_root_link_velocity_to_sim_mask(                                                <L 726>
        // i = wp.tid()                                                                           <L 756>
        var_0 = builtin_tid1d();
        // if env_mask[i]:                                                                        <L 757>
        var_1 = wp::address(var_env_mask, var_0);
        var_2 = wp::load(var_1);
        if (var_2) {
            // root_link_velocity_w[i] = data[i]                                                  <L 758>
            var_3 = wp::address(var_data, var_0);
            var_4 = wp::load(var_3);
            // wp::array_store(var_root_link_velocity_w, var_0, var_4);
            // root_com_velocity_w[i] = get_link_velocity_in_com_frame_func(                      <L 760>
            // root_link_velocity_w[i], link_pose_w[i], body_com_pos_b[i, 0]                      <L 761>
            var_5 = wp::address(var_root_link_velocity_w, var_0);
            var_6 = wp::address(var_link_pose_w, var_0);
            var_8 = wp::address(var_body_com_pos_b, var_0, var_7);
            var_10 = wp::load(var_5);
            var_11 = wp::load(var_6);
            var_12 = wp::load(var_8);
            var_9 = get_link_velocity_in_com_frame_func_0(var_10, var_11, var_12);
            // root_com_velocity_w[i] = get_link_velocity_in_com_frame_func(                      <L 760>
            // wp::array_store(var_root_com_velocity_w, var_0, var_9);
            // for j in range(num_bodies):                                                        <L 764>
            var_13 = wp::range(var_num_bodies);
        }
        var_22 = wp::load(var_1);
        //---------
        // reverse
        if (var_22) {
            var_13 = wp::iter_reverse(var_13);
            start_for_0:;
                if (iter_cmp(var_13) == 0) goto end_for_0;
                var_14 = wp::iter_next(var_13);
            	adj_15 = {};
            	adj_16 = {};
            	adj_17 = {};
            	adj_18 = {};
            	adj_19 = {};
            	adj_20 = {};
            	adj_21 = {};
                // body_acc_w[i, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)            <L 765>
                var_21 = wp::vec_t<6, wp::float32>({var_15, var_16, var_17, var_18, var_19, var_20});
                // wp::array_store(var_body_acc_w, var_0, var_14, var_21);
                wp::adj_array_store(var_body_acc_w, var_0, var_14, var_21, adj_body_acc_w, adj_0, adj_14, adj_21);
                wp::adj_vec_t({var_15, var_16, var_17, var_18, var_19, var_20}, {&adj_15, &adj_16, &adj_17, &adj_18, &adj_19, &adj_20}, adj_21);
                // adj: body_acc_w[i, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)       <L 765>
            	goto start_for_0;
            end_for_0:;
            // adj: for j in range(num_bodies):                                                   <L 764>
            wp::adj_array_store(var_root_com_velocity_w, var_0, var_9, adj_root_com_velocity_w, adj_0, adj_9);
            // adj: root_com_velocity_w[i] = get_link_velocity_in_com_frame_func(                 <L 760>
            adj_get_link_velocity_in_com_frame_func_0(var_10, var_11, var_12, adj_5, adj_6, adj_8, adj_9);
            wp::adj_address(var_body_com_pos_b, var_0, var_7, adj_body_com_pos_b, adj_0, adj_7, adj_8);
            wp::adj_address(var_link_pose_w, var_0, adj_link_pose_w, adj_0, adj_6);
            wp::adj_address(var_root_link_velocity_w, var_0, adj_root_link_velocity_w, adj_0, adj_5);
            // adj: root_link_velocity_w[i], link_pose_w[i], body_com_pos_b[i, 0]                 <L 761>
            // adj: root_com_velocity_w[i] = get_link_velocity_in_com_frame_func(                 <L 760>
            wp::adj_array_store(var_root_link_velocity_w, var_0, var_4, adj_root_link_velocity_w, adj_0, adj_3);
            wp::adj_address(var_data, var_0, adj_data, adj_0, adj_3);
            // adj: root_link_velocity_w[i] = data[i]                                             <L 758>
        }
        wp::adj_address(var_env_mask, var_0, adj_env_mask, adj_0, adj_1);
        // adj: if env_mask[i]:                                                                   <L 757>
        // adj: i = wp.tid()                                                                      <L 756>
        // adj: def set_root_link_velocity_to_sim_mask(                                           <L 726>
        continue;
    }
}



extern "C" __global__ void set_body_link_pose_to_sim_3eb3ce62_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::transform_t<wp::float32>* var_8;
        wp::transform_t<wp::float32> var_9;
        wp::transform_t<wp::float32>* var_10;
        wp::transform_t<wp::float32> var_11;
        //---------
        // forward
        // def set_body_link_pose_to_sim(                                                         <L 1>
        // i, j = wp.tid()                                                                        <L 22>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 23>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 24>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 25>
        if (var_from_mask) {
            // body_link_pose_w[env_id, body_id] = data[env_id, body_id]                          <L 26>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_body_link_pose_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_link_pose_w[env_id, body_id] = data[i, j]                                     <L 28>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            wp::array_store(var_body_link_pose_w, var_3, var_6, var_11);
        }
    }
}



extern "C" __global__ void set_body_link_pose_to_sim_3eb3ce62_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_data,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int64> adj_body_ids,
    bool adj_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_link_pose_w)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::transform_t<wp::float32>* var_8;
        wp::transform_t<wp::float32> var_9;
        wp::transform_t<wp::float32>* var_10;
        wp::transform_t<wp::float32> var_11;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int64 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int64 adj_4 = {};
        wp::int64 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int64 adj_7 = {};
        wp::transform_t<wp::float32> adj_8 = {};
        wp::transform_t<wp::float32> adj_9 = {};
        wp::transform_t<wp::float32> adj_10 = {};
        wp::transform_t<wp::float32> adj_11 = {};
        //---------
        // forward
        // def set_body_link_pose_to_sim(                                                         <L 1>
        // i, j = wp.tid()                                                                        <L 22>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 23>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 24>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 25>
        if (var_from_mask) {
            // body_link_pose_w[env_id, body_id] = data[env_id, body_id]                          <L 26>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            // wp::array_store(var_body_link_pose_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_link_pose_w[env_id, body_id] = data[i, j]                                     <L 28>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            // wp::array_store(var_body_link_pose_w, var_3, var_6, var_11);
        }
        //---------
        // reverse
        if (!var_from_mask) {
            wp::adj_array_store(var_body_link_pose_w, var_3, var_6, var_11, adj_body_link_pose_w, adj_3, adj_6, adj_10);
            wp::adj_address(var_data, var_0, var_1, adj_data, adj_0, adj_1, adj_10);
            // adj: body_link_pose_w[env_id, body_id] = data[i, j]                                <L 28>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_body_link_pose_w, var_3, var_6, var_9, adj_body_link_pose_w, adj_3, adj_6, adj_8);
            wp::adj_address(var_data, var_3, var_6, adj_data, adj_3, adj_6, adj_8);
            // adj: body_link_pose_w[env_id, body_id] = data[env_id, body_id]                     <L 26>
        }
        // adj: if from_mask:                                                                     <L 25>
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_5);
        // adj: body_id = wp.int32(body_ids[j])                                                   <L 24>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 23>
        // adj: i, j = wp.tid()                                                                   <L 22>
        // adj: def set_body_link_pose_to_sim(                                                    <L 1>
        continue;
    }
}



extern "C" __global__ void quat_apply_inverse_1D_kernel_411f1020_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_gravity,
    wp::array_t<wp::quat_t<wp::float32>> var_quat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_projected_gravity)
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
        wp::quat_t<wp::float32>* var_1;
        wp::vec_t<3, wp::float32>* var_2;
        wp::vec_t<3, wp::float32> var_3;
        wp::quat_t<wp::float32> var_4;
        wp::vec_t<3, wp::float32> var_5;
        //---------
        // forward
        // def quat_apply_inverse_1D_kernel(                                                      <L 400>
        // i = wp.tid()                                                                           <L 416>
        var_0 = builtin_tid1d();
        // projected_gravity[i] = wp.quat_rotate_inv(quat[i], gravity[i])                         <L 417>
        var_1 = wp::address(var_quat, var_0);
        var_2 = wp::address(var_gravity, var_0);
        var_4 = wp::load(var_1);
        var_5 = wp::load(var_2);
        var_3 = wp::quat_rotate_inv(var_4, var_5);
        wp::array_store(var_projected_gravity, var_0, var_3);
    }
}



extern "C" __global__ void quat_apply_inverse_1D_kernel_411f1020_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_gravity,
    wp::array_t<wp::quat_t<wp::float32>> var_quat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_projected_gravity,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_gravity,
    wp::array_t<wp::quat_t<wp::float32>> adj_quat,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_projected_gravity)
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
        wp::quat_t<wp::float32>* var_1;
        wp::vec_t<3, wp::float32>* var_2;
        wp::vec_t<3, wp::float32> var_3;
        wp::quat_t<wp::float32> var_4;
        wp::vec_t<3, wp::float32> var_5;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::quat_t<wp::float32> adj_1 = {};
        wp::vec_t<3, wp::float32> adj_2 = {};
        wp::vec_t<3, wp::float32> adj_3 = {};
        wp::quat_t<wp::float32> adj_4 = {};
        wp::vec_t<3, wp::float32> adj_5 = {};
        //---------
        // forward
        // def quat_apply_inverse_1D_kernel(                                                      <L 400>
        // i = wp.tid()                                                                           <L 416>
        var_0 = builtin_tid1d();
        // projected_gravity[i] = wp.quat_rotate_inv(quat[i], gravity[i])                         <L 417>
        var_1 = wp::address(var_quat, var_0);
        var_2 = wp::address(var_gravity, var_0);
        var_4 = wp::load(var_1);
        var_5 = wp::load(var_2);
        var_3 = wp::quat_rotate_inv(var_4, var_5);
        // wp::array_store(var_projected_gravity, var_0, var_3);
        //---------
        // reverse
        wp::adj_array_store(var_projected_gravity, var_0, var_3, adj_projected_gravity, adj_0, adj_3);
        wp::adj_quat_rotate_inv(var_4, var_5, adj_1, adj_2, adj_3);
        wp::adj_address(var_gravity, var_0, adj_gravity, adj_0, adj_2);
        wp::adj_address(var_quat, var_0, adj_quat, adj_0, adj_1);
        // adj: projected_gravity[i] = wp.quat_rotate_inv(quat[i], gravity[i])                    <L 417>
        // adj: i = wp.tid()                                                                      <L 416>
        // adj: def quat_apply_inverse_1D_kernel(                                                 <L 400>
        continue;
    }
}



extern "C" __global__ void set_root_com_velocity_to_sim_index_4369c84d_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::int32 var_num_bodies,
    wp::array_t<wp::vec_t<6, wp::float32>> var_root_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w)
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
        wp::vec_t<6, wp::float32>* var_4;
        wp::vec_t<6, wp::float32> var_5;
        wp::range_t var_6;
        wp::int32 var_7;
        const wp::float32 var_8 = 0.0;
        const wp::float32 var_9 = 0.0;
        const wp::float32 var_10 = 0.0;
        const wp::float32 var_11 = 0.0;
        const wp::float32 var_12 = 0.0;
        const wp::float32 var_13 = 0.0;
        wp::vec_t<6, wp::float32> var_14;
        //---------
        // forward
        // def set_root_com_velocity_to_sim_index(                                                <L 1>
        // i = wp.tid()                                                                           <L 21>
        var_0 = builtin_tid1d();
        // env_id = wp.int32(env_ids[i])                                                          <L 22>
        var_1 = wp::address(var_env_ids, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::int32(var_3);
        // root_com_velocity_w[env_id] = data[i]                                                  <L 23>
        var_4 = wp::address(var_data, var_0);
        var_5 = wp::load(var_4);
        wp::array_store(var_root_com_velocity_w, var_2, var_5);
        // for j in range(num_bodies):                                                            <L 25>
        var_6 = wp::range(var_num_bodies);
        start_for_0:;
            if (iter_cmp(var_6) == 0) goto end_for_0;
            var_7 = wp::iter_next(var_6);
            // body_acc_w[env_id, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)           <L 26>
            var_14 = wp::vec_t<6, wp::float32>({var_8, var_9, var_10, var_11, var_12, var_13});
            wp::array_store(var_body_acc_w, var_2, var_7, var_14);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void set_root_com_velocity_to_sim_index_4369c84d_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::int32 var_num_bodies,
    wp::array_t<wp::vec_t<6, wp::float32>> var_root_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::int32 adj_num_bodies,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_root_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_acc_w)
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
        wp::vec_t<6, wp::float32>* var_4;
        wp::vec_t<6, wp::float32> var_5;
        wp::range_t var_6;
        wp::int32 var_7;
        const wp::float32 var_8 = 0.0;
        const wp::float32 var_9 = 0.0;
        const wp::float32 var_10 = 0.0;
        const wp::float32 var_11 = 0.0;
        const wp::float32 var_12 = 0.0;
        const wp::float32 var_13 = 0.0;
        wp::vec_t<6, wp::float32> var_14;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::vec_t<6, wp::float32> adj_4 = {};
        wp::vec_t<6, wp::float32> adj_5 = {};
        wp::range_t adj_6 = {};
        wp::int32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::vec_t<6, wp::float32> adj_14 = {};
        //---------
        // forward
        // def set_root_com_velocity_to_sim_index(                                                <L 1>
        // i = wp.tid()                                                                           <L 21>
        var_0 = builtin_tid1d();
        // env_id = wp.int32(env_ids[i])                                                          <L 22>
        var_1 = wp::address(var_env_ids, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::int32(var_3);
        // root_com_velocity_w[env_id] = data[i]                                                  <L 23>
        var_4 = wp::address(var_data, var_0);
        var_5 = wp::load(var_4);
        // wp::array_store(var_root_com_velocity_w, var_2, var_5);
        // for j in range(num_bodies):                                                            <L 25>
        var_6 = wp::range(var_num_bodies);
        //---------
        // reverse
        var_6 = wp::iter_reverse(var_6);
        start_for_0:;
            if (iter_cmp(var_6) == 0) goto end_for_0;
            var_7 = wp::iter_next(var_6);
        	adj_8 = {};
        	adj_9 = {};
        	adj_10 = {};
        	adj_11 = {};
        	adj_12 = {};
        	adj_13 = {};
        	adj_14 = {};
            // body_acc_w[env_id, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)           <L 26>
            var_14 = wp::vec_t<6, wp::float32>({var_8, var_9, var_10, var_11, var_12, var_13});
            // wp::array_store(var_body_acc_w, var_2, var_7, var_14);
            wp::adj_array_store(var_body_acc_w, var_2, var_7, var_14, adj_body_acc_w, adj_2, adj_7, adj_14);
            wp::adj_vec_t({var_8, var_9, var_10, var_11, var_12, var_13}, {&adj_8, &adj_9, &adj_10, &adj_11, &adj_12, &adj_13}, adj_14);
            // adj: body_acc_w[env_id, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)      <L 26>
        	goto start_for_0;
        end_for_0:;
        // adj: for j in range(num_bodies):                                                       <L 25>
        wp::adj_array_store(var_root_com_velocity_w, var_2, var_5, adj_root_com_velocity_w, adj_2, adj_4);
        wp::adj_address(var_data, var_0, adj_data, adj_0, adj_4);
        // adj: root_com_velocity_w[env_id] = data[i]                                             <L 23>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_1);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 22>
        // adj: i = wp.tid()                                                                      <L 21>
        // adj: def set_root_com_velocity_to_sim_index(                                           <L 1>
        continue;
    }
}



extern "C" __global__ void get_root_link_vel_from_root_com_vel_47adfdbe_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_com_vel,
    wp::array_t<wp::transform_t<wp::float32>> var_link_pose,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::vec_t<6, wp::float32>> var_link_vel)
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
        wp::vec_t<6, wp::float32>* var_1;
        wp::transform_t<wp::float32>* var_2;
        const wp::int32 var_3 = 0;
        wp::vec_t<3, wp::float32>* var_4;
        wp::vec_t<6, wp::float32> var_5;
        wp::vec_t<6, wp::float32> var_6;
        wp::transform_t<wp::float32> var_7;
        wp::vec_t<3, wp::float32> var_8;
        //---------
        // forward
        // def get_root_link_vel_from_root_com_vel(                                               <L 234>
        // i = wp.tid()                                                                           <L 252>
        var_0 = builtin_tid1d();
        // link_vel[i] = get_link_vel_from_root_com_vel_func(com_vel[i], link_pose[i], body_com_pos_b[i, 0])       <L 253>
        var_1 = wp::address(var_com_vel, var_0);
        var_2 = wp::address(var_link_pose, var_0);
        var_4 = wp::address(var_body_com_pos_b, var_0, var_3);
        var_6 = wp::load(var_1);
        var_7 = wp::load(var_2);
        var_8 = wp::load(var_4);
        var_5 = get_link_vel_from_root_com_vel_func_0(var_6, var_7, var_8);
        wp::array_store(var_link_vel, var_0, var_5);
    }
}



extern "C" __global__ void get_root_link_vel_from_root_com_vel_47adfdbe_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_com_vel,
    wp::array_t<wp::transform_t<wp::float32>> var_link_pose,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::vec_t<6, wp::float32>> var_link_vel,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_com_vel,
    wp::array_t<wp::transform_t<wp::float32>> adj_link_pose,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_body_com_pos_b,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_link_vel)
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
        wp::vec_t<6, wp::float32>* var_1;
        wp::transform_t<wp::float32>* var_2;
        const wp::int32 var_3 = 0;
        wp::vec_t<3, wp::float32>* var_4;
        wp::vec_t<6, wp::float32> var_5;
        wp::vec_t<6, wp::float32> var_6;
        wp::transform_t<wp::float32> var_7;
        wp::vec_t<3, wp::float32> var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::vec_t<6, wp::float32> adj_1 = {};
        wp::transform_t<wp::float32> adj_2 = {};
        wp::int32 adj_3 = {};
        wp::vec_t<3, wp::float32> adj_4 = {};
        wp::vec_t<6, wp::float32> adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        wp::transform_t<wp::float32> adj_7 = {};
        wp::vec_t<3, wp::float32> adj_8 = {};
        //---------
        // forward
        // def get_root_link_vel_from_root_com_vel(                                               <L 234>
        // i = wp.tid()                                                                           <L 252>
        var_0 = builtin_tid1d();
        // link_vel[i] = get_link_vel_from_root_com_vel_func(com_vel[i], link_pose[i], body_com_pos_b[i, 0])       <L 253>
        var_1 = wp::address(var_com_vel, var_0);
        var_2 = wp::address(var_link_pose, var_0);
        var_4 = wp::address(var_body_com_pos_b, var_0, var_3);
        var_6 = wp::load(var_1);
        var_7 = wp::load(var_2);
        var_8 = wp::load(var_4);
        var_5 = get_link_vel_from_root_com_vel_func_0(var_6, var_7, var_8);
        // wp::array_store(var_link_vel, var_0, var_5);
        //---------
        // reverse
        wp::adj_array_store(var_link_vel, var_0, var_5, adj_link_vel, adj_0, adj_5);
        adj_get_link_vel_from_root_com_vel_func_0(var_6, var_7, var_8, adj_1, adj_2, adj_4, adj_5);
        wp::adj_address(var_body_com_pos_b, var_0, var_3, adj_body_com_pos_b, adj_0, adj_3, adj_4);
        wp::adj_address(var_link_pose, var_0, adj_link_pose, adj_0, adj_2);
        wp::adj_address(var_com_vel, var_0, adj_com_vel, adj_0, adj_1);
        // adj: link_vel[i] = get_link_vel_from_root_com_vel_func(com_vel[i], link_pose[i], body_com_pos_b[i, 0])  <L 253>
        // adj: i = wp.tid()                                                                      <L 252>
        // adj: def get_root_link_vel_from_root_com_vel(                                          <L 234>
        continue;
    }
}



extern "C" __global__ void write_2d_data_to_buffer_with_indices_47cc0614_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::array_t<wp::float32> var_out_data)
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
        wp::float32* var_8;
        wp::float32 var_9;
        //---------
        // forward
        // def write_2d_data_to_buffer_with_indices(                                              <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 19>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 20>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // out_data[env_id, joint_id] = in_data[i, j]                                             <L 21>
        var_8 = wp::address(var_in_data, var_0, var_1);
        var_9 = wp::load(var_8);
        wp::array_store(var_out_data, var_3, var_6, var_9);
    }
}



extern "C" __global__ void write_2d_data_to_buffer_with_indices_47cc0614_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::array_t<wp::float32> var_out_data,
    wp::array_t<wp::float32> adj_in_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_joint_ids,
    wp::array_t<wp::float32> adj_out_data)
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
        wp::float32* var_8;
        wp::float32 var_9;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        //---------
        // forward
        // def write_2d_data_to_buffer_with_indices(                                              <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 19>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 20>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // out_data[env_id, joint_id] = in_data[i, j]                                             <L 21>
        var_8 = wp::address(var_in_data, var_0, var_1);
        var_9 = wp::load(var_8);
        // wp::array_store(var_out_data, var_3, var_6, var_9);
        //---------
        // reverse
        wp::adj_array_store(var_out_data, var_3, var_6, var_9, adj_out_data, adj_3, adj_6, adj_8);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_8);
        // adj: out_data[env_id, joint_id] = in_data[i, j]                                        <L 21>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 20>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 19>
        // adj: i, j = wp.tid()                                                                   <L 18>
        // adj: def write_2d_data_to_buffer_with_indices(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void projected_gravity_b_kernel_59062a57_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_gravity_w,
    wp::array_t<wp::quat_t<wp::float32>> var_quat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_projected_gravity_b)
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
        wp::quat_t<wp::float32>* var_1;
        wp::vec_t<3, wp::float32>* var_2;
        wp::vec_t<3, wp::float32> var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::vec_t<3, wp::float32> var_5;
        wp::quat_t<wp::float32> var_6;
        //---------
        // forward
        // def projected_gravity_b_kernel(                                                        <L 461>
        // i = wp.tid()                                                                           <L 477>
        var_0 = builtin_tid1d();
        // projected_gravity_b[i] = wp.quat_rotate_inv(quat[i], wp.normalize(gravity_w[i]))       <L 478>
        var_1 = wp::address(var_quat, var_0);
        var_2 = wp::address(var_gravity_w, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::normalize(var_4);
        var_6 = wp::load(var_1);
        var_5 = wp::quat_rotate_inv(var_6, var_3);
        wp::array_store(var_projected_gravity_b, var_0, var_5);
    }
}



extern "C" __global__ void projected_gravity_b_kernel_59062a57_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_gravity_w,
    wp::array_t<wp::quat_t<wp::float32>> var_quat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_projected_gravity_b,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_gravity_w,
    wp::array_t<wp::quat_t<wp::float32>> adj_quat,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_projected_gravity_b)
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
        wp::quat_t<wp::float32>* var_1;
        wp::vec_t<3, wp::float32>* var_2;
        wp::vec_t<3, wp::float32> var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::vec_t<3, wp::float32> var_5;
        wp::quat_t<wp::float32> var_6;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::quat_t<wp::float32> adj_1 = {};
        wp::vec_t<3, wp::float32> adj_2 = {};
        wp::vec_t<3, wp::float32> adj_3 = {};
        wp::vec_t<3, wp::float32> adj_4 = {};
        wp::vec_t<3, wp::float32> adj_5 = {};
        wp::quat_t<wp::float32> adj_6 = {};
        //---------
        // forward
        // def projected_gravity_b_kernel(                                                        <L 461>
        // i = wp.tid()                                                                           <L 477>
        var_0 = builtin_tid1d();
        // projected_gravity_b[i] = wp.quat_rotate_inv(quat[i], wp.normalize(gravity_w[i]))       <L 478>
        var_1 = wp::address(var_quat, var_0);
        var_2 = wp::address(var_gravity_w, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::normalize(var_4);
        var_6 = wp::load(var_1);
        var_5 = wp::quat_rotate_inv(var_6, var_3);
        // wp::array_store(var_projected_gravity_b, var_0, var_5);
        //---------
        // reverse
        wp::adj_array_store(var_projected_gravity_b, var_0, var_5, adj_projected_gravity_b, adj_0, adj_5);
        wp::adj_quat_rotate_inv(var_6, var_3, adj_1, adj_3, adj_5);
        wp::adj_normalize(var_4, var_3, adj_2, adj_3);
        wp::adj_address(var_gravity_w, var_0, adj_gravity_w, adj_0, adj_2);
        wp::adj_address(var_quat, var_0, adj_quat, adj_0, adj_1);
        // adj: projected_gravity_b[i] = wp.quat_rotate_inv(quat[i], wp.normalize(gravity_w[i]))  <L 478>
        // adj: i = wp.tid()                                                                      <L 477>
        // adj: def projected_gravity_b_kernel(                                                   <L 461>
        continue;
    }
}



extern "C" __global__ void get_body_com_pose_from_body_link_pose_5940b76d_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_w)
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
        wp::transform_t<wp::float32>* var_2;
        wp::vec_t<3, wp::float32>* var_3;
        wp::transform_t<wp::float32> var_4;
        wp::transform_t<wp::float32> var_5;
        wp::vec_t<3, wp::float32> var_6;
        //---------
        // forward
        // def get_body_com_pose_from_body_link_pose(                                             <L 354>
        // i, j = wp.tid()                                                                        <L 369>
        builtin_tid2d(var_0, var_1);
        // body_com_pose_w[i, j] = get_com_pose_from_link_pose_func(body_link_pose[i, j], body_com_pos_b[i, j])       <L 370>
        var_2 = wp::address(var_body_link_pose, var_0, var_1);
        var_3 = wp::address(var_body_com_pos_b, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = get_com_pose_from_link_pose_func_0(var_5, var_6);
        wp::array_store(var_body_com_pose_w, var_0, var_1, var_4);
    }
}



extern "C" __global__ void get_body_com_pose_from_body_link_pose_5940b76d_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_link_pose,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_com_pose_w)
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
        wp::transform_t<wp::float32>* var_2;
        wp::vec_t<3, wp::float32>* var_3;
        wp::transform_t<wp::float32> var_4;
        wp::transform_t<wp::float32> var_5;
        wp::vec_t<3, wp::float32> var_6;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::transform_t<wp::float32> adj_2 = {};
        wp::vec_t<3, wp::float32> adj_3 = {};
        wp::transform_t<wp::float32> adj_4 = {};
        wp::transform_t<wp::float32> adj_5 = {};
        wp::vec_t<3, wp::float32> adj_6 = {};
        //---------
        // forward
        // def get_body_com_pose_from_body_link_pose(                                             <L 354>
        // i, j = wp.tid()                                                                        <L 369>
        builtin_tid2d(var_0, var_1);
        // body_com_pose_w[i, j] = get_com_pose_from_link_pose_func(body_link_pose[i, j], body_com_pos_b[i, j])       <L 370>
        var_2 = wp::address(var_body_link_pose, var_0, var_1);
        var_3 = wp::address(var_body_com_pos_b, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = get_com_pose_from_link_pose_func_0(var_5, var_6);
        // wp::array_store(var_body_com_pose_w, var_0, var_1, var_4);
        //---------
        // reverse
        wp::adj_array_store(var_body_com_pose_w, var_0, var_1, var_4, adj_body_com_pose_w, adj_0, adj_1, adj_4);
        adj_get_com_pose_from_link_pose_func_0(var_5, var_6, adj_2, adj_3, adj_4);
        wp::adj_address(var_body_com_pos_b, var_0, var_1, adj_body_com_pos_b, adj_0, adj_1, adj_3);
        wp::adj_address(var_body_link_pose, var_0, var_1, adj_body_link_pose, adj_0, adj_1, adj_2);
        // adj: body_com_pose_w[i, j] = get_com_pose_from_link_pose_func(body_link_pose[i, j], body_com_pos_b[i, j])  <L 370>
        // adj: i, j = wp.tid()                                                                   <L 369>
        // adj: def get_body_com_pose_from_body_link_pose(                                        <L 354>
        continue;
    }
}



extern "C" __global__ void write_single_body_inertia_to_buffer_5a600b96_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    bool var_from_mask,
    wp::array_t<wp::float32> var_out_data)
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
        const wp::int32 var_4 = 0;
        wp::float32* var_5;
        wp::float32 var_6;
        const wp::int32 var_7 = 1;
        wp::float32* var_8;
        wp::float32 var_9;
        const wp::int32 var_10 = 2;
        wp::float32* var_11;
        wp::float32 var_12;
        const wp::int32 var_13 = 3;
        wp::float32* var_14;
        wp::float32 var_15;
        const wp::int32 var_16 = 4;
        wp::float32* var_17;
        wp::float32 var_18;
        const wp::int32 var_19 = 5;
        wp::float32* var_20;
        wp::float32 var_21;
        const wp::int32 var_22 = 6;
        wp::float32* var_23;
        wp::float32 var_24;
        const wp::int32 var_25 = 7;
        wp::float32* var_26;
        wp::float32 var_27;
        const wp::int32 var_28 = 8;
        wp::float32* var_29;
        wp::float32 var_30;
        const wp::int32 var_31 = 0;
        wp::float32* var_32;
        wp::float32 var_33;
        const wp::int32 var_34 = 1;
        wp::float32* var_35;
        wp::float32 var_36;
        const wp::int32 var_37 = 2;
        wp::float32* var_38;
        wp::float32 var_39;
        const wp::int32 var_40 = 3;
        wp::float32* var_41;
        wp::float32 var_42;
        const wp::int32 var_43 = 4;
        wp::float32* var_44;
        wp::float32 var_45;
        const wp::int32 var_46 = 5;
        wp::float32* var_47;
        wp::float32 var_48;
        const wp::int32 var_49 = 6;
        wp::float32* var_50;
        wp::float32 var_51;
        const wp::int32 var_52 = 7;
        wp::float32* var_53;
        wp::float32 var_54;
        const wp::int32 var_55 = 8;
        wp::float32* var_56;
        wp::float32 var_57;
        wp::int32 var_58;
        //---------
        // forward
        // def write_single_body_inertia_to_buffer(                                               <L 1>
        // i = wp.tid()                                                                           <L 19>
        var_0 = builtin_tid1d();
        // env_id = wp.int32(env_ids[i])                                                          <L 20>
        var_1 = wp::address(var_env_ids, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::int32(var_3);
        // if from_mask:                                                                          <L 21>
        if (var_from_mask) {
            // for k in range(9):                                                                 <L 22>
            // out_data[env_id, k] = in_data[env_id, k]                                           <L 23>
            var_5 = wp::address(var_in_data, var_2, var_4);
            var_6 = wp::load(var_5);
            wp::array_store(var_out_data, var_2, var_4, var_6);
            var_8 = wp::address(var_in_data, var_2, var_7);
            var_9 = wp::load(var_8);
            wp::array_store(var_out_data, var_2, var_7, var_9);
            var_11 = wp::address(var_in_data, var_2, var_10);
            var_12 = wp::load(var_11);
            wp::array_store(var_out_data, var_2, var_10, var_12);
            var_14 = wp::address(var_in_data, var_2, var_13);
            var_15 = wp::load(var_14);
            wp::array_store(var_out_data, var_2, var_13, var_15);
            var_17 = wp::address(var_in_data, var_2, var_16);
            var_18 = wp::load(var_17);
            wp::array_store(var_out_data, var_2, var_16, var_18);
            var_20 = wp::address(var_in_data, var_2, var_19);
            var_21 = wp::load(var_20);
            wp::array_store(var_out_data, var_2, var_19, var_21);
            var_23 = wp::address(var_in_data, var_2, var_22);
            var_24 = wp::load(var_23);
            wp::array_store(var_out_data, var_2, var_22, var_24);
            var_26 = wp::address(var_in_data, var_2, var_25);
            var_27 = wp::load(var_26);
            wp::array_store(var_out_data, var_2, var_25, var_27);
            var_29 = wp::address(var_in_data, var_2, var_28);
            var_30 = wp::load(var_29);
            wp::array_store(var_out_data, var_2, var_28, var_30);
        }
        if (!var_from_mask) {
            // for k in range(9):                                                                 <L 25>
            // out_data[env_id, k] = in_data[i, k]                                                <L 26>
            var_32 = wp::address(var_in_data, var_0, var_31);
            var_33 = wp::load(var_32);
            wp::array_store(var_out_data, var_2, var_31, var_33);
            var_35 = wp::address(var_in_data, var_0, var_34);
            var_36 = wp::load(var_35);
            wp::array_store(var_out_data, var_2, var_34, var_36);
            var_38 = wp::address(var_in_data, var_0, var_37);
            var_39 = wp::load(var_38);
            wp::array_store(var_out_data, var_2, var_37, var_39);
            var_41 = wp::address(var_in_data, var_0, var_40);
            var_42 = wp::load(var_41);
            wp::array_store(var_out_data, var_2, var_40, var_42);
            var_44 = wp::address(var_in_data, var_0, var_43);
            var_45 = wp::load(var_44);
            wp::array_store(var_out_data, var_2, var_43, var_45);
            var_47 = wp::address(var_in_data, var_0, var_46);
            var_48 = wp::load(var_47);
            wp::array_store(var_out_data, var_2, var_46, var_48);
            var_50 = wp::address(var_in_data, var_0, var_49);
            var_51 = wp::load(var_50);
            wp::array_store(var_out_data, var_2, var_49, var_51);
            var_53 = wp::address(var_in_data, var_0, var_52);
            var_54 = wp::load(var_53);
            wp::array_store(var_out_data, var_2, var_52, var_54);
            var_56 = wp::address(var_in_data, var_0, var_55);
            var_57 = wp::load(var_56);
            wp::array_store(var_out_data, var_2, var_55, var_57);
        }
        var_58 = wp::where(var_from_mask, var_28, var_55);
    }
}



extern "C" __global__ void write_single_body_inertia_to_buffer_5a600b96_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    bool var_from_mask,
    wp::array_t<wp::float32> var_out_data,
    wp::array_t<wp::float32> adj_in_data,
    wp::array_t<wp::int32> adj_env_ids,
    bool adj_from_mask,
    wp::array_t<wp::float32> adj_out_data)
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
        const wp::int32 var_4 = 0;
        wp::float32* var_5;
        wp::float32 var_6;
        const wp::int32 var_7 = 1;
        wp::float32* var_8;
        wp::float32 var_9;
        const wp::int32 var_10 = 2;
        wp::float32* var_11;
        wp::float32 var_12;
        const wp::int32 var_13 = 3;
        wp::float32* var_14;
        wp::float32 var_15;
        const wp::int32 var_16 = 4;
        wp::float32* var_17;
        wp::float32 var_18;
        const wp::int32 var_19 = 5;
        wp::float32* var_20;
        wp::float32 var_21;
        const wp::int32 var_22 = 6;
        wp::float32* var_23;
        wp::float32 var_24;
        const wp::int32 var_25 = 7;
        wp::float32* var_26;
        wp::float32 var_27;
        const wp::int32 var_28 = 8;
        wp::float32* var_29;
        wp::float32 var_30;
        const wp::int32 var_31 = 0;
        wp::float32* var_32;
        wp::float32 var_33;
        const wp::int32 var_34 = 1;
        wp::float32* var_35;
        wp::float32 var_36;
        const wp::int32 var_37 = 2;
        wp::float32* var_38;
        wp::float32 var_39;
        const wp::int32 var_40 = 3;
        wp::float32* var_41;
        wp::float32 var_42;
        const wp::int32 var_43 = 4;
        wp::float32* var_44;
        wp::float32 var_45;
        const wp::int32 var_46 = 5;
        wp::float32* var_47;
        wp::float32 var_48;
        const wp::int32 var_49 = 6;
        wp::float32* var_50;
        wp::float32 var_51;
        const wp::int32 var_52 = 7;
        wp::float32* var_53;
        wp::float32 var_54;
        const wp::int32 var_55 = 8;
        wp::float32* var_56;
        wp::float32 var_57;
        wp::int32 var_58;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::float32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::float32 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::float32 adj_41 = {};
        wp::float32 adj_42 = {};
        wp::int32 adj_43 = {};
        wp::float32 adj_44 = {};
        wp::float32 adj_45 = {};
        wp::int32 adj_46 = {};
        wp::float32 adj_47 = {};
        wp::float32 adj_48 = {};
        wp::int32 adj_49 = {};
        wp::float32 adj_50 = {};
        wp::float32 adj_51 = {};
        wp::int32 adj_52 = {};
        wp::float32 adj_53 = {};
        wp::float32 adj_54 = {};
        wp::int32 adj_55 = {};
        wp::float32 adj_56 = {};
        wp::float32 adj_57 = {};
        wp::int32 adj_58 = {};
        //---------
        // forward
        // def write_single_body_inertia_to_buffer(                                               <L 1>
        // i = wp.tid()                                                                           <L 19>
        var_0 = builtin_tid1d();
        // env_id = wp.int32(env_ids[i])                                                          <L 20>
        var_1 = wp::address(var_env_ids, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::int32(var_3);
        // if from_mask:                                                                          <L 21>
        if (var_from_mask) {
            // for k in range(9):                                                                 <L 22>
            // out_data[env_id, k] = in_data[env_id, k]                                           <L 23>
            var_5 = wp::address(var_in_data, var_2, var_4);
            var_6 = wp::load(var_5);
            // wp::array_store(var_out_data, var_2, var_4, var_6);
            var_8 = wp::address(var_in_data, var_2, var_7);
            var_9 = wp::load(var_8);
            // wp::array_store(var_out_data, var_2, var_7, var_9);
            var_11 = wp::address(var_in_data, var_2, var_10);
            var_12 = wp::load(var_11);
            // wp::array_store(var_out_data, var_2, var_10, var_12);
            var_14 = wp::address(var_in_data, var_2, var_13);
            var_15 = wp::load(var_14);
            // wp::array_store(var_out_data, var_2, var_13, var_15);
            var_17 = wp::address(var_in_data, var_2, var_16);
            var_18 = wp::load(var_17);
            // wp::array_store(var_out_data, var_2, var_16, var_18);
            var_20 = wp::address(var_in_data, var_2, var_19);
            var_21 = wp::load(var_20);
            // wp::array_store(var_out_data, var_2, var_19, var_21);
            var_23 = wp::address(var_in_data, var_2, var_22);
            var_24 = wp::load(var_23);
            // wp::array_store(var_out_data, var_2, var_22, var_24);
            var_26 = wp::address(var_in_data, var_2, var_25);
            var_27 = wp::load(var_26);
            // wp::array_store(var_out_data, var_2, var_25, var_27);
            var_29 = wp::address(var_in_data, var_2, var_28);
            var_30 = wp::load(var_29);
            // wp::array_store(var_out_data, var_2, var_28, var_30);
        }
        if (!var_from_mask) {
            // for k in range(9):                                                                 <L 25>
            // out_data[env_id, k] = in_data[i, k]                                                <L 26>
            var_32 = wp::address(var_in_data, var_0, var_31);
            var_33 = wp::load(var_32);
            // wp::array_store(var_out_data, var_2, var_31, var_33);
            var_35 = wp::address(var_in_data, var_0, var_34);
            var_36 = wp::load(var_35);
            // wp::array_store(var_out_data, var_2, var_34, var_36);
            var_38 = wp::address(var_in_data, var_0, var_37);
            var_39 = wp::load(var_38);
            // wp::array_store(var_out_data, var_2, var_37, var_39);
            var_41 = wp::address(var_in_data, var_0, var_40);
            var_42 = wp::load(var_41);
            // wp::array_store(var_out_data, var_2, var_40, var_42);
            var_44 = wp::address(var_in_data, var_0, var_43);
            var_45 = wp::load(var_44);
            // wp::array_store(var_out_data, var_2, var_43, var_45);
            var_47 = wp::address(var_in_data, var_0, var_46);
            var_48 = wp::load(var_47);
            // wp::array_store(var_out_data, var_2, var_46, var_48);
            var_50 = wp::address(var_in_data, var_0, var_49);
            var_51 = wp::load(var_50);
            // wp::array_store(var_out_data, var_2, var_49, var_51);
            var_53 = wp::address(var_in_data, var_0, var_52);
            var_54 = wp::load(var_53);
            // wp::array_store(var_out_data, var_2, var_52, var_54);
            var_56 = wp::address(var_in_data, var_0, var_55);
            var_57 = wp::load(var_56);
            // wp::array_store(var_out_data, var_2, var_55, var_57);
        }
        var_58 = wp::where(var_from_mask, var_28, var_55);
        //---------
        // reverse
        wp::adj_where(var_from_mask, var_28, var_55, adj_from_mask, adj_28, adj_55, adj_58);
        if (!var_from_mask) {
            wp::adj_array_store(var_out_data, var_2, var_55, var_57, adj_out_data, adj_2, adj_55, adj_56);
            wp::adj_address(var_in_data, var_0, var_55, adj_in_data, adj_0, adj_55, adj_56);
            wp::adj_array_store(var_out_data, var_2, var_52, var_54, adj_out_data, adj_2, adj_52, adj_53);
            wp::adj_address(var_in_data, var_0, var_52, adj_in_data, adj_0, adj_52, adj_53);
            wp::adj_array_store(var_out_data, var_2, var_49, var_51, adj_out_data, adj_2, adj_49, adj_50);
            wp::adj_address(var_in_data, var_0, var_49, adj_in_data, adj_0, adj_49, adj_50);
            wp::adj_array_store(var_out_data, var_2, var_46, var_48, adj_out_data, adj_2, adj_46, adj_47);
            wp::adj_address(var_in_data, var_0, var_46, adj_in_data, adj_0, adj_46, adj_47);
            wp::adj_array_store(var_out_data, var_2, var_43, var_45, adj_out_data, adj_2, adj_43, adj_44);
            wp::adj_address(var_in_data, var_0, var_43, adj_in_data, adj_0, adj_43, adj_44);
            wp::adj_array_store(var_out_data, var_2, var_40, var_42, adj_out_data, adj_2, adj_40, adj_41);
            wp::adj_address(var_in_data, var_0, var_40, adj_in_data, adj_0, adj_40, adj_41);
            wp::adj_array_store(var_out_data, var_2, var_37, var_39, adj_out_data, adj_2, adj_37, adj_38);
            wp::adj_address(var_in_data, var_0, var_37, adj_in_data, adj_0, adj_37, adj_38);
            wp::adj_array_store(var_out_data, var_2, var_34, var_36, adj_out_data, adj_2, adj_34, adj_35);
            wp::adj_address(var_in_data, var_0, var_34, adj_in_data, adj_0, adj_34, adj_35);
            wp::adj_array_store(var_out_data, var_2, var_31, var_33, adj_out_data, adj_2, adj_31, adj_32);
            wp::adj_address(var_in_data, var_0, var_31, adj_in_data, adj_0, adj_31, adj_32);
            // adj: out_data[env_id, k] = in_data[i, k]                                           <L 26>
            // adj: for k in range(9):                                                            <L 25>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_out_data, var_2, var_28, var_30, adj_out_data, adj_2, adj_28, adj_29);
            wp::adj_address(var_in_data, var_2, var_28, adj_in_data, adj_2, adj_28, adj_29);
            wp::adj_array_store(var_out_data, var_2, var_25, var_27, adj_out_data, adj_2, adj_25, adj_26);
            wp::adj_address(var_in_data, var_2, var_25, adj_in_data, adj_2, adj_25, adj_26);
            wp::adj_array_store(var_out_data, var_2, var_22, var_24, adj_out_data, adj_2, adj_22, adj_23);
            wp::adj_address(var_in_data, var_2, var_22, adj_in_data, adj_2, adj_22, adj_23);
            wp::adj_array_store(var_out_data, var_2, var_19, var_21, adj_out_data, adj_2, adj_19, adj_20);
            wp::adj_address(var_in_data, var_2, var_19, adj_in_data, adj_2, adj_19, adj_20);
            wp::adj_array_store(var_out_data, var_2, var_16, var_18, adj_out_data, adj_2, adj_16, adj_17);
            wp::adj_address(var_in_data, var_2, var_16, adj_in_data, adj_2, adj_16, adj_17);
            wp::adj_array_store(var_out_data, var_2, var_13, var_15, adj_out_data, adj_2, adj_13, adj_14);
            wp::adj_address(var_in_data, var_2, var_13, adj_in_data, adj_2, adj_13, adj_14);
            wp::adj_array_store(var_out_data, var_2, var_10, var_12, adj_out_data, adj_2, adj_10, adj_11);
            wp::adj_address(var_in_data, var_2, var_10, adj_in_data, adj_2, adj_10, adj_11);
            wp::adj_array_store(var_out_data, var_2, var_7, var_9, adj_out_data, adj_2, adj_7, adj_8);
            wp::adj_address(var_in_data, var_2, var_7, adj_in_data, adj_2, adj_7, adj_8);
            wp::adj_array_store(var_out_data, var_2, var_4, var_6, adj_out_data, adj_2, adj_4, adj_5);
            wp::adj_address(var_in_data, var_2, var_4, adj_in_data, adj_2, adj_4, adj_5);
            // adj: out_data[env_id, k] = in_data[env_id, k]                                      <L 23>
            // adj: for k in range(9):                                                            <L 22>
        }
        // adj: if from_mask:                                                                     <L 21>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_1);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 20>
        // adj: i = wp.tid()                                                                      <L 19>
        // adj: def write_single_body_inertia_to_buffer(                                          <L 1>
        continue;
    }
}



extern "C" __global__ void set_body_com_pose_to_sim_5e303afd_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w)
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
        wp::transform_t<wp::float32>* var_8;
        wp::transform_t<wp::float32> var_9;
        wp::transform_t<wp::float32>* var_10;
        wp::transform_t<wp::float32> var_11;
        wp::transform_t<wp::float32>* var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::transform_t<wp::float32> var_14;
        wp::transform_t<wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        //---------
        // forward
        // def set_body_com_pose_to_sim(                                                          <L 1>
        // i, j = wp.tid()                                                                        <L 29>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 30>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 31>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 32>
        if (var_from_mask) {
            // body_com_pose_w[env_id, body_id] = data[env_id, body_id]                           <L 33>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_body_com_pose_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_com_pose_w[env_id, body_id] = data[i, j]                                      <L 35>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            wp::array_store(var_body_com_pose_w, var_3, var_6, var_11);
        }
        // body_link_pose_w[env_id, body_id] = get_com_pose_in_link_frame_func(                   <L 37>
        // body_com_pose_w[env_id, body_id], body_com_pos_b[env_id, body_id]                      <L 38>
        var_12 = wp::address(var_body_com_pose_w, var_3, var_6);
        var_13 = wp::address(var_body_com_pos_b, var_3, var_6);
        var_15 = wp::load(var_12);
        var_16 = wp::load(var_13);
        var_14 = get_com_pose_in_link_frame_func_0(var_15, var_16);
        // body_link_pose_w[env_id, body_id] = get_com_pose_in_link_frame_func(                   <L 37>
        wp::array_store(var_body_link_pose_w, var_3, var_6, var_14);
    }
}



extern "C" __global__ void set_body_com_pose_to_sim_5e303afd_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_data,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_body_com_pos_b,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_body_ids,
    bool adj_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_link_pose_w)
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
        wp::transform_t<wp::float32>* var_8;
        wp::transform_t<wp::float32> var_9;
        wp::transform_t<wp::float32>* var_10;
        wp::transform_t<wp::float32> var_11;
        wp::transform_t<wp::float32>* var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::transform_t<wp::float32> var_14;
        wp::transform_t<wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::transform_t<wp::float32> adj_8 = {};
        wp::transform_t<wp::float32> adj_9 = {};
        wp::transform_t<wp::float32> adj_10 = {};
        wp::transform_t<wp::float32> adj_11 = {};
        wp::transform_t<wp::float32> adj_12 = {};
        wp::vec_t<3, wp::float32> adj_13 = {};
        wp::transform_t<wp::float32> adj_14 = {};
        wp::transform_t<wp::float32> adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        //---------
        // forward
        // def set_body_com_pose_to_sim(                                                          <L 1>
        // i, j = wp.tid()                                                                        <L 29>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 30>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 31>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 32>
        if (var_from_mask) {
            // body_com_pose_w[env_id, body_id] = data[env_id, body_id]                           <L 33>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            // wp::array_store(var_body_com_pose_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_com_pose_w[env_id, body_id] = data[i, j]                                      <L 35>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            // wp::array_store(var_body_com_pose_w, var_3, var_6, var_11);
        }
        // body_link_pose_w[env_id, body_id] = get_com_pose_in_link_frame_func(                   <L 37>
        // body_com_pose_w[env_id, body_id], body_com_pos_b[env_id, body_id]                      <L 38>
        var_12 = wp::address(var_body_com_pose_w, var_3, var_6);
        var_13 = wp::address(var_body_com_pos_b, var_3, var_6);
        var_15 = wp::load(var_12);
        var_16 = wp::load(var_13);
        var_14 = get_com_pose_in_link_frame_func_0(var_15, var_16);
        // body_link_pose_w[env_id, body_id] = get_com_pose_in_link_frame_func(                   <L 37>
        // wp::array_store(var_body_link_pose_w, var_3, var_6, var_14);
        //---------
        // reverse
        wp::adj_array_store(var_body_link_pose_w, var_3, var_6, var_14, adj_body_link_pose_w, adj_3, adj_6, adj_14);
        // adj: body_link_pose_w[env_id, body_id] = get_com_pose_in_link_frame_func(              <L 37>
        adj_get_com_pose_in_link_frame_func_0(var_15, var_16, adj_12, adj_13, adj_14);
        wp::adj_address(var_body_com_pos_b, var_3, var_6, adj_body_com_pos_b, adj_3, adj_6, adj_13);
        wp::adj_address(var_body_com_pose_w, var_3, var_6, adj_body_com_pose_w, adj_3, adj_6, adj_12);
        // adj: body_com_pose_w[env_id, body_id], body_com_pos_b[env_id, body_id]                 <L 38>
        // adj: body_link_pose_w[env_id, body_id] = get_com_pose_in_link_frame_func(              <L 37>
        if (!var_from_mask) {
            wp::adj_array_store(var_body_com_pose_w, var_3, var_6, var_11, adj_body_com_pose_w, adj_3, adj_6, adj_10);
            wp::adj_address(var_data, var_0, var_1, adj_data, adj_0, adj_1, adj_10);
            // adj: body_com_pose_w[env_id, body_id] = data[i, j]                                 <L 35>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_body_com_pose_w, var_3, var_6, var_9, adj_body_com_pose_w, adj_3, adj_6, adj_8);
            wp::adj_address(var_data, var_3, var_6, adj_data, adj_3, adj_6, adj_8);
            // adj: body_com_pose_w[env_id, body_id] = data[env_id, body_id]                      <L 33>
        }
        // adj: if from_mask:                                                                     <L 32>
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_5);
        // adj: body_id = wp.int32(body_ids[j])                                                   <L 31>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 30>
        // adj: i, j = wp.tid()                                                                   <L 29>
        // adj: def set_body_com_pose_to_sim(                                                     <L 1>
        continue;
    }
}



extern "C" __global__ void write_body_mass_and_inverse_index_683fe04e_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_masses,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    wp::array_t<wp::int32> var_body_user_to_backend,
    bool var_has_body_ordering,
    wp::array_t<wp::float32> var_body_inertia_backend,
    wp::array_t<wp::float32> var_body_mass_user,
    wp::array_t<wp::float32> var_body_mass_backend,
    wp::array_t<wp::float32> var_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_body_inv_inertia)
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
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        const bool var_16 = true;
        //---------
        // forward
        // def write_body_mass_and_inverse_index(                                                 <L 1>
        // i, j = wp.tid()                                                                        <L 27>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 28>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_body_id = wp.int32(body_ids[j])                                                   <L 29>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // backend_body_id = user_body_id                                                         <L 30>
        var_8 = wp::copy(var_6);
        // if has_body_ordering:                                                                  <L 31>
        if (var_has_body_ordering) {
            // backend_body_id = body_user_to_backend[user_body_id]                               <L 32>
            var_9 = wp::address(var_body_user_to_backend, var_6);
            var_11 = wp::load(var_9);
            var_10 = wp::copy(var_11);
        }
        var_12 = wp::where(var_has_body_ordering, var_10, var_8);
        // mass = masses[i, j]                                                                    <L 33>
        var_13 = wp::address(var_masses, var_0, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // body_mass_user[env_id, user_body_id] = mass                                            <L 34>
        wp::array_store(var_body_mass_user, var_3, var_6, var_14);
        // if has_body_ordering:                                                                  <L 35>
        if (var_has_body_ordering) {
            // body_mass_backend[env_id, backend_body_id] = mass                                  <L 36>
            wp::array_store(var_body_mass_backend, var_3, var_12, var_14);
        }
        // _update_body_inertial_properties_inverse(                                              <L 37>
        // env_id, backend_body_id, body_mass_backend, body_inertia_backend, True, body_inv_mass, body_inv_inertia       <L 38>
        _update_body_inertial_properties_inverse_0(var_3, var_12, var_body_mass_backend, var_body_inertia_backend, var_16, var_body_inv_mass, var_body_inv_inertia);
    }
}



extern "C" __global__ void write_body_mass_and_inverse_index_683fe04e_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_masses,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    wp::array_t<wp::int32> var_body_user_to_backend,
    bool var_has_body_ordering,
    wp::array_t<wp::float32> var_body_inertia_backend,
    wp::array_t<wp::float32> var_body_mass_user,
    wp::array_t<wp::float32> var_body_mass_backend,
    wp::array_t<wp::float32> var_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_body_inv_inertia,
    wp::array_t<wp::float32> adj_masses,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_body_ids,
    wp::array_t<wp::int32> adj_body_user_to_backend,
    bool adj_has_body_ordering,
    wp::array_t<wp::float32> adj_body_inertia_backend,
    wp::array_t<wp::float32> adj_body_mass_user,
    wp::array_t<wp::float32> adj_body_mass_backend,
    wp::array_t<wp::float32> adj_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> adj_body_inv_inertia)
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
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        const bool var_16 = true;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        bool adj_16 = {};
        //---------
        // forward
        // def write_body_mass_and_inverse_index(                                                 <L 1>
        // i, j = wp.tid()                                                                        <L 27>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 28>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_body_id = wp.int32(body_ids[j])                                                   <L 29>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // backend_body_id = user_body_id                                                         <L 30>
        var_8 = wp::copy(var_6);
        // if has_body_ordering:                                                                  <L 31>
        if (var_has_body_ordering) {
            // backend_body_id = body_user_to_backend[user_body_id]                               <L 32>
            var_9 = wp::address(var_body_user_to_backend, var_6);
            var_11 = wp::load(var_9);
            var_10 = wp::copy(var_11);
        }
        var_12 = wp::where(var_has_body_ordering, var_10, var_8);
        // mass = masses[i, j]                                                                    <L 33>
        var_13 = wp::address(var_masses, var_0, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // body_mass_user[env_id, user_body_id] = mass                                            <L 34>
        // wp::array_store(var_body_mass_user, var_3, var_6, var_14);
        // if has_body_ordering:                                                                  <L 35>
        if (var_has_body_ordering) {
            // body_mass_backend[env_id, backend_body_id] = mass                                  <L 36>
            // wp::array_store(var_body_mass_backend, var_3, var_12, var_14);
        }
        // _update_body_inertial_properties_inverse(                                              <L 37>
        // env_id, backend_body_id, body_mass_backend, body_inertia_backend, True, body_inv_mass, body_inv_inertia       <L 38>
        _update_body_inertial_properties_inverse_0(var_3, var_12, var_body_mass_backend, var_body_inertia_backend, var_16, var_body_inv_mass, var_body_inv_inertia);
        //---------
        // reverse
        adj__update_body_inertial_properties_inverse_0(var_3, var_12, var_body_mass_backend, var_body_inertia_backend, var_16, var_body_inv_mass, var_body_inv_inertia, adj_3, adj_12, adj_body_mass_backend, adj_body_inertia_backend, adj_16, adj_body_inv_mass, adj_body_inv_inertia);
        // adj: env_id, backend_body_id, body_mass_backend, body_inertia_backend, True, body_inv_mass, body_inv_inertia  <L 38>
        // adj: _update_body_inertial_properties_inverse(                                         <L 37>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_mass_backend, var_3, var_12, var_14, adj_body_mass_backend, adj_3, adj_12, adj_14);
            // adj: body_mass_backend[env_id, backend_body_id] = mass                             <L 36>
        }
        // adj: if has_body_ordering:                                                             <L 35>
        wp::adj_array_store(var_body_mass_user, var_3, var_6, var_14, adj_body_mass_user, adj_3, adj_6, adj_14);
        // adj: body_mass_user[env_id, user_body_id] = mass                                       <L 34>
        wp::adj_copy(var_15, adj_13, adj_14);
        wp::adj_address(var_masses, var_0, var_1, adj_masses, adj_0, adj_1, adj_13);
        // adj: mass = masses[i, j]                                                               <L 33>
        wp::adj_where(var_has_body_ordering, var_10, var_8, adj_has_body_ordering, adj_10, adj_8, adj_12);
        if (var_has_body_ordering) {
            wp::adj_copy(var_11, adj_9, adj_10);
            wp::adj_address(var_body_user_to_backend, var_6, adj_body_user_to_backend, adj_6, adj_9);
            // adj: backend_body_id = body_user_to_backend[user_body_id]                          <L 32>
        }
        // adj: if has_body_ordering:                                                             <L 31>
        wp::adj_copy(var_6, adj_6, adj_8);
        // adj: backend_body_id = user_body_id                                                    <L 30>
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_5);
        // adj: user_body_id = wp.int32(body_ids[j])                                              <L 29>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 28>
        // adj: i, j = wp.tid()                                                                   <L 27>
        // adj: def write_body_mass_and_inverse_index(                                            <L 1>
        continue;
    }
}



extern "C" __global__ void set_root_com_velocity_to_sim_mask_6e527579_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<bool> var_env_mask,
    wp::int32 var_num_bodies,
    wp::array_t<wp::vec_t<6, wp::float32>> var_root_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w)
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
        bool* var_1;
        bool var_2;
        wp::vec_t<6, wp::float32>* var_3;
        wp::vec_t<6, wp::float32> var_4;
        wp::range_t var_5;
        wp::int32 var_6;
        const wp::float32 var_7 = 0.0;
        const wp::float32 var_8 = 0.0;
        const wp::float32 var_9 = 0.0;
        const wp::float32 var_10 = 0.0;
        const wp::float32 var_11 = 0.0;
        const wp::float32 var_12 = 0.0;
        wp::vec_t<6, wp::float32> var_13;
        bool var_14;
        //---------
        // forward
        // def set_root_com_velocity_to_sim_mask(                                                 <L 654>
        // i = wp.tid()                                                                           <L 674>
        var_0 = builtin_tid1d();
        // if env_mask[i]:                                                                        <L 675>
        var_1 = wp::address(var_env_mask, var_0);
        var_2 = wp::load(var_1);
        if (var_2) {
            // root_com_velocity_w[i] = data[i]                                                   <L 676>
            var_3 = wp::address(var_data, var_0);
            var_4 = wp::load(var_3);
            wp::array_store(var_root_com_velocity_w, var_0, var_4);
            // for j in range(num_bodies):                                                        <L 678>
            var_5 = wp::range(var_num_bodies);
            start_for_0:;
                if (iter_cmp(var_5) == 0) goto end_for_0;
                var_6 = wp::iter_next(var_5);
                // body_acc_w[i, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)            <L 679>
                var_13 = wp::vec_t<6, wp::float32>({var_7, var_8, var_9, var_10, var_11, var_12});
                wp::array_store(var_body_acc_w, var_0, var_6, var_13);
                goto start_for_0;
            end_for_0:;
        }
        var_14 = wp::load(var_1);
    }
}



extern "C" __global__ void set_root_com_velocity_to_sim_mask_6e527579_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<bool> var_env_mask,
    wp::int32 var_num_bodies,
    wp::array_t<wp::vec_t<6, wp::float32>> var_root_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_data,
    wp::array_t<bool> adj_env_mask,
    wp::int32 adj_num_bodies,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_root_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_acc_w)
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
        bool* var_1;
        bool var_2;
        wp::vec_t<6, wp::float32>* var_3;
        wp::vec_t<6, wp::float32> var_4;
        wp::range_t var_5;
        wp::int32 var_6;
        const wp::float32 var_7 = 0.0;
        const wp::float32 var_8 = 0.0;
        const wp::float32 var_9 = 0.0;
        const wp::float32 var_10 = 0.0;
        const wp::float32 var_11 = 0.0;
        const wp::float32 var_12 = 0.0;
        wp::vec_t<6, wp::float32> var_13;
        bool var_14;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        bool adj_2 = {};
        wp::vec_t<6, wp::float32> adj_3 = {};
        wp::vec_t<6, wp::float32> adj_4 = {};
        wp::range_t adj_5 = {};
        wp::int32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::vec_t<6, wp::float32> adj_13 = {};
        bool adj_14 = {};
        //---------
        // forward
        // def set_root_com_velocity_to_sim_mask(                                                 <L 654>
        // i = wp.tid()                                                                           <L 674>
        var_0 = builtin_tid1d();
        // if env_mask[i]:                                                                        <L 675>
        var_1 = wp::address(var_env_mask, var_0);
        var_2 = wp::load(var_1);
        if (var_2) {
            // root_com_velocity_w[i] = data[i]                                                   <L 676>
            var_3 = wp::address(var_data, var_0);
            var_4 = wp::load(var_3);
            // wp::array_store(var_root_com_velocity_w, var_0, var_4);
            // for j in range(num_bodies):                                                        <L 678>
            var_5 = wp::range(var_num_bodies);
        }
        var_14 = wp::load(var_1);
        //---------
        // reverse
        if (var_14) {
            var_5 = wp::iter_reverse(var_5);
            start_for_0:;
                if (iter_cmp(var_5) == 0) goto end_for_0;
                var_6 = wp::iter_next(var_5);
            	adj_7 = {};
            	adj_8 = {};
            	adj_9 = {};
            	adj_10 = {};
            	adj_11 = {};
            	adj_12 = {};
            	adj_13 = {};
                // body_acc_w[i, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)            <L 679>
                var_13 = wp::vec_t<6, wp::float32>({var_7, var_8, var_9, var_10, var_11, var_12});
                // wp::array_store(var_body_acc_w, var_0, var_6, var_13);
                wp::adj_array_store(var_body_acc_w, var_0, var_6, var_13, adj_body_acc_w, adj_0, adj_6, adj_13);
                wp::adj_vec_t({var_7, var_8, var_9, var_10, var_11, var_12}, {&adj_7, &adj_8, &adj_9, &adj_10, &adj_11, &adj_12}, adj_13);
                // adj: body_acc_w[i, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)       <L 679>
            	goto start_for_0;
            end_for_0:;
            // adj: for j in range(num_bodies):                                                   <L 678>
            wp::adj_array_store(var_root_com_velocity_w, var_0, var_4, adj_root_com_velocity_w, adj_0, adj_3);
            wp::adj_address(var_data, var_0, adj_data, adj_0, adj_3);
            // adj: root_com_velocity_w[i] = data[i]                                              <L 676>
        }
        wp::adj_address(var_env_mask, var_0, adj_env_mask, adj_0, adj_1);
        // adj: if env_mask[i]:                                                                   <L 675>
        // adj: i = wp.tid()                                                                      <L 674>
        // adj: def set_root_com_velocity_to_sim_mask(                                            <L 654>
        continue;
    }
}



extern "C" __global__ void set_body_link_velocity_to_sim_72616017_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_link_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::vec_t<6, wp::float32>* var_8;
        wp::vec_t<6, wp::float32> var_9;
        wp::vec_t<6, wp::float32>* var_10;
        wp::vec_t<6, wp::float32> var_11;
        wp::vec_t<6, wp::float32>* var_12;
        wp::transform_t<wp::float32>* var_13;
        wp::vec_t<3, wp::float32>* var_14;
        wp::vec_t<6, wp::float32> var_15;
        wp::vec_t<6, wp::float32> var_16;
        wp::transform_t<wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        const wp::float32 var_19 = 0.0;
        const wp::float32 var_20 = 0.0;
        const wp::float32 var_21 = 0.0;
        const wp::float32 var_22 = 0.0;
        const wp::float32 var_23 = 0.0;
        const wp::float32 var_24 = 0.0;
        wp::vec_t<6, wp::float32> var_25;
        //---------
        // forward
        // def set_body_link_velocity_to_sim(                                                     <L 1>
        // i, j = wp.tid()                                                                        <L 35>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 36>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 37>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 38>
        if (var_from_mask) {
            // body_link_velocity_w[env_id, body_id] = data[env_id, body_id]                      <L 39>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_body_link_velocity_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_link_velocity_w[env_id, body_id] = data[i, j]                                 <L 41>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            wp::array_store(var_body_link_velocity_w, var_3, var_6, var_11);
        }
        // body_com_velocity_w[env_id, body_id] = get_link_velocity_in_com_frame_func(            <L 43>
        // body_link_velocity_w[env_id, body_id],                                                 <L 44>
        var_12 = wp::address(var_body_link_velocity_w, var_3, var_6);
        // body_link_pose_w[env_id, body_id],                                                     <L 45>
        var_13 = wp::address(var_body_link_pose_w, var_3, var_6);
        // body_com_pos_b[env_id, body_id],                                                       <L 46>
        var_14 = wp::address(var_body_com_pos_b, var_3, var_6);
        var_16 = wp::load(var_12);
        var_17 = wp::load(var_13);
        var_18 = wp::load(var_14);
        var_15 = get_link_velocity_in_com_frame_func_0(var_16, var_17, var_18);
        // body_com_velocity_w[env_id, body_id] = get_link_velocity_in_com_frame_func(            <L 43>
        wp::array_store(var_body_com_velocity_w, var_3, var_6, var_15);
        // body_acc_w[env_id, body_id] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)         <L 49>
        var_25 = wp::vec_t<6, wp::float32>({var_19, var_20, var_21, var_22, var_23, var_24});
        wp::array_store(var_body_acc_w, var_3, var_6, var_25);
    }
}



extern "C" __global__ void set_body_link_velocity_to_sim_72616017_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_link_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_data,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_link_pose_w,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int64> adj_body_ids,
    bool adj_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_link_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_acc_w)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::vec_t<6, wp::float32>* var_8;
        wp::vec_t<6, wp::float32> var_9;
        wp::vec_t<6, wp::float32>* var_10;
        wp::vec_t<6, wp::float32> var_11;
        wp::vec_t<6, wp::float32>* var_12;
        wp::transform_t<wp::float32>* var_13;
        wp::vec_t<3, wp::float32>* var_14;
        wp::vec_t<6, wp::float32> var_15;
        wp::vec_t<6, wp::float32> var_16;
        wp::transform_t<wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        const wp::float32 var_19 = 0.0;
        const wp::float32 var_20 = 0.0;
        const wp::float32 var_21 = 0.0;
        const wp::float32 var_22 = 0.0;
        const wp::float32 var_23 = 0.0;
        const wp::float32 var_24 = 0.0;
        wp::vec_t<6, wp::float32> var_25;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int64 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int64 adj_7 = {};
        wp::vec_t<6, wp::float32> adj_8 = {};
        wp::vec_t<6, wp::float32> adj_9 = {};
        wp::vec_t<6, wp::float32> adj_10 = {};
        wp::vec_t<6, wp::float32> adj_11 = {};
        wp::vec_t<6, wp::float32> adj_12 = {};
        wp::transform_t<wp::float32> adj_13 = {};
        wp::vec_t<3, wp::float32> adj_14 = {};
        wp::vec_t<6, wp::float32> adj_15 = {};
        wp::vec_t<6, wp::float32> adj_16 = {};
        wp::transform_t<wp::float32> adj_17 = {};
        wp::vec_t<3, wp::float32> adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::vec_t<6, wp::float32> adj_25 = {};
        //---------
        // forward
        // def set_body_link_velocity_to_sim(                                                     <L 1>
        // i, j = wp.tid()                                                                        <L 35>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 36>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 37>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 38>
        if (var_from_mask) {
            // body_link_velocity_w[env_id, body_id] = data[env_id, body_id]                      <L 39>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            // wp::array_store(var_body_link_velocity_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_link_velocity_w[env_id, body_id] = data[i, j]                                 <L 41>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            // wp::array_store(var_body_link_velocity_w, var_3, var_6, var_11);
        }
        // body_com_velocity_w[env_id, body_id] = get_link_velocity_in_com_frame_func(            <L 43>
        // body_link_velocity_w[env_id, body_id],                                                 <L 44>
        var_12 = wp::address(var_body_link_velocity_w, var_3, var_6);
        // body_link_pose_w[env_id, body_id],                                                     <L 45>
        var_13 = wp::address(var_body_link_pose_w, var_3, var_6);
        // body_com_pos_b[env_id, body_id],                                                       <L 46>
        var_14 = wp::address(var_body_com_pos_b, var_3, var_6);
        var_16 = wp::load(var_12);
        var_17 = wp::load(var_13);
        var_18 = wp::load(var_14);
        var_15 = get_link_velocity_in_com_frame_func_0(var_16, var_17, var_18);
        // body_com_velocity_w[env_id, body_id] = get_link_velocity_in_com_frame_func(            <L 43>
        // wp::array_store(var_body_com_velocity_w, var_3, var_6, var_15);
        // body_acc_w[env_id, body_id] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)         <L 49>
        var_25 = wp::vec_t<6, wp::float32>({var_19, var_20, var_21, var_22, var_23, var_24});
        // wp::array_store(var_body_acc_w, var_3, var_6, var_25);
        //---------
        // reverse
        wp::adj_array_store(var_body_acc_w, var_3, var_6, var_25, adj_body_acc_w, adj_3, adj_6, adj_25);
        wp::adj_vec_t({var_19, var_20, var_21, var_22, var_23, var_24}, {&adj_19, &adj_20, &adj_21, &adj_22, &adj_23, &adj_24}, adj_25);
        // adj: body_acc_w[env_id, body_id] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)    <L 49>
        wp::adj_array_store(var_body_com_velocity_w, var_3, var_6, var_15, adj_body_com_velocity_w, adj_3, adj_6, adj_15);
        // adj: body_com_velocity_w[env_id, body_id] = get_link_velocity_in_com_frame_func(       <L 43>
        adj_get_link_velocity_in_com_frame_func_0(var_16, var_17, var_18, adj_12, adj_13, adj_14, adj_15);
        wp::adj_address(var_body_com_pos_b, var_3, var_6, adj_body_com_pos_b, adj_3, adj_6, adj_14);
        // adj: body_com_pos_b[env_id, body_id],                                                  <L 46>
        wp::adj_address(var_body_link_pose_w, var_3, var_6, adj_body_link_pose_w, adj_3, adj_6, adj_13);
        // adj: body_link_pose_w[env_id, body_id],                                                <L 45>
        wp::adj_address(var_body_link_velocity_w, var_3, var_6, adj_body_link_velocity_w, adj_3, adj_6, adj_12);
        // adj: body_link_velocity_w[env_id, body_id],                                            <L 44>
        // adj: body_com_velocity_w[env_id, body_id] = get_link_velocity_in_com_frame_func(       <L 43>
        if (!var_from_mask) {
            wp::adj_array_store(var_body_link_velocity_w, var_3, var_6, var_11, adj_body_link_velocity_w, adj_3, adj_6, adj_10);
            wp::adj_address(var_data, var_0, var_1, adj_data, adj_0, adj_1, adj_10);
            // adj: body_link_velocity_w[env_id, body_id] = data[i, j]                            <L 41>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_body_link_velocity_w, var_3, var_6, var_9, adj_body_link_velocity_w, adj_3, adj_6, adj_8);
            wp::adj_address(var_data, var_3, var_6, adj_data, adj_3, adj_6, adj_8);
            // adj: body_link_velocity_w[env_id, body_id] = data[env_id, body_id]                 <L 39>
        }
        // adj: if from_mask:                                                                     <L 38>
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_5);
        // adj: body_id = wp.int32(body_ids[j])                                                   <L 37>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 36>
        // adj: i, j = wp.tid()                                                                   <L 35>
        // adj: def set_body_link_velocity_to_sim(                                                <L 1>
        continue;
    }
}



extern "C" __global__ void update_wrench_array_with_force_and_torque_7313e446_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> var_torques,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_wrench,
    wp::array_t<bool> var_env_ids,
    wp::array_t<bool> var_body_ids)
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
        bool var_2;
        bool* var_3;
        bool var_4;
        bool* var_5;
        bool var_6;
        wp::transform_t<wp::float32>* var_7;
        wp::quat_t<wp::float32> var_8;
        wp::transform_t<wp::float32> var_9;
        wp::vec_t<3, wp::float32>* var_10;
        wp::vec_t<3, wp::float32>* var_11;
        wp::vec_t<6, wp::float32> var_12;
        wp::vec_t<3, wp::float32> var_13;
        wp::vec_t<3, wp::float32> var_14;
        //---------
        // forward
        // def update_wrench_array_with_force_and_torque(                                         <L 1393>
        // env_index, body_index = wp.tid()                                                       <L 1416>
        builtin_tid2d(var_0, var_1);
        // if env_ids[env_index] and body_ids[body_index]:                                        <L 1417>
        var_3 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_3);
        var_2 = var_4;
        if (var_2) {
            var_5 = wp::address(var_body_ids, var_1);
            var_6 = wp::load(var_5);
            var_2 = var_2 && var_6;
        }
        if (var_2) {
            // body_rot_w = wp.transform_get_rotation(body_link_pose_w[env_index, body_index])       <L 1418>
            var_7 = wp::address(var_body_link_pose_w, var_0, var_1);
            var_9 = wp::load(var_7);
            var_8 = wp::transform_get_rotation(var_9);
            // wrench[env_index, body_index] = pack_body_wrench_to_world(                         <L 1419>
            // forces[env_index, body_index],                                                     <L 1420>
            var_10 = wp::address(var_forces, var_0, var_1);
            // torques[env_index, body_index],                                                    <L 1421>
            var_11 = wp::address(var_torques, var_0, var_1);
            // body_rot_w,                                                                        <L 1422>
            var_13 = wp::load(var_10);
            var_14 = wp::load(var_11);
            var_12 = pack_body_wrench_to_world_0(var_13, var_14, var_8);
            // wrench[env_index, body_index] = pack_body_wrench_to_world(                         <L 1419>
            wp::array_store(var_wrench, var_0, var_1, var_12);
        }
    }
}



extern "C" __global__ void update_wrench_array_with_force_and_torque_7313e446_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> var_torques,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_wrench,
    wp::array_t<bool> var_env_ids,
    wp::array_t<bool> var_body_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_torques,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_link_pose_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_wrench,
    wp::array_t<bool> adj_env_ids,
    wp::array_t<bool> adj_body_ids)
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
        bool var_2;
        bool* var_3;
        bool var_4;
        bool* var_5;
        bool var_6;
        wp::transform_t<wp::float32>* var_7;
        wp::quat_t<wp::float32> var_8;
        wp::transform_t<wp::float32> var_9;
        wp::vec_t<3, wp::float32>* var_10;
        wp::vec_t<3, wp::float32>* var_11;
        wp::vec_t<6, wp::float32> var_12;
        wp::vec_t<3, wp::float32> var_13;
        wp::vec_t<3, wp::float32> var_14;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        bool adj_6 = {};
        wp::transform_t<wp::float32> adj_7 = {};
        wp::quat_t<wp::float32> adj_8 = {};
        wp::transform_t<wp::float32> adj_9 = {};
        wp::vec_t<3, wp::float32> adj_10 = {};
        wp::vec_t<3, wp::float32> adj_11 = {};
        wp::vec_t<6, wp::float32> adj_12 = {};
        wp::vec_t<3, wp::float32> adj_13 = {};
        wp::vec_t<3, wp::float32> adj_14 = {};
        //---------
        // forward
        // def update_wrench_array_with_force_and_torque(                                         <L 1393>
        // env_index, body_index = wp.tid()                                                       <L 1416>
        builtin_tid2d(var_0, var_1);
        // if env_ids[env_index] and body_ids[body_index]:                                        <L 1417>
        var_3 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_3);
        var_2 = var_4;
        if (var_2) {
            var_5 = wp::address(var_body_ids, var_1);
            var_6 = wp::load(var_5);
            var_2 = var_2 && var_6;
        }
        if (var_2) {
            // body_rot_w = wp.transform_get_rotation(body_link_pose_w[env_index, body_index])       <L 1418>
            var_7 = wp::address(var_body_link_pose_w, var_0, var_1);
            var_9 = wp::load(var_7);
            var_8 = wp::transform_get_rotation(var_9);
            // wrench[env_index, body_index] = pack_body_wrench_to_world(                         <L 1419>
            // forces[env_index, body_index],                                                     <L 1420>
            var_10 = wp::address(var_forces, var_0, var_1);
            // torques[env_index, body_index],                                                    <L 1421>
            var_11 = wp::address(var_torques, var_0, var_1);
            // body_rot_w,                                                                        <L 1422>
            var_13 = wp::load(var_10);
            var_14 = wp::load(var_11);
            var_12 = pack_body_wrench_to_world_0(var_13, var_14, var_8);
            // wrench[env_index, body_index] = pack_body_wrench_to_world(                         <L 1419>
            // wp::array_store(var_wrench, var_0, var_1, var_12);
        }
        //---------
        // reverse
        if (var_2) {
            wp::adj_array_store(var_wrench, var_0, var_1, var_12, adj_wrench, adj_0, adj_1, adj_12);
            // adj: wrench[env_index, body_index] = pack_body_wrench_to_world(                    <L 1419>
            adj_pack_body_wrench_to_world_0(var_13, var_14, var_8, adj_10, adj_11, adj_8, adj_12);
            // adj: body_rot_w,                                                                   <L 1422>
            wp::adj_address(var_torques, var_0, var_1, adj_torques, adj_0, adj_1, adj_11);
            // adj: torques[env_index, body_index],                                               <L 1421>
            wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_10);
            // adj: forces[env_index, body_index],                                                <L 1420>
            // adj: wrench[env_index, body_index] = pack_body_wrench_to_world(                    <L 1419>
            wp::adj_transform_get_rotation(var_9, adj_7, adj_8);
            wp::adj_address(var_body_link_pose_w, var_0, var_1, adj_body_link_pose_w, adj_0, adj_1, adj_7);
            // adj: body_rot_w = wp.transform_get_rotation(body_link_pose_w[env_index, body_index])  <L 1418>
        }
        if (var_2) {
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_5);
        }
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_3);
        // adj: if env_ids[env_index] and body_ids[body_index]:                                   <L 1417>
        // adj: env_index, body_index = wp.tid()                                                  <L 1416>
        // adj: def update_wrench_array_with_force_and_torque(                                    <L 1393>
        continue;
    }
}



extern "C" __global__ void set_root_com_velocity_to_sim_index_7aa6b69e_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::int32 var_num_bodies,
    wp::array_t<wp::vec_t<6, wp::float32>> var_root_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w)
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
        wp::int64* var_1;
        wp::int32 var_2;
        wp::int64 var_3;
        wp::vec_t<6, wp::float32>* var_4;
        wp::vec_t<6, wp::float32> var_5;
        wp::range_t var_6;
        wp::int32 var_7;
        const wp::float32 var_8 = 0.0;
        const wp::float32 var_9 = 0.0;
        const wp::float32 var_10 = 0.0;
        const wp::float32 var_11 = 0.0;
        const wp::float32 var_12 = 0.0;
        const wp::float32 var_13 = 0.0;
        wp::vec_t<6, wp::float32> var_14;
        //---------
        // forward
        // def set_root_com_velocity_to_sim_index(                                                <L 1>
        // i = wp.tid()                                                                           <L 21>
        var_0 = builtin_tid1d();
        // env_id = wp.int32(env_ids[i])                                                          <L 22>
        var_1 = wp::address(var_env_ids, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::int32(var_3);
        // root_com_velocity_w[env_id] = data[i]                                                  <L 23>
        var_4 = wp::address(var_data, var_0);
        var_5 = wp::load(var_4);
        wp::array_store(var_root_com_velocity_w, var_2, var_5);
        // for j in range(num_bodies):                                                            <L 25>
        var_6 = wp::range(var_num_bodies);
        start_for_0:;
            if (iter_cmp(var_6) == 0) goto end_for_0;
            var_7 = wp::iter_next(var_6);
            // body_acc_w[env_id, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)           <L 26>
            var_14 = wp::vec_t<6, wp::float32>({var_8, var_9, var_10, var_11, var_12, var_13});
            wp::array_store(var_body_acc_w, var_2, var_7, var_14);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void set_root_com_velocity_to_sim_index_7aa6b69e_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::int32 var_num_bodies,
    wp::array_t<wp::vec_t<6, wp::float32>> var_root_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_data,
    wp::array_t<wp::int64> adj_env_ids,
    wp::int32 adj_num_bodies,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_root_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_acc_w)
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
        wp::int64* var_1;
        wp::int32 var_2;
        wp::int64 var_3;
        wp::vec_t<6, wp::float32>* var_4;
        wp::vec_t<6, wp::float32> var_5;
        wp::range_t var_6;
        wp::int32 var_7;
        const wp::float32 var_8 = 0.0;
        const wp::float32 var_9 = 0.0;
        const wp::float32 var_10 = 0.0;
        const wp::float32 var_11 = 0.0;
        const wp::float32 var_12 = 0.0;
        const wp::float32 var_13 = 0.0;
        wp::vec_t<6, wp::float32> var_14;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int64 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int64 adj_3 = {};
        wp::vec_t<6, wp::float32> adj_4 = {};
        wp::vec_t<6, wp::float32> adj_5 = {};
        wp::range_t adj_6 = {};
        wp::int32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::vec_t<6, wp::float32> adj_14 = {};
        //---------
        // forward
        // def set_root_com_velocity_to_sim_index(                                                <L 1>
        // i = wp.tid()                                                                           <L 21>
        var_0 = builtin_tid1d();
        // env_id = wp.int32(env_ids[i])                                                          <L 22>
        var_1 = wp::address(var_env_ids, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::int32(var_3);
        // root_com_velocity_w[env_id] = data[i]                                                  <L 23>
        var_4 = wp::address(var_data, var_0);
        var_5 = wp::load(var_4);
        // wp::array_store(var_root_com_velocity_w, var_2, var_5);
        // for j in range(num_bodies):                                                            <L 25>
        var_6 = wp::range(var_num_bodies);
        //---------
        // reverse
        var_6 = wp::iter_reverse(var_6);
        start_for_0:;
            if (iter_cmp(var_6) == 0) goto end_for_0;
            var_7 = wp::iter_next(var_6);
        	adj_8 = {};
        	adj_9 = {};
        	adj_10 = {};
        	adj_11 = {};
        	adj_12 = {};
        	adj_13 = {};
        	adj_14 = {};
            // body_acc_w[env_id, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)           <L 26>
            var_14 = wp::vec_t<6, wp::float32>({var_8, var_9, var_10, var_11, var_12, var_13});
            // wp::array_store(var_body_acc_w, var_2, var_7, var_14);
            wp::adj_array_store(var_body_acc_w, var_2, var_7, var_14, adj_body_acc_w, adj_2, adj_7, adj_14);
            wp::adj_vec_t({var_8, var_9, var_10, var_11, var_12, var_13}, {&adj_8, &adj_9, &adj_10, &adj_11, &adj_12, &adj_13}, adj_14);
            // adj: body_acc_w[env_id, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)      <L 26>
        	goto start_for_0;
        end_for_0:;
        // adj: for j in range(num_bodies):                                                       <L 25>
        wp::adj_array_store(var_root_com_velocity_w, var_2, var_5, adj_root_com_velocity_w, adj_2, adj_4);
        wp::adj_address(var_data, var_0, adj_data, adj_0, adj_4);
        // adj: root_com_velocity_w[env_id] = data[i]                                             <L 23>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_1);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 22>
        // adj: i = wp.tid()                                                                      <L 21>
        // adj: def set_root_com_velocity_to_sim_index(                                           <L 1>
        continue;
    }
}



extern "C" __global__ void write_2d_data_to_buffer_with_indices_7d8f1871_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    wp::array_t<wp::float32> var_out_data)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        //---------
        // forward
        // def write_2d_data_to_buffer_with_indices(                                              <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 19>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 20>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // out_data[env_id, joint_id] = in_data[i, j]                                             <L 21>
        var_8 = wp::address(var_in_data, var_0, var_1);
        var_9 = wp::load(var_8);
        wp::array_store(var_out_data, var_3, var_6, var_9);
    }
}



extern "C" __global__ void write_2d_data_to_buffer_with_indices_7d8f1871_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    wp::array_t<wp::float32> var_out_data,
    wp::array_t<wp::float32> adj_in_data,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int64> adj_joint_ids,
    wp::array_t<wp::float32> adj_out_data)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int64 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int64 adj_4 = {};
        wp::int64 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int64 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        //---------
        // forward
        // def write_2d_data_to_buffer_with_indices(                                              <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 19>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 20>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // out_data[env_id, joint_id] = in_data[i, j]                                             <L 21>
        var_8 = wp::address(var_in_data, var_0, var_1);
        var_9 = wp::load(var_8);
        // wp::array_store(var_out_data, var_3, var_6, var_9);
        //---------
        // reverse
        wp::adj_array_store(var_out_data, var_3, var_6, var_9, adj_out_data, adj_3, adj_6, adj_8);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_8);
        // adj: out_data[env_id, joint_id] = in_data[i, j]                                        <L 21>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 20>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 19>
        // adj: i, j = wp.tid()                                                                   <L 18>
        // adj: def write_2d_data_to_buffer_with_indices(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_body_com_pose_to_sim_8072226e_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::transform_t<wp::float32>* var_8;
        wp::transform_t<wp::float32> var_9;
        wp::transform_t<wp::float32>* var_10;
        wp::transform_t<wp::float32> var_11;
        wp::transform_t<wp::float32>* var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::transform_t<wp::float32> var_14;
        wp::transform_t<wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        //---------
        // forward
        // def set_body_com_pose_to_sim(                                                          <L 1>
        // i, j = wp.tid()                                                                        <L 29>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 30>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 31>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 32>
        if (var_from_mask) {
            // body_com_pose_w[env_id, body_id] = data[env_id, body_id]                           <L 33>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_body_com_pose_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_com_pose_w[env_id, body_id] = data[i, j]                                      <L 35>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            wp::array_store(var_body_com_pose_w, var_3, var_6, var_11);
        }
        // body_link_pose_w[env_id, body_id] = get_com_pose_in_link_frame_func(                   <L 37>
        // body_com_pose_w[env_id, body_id], body_com_pos_b[env_id, body_id]                      <L 38>
        var_12 = wp::address(var_body_com_pose_w, var_3, var_6);
        var_13 = wp::address(var_body_com_pos_b, var_3, var_6);
        var_15 = wp::load(var_12);
        var_16 = wp::load(var_13);
        var_14 = get_com_pose_in_link_frame_func_0(var_15, var_16);
        // body_link_pose_w[env_id, body_id] = get_com_pose_in_link_frame_func(                   <L 37>
        wp::array_store(var_body_link_pose_w, var_3, var_6, var_14);
    }
}



extern "C" __global__ void set_body_com_pose_to_sim_8072226e_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_data,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_body_com_pos_b,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int64> adj_body_ids,
    bool adj_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_link_pose_w)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::transform_t<wp::float32>* var_8;
        wp::transform_t<wp::float32> var_9;
        wp::transform_t<wp::float32>* var_10;
        wp::transform_t<wp::float32> var_11;
        wp::transform_t<wp::float32>* var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::transform_t<wp::float32> var_14;
        wp::transform_t<wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int64 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int64 adj_7 = {};
        wp::transform_t<wp::float32> adj_8 = {};
        wp::transform_t<wp::float32> adj_9 = {};
        wp::transform_t<wp::float32> adj_10 = {};
        wp::transform_t<wp::float32> adj_11 = {};
        wp::transform_t<wp::float32> adj_12 = {};
        wp::vec_t<3, wp::float32> adj_13 = {};
        wp::transform_t<wp::float32> adj_14 = {};
        wp::transform_t<wp::float32> adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        //---------
        // forward
        // def set_body_com_pose_to_sim(                                                          <L 1>
        // i, j = wp.tid()                                                                        <L 29>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 30>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 31>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 32>
        if (var_from_mask) {
            // body_com_pose_w[env_id, body_id] = data[env_id, body_id]                           <L 33>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            // wp::array_store(var_body_com_pose_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_com_pose_w[env_id, body_id] = data[i, j]                                      <L 35>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            // wp::array_store(var_body_com_pose_w, var_3, var_6, var_11);
        }
        // body_link_pose_w[env_id, body_id] = get_com_pose_in_link_frame_func(                   <L 37>
        // body_com_pose_w[env_id, body_id], body_com_pos_b[env_id, body_id]                      <L 38>
        var_12 = wp::address(var_body_com_pose_w, var_3, var_6);
        var_13 = wp::address(var_body_com_pos_b, var_3, var_6);
        var_15 = wp::load(var_12);
        var_16 = wp::load(var_13);
        var_14 = get_com_pose_in_link_frame_func_0(var_15, var_16);
        // body_link_pose_w[env_id, body_id] = get_com_pose_in_link_frame_func(                   <L 37>
        // wp::array_store(var_body_link_pose_w, var_3, var_6, var_14);
        //---------
        // reverse
        wp::adj_array_store(var_body_link_pose_w, var_3, var_6, var_14, adj_body_link_pose_w, adj_3, adj_6, adj_14);
        // adj: body_link_pose_w[env_id, body_id] = get_com_pose_in_link_frame_func(              <L 37>
        adj_get_com_pose_in_link_frame_func_0(var_15, var_16, adj_12, adj_13, adj_14);
        wp::adj_address(var_body_com_pos_b, var_3, var_6, adj_body_com_pos_b, adj_3, adj_6, adj_13);
        wp::adj_address(var_body_com_pose_w, var_3, var_6, adj_body_com_pose_w, adj_3, adj_6, adj_12);
        // adj: body_com_pose_w[env_id, body_id], body_com_pos_b[env_id, body_id]                 <L 38>
        // adj: body_link_pose_w[env_id, body_id] = get_com_pose_in_link_frame_func(              <L 37>
        if (!var_from_mask) {
            wp::adj_array_store(var_body_com_pose_w, var_3, var_6, var_11, adj_body_com_pose_w, adj_3, adj_6, adj_10);
            wp::adj_address(var_data, var_0, var_1, adj_data, adj_0, adj_1, adj_10);
            // adj: body_com_pose_w[env_id, body_id] = data[i, j]                                 <L 35>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_body_com_pose_w, var_3, var_6, var_9, adj_body_com_pose_w, adj_3, adj_6, adj_8);
            wp::adj_address(var_data, var_3, var_6, adj_data, adj_3, adj_6, adj_8);
            // adj: body_com_pose_w[env_id, body_id] = data[env_id, body_id]                      <L 33>
        }
        // adj: if from_mask:                                                                     <L 32>
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_5);
        // adj: body_id = wp.int32(body_ids[j])                                                   <L 31>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 30>
        // adj: i, j = wp.tid()                                                                   <L 29>
        // adj: def set_body_com_pose_to_sim(                                                     <L 1>
        continue;
    }
}



extern "C" __global__ void write_2d_data_to_buffer_with_indices_80812dfc_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    wp::array_t<wp::float32> var_out_data)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        //---------
        // forward
        // def write_2d_data_to_buffer_with_indices(                                              <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 19>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 20>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // out_data[env_id, joint_id] = in_data[i, j]                                             <L 21>
        var_8 = wp::address(var_in_data, var_0, var_1);
        var_9 = wp::load(var_8);
        wp::array_store(var_out_data, var_3, var_6, var_9);
    }
}



extern "C" __global__ void write_2d_data_to_buffer_with_indices_80812dfc_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_joint_ids,
    wp::array_t<wp::float32> var_out_data,
    wp::array_t<wp::float32> adj_in_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int64> adj_joint_ids,
    wp::array_t<wp::float32> adj_out_data)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int64 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int64 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        //---------
        // forward
        // def write_2d_data_to_buffer_with_indices(                                              <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 19>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // joint_id = wp.int32(joint_ids[j])                                                      <L 20>
        var_5 = wp::address(var_joint_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // out_data[env_id, joint_id] = in_data[i, j]                                             <L 21>
        var_8 = wp::address(var_in_data, var_0, var_1);
        var_9 = wp::load(var_8);
        // wp::array_store(var_out_data, var_3, var_6, var_9);
        //---------
        // reverse
        wp::adj_array_store(var_out_data, var_3, var_6, var_9, adj_out_data, adj_3, adj_6, adj_8);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_8);
        // adj: out_data[env_id, joint_id] = in_data[i, j]                                        <L 21>
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_5);
        // adj: joint_id = wp.int32(joint_ids[j])                                                 <L 20>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 19>
        // adj: i, j = wp.tid()                                                                   <L 18>
        // adj: def write_2d_data_to_buffer_with_indices(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void write_body_inertia_and_inverse_index_8291b6e6_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_inertias,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    wp::array_t<wp::int32> var_body_user_to_backend,
    bool var_has_body_ordering,
    wp::array_t<wp::float32> var_body_mass_backend,
    wp::array_t<wp::float32> var_body_inertia_user,
    wp::array_t<wp::float32> var_body_inertia_backend,
    wp::array_t<wp::float32> var_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_body_inv_inertia)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 0;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        const wp::int32 var_17 = 1;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        const wp::int32 var_21 = 2;
        wp::float32* var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        const wp::int32 var_25 = 3;
        wp::float32* var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        const wp::int32 var_29 = 4;
        wp::float32* var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        const wp::int32 var_33 = 5;
        wp::float32* var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        const wp::int32 var_37 = 6;
        wp::float32* var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        const wp::int32 var_41 = 7;
        wp::float32* var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        const wp::int32 var_45 = 8;
        wp::float32* var_46;
        wp::float32 var_47;
        wp::float32 var_48;
        const bool var_49 = false;
        //---------
        // forward
        // def write_body_inertia_and_inverse_index(                                              <L 1>
        // i, j = wp.tid()                                                                        <L 14>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 15>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_body_id = wp.int32(body_ids[j])                                                   <L 16>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // backend_body_id = user_body_id                                                         <L 17>
        var_8 = wp::copy(var_6);
        // if has_body_ordering:                                                                  <L 18>
        if (var_has_body_ordering) {
            // backend_body_id = body_user_to_backend[user_body_id]                               <L 19>
            var_9 = wp::address(var_body_user_to_backend, var_6);
            var_11 = wp::load(var_9);
            var_10 = wp::copy(var_11);
        }
        var_12 = wp::where(var_has_body_ordering, var_10, var_8);
        // for k in range(9):                                                                     <L 20>
        // value = inertias[i, j, k]                                                              <L 21>
        var_14 = wp::address(var_inertias, var_0, var_1, var_13);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_13, var_15);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_13, var_15);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_18 = wp::address(var_inertias, var_0, var_1, var_17);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_17, var_19);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_17, var_19);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_22 = wp::address(var_inertias, var_0, var_1, var_21);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_21, var_23);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_21, var_23);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_26 = wp::address(var_inertias, var_0, var_1, var_25);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_25, var_27);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_25, var_27);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_30 = wp::address(var_inertias, var_0, var_1, var_29);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_29, var_31);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_29, var_31);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_34 = wp::address(var_inertias, var_0, var_1, var_33);
        var_36 = wp::load(var_34);
        var_35 = wp::copy(var_36);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_33, var_35);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_33, var_35);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_38 = wp::address(var_inertias, var_0, var_1, var_37);
        var_40 = wp::load(var_38);
        var_39 = wp::copy(var_40);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_37, var_39);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_37, var_39);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_42 = wp::address(var_inertias, var_0, var_1, var_41);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_41, var_43);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_41, var_43);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_46 = wp::address(var_inertias, var_0, var_1, var_45);
        var_48 = wp::load(var_46);
        var_47 = wp::copy(var_48);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_45, var_47);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_45, var_47);
        }
        // _update_body_inertial_properties_inverse(                                              <L 25>
        // env_id, backend_body_id, body_mass_backend, body_inertia_backend, False, body_inv_mass, body_inv_inertia       <L 26>
        _update_body_inertial_properties_inverse_0(var_3, var_12, var_body_mass_backend, var_body_inertia_backend, var_49, var_body_inv_mass, var_body_inv_inertia);
    }
}



extern "C" __global__ void write_body_inertia_and_inverse_index_8291b6e6_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_inertias,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    wp::array_t<wp::int32> var_body_user_to_backend,
    bool var_has_body_ordering,
    wp::array_t<wp::float32> var_body_mass_backend,
    wp::array_t<wp::float32> var_body_inertia_user,
    wp::array_t<wp::float32> var_body_inertia_backend,
    wp::array_t<wp::float32> var_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_body_inv_inertia,
    wp::array_t<wp::float32> adj_inertias,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int64> adj_body_ids,
    wp::array_t<wp::int32> adj_body_user_to_backend,
    bool adj_has_body_ordering,
    wp::array_t<wp::float32> adj_body_mass_backend,
    wp::array_t<wp::float32> adj_body_inertia_user,
    wp::array_t<wp::float32> adj_body_inertia_backend,
    wp::array_t<wp::float32> adj_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> adj_body_inv_inertia)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 0;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        const wp::int32 var_17 = 1;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        const wp::int32 var_21 = 2;
        wp::float32* var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        const wp::int32 var_25 = 3;
        wp::float32* var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        const wp::int32 var_29 = 4;
        wp::float32* var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        const wp::int32 var_33 = 5;
        wp::float32* var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        const wp::int32 var_37 = 6;
        wp::float32* var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        const wp::int32 var_41 = 7;
        wp::float32* var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        const wp::int32 var_45 = 8;
        wp::float32* var_46;
        wp::float32 var_47;
        wp::float32 var_48;
        const bool var_49 = false;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int64 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int64 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::float32 adj_39 = {};
        wp::float32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::float32 adj_42 = {};
        wp::float32 adj_43 = {};
        wp::float32 adj_44 = {};
        wp::int32 adj_45 = {};
        wp::float32 adj_46 = {};
        wp::float32 adj_47 = {};
        wp::float32 adj_48 = {};
        bool adj_49 = {};
        //---------
        // forward
        // def write_body_inertia_and_inverse_index(                                              <L 1>
        // i, j = wp.tid()                                                                        <L 14>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 15>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_body_id = wp.int32(body_ids[j])                                                   <L 16>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // backend_body_id = user_body_id                                                         <L 17>
        var_8 = wp::copy(var_6);
        // if has_body_ordering:                                                                  <L 18>
        if (var_has_body_ordering) {
            // backend_body_id = body_user_to_backend[user_body_id]                               <L 19>
            var_9 = wp::address(var_body_user_to_backend, var_6);
            var_11 = wp::load(var_9);
            var_10 = wp::copy(var_11);
        }
        var_12 = wp::where(var_has_body_ordering, var_10, var_8);
        // for k in range(9):                                                                     <L 20>
        // value = inertias[i, j, k]                                                              <L 21>
        var_14 = wp::address(var_inertias, var_0, var_1, var_13);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_13, var_15);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_13, var_15);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_18 = wp::address(var_inertias, var_0, var_1, var_17);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_17, var_19);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_17, var_19);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_22 = wp::address(var_inertias, var_0, var_1, var_21);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_21, var_23);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_21, var_23);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_26 = wp::address(var_inertias, var_0, var_1, var_25);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_25, var_27);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_25, var_27);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_30 = wp::address(var_inertias, var_0, var_1, var_29);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_29, var_31);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_29, var_31);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_34 = wp::address(var_inertias, var_0, var_1, var_33);
        var_36 = wp::load(var_34);
        var_35 = wp::copy(var_36);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_33, var_35);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_33, var_35);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_38 = wp::address(var_inertias, var_0, var_1, var_37);
        var_40 = wp::load(var_38);
        var_39 = wp::copy(var_40);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_37, var_39);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_37, var_39);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_42 = wp::address(var_inertias, var_0, var_1, var_41);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_41, var_43);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_41, var_43);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_46 = wp::address(var_inertias, var_0, var_1, var_45);
        var_48 = wp::load(var_46);
        var_47 = wp::copy(var_48);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_45, var_47);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_45, var_47);
        }
        // _update_body_inertial_properties_inverse(                                              <L 25>
        // env_id, backend_body_id, body_mass_backend, body_inertia_backend, False, body_inv_mass, body_inv_inertia       <L 26>
        _update_body_inertial_properties_inverse_0(var_3, var_12, var_body_mass_backend, var_body_inertia_backend, var_49, var_body_inv_mass, var_body_inv_inertia);
        //---------
        // reverse
        adj__update_body_inertial_properties_inverse_0(var_3, var_12, var_body_mass_backend, var_body_inertia_backend, var_49, var_body_inv_mass, var_body_inv_inertia, adj_3, adj_12, adj_body_mass_backend, adj_body_inertia_backend, adj_49, adj_body_inv_mass, adj_body_inv_inertia);
        // adj: env_id, backend_body_id, body_mass_backend, body_inertia_backend, False, body_inv_mass, body_inv_inertia  <L 26>
        // adj: _update_body_inertial_properties_inverse(                                         <L 25>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_45, var_47, adj_body_inertia_backend, adj_3, adj_12, adj_45, adj_47);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_45, var_47, adj_body_inertia_user, adj_3, adj_6, adj_45, adj_47);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_48, adj_46, adj_47);
        wp::adj_address(var_inertias, var_0, var_1, var_45, adj_inertias, adj_0, adj_1, adj_45, adj_46);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_41, var_43, adj_body_inertia_backend, adj_3, adj_12, adj_41, adj_43);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_41, var_43, adj_body_inertia_user, adj_3, adj_6, adj_41, adj_43);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_44, adj_42, adj_43);
        wp::adj_address(var_inertias, var_0, var_1, var_41, adj_inertias, adj_0, adj_1, adj_41, adj_42);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_37, var_39, adj_body_inertia_backend, adj_3, adj_12, adj_37, adj_39);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_37, var_39, adj_body_inertia_user, adj_3, adj_6, adj_37, adj_39);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_40, adj_38, adj_39);
        wp::adj_address(var_inertias, var_0, var_1, var_37, adj_inertias, adj_0, adj_1, adj_37, adj_38);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_33, var_35, adj_body_inertia_backend, adj_3, adj_12, adj_33, adj_35);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_33, var_35, adj_body_inertia_user, adj_3, adj_6, adj_33, adj_35);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_36, adj_34, adj_35);
        wp::adj_address(var_inertias, var_0, var_1, var_33, adj_inertias, adj_0, adj_1, adj_33, adj_34);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_29, var_31, adj_body_inertia_backend, adj_3, adj_12, adj_29, adj_31);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_29, var_31, adj_body_inertia_user, adj_3, adj_6, adj_29, adj_31);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_32, adj_30, adj_31);
        wp::adj_address(var_inertias, var_0, var_1, var_29, adj_inertias, adj_0, adj_1, adj_29, adj_30);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_25, var_27, adj_body_inertia_backend, adj_3, adj_12, adj_25, adj_27);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_25, var_27, adj_body_inertia_user, adj_3, adj_6, adj_25, adj_27);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_28, adj_26, adj_27);
        wp::adj_address(var_inertias, var_0, var_1, var_25, adj_inertias, adj_0, adj_1, adj_25, adj_26);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_21, var_23, adj_body_inertia_backend, adj_3, adj_12, adj_21, adj_23);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_21, var_23, adj_body_inertia_user, adj_3, adj_6, adj_21, adj_23);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_24, adj_22, adj_23);
        wp::adj_address(var_inertias, var_0, var_1, var_21, adj_inertias, adj_0, adj_1, adj_21, adj_22);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_17, var_19, adj_body_inertia_backend, adj_3, adj_12, adj_17, adj_19);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_17, var_19, adj_body_inertia_user, adj_3, adj_6, adj_17, adj_19);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_20, adj_18, adj_19);
        wp::adj_address(var_inertias, var_0, var_1, var_17, adj_inertias, adj_0, adj_1, adj_17, adj_18);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_13, var_15, adj_body_inertia_backend, adj_3, adj_12, adj_13, adj_15);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_13, var_15, adj_body_inertia_user, adj_3, adj_6, adj_13, adj_15);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_16, adj_14, adj_15);
        wp::adj_address(var_inertias, var_0, var_1, var_13, adj_inertias, adj_0, adj_1, adj_13, adj_14);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        // adj: for k in range(9):                                                                <L 20>
        wp::adj_where(var_has_body_ordering, var_10, var_8, adj_has_body_ordering, adj_10, adj_8, adj_12);
        if (var_has_body_ordering) {
            wp::adj_copy(var_11, adj_9, adj_10);
            wp::adj_address(var_body_user_to_backend, var_6, adj_body_user_to_backend, adj_6, adj_9);
            // adj: backend_body_id = body_user_to_backend[user_body_id]                          <L 19>
        }
        // adj: if has_body_ordering:                                                             <L 18>
        wp::adj_copy(var_6, adj_6, adj_8);
        // adj: backend_body_id = user_body_id                                                    <L 17>
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_5);
        // adj: user_body_id = wp.int32(body_ids[j])                                              <L 16>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 15>
        // adj: i, j = wp.tid()                                                                   <L 14>
        // adj: def write_body_inertia_and_inverse_index(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_root_link_pose_to_sim_mask_85050265_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<bool> var_env_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_root_link_pose_w)
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
        bool* var_1;
        bool var_2;
        wp::transform_t<wp::float32>* var_3;
        wp::transform_t<wp::float32> var_4;
        bool var_5;
        //---------
        // forward
        // def set_root_link_pose_to_sim_mask(                                                    <L 547>
        // i = wp.tid()                                                                           <L 561>
        var_0 = builtin_tid1d();
        // if env_mask[i]:                                                                        <L 562>
        var_1 = wp::address(var_env_mask, var_0);
        var_2 = wp::load(var_1);
        if (var_2) {
            // root_link_pose_w[i] = data[i]                                                      <L 563>
            var_3 = wp::address(var_data, var_0);
            var_4 = wp::load(var_3);
            wp::array_store(var_root_link_pose_w, var_0, var_4);
        }
        var_5 = wp::load(var_1);
    }
}



extern "C" __global__ void set_root_link_pose_to_sim_mask_85050265_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<bool> var_env_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_root_link_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_data,
    wp::array_t<bool> adj_env_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_root_link_pose_w)
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
        bool* var_1;
        bool var_2;
        wp::transform_t<wp::float32>* var_3;
        wp::transform_t<wp::float32> var_4;
        bool var_5;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        bool adj_2 = {};
        wp::transform_t<wp::float32> adj_3 = {};
        wp::transform_t<wp::float32> adj_4 = {};
        bool adj_5 = {};
        //---------
        // forward
        // def set_root_link_pose_to_sim_mask(                                                    <L 547>
        // i = wp.tid()                                                                           <L 561>
        var_0 = builtin_tid1d();
        // if env_mask[i]:                                                                        <L 562>
        var_1 = wp::address(var_env_mask, var_0);
        var_2 = wp::load(var_1);
        if (var_2) {
            // root_link_pose_w[i] = data[i]                                                      <L 563>
            var_3 = wp::address(var_data, var_0);
            var_4 = wp::load(var_3);
            // wp::array_store(var_root_link_pose_w, var_0, var_4);
        }
        var_5 = wp::load(var_1);
        //---------
        // reverse
        if (var_5) {
            wp::adj_array_store(var_root_link_pose_w, var_0, var_4, adj_root_link_pose_w, adj_0, adj_3);
            wp::adj_address(var_data, var_0, adj_data, adj_0, adj_3);
            // adj: root_link_pose_w[i] = data[i]                                                 <L 563>
        }
        wp::adj_address(var_env_mask, var_0, adj_env_mask, adj_0, adj_1);
        // adj: if env_mask[i]:                                                                   <L 562>
        // adj: i = wp.tid()                                                                      <L 561>
        // adj: def set_root_link_pose_to_sim_mask(                                               <L 547>
        continue;
    }
}



extern "C" __global__ void write_body_com_position_to_buffer_index_91063869_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_in_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_data)
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
        wp::vec_t<3, wp::float32>* var_2;
        wp::int64* var_3;
        wp::int32 var_4;
        wp::int64 var_5;
        wp::int64* var_6;
        wp::int32 var_7;
        wp::int64 var_8;
        wp::vec_t<3, wp::float32> var_9;
        //---------
        // forward
        // def write_body_com_position_to_buffer_index(                                           <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // out_data[wp.int32(env_ids[i]), wp.int32(body_ids[j])] = in_data[i, j]                  <L 19>
        var_2 = wp::address(var_in_data, var_0, var_1);
        var_3 = wp::address(var_env_ids, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::int32(var_5);
        var_6 = wp::address(var_body_ids, var_1);
        var_8 = wp::load(var_6);
        var_7 = wp::int32(var_8);
        var_9 = wp::load(var_2);
        wp::array_store(var_out_data, var_4, var_7, var_9);
    }
}



extern "C" __global__ void write_body_com_position_to_buffer_index_91063869_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_in_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_data,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_in_data,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int64> adj_body_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_out_data)
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
        wp::vec_t<3, wp::float32>* var_2;
        wp::int64* var_3;
        wp::int32 var_4;
        wp::int64 var_5;
        wp::int64* var_6;
        wp::int32 var_7;
        wp::int64 var_8;
        wp::vec_t<3, wp::float32> var_9;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::vec_t<3, wp::float32> adj_2 = {};
        wp::int64 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int64 adj_5 = {};
        wp::int64 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int64 adj_8 = {};
        wp::vec_t<3, wp::float32> adj_9 = {};
        //---------
        // forward
        // def write_body_com_position_to_buffer_index(                                           <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // out_data[wp.int32(env_ids[i]), wp.int32(body_ids[j])] = in_data[i, j]                  <L 19>
        var_2 = wp::address(var_in_data, var_0, var_1);
        var_3 = wp::address(var_env_ids, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::int32(var_5);
        var_6 = wp::address(var_body_ids, var_1);
        var_8 = wp::load(var_6);
        var_7 = wp::int32(var_8);
        var_9 = wp::load(var_2);
        // wp::array_store(var_out_data, var_4, var_7, var_9);
        //---------
        // reverse
        wp::adj_array_store(var_out_data, var_4, var_7, var_9, adj_out_data, adj_4, adj_7, adj_2);
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_6);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_3);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_2);
        // adj: out_data[wp.int32(env_ids[i]), wp.int32(body_ids[j])] = in_data[i, j]             <L 19>
        // adj: i, j = wp.tid()                                                                   <L 18>
        // adj: def write_body_com_position_to_buffer_index(                                      <L 1>
        continue;
    }
}



extern "C" __global__ void write_body_inertia_and_inverse_index_9a8468be_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_inertias,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    wp::array_t<wp::int32> var_body_user_to_backend,
    bool var_has_body_ordering,
    wp::array_t<wp::float32> var_body_mass_backend,
    wp::array_t<wp::float32> var_body_inertia_user,
    wp::array_t<wp::float32> var_body_inertia_backend,
    wp::array_t<wp::float32> var_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_body_inv_inertia)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 0;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        const wp::int32 var_17 = 1;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        const wp::int32 var_21 = 2;
        wp::float32* var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        const wp::int32 var_25 = 3;
        wp::float32* var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        const wp::int32 var_29 = 4;
        wp::float32* var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        const wp::int32 var_33 = 5;
        wp::float32* var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        const wp::int32 var_37 = 6;
        wp::float32* var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        const wp::int32 var_41 = 7;
        wp::float32* var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        const wp::int32 var_45 = 8;
        wp::float32* var_46;
        wp::float32 var_47;
        wp::float32 var_48;
        const bool var_49 = false;
        //---------
        // forward
        // def write_body_inertia_and_inverse_index(                                              <L 1>
        // i, j = wp.tid()                                                                        <L 14>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 15>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_body_id = wp.int32(body_ids[j])                                                   <L 16>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // backend_body_id = user_body_id                                                         <L 17>
        var_8 = wp::copy(var_6);
        // if has_body_ordering:                                                                  <L 18>
        if (var_has_body_ordering) {
            // backend_body_id = body_user_to_backend[user_body_id]                               <L 19>
            var_9 = wp::address(var_body_user_to_backend, var_6);
            var_11 = wp::load(var_9);
            var_10 = wp::copy(var_11);
        }
        var_12 = wp::where(var_has_body_ordering, var_10, var_8);
        // for k in range(9):                                                                     <L 20>
        // value = inertias[i, j, k]                                                              <L 21>
        var_14 = wp::address(var_inertias, var_0, var_1, var_13);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_13, var_15);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_13, var_15);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_18 = wp::address(var_inertias, var_0, var_1, var_17);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_17, var_19);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_17, var_19);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_22 = wp::address(var_inertias, var_0, var_1, var_21);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_21, var_23);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_21, var_23);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_26 = wp::address(var_inertias, var_0, var_1, var_25);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_25, var_27);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_25, var_27);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_30 = wp::address(var_inertias, var_0, var_1, var_29);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_29, var_31);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_29, var_31);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_34 = wp::address(var_inertias, var_0, var_1, var_33);
        var_36 = wp::load(var_34);
        var_35 = wp::copy(var_36);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_33, var_35);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_33, var_35);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_38 = wp::address(var_inertias, var_0, var_1, var_37);
        var_40 = wp::load(var_38);
        var_39 = wp::copy(var_40);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_37, var_39);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_37, var_39);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_42 = wp::address(var_inertias, var_0, var_1, var_41);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_41, var_43);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_41, var_43);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_46 = wp::address(var_inertias, var_0, var_1, var_45);
        var_48 = wp::load(var_46);
        var_47 = wp::copy(var_48);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_45, var_47);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_45, var_47);
        }
        // _update_body_inertial_properties_inverse(                                              <L 25>
        // env_id, backend_body_id, body_mass_backend, body_inertia_backend, False, body_inv_mass, body_inv_inertia       <L 26>
        _update_body_inertial_properties_inverse_0(var_3, var_12, var_body_mass_backend, var_body_inertia_backend, var_49, var_body_inv_mass, var_body_inv_inertia);
    }
}



extern "C" __global__ void write_body_inertia_and_inverse_index_9a8468be_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_inertias,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    wp::array_t<wp::int32> var_body_user_to_backend,
    bool var_has_body_ordering,
    wp::array_t<wp::float32> var_body_mass_backend,
    wp::array_t<wp::float32> var_body_inertia_user,
    wp::array_t<wp::float32> var_body_inertia_backend,
    wp::array_t<wp::float32> var_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_body_inv_inertia,
    wp::array_t<wp::float32> adj_inertias,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int64> adj_body_ids,
    wp::array_t<wp::int32> adj_body_user_to_backend,
    bool adj_has_body_ordering,
    wp::array_t<wp::float32> adj_body_mass_backend,
    wp::array_t<wp::float32> adj_body_inertia_user,
    wp::array_t<wp::float32> adj_body_inertia_backend,
    wp::array_t<wp::float32> adj_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> adj_body_inv_inertia)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 0;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        const wp::int32 var_17 = 1;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        const wp::int32 var_21 = 2;
        wp::float32* var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        const wp::int32 var_25 = 3;
        wp::float32* var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        const wp::int32 var_29 = 4;
        wp::float32* var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        const wp::int32 var_33 = 5;
        wp::float32* var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        const wp::int32 var_37 = 6;
        wp::float32* var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        const wp::int32 var_41 = 7;
        wp::float32* var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        const wp::int32 var_45 = 8;
        wp::float32* var_46;
        wp::float32 var_47;
        wp::float32 var_48;
        const bool var_49 = false;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int64 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int64 adj_4 = {};
        wp::int64 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int64 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::float32 adj_39 = {};
        wp::float32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::float32 adj_42 = {};
        wp::float32 adj_43 = {};
        wp::float32 adj_44 = {};
        wp::int32 adj_45 = {};
        wp::float32 adj_46 = {};
        wp::float32 adj_47 = {};
        wp::float32 adj_48 = {};
        bool adj_49 = {};
        //---------
        // forward
        // def write_body_inertia_and_inverse_index(                                              <L 1>
        // i, j = wp.tid()                                                                        <L 14>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 15>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_body_id = wp.int32(body_ids[j])                                                   <L 16>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // backend_body_id = user_body_id                                                         <L 17>
        var_8 = wp::copy(var_6);
        // if has_body_ordering:                                                                  <L 18>
        if (var_has_body_ordering) {
            // backend_body_id = body_user_to_backend[user_body_id]                               <L 19>
            var_9 = wp::address(var_body_user_to_backend, var_6);
            var_11 = wp::load(var_9);
            var_10 = wp::copy(var_11);
        }
        var_12 = wp::where(var_has_body_ordering, var_10, var_8);
        // for k in range(9):                                                                     <L 20>
        // value = inertias[i, j, k]                                                              <L 21>
        var_14 = wp::address(var_inertias, var_0, var_1, var_13);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_13, var_15);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_13, var_15);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_18 = wp::address(var_inertias, var_0, var_1, var_17);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_17, var_19);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_17, var_19);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_22 = wp::address(var_inertias, var_0, var_1, var_21);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_21, var_23);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_21, var_23);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_26 = wp::address(var_inertias, var_0, var_1, var_25);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_25, var_27);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_25, var_27);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_30 = wp::address(var_inertias, var_0, var_1, var_29);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_29, var_31);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_29, var_31);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_34 = wp::address(var_inertias, var_0, var_1, var_33);
        var_36 = wp::load(var_34);
        var_35 = wp::copy(var_36);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_33, var_35);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_33, var_35);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_38 = wp::address(var_inertias, var_0, var_1, var_37);
        var_40 = wp::load(var_38);
        var_39 = wp::copy(var_40);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_37, var_39);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_37, var_39);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_42 = wp::address(var_inertias, var_0, var_1, var_41);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_41, var_43);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_41, var_43);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_46 = wp::address(var_inertias, var_0, var_1, var_45);
        var_48 = wp::load(var_46);
        var_47 = wp::copy(var_48);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_45, var_47);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_45, var_47);
        }
        // _update_body_inertial_properties_inverse(                                              <L 25>
        // env_id, backend_body_id, body_mass_backend, body_inertia_backend, False, body_inv_mass, body_inv_inertia       <L 26>
        _update_body_inertial_properties_inverse_0(var_3, var_12, var_body_mass_backend, var_body_inertia_backend, var_49, var_body_inv_mass, var_body_inv_inertia);
        //---------
        // reverse
        adj__update_body_inertial_properties_inverse_0(var_3, var_12, var_body_mass_backend, var_body_inertia_backend, var_49, var_body_inv_mass, var_body_inv_inertia, adj_3, adj_12, adj_body_mass_backend, adj_body_inertia_backend, adj_49, adj_body_inv_mass, adj_body_inv_inertia);
        // adj: env_id, backend_body_id, body_mass_backend, body_inertia_backend, False, body_inv_mass, body_inv_inertia  <L 26>
        // adj: _update_body_inertial_properties_inverse(                                         <L 25>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_45, var_47, adj_body_inertia_backend, adj_3, adj_12, adj_45, adj_47);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_45, var_47, adj_body_inertia_user, adj_3, adj_6, adj_45, adj_47);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_48, adj_46, adj_47);
        wp::adj_address(var_inertias, var_0, var_1, var_45, adj_inertias, adj_0, adj_1, adj_45, adj_46);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_41, var_43, adj_body_inertia_backend, adj_3, adj_12, adj_41, adj_43);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_41, var_43, adj_body_inertia_user, adj_3, adj_6, adj_41, adj_43);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_44, adj_42, adj_43);
        wp::adj_address(var_inertias, var_0, var_1, var_41, adj_inertias, adj_0, adj_1, adj_41, adj_42);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_37, var_39, adj_body_inertia_backend, adj_3, adj_12, adj_37, adj_39);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_37, var_39, adj_body_inertia_user, adj_3, adj_6, adj_37, adj_39);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_40, adj_38, adj_39);
        wp::adj_address(var_inertias, var_0, var_1, var_37, adj_inertias, adj_0, adj_1, adj_37, adj_38);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_33, var_35, adj_body_inertia_backend, adj_3, adj_12, adj_33, adj_35);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_33, var_35, adj_body_inertia_user, adj_3, adj_6, adj_33, adj_35);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_36, adj_34, adj_35);
        wp::adj_address(var_inertias, var_0, var_1, var_33, adj_inertias, adj_0, adj_1, adj_33, adj_34);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_29, var_31, adj_body_inertia_backend, adj_3, adj_12, adj_29, adj_31);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_29, var_31, adj_body_inertia_user, adj_3, adj_6, adj_29, adj_31);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_32, adj_30, adj_31);
        wp::adj_address(var_inertias, var_0, var_1, var_29, adj_inertias, adj_0, adj_1, adj_29, adj_30);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_25, var_27, adj_body_inertia_backend, adj_3, adj_12, adj_25, adj_27);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_25, var_27, adj_body_inertia_user, adj_3, adj_6, adj_25, adj_27);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_28, adj_26, adj_27);
        wp::adj_address(var_inertias, var_0, var_1, var_25, adj_inertias, adj_0, adj_1, adj_25, adj_26);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_21, var_23, adj_body_inertia_backend, adj_3, adj_12, adj_21, adj_23);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_21, var_23, adj_body_inertia_user, adj_3, adj_6, adj_21, adj_23);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_24, adj_22, adj_23);
        wp::adj_address(var_inertias, var_0, var_1, var_21, adj_inertias, adj_0, adj_1, adj_21, adj_22);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_17, var_19, adj_body_inertia_backend, adj_3, adj_12, adj_17, adj_19);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_17, var_19, adj_body_inertia_user, adj_3, adj_6, adj_17, adj_19);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_20, adj_18, adj_19);
        wp::adj_address(var_inertias, var_0, var_1, var_17, adj_inertias, adj_0, adj_1, adj_17, adj_18);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_13, var_15, adj_body_inertia_backend, adj_3, adj_12, adj_13, adj_15);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_13, var_15, adj_body_inertia_user, adj_3, adj_6, adj_13, adj_15);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_16, adj_14, adj_15);
        wp::adj_address(var_inertias, var_0, var_1, var_13, adj_inertias, adj_0, adj_1, adj_13, adj_14);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        // adj: for k in range(9):                                                                <L 20>
        wp::adj_where(var_has_body_ordering, var_10, var_8, adj_has_body_ordering, adj_10, adj_8, adj_12);
        if (var_has_body_ordering) {
            wp::adj_copy(var_11, adj_9, adj_10);
            wp::adj_address(var_body_user_to_backend, var_6, adj_body_user_to_backend, adj_6, adj_9);
            // adj: backend_body_id = body_user_to_backend[user_body_id]                          <L 19>
        }
        // adj: if has_body_ordering:                                                             <L 18>
        wp::adj_copy(var_6, adj_6, adj_8);
        // adj: backend_body_id = user_body_id                                                    <L 17>
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_5);
        // adj: user_body_id = wp.int32(body_ids[j])                                              <L 16>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 15>
        // adj: i, j = wp.tid()                                                                   <L 14>
        // adj: def write_body_inertia_and_inverse_index(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void set_root_com_pose_to_sim_mask_9bbd321f_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<bool> var_env_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_root_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> var_root_link_pose_w)
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
        bool* var_1;
        bool var_2;
        wp::transform_t<wp::float32>* var_3;
        wp::transform_t<wp::float32> var_4;
        wp::transform_t<wp::float32>* var_5;
        const wp::int32 var_6 = 0;
        wp::vec_t<3, wp::float32>* var_7;
        wp::transform_t<wp::float32> var_8;
        wp::transform_t<wp::float32> var_9;
        wp::vec_t<3, wp::float32> var_10;
        bool var_11;
        //---------
        // forward
        // def set_root_com_pose_to_sim_mask(                                                     <L 596>
        // i = wp.tid()                                                                           <L 617>
        var_0 = builtin_tid1d();
        // if env_mask[i]:                                                                        <L 618>
        var_1 = wp::address(var_env_mask, var_0);
        var_2 = wp::load(var_1);
        if (var_2) {
            // root_com_pose_w[i] = data[i]                                                       <L 619>
            var_3 = wp::address(var_data, var_0);
            var_4 = wp::load(var_3);
            wp::array_store(var_root_com_pose_w, var_0, var_4);
            // root_link_pose_w[i] = get_com_pose_in_link_frame_func(root_com_pose_w[i], body_com_pos_b[i, 0])       <L 621>
            var_5 = wp::address(var_root_com_pose_w, var_0);
            var_7 = wp::address(var_body_com_pos_b, var_0, var_6);
            var_9 = wp::load(var_5);
            var_10 = wp::load(var_7);
            var_8 = get_com_pose_in_link_frame_func_0(var_9, var_10);
            wp::array_store(var_root_link_pose_w, var_0, var_8);
        }
        var_11 = wp::load(var_1);
    }
}



extern "C" __global__ void set_root_com_pose_to_sim_mask_9bbd321f_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<bool> var_env_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_root_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> var_root_link_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_data,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_body_com_pos_b,
    wp::array_t<bool> adj_env_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_root_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_root_link_pose_w)
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
        bool* var_1;
        bool var_2;
        wp::transform_t<wp::float32>* var_3;
        wp::transform_t<wp::float32> var_4;
        wp::transform_t<wp::float32>* var_5;
        const wp::int32 var_6 = 0;
        wp::vec_t<3, wp::float32>* var_7;
        wp::transform_t<wp::float32> var_8;
        wp::transform_t<wp::float32> var_9;
        wp::vec_t<3, wp::float32> var_10;
        bool var_11;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        bool adj_2 = {};
        wp::transform_t<wp::float32> adj_3 = {};
        wp::transform_t<wp::float32> adj_4 = {};
        wp::transform_t<wp::float32> adj_5 = {};
        wp::int32 adj_6 = {};
        wp::vec_t<3, wp::float32> adj_7 = {};
        wp::transform_t<wp::float32> adj_8 = {};
        wp::transform_t<wp::float32> adj_9 = {};
        wp::vec_t<3, wp::float32> adj_10 = {};
        bool adj_11 = {};
        //---------
        // forward
        // def set_root_com_pose_to_sim_mask(                                                     <L 596>
        // i = wp.tid()                                                                           <L 617>
        var_0 = builtin_tid1d();
        // if env_mask[i]:                                                                        <L 618>
        var_1 = wp::address(var_env_mask, var_0);
        var_2 = wp::load(var_1);
        if (var_2) {
            // root_com_pose_w[i] = data[i]                                                       <L 619>
            var_3 = wp::address(var_data, var_0);
            var_4 = wp::load(var_3);
            // wp::array_store(var_root_com_pose_w, var_0, var_4);
            // root_link_pose_w[i] = get_com_pose_in_link_frame_func(root_com_pose_w[i], body_com_pos_b[i, 0])       <L 621>
            var_5 = wp::address(var_root_com_pose_w, var_0);
            var_7 = wp::address(var_body_com_pos_b, var_0, var_6);
            var_9 = wp::load(var_5);
            var_10 = wp::load(var_7);
            var_8 = get_com_pose_in_link_frame_func_0(var_9, var_10);
            // wp::array_store(var_root_link_pose_w, var_0, var_8);
        }
        var_11 = wp::load(var_1);
        //---------
        // reverse
        if (var_11) {
            wp::adj_array_store(var_root_link_pose_w, var_0, var_8, adj_root_link_pose_w, adj_0, adj_8);
            adj_get_com_pose_in_link_frame_func_0(var_9, var_10, adj_5, adj_7, adj_8);
            wp::adj_address(var_body_com_pos_b, var_0, var_6, adj_body_com_pos_b, adj_0, adj_6, adj_7);
            wp::adj_address(var_root_com_pose_w, var_0, adj_root_com_pose_w, adj_0, adj_5);
            // adj: root_link_pose_w[i] = get_com_pose_in_link_frame_func(root_com_pose_w[i], body_com_pos_b[i, 0])  <L 621>
            wp::adj_array_store(var_root_com_pose_w, var_0, var_4, adj_root_com_pose_w, adj_0, adj_3);
            wp::adj_address(var_data, var_0, adj_data, adj_0, adj_3);
            // adj: root_com_pose_w[i] = data[i]                                                  <L 619>
        }
        wp::adj_address(var_env_mask, var_0, adj_env_mask, adj_0, adj_1);
        // adj: if env_mask[i]:                                                                   <L 618>
        // adj: i = wp.tid()                                                                      <L 617>
        // adj: def set_root_com_pose_to_sim_mask(                                                <L 596>
        continue;
    }
}



extern "C" __global__ void write_body_mass_and_inverse_index_9eace17f_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_masses,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    wp::array_t<wp::int32> var_body_user_to_backend,
    bool var_has_body_ordering,
    wp::array_t<wp::float32> var_body_inertia_backend,
    wp::array_t<wp::float32> var_body_mass_user,
    wp::array_t<wp::float32> var_body_mass_backend,
    wp::array_t<wp::float32> var_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_body_inv_inertia)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        const bool var_16 = true;
        //---------
        // forward
        // def write_body_mass_and_inverse_index(                                                 <L 1>
        // i, j = wp.tid()                                                                        <L 27>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 28>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_body_id = wp.int32(body_ids[j])                                                   <L 29>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // backend_body_id = user_body_id                                                         <L 30>
        var_8 = wp::copy(var_6);
        // if has_body_ordering:                                                                  <L 31>
        if (var_has_body_ordering) {
            // backend_body_id = body_user_to_backend[user_body_id]                               <L 32>
            var_9 = wp::address(var_body_user_to_backend, var_6);
            var_11 = wp::load(var_9);
            var_10 = wp::copy(var_11);
        }
        var_12 = wp::where(var_has_body_ordering, var_10, var_8);
        // mass = masses[i, j]                                                                    <L 33>
        var_13 = wp::address(var_masses, var_0, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // body_mass_user[env_id, user_body_id] = mass                                            <L 34>
        wp::array_store(var_body_mass_user, var_3, var_6, var_14);
        // if has_body_ordering:                                                                  <L 35>
        if (var_has_body_ordering) {
            // body_mass_backend[env_id, backend_body_id] = mass                                  <L 36>
            wp::array_store(var_body_mass_backend, var_3, var_12, var_14);
        }
        // _update_body_inertial_properties_inverse(                                              <L 37>
        // env_id, backend_body_id, body_mass_backend, body_inertia_backend, True, body_inv_mass, body_inv_inertia       <L 38>
        _update_body_inertial_properties_inverse_0(var_3, var_12, var_body_mass_backend, var_body_inertia_backend, var_16, var_body_inv_mass, var_body_inv_inertia);
    }
}



extern "C" __global__ void write_body_mass_and_inverse_index_9eace17f_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_masses,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    wp::array_t<wp::int32> var_body_user_to_backend,
    bool var_has_body_ordering,
    wp::array_t<wp::float32> var_body_inertia_backend,
    wp::array_t<wp::float32> var_body_mass_user,
    wp::array_t<wp::float32> var_body_mass_backend,
    wp::array_t<wp::float32> var_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_body_inv_inertia,
    wp::array_t<wp::float32> adj_masses,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int64> adj_body_ids,
    wp::array_t<wp::int32> adj_body_user_to_backend,
    bool adj_has_body_ordering,
    wp::array_t<wp::float32> adj_body_inertia_backend,
    wp::array_t<wp::float32> adj_body_mass_user,
    wp::array_t<wp::float32> adj_body_mass_backend,
    wp::array_t<wp::float32> adj_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> adj_body_inv_inertia)
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
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        const bool var_16 = true;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int64 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int64 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        bool adj_16 = {};
        //---------
        // forward
        // def write_body_mass_and_inverse_index(                                                 <L 1>
        // i, j = wp.tid()                                                                        <L 27>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 28>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_body_id = wp.int32(body_ids[j])                                                   <L 29>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // backend_body_id = user_body_id                                                         <L 30>
        var_8 = wp::copy(var_6);
        // if has_body_ordering:                                                                  <L 31>
        if (var_has_body_ordering) {
            // backend_body_id = body_user_to_backend[user_body_id]                               <L 32>
            var_9 = wp::address(var_body_user_to_backend, var_6);
            var_11 = wp::load(var_9);
            var_10 = wp::copy(var_11);
        }
        var_12 = wp::where(var_has_body_ordering, var_10, var_8);
        // mass = masses[i, j]                                                                    <L 33>
        var_13 = wp::address(var_masses, var_0, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // body_mass_user[env_id, user_body_id] = mass                                            <L 34>
        // wp::array_store(var_body_mass_user, var_3, var_6, var_14);
        // if has_body_ordering:                                                                  <L 35>
        if (var_has_body_ordering) {
            // body_mass_backend[env_id, backend_body_id] = mass                                  <L 36>
            // wp::array_store(var_body_mass_backend, var_3, var_12, var_14);
        }
        // _update_body_inertial_properties_inverse(                                              <L 37>
        // env_id, backend_body_id, body_mass_backend, body_inertia_backend, True, body_inv_mass, body_inv_inertia       <L 38>
        _update_body_inertial_properties_inverse_0(var_3, var_12, var_body_mass_backend, var_body_inertia_backend, var_16, var_body_inv_mass, var_body_inv_inertia);
        //---------
        // reverse
        adj__update_body_inertial_properties_inverse_0(var_3, var_12, var_body_mass_backend, var_body_inertia_backend, var_16, var_body_inv_mass, var_body_inv_inertia, adj_3, adj_12, adj_body_mass_backend, adj_body_inertia_backend, adj_16, adj_body_inv_mass, adj_body_inv_inertia);
        // adj: env_id, backend_body_id, body_mass_backend, body_inertia_backend, True, body_inv_mass, body_inv_inertia  <L 38>
        // adj: _update_body_inertial_properties_inverse(                                         <L 37>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_mass_backend, var_3, var_12, var_14, adj_body_mass_backend, adj_3, adj_12, adj_14);
            // adj: body_mass_backend[env_id, backend_body_id] = mass                             <L 36>
        }
        // adj: if has_body_ordering:                                                             <L 35>
        wp::adj_array_store(var_body_mass_user, var_3, var_6, var_14, adj_body_mass_user, adj_3, adj_6, adj_14);
        // adj: body_mass_user[env_id, user_body_id] = mass                                       <L 34>
        wp::adj_copy(var_15, adj_13, adj_14);
        wp::adj_address(var_masses, var_0, var_1, adj_masses, adj_0, adj_1, adj_13);
        // adj: mass = masses[i, j]                                                               <L 33>
        wp::adj_where(var_has_body_ordering, var_10, var_8, adj_has_body_ordering, adj_10, adj_8, adj_12);
        if (var_has_body_ordering) {
            wp::adj_copy(var_11, adj_9, adj_10);
            wp::adj_address(var_body_user_to_backend, var_6, adj_body_user_to_backend, adj_6, adj_9);
            // adj: backend_body_id = body_user_to_backend[user_body_id]                          <L 32>
        }
        // adj: if has_body_ordering:                                                             <L 31>
        wp::adj_copy(var_6, adj_6, adj_8);
        // adj: backend_body_id = user_body_id                                                    <L 30>
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_5);
        // adj: user_body_id = wp.int32(body_ids[j])                                              <L 29>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 28>
        // adj: i, j = wp.tid()                                                                   <L 27>
        // adj: def write_body_mass_and_inverse_index(                                            <L 1>
        continue;
    }
}



extern "C" __global__ void set_root_com_pose_to_sim_index_a1545ae3_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::transform_t<wp::float32>> var_root_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> var_root_link_pose_w)
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
        wp::int64* var_1;
        wp::int32 var_2;
        wp::int64 var_3;
        wp::transform_t<wp::float32>* var_4;
        wp::transform_t<wp::float32> var_5;
        wp::transform_t<wp::float32>* var_6;
        const wp::int32 var_7 = 0;
        wp::vec_t<3, wp::float32>* var_8;
        wp::transform_t<wp::float32> var_9;
        wp::transform_t<wp::float32> var_10;
        wp::vec_t<3, wp::float32> var_11;
        //---------
        // forward
        // def set_root_com_pose_to_sim_index(                                                    <L 1>
        // i = wp.tid()                                                                           <L 22>
        var_0 = builtin_tid1d();
        // env_id = wp.int32(env_ids[i])                                                          <L 23>
        var_1 = wp::address(var_env_ids, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::int32(var_3);
        // root_com_pose_w[env_id] = data[i]                                                      <L 24>
        var_4 = wp::address(var_data, var_0);
        var_5 = wp::load(var_4);
        wp::array_store(var_root_com_pose_w, var_2, var_5);
        // root_link_pose_w[env_id] = get_com_pose_in_link_frame_func(root_com_pose_w[env_id], body_com_pos_b[env_id, 0])       <L 26>
        var_6 = wp::address(var_root_com_pose_w, var_2);
        var_8 = wp::address(var_body_com_pos_b, var_2, var_7);
        var_10 = wp::load(var_6);
        var_11 = wp::load(var_8);
        var_9 = get_com_pose_in_link_frame_func_0(var_10, var_11);
        wp::array_store(var_root_link_pose_w, var_2, var_9);
    }
}



extern "C" __global__ void set_root_com_pose_to_sim_index_a1545ae3_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::transform_t<wp::float32>> var_root_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> var_root_link_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_data,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_body_com_pos_b,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::transform_t<wp::float32>> adj_root_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_root_link_pose_w)
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
        wp::int64* var_1;
        wp::int32 var_2;
        wp::int64 var_3;
        wp::transform_t<wp::float32>* var_4;
        wp::transform_t<wp::float32> var_5;
        wp::transform_t<wp::float32>* var_6;
        const wp::int32 var_7 = 0;
        wp::vec_t<3, wp::float32>* var_8;
        wp::transform_t<wp::float32> var_9;
        wp::transform_t<wp::float32> var_10;
        wp::vec_t<3, wp::float32> var_11;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int64 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int64 adj_3 = {};
        wp::transform_t<wp::float32> adj_4 = {};
        wp::transform_t<wp::float32> adj_5 = {};
        wp::transform_t<wp::float32> adj_6 = {};
        wp::int32 adj_7 = {};
        wp::vec_t<3, wp::float32> adj_8 = {};
        wp::transform_t<wp::float32> adj_9 = {};
        wp::transform_t<wp::float32> adj_10 = {};
        wp::vec_t<3, wp::float32> adj_11 = {};
        //---------
        // forward
        // def set_root_com_pose_to_sim_index(                                                    <L 1>
        // i = wp.tid()                                                                           <L 22>
        var_0 = builtin_tid1d();
        // env_id = wp.int32(env_ids[i])                                                          <L 23>
        var_1 = wp::address(var_env_ids, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::int32(var_3);
        // root_com_pose_w[env_id] = data[i]                                                      <L 24>
        var_4 = wp::address(var_data, var_0);
        var_5 = wp::load(var_4);
        // wp::array_store(var_root_com_pose_w, var_2, var_5);
        // root_link_pose_w[env_id] = get_com_pose_in_link_frame_func(root_com_pose_w[env_id], body_com_pos_b[env_id, 0])       <L 26>
        var_6 = wp::address(var_root_com_pose_w, var_2);
        var_8 = wp::address(var_body_com_pos_b, var_2, var_7);
        var_10 = wp::load(var_6);
        var_11 = wp::load(var_8);
        var_9 = get_com_pose_in_link_frame_func_0(var_10, var_11);
        // wp::array_store(var_root_link_pose_w, var_2, var_9);
        //---------
        // reverse
        wp::adj_array_store(var_root_link_pose_w, var_2, var_9, adj_root_link_pose_w, adj_2, adj_9);
        adj_get_com_pose_in_link_frame_func_0(var_10, var_11, adj_6, adj_8, adj_9);
        wp::adj_address(var_body_com_pos_b, var_2, var_7, adj_body_com_pos_b, adj_2, adj_7, adj_8);
        wp::adj_address(var_root_com_pose_w, var_2, adj_root_com_pose_w, adj_2, adj_6);
        // adj: root_link_pose_w[env_id] = get_com_pose_in_link_frame_func(root_com_pose_w[env_id], body_com_pos_b[env_id, 0])  <L 26>
        wp::adj_array_store(var_root_com_pose_w, var_2, var_5, adj_root_com_pose_w, adj_2, adj_4);
        wp::adj_address(var_data, var_0, adj_data, adj_0, adj_4);
        // adj: root_com_pose_w[env_id] = data[i]                                                 <L 24>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_1);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 23>
        // adj: i = wp.tid()                                                                      <L 22>
        // adj: def set_root_com_pose_to_sim_index(                                               <L 1>
        continue;
    }
}



extern "C" __global__ void set_body_com_velocity_to_sim_b3e1efb6_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::vec_t<6, wp::float32>* var_8;
        wp::vec_t<6, wp::float32> var_9;
        wp::vec_t<6, wp::float32>* var_10;
        wp::vec_t<6, wp::float32> var_11;
        const wp::float32 var_12 = 0.0;
        const wp::float32 var_13 = 0.0;
        const wp::float32 var_14 = 0.0;
        const wp::float32 var_15 = 0.0;
        const wp::float32 var_16 = 0.0;
        const wp::float32 var_17 = 0.0;
        wp::vec_t<6, wp::float32> var_18;
        //---------
        // forward
        // def set_body_com_velocity_to_sim(                                                      <L 1>
        // i, j = wp.tid()                                                                        <L 26>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 27>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 28>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 29>
        if (var_from_mask) {
            // body_com_velocity_w[env_id, body_id] = data[env_id, body_id]                       <L 30>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_body_com_velocity_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_com_velocity_w[env_id, body_id] = data[i, j]                                  <L 32>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            wp::array_store(var_body_com_velocity_w, var_3, var_6, var_11);
        }
        // body_acc_w[env_id, body_id] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)         <L 34>
        var_18 = wp::vec_t<6, wp::float32>({var_12, var_13, var_14, var_15, var_16, var_17});
        wp::array_store(var_body_acc_w, var_3, var_6, var_18);
    }
}



extern "C" __global__ void set_body_com_velocity_to_sim_b3e1efb6_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_data,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int32> adj_body_ids,
    bool adj_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_acc_w)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::vec_t<6, wp::float32>* var_8;
        wp::vec_t<6, wp::float32> var_9;
        wp::vec_t<6, wp::float32>* var_10;
        wp::vec_t<6, wp::float32> var_11;
        const wp::float32 var_12 = 0.0;
        const wp::float32 var_13 = 0.0;
        const wp::float32 var_14 = 0.0;
        const wp::float32 var_15 = 0.0;
        const wp::float32 var_16 = 0.0;
        const wp::float32 var_17 = 0.0;
        wp::vec_t<6, wp::float32> var_18;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int64 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int64 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::vec_t<6, wp::float32> adj_8 = {};
        wp::vec_t<6, wp::float32> adj_9 = {};
        wp::vec_t<6, wp::float32> adj_10 = {};
        wp::vec_t<6, wp::float32> adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::vec_t<6, wp::float32> adj_18 = {};
        //---------
        // forward
        // def set_body_com_velocity_to_sim(                                                      <L 1>
        // i, j = wp.tid()                                                                        <L 26>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 27>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 28>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 29>
        if (var_from_mask) {
            // body_com_velocity_w[env_id, body_id] = data[env_id, body_id]                       <L 30>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            // wp::array_store(var_body_com_velocity_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_com_velocity_w[env_id, body_id] = data[i, j]                                  <L 32>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            // wp::array_store(var_body_com_velocity_w, var_3, var_6, var_11);
        }
        // body_acc_w[env_id, body_id] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)         <L 34>
        var_18 = wp::vec_t<6, wp::float32>({var_12, var_13, var_14, var_15, var_16, var_17});
        // wp::array_store(var_body_acc_w, var_3, var_6, var_18);
        //---------
        // reverse
        wp::adj_array_store(var_body_acc_w, var_3, var_6, var_18, adj_body_acc_w, adj_3, adj_6, adj_18);
        wp::adj_vec_t({var_12, var_13, var_14, var_15, var_16, var_17}, {&adj_12, &adj_13, &adj_14, &adj_15, &adj_16, &adj_17}, adj_18);
        // adj: body_acc_w[env_id, body_id] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)    <L 34>
        if (!var_from_mask) {
            wp::adj_array_store(var_body_com_velocity_w, var_3, var_6, var_11, adj_body_com_velocity_w, adj_3, adj_6, adj_10);
            wp::adj_address(var_data, var_0, var_1, adj_data, adj_0, adj_1, adj_10);
            // adj: body_com_velocity_w[env_id, body_id] = data[i, j]                             <L 32>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_body_com_velocity_w, var_3, var_6, var_9, adj_body_com_velocity_w, adj_3, adj_6, adj_8);
            wp::adj_address(var_data, var_3, var_6, adj_data, adj_3, adj_6, adj_8);
            // adj: body_com_velocity_w[env_id, body_id] = data[env_id, body_id]                  <L 30>
        }
        // adj: if from_mask:                                                                     <L 29>
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_5);
        // adj: body_id = wp.int32(body_ids[j])                                                   <L 28>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 27>
        // adj: i, j = wp.tid()                                                                   <L 26>
        // adj: def set_body_com_velocity_to_sim(                                                 <L 1>
        continue;
    }
}



extern "C" __global__ void set_root_com_pose_to_sim_index_b5c24cae_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::transform_t<wp::float32>> var_root_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> var_root_link_pose_w)
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
        wp::transform_t<wp::float32>* var_4;
        wp::transform_t<wp::float32> var_5;
        wp::transform_t<wp::float32>* var_6;
        const wp::int32 var_7 = 0;
        wp::vec_t<3, wp::float32>* var_8;
        wp::transform_t<wp::float32> var_9;
        wp::transform_t<wp::float32> var_10;
        wp::vec_t<3, wp::float32> var_11;
        //---------
        // forward
        // def set_root_com_pose_to_sim_index(                                                    <L 1>
        // i = wp.tid()                                                                           <L 22>
        var_0 = builtin_tid1d();
        // env_id = wp.int32(env_ids[i])                                                          <L 23>
        var_1 = wp::address(var_env_ids, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::int32(var_3);
        // root_com_pose_w[env_id] = data[i]                                                      <L 24>
        var_4 = wp::address(var_data, var_0);
        var_5 = wp::load(var_4);
        wp::array_store(var_root_com_pose_w, var_2, var_5);
        // root_link_pose_w[env_id] = get_com_pose_in_link_frame_func(root_com_pose_w[env_id], body_com_pos_b[env_id, 0])       <L 26>
        var_6 = wp::address(var_root_com_pose_w, var_2);
        var_8 = wp::address(var_body_com_pos_b, var_2, var_7);
        var_10 = wp::load(var_6);
        var_11 = wp::load(var_8);
        var_9 = get_com_pose_in_link_frame_func_0(var_10, var_11);
        wp::array_store(var_root_link_pose_w, var_2, var_9);
    }
}



extern "C" __global__ void set_root_com_pose_to_sim_index_b5c24cae_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::transform_t<wp::float32>> var_root_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> var_root_link_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_data,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_body_com_pos_b,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::transform_t<wp::float32>> adj_root_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_root_link_pose_w)
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
        wp::transform_t<wp::float32>* var_4;
        wp::transform_t<wp::float32> var_5;
        wp::transform_t<wp::float32>* var_6;
        const wp::int32 var_7 = 0;
        wp::vec_t<3, wp::float32>* var_8;
        wp::transform_t<wp::float32> var_9;
        wp::transform_t<wp::float32> var_10;
        wp::vec_t<3, wp::float32> var_11;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::transform_t<wp::float32> adj_4 = {};
        wp::transform_t<wp::float32> adj_5 = {};
        wp::transform_t<wp::float32> adj_6 = {};
        wp::int32 adj_7 = {};
        wp::vec_t<3, wp::float32> adj_8 = {};
        wp::transform_t<wp::float32> adj_9 = {};
        wp::transform_t<wp::float32> adj_10 = {};
        wp::vec_t<3, wp::float32> adj_11 = {};
        //---------
        // forward
        // def set_root_com_pose_to_sim_index(                                                    <L 1>
        // i = wp.tid()                                                                           <L 22>
        var_0 = builtin_tid1d();
        // env_id = wp.int32(env_ids[i])                                                          <L 23>
        var_1 = wp::address(var_env_ids, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::int32(var_3);
        // root_com_pose_w[env_id] = data[i]                                                      <L 24>
        var_4 = wp::address(var_data, var_0);
        var_5 = wp::load(var_4);
        // wp::array_store(var_root_com_pose_w, var_2, var_5);
        // root_link_pose_w[env_id] = get_com_pose_in_link_frame_func(root_com_pose_w[env_id], body_com_pos_b[env_id, 0])       <L 26>
        var_6 = wp::address(var_root_com_pose_w, var_2);
        var_8 = wp::address(var_body_com_pos_b, var_2, var_7);
        var_10 = wp::load(var_6);
        var_11 = wp::load(var_8);
        var_9 = get_com_pose_in_link_frame_func_0(var_10, var_11);
        // wp::array_store(var_root_link_pose_w, var_2, var_9);
        //---------
        // reverse
        wp::adj_array_store(var_root_link_pose_w, var_2, var_9, adj_root_link_pose_w, adj_2, adj_9);
        adj_get_com_pose_in_link_frame_func_0(var_10, var_11, adj_6, adj_8, adj_9);
        wp::adj_address(var_body_com_pos_b, var_2, var_7, adj_body_com_pos_b, adj_2, adj_7, adj_8);
        wp::adj_address(var_root_com_pose_w, var_2, adj_root_com_pose_w, adj_2, adj_6);
        // adj: root_link_pose_w[env_id] = get_com_pose_in_link_frame_func(root_com_pose_w[env_id], body_com_pos_b[env_id, 0])  <L 26>
        wp::adj_array_store(var_root_com_pose_w, var_2, var_5, adj_root_com_pose_w, adj_2, adj_4);
        wp::adj_address(var_data, var_0, adj_data, adj_0, adj_4);
        // adj: root_com_pose_w[env_id] = data[i]                                                 <L 24>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_1);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 23>
        // adj: i = wp.tid()                                                                      <L 22>
        // adj: def set_root_com_pose_to_sim_index(                                               <L 1>
        continue;
    }
}



extern "C" __global__ void get_root_com_pose_from_root_link_pose_b706d044_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_link_pose,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> var_com_pose_w)
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
        wp::transform_t<wp::float32>* var_1;
        const wp::int32 var_2 = 0;
        wp::vec_t<3, wp::float32>* var_3;
        wp::transform_t<wp::float32> var_4;
        wp::transform_t<wp::float32> var_5;
        wp::vec_t<3, wp::float32> var_6;
        //---------
        // forward
        // def get_root_com_pose_from_root_link_pose(                                             <L 257>
        // i = wp.tid()                                                                           <L 272>
        var_0 = builtin_tid1d();
        // com_pose_w[i] = get_com_pose_from_link_pose_func(link_pose[i], body_com_pos_b[i, 0])       <L 273>
        var_1 = wp::address(var_link_pose, var_0);
        var_3 = wp::address(var_body_com_pos_b, var_0, var_2);
        var_5 = wp::load(var_1);
        var_6 = wp::load(var_3);
        var_4 = get_com_pose_from_link_pose_func_0(var_5, var_6);
        wp::array_store(var_com_pose_w, var_0, var_4);
    }
}



extern "C" __global__ void get_root_com_pose_from_root_link_pose_b706d044_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_link_pose,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> var_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_link_pose,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> adj_com_pose_w)
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
        wp::transform_t<wp::float32>* var_1;
        const wp::int32 var_2 = 0;
        wp::vec_t<3, wp::float32>* var_3;
        wp::transform_t<wp::float32> var_4;
        wp::transform_t<wp::float32> var_5;
        wp::vec_t<3, wp::float32> var_6;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::transform_t<wp::float32> adj_1 = {};
        wp::int32 adj_2 = {};
        wp::vec_t<3, wp::float32> adj_3 = {};
        wp::transform_t<wp::float32> adj_4 = {};
        wp::transform_t<wp::float32> adj_5 = {};
        wp::vec_t<3, wp::float32> adj_6 = {};
        //---------
        // forward
        // def get_root_com_pose_from_root_link_pose(                                             <L 257>
        // i = wp.tid()                                                                           <L 272>
        var_0 = builtin_tid1d();
        // com_pose_w[i] = get_com_pose_from_link_pose_func(link_pose[i], body_com_pos_b[i, 0])       <L 273>
        var_1 = wp::address(var_link_pose, var_0);
        var_3 = wp::address(var_body_com_pos_b, var_0, var_2);
        var_5 = wp::load(var_1);
        var_6 = wp::load(var_3);
        var_4 = get_com_pose_from_link_pose_func_0(var_5, var_6);
        // wp::array_store(var_com_pose_w, var_0, var_4);
        //---------
        // reverse
        wp::adj_array_store(var_com_pose_w, var_0, var_4, adj_com_pose_w, adj_0, adj_4);
        adj_get_com_pose_from_link_pose_func_0(var_5, var_6, adj_1, adj_3, adj_4);
        wp::adj_address(var_body_com_pos_b, var_0, var_2, adj_body_com_pos_b, adj_0, adj_2, adj_3);
        wp::adj_address(var_link_pose, var_0, adj_link_pose, adj_0, adj_1);
        // adj: com_pose_w[i] = get_com_pose_from_link_pose_func(link_pose[i], body_com_pos_b[i, 0])  <L 273>
        // adj: i = wp.tid()                                                                      <L 272>
        // adj: def get_root_com_pose_from_root_link_pose(                                        <L 257>
        continue;
    }
}



extern "C" __global__ void set_body_link_velocity_to_sim_ba14d0c9_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_link_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::vec_t<6, wp::float32>* var_8;
        wp::vec_t<6, wp::float32> var_9;
        wp::vec_t<6, wp::float32>* var_10;
        wp::vec_t<6, wp::float32> var_11;
        wp::vec_t<6, wp::float32>* var_12;
        wp::transform_t<wp::float32>* var_13;
        wp::vec_t<3, wp::float32>* var_14;
        wp::vec_t<6, wp::float32> var_15;
        wp::vec_t<6, wp::float32> var_16;
        wp::transform_t<wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        const wp::float32 var_19 = 0.0;
        const wp::float32 var_20 = 0.0;
        const wp::float32 var_21 = 0.0;
        const wp::float32 var_22 = 0.0;
        const wp::float32 var_23 = 0.0;
        const wp::float32 var_24 = 0.0;
        wp::vec_t<6, wp::float32> var_25;
        //---------
        // forward
        // def set_body_link_velocity_to_sim(                                                     <L 1>
        // i, j = wp.tid()                                                                        <L 35>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 36>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 37>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 38>
        if (var_from_mask) {
            // body_link_velocity_w[env_id, body_id] = data[env_id, body_id]                      <L 39>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_body_link_velocity_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_link_velocity_w[env_id, body_id] = data[i, j]                                 <L 41>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            wp::array_store(var_body_link_velocity_w, var_3, var_6, var_11);
        }
        // body_com_velocity_w[env_id, body_id] = get_link_velocity_in_com_frame_func(            <L 43>
        // body_link_velocity_w[env_id, body_id],                                                 <L 44>
        var_12 = wp::address(var_body_link_velocity_w, var_3, var_6);
        // body_link_pose_w[env_id, body_id],                                                     <L 45>
        var_13 = wp::address(var_body_link_pose_w, var_3, var_6);
        // body_com_pos_b[env_id, body_id],                                                       <L 46>
        var_14 = wp::address(var_body_com_pos_b, var_3, var_6);
        var_16 = wp::load(var_12);
        var_17 = wp::load(var_13);
        var_18 = wp::load(var_14);
        var_15 = get_link_velocity_in_com_frame_func_0(var_16, var_17, var_18);
        // body_com_velocity_w[env_id, body_id] = get_link_velocity_in_com_frame_func(            <L 43>
        wp::array_store(var_body_com_velocity_w, var_3, var_6, var_15);
        // body_acc_w[env_id, body_id] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)         <L 49>
        var_25 = wp::vec_t<6, wp::float32>({var_19, var_20, var_21, var_22, var_23, var_24});
        wp::array_store(var_body_acc_w, var_3, var_6, var_25);
    }
}



extern "C" __global__ void set_body_link_velocity_to_sim_ba14d0c9_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_link_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_data,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_link_pose_w,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int64> adj_body_ids,
    bool adj_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_link_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_acc_w)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::vec_t<6, wp::float32>* var_8;
        wp::vec_t<6, wp::float32> var_9;
        wp::vec_t<6, wp::float32>* var_10;
        wp::vec_t<6, wp::float32> var_11;
        wp::vec_t<6, wp::float32>* var_12;
        wp::transform_t<wp::float32>* var_13;
        wp::vec_t<3, wp::float32>* var_14;
        wp::vec_t<6, wp::float32> var_15;
        wp::vec_t<6, wp::float32> var_16;
        wp::transform_t<wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        const wp::float32 var_19 = 0.0;
        const wp::float32 var_20 = 0.0;
        const wp::float32 var_21 = 0.0;
        const wp::float32 var_22 = 0.0;
        const wp::float32 var_23 = 0.0;
        const wp::float32 var_24 = 0.0;
        wp::vec_t<6, wp::float32> var_25;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int64 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int64 adj_4 = {};
        wp::int64 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int64 adj_7 = {};
        wp::vec_t<6, wp::float32> adj_8 = {};
        wp::vec_t<6, wp::float32> adj_9 = {};
        wp::vec_t<6, wp::float32> adj_10 = {};
        wp::vec_t<6, wp::float32> adj_11 = {};
        wp::vec_t<6, wp::float32> adj_12 = {};
        wp::transform_t<wp::float32> adj_13 = {};
        wp::vec_t<3, wp::float32> adj_14 = {};
        wp::vec_t<6, wp::float32> adj_15 = {};
        wp::vec_t<6, wp::float32> adj_16 = {};
        wp::transform_t<wp::float32> adj_17 = {};
        wp::vec_t<3, wp::float32> adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::vec_t<6, wp::float32> adj_25 = {};
        //---------
        // forward
        // def set_body_link_velocity_to_sim(                                                     <L 1>
        // i, j = wp.tid()                                                                        <L 35>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 36>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 37>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 38>
        if (var_from_mask) {
            // body_link_velocity_w[env_id, body_id] = data[env_id, body_id]                      <L 39>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            // wp::array_store(var_body_link_velocity_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_link_velocity_w[env_id, body_id] = data[i, j]                                 <L 41>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            // wp::array_store(var_body_link_velocity_w, var_3, var_6, var_11);
        }
        // body_com_velocity_w[env_id, body_id] = get_link_velocity_in_com_frame_func(            <L 43>
        // body_link_velocity_w[env_id, body_id],                                                 <L 44>
        var_12 = wp::address(var_body_link_velocity_w, var_3, var_6);
        // body_link_pose_w[env_id, body_id],                                                     <L 45>
        var_13 = wp::address(var_body_link_pose_w, var_3, var_6);
        // body_com_pos_b[env_id, body_id],                                                       <L 46>
        var_14 = wp::address(var_body_com_pos_b, var_3, var_6);
        var_16 = wp::load(var_12);
        var_17 = wp::load(var_13);
        var_18 = wp::load(var_14);
        var_15 = get_link_velocity_in_com_frame_func_0(var_16, var_17, var_18);
        // body_com_velocity_w[env_id, body_id] = get_link_velocity_in_com_frame_func(            <L 43>
        // wp::array_store(var_body_com_velocity_w, var_3, var_6, var_15);
        // body_acc_w[env_id, body_id] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)         <L 49>
        var_25 = wp::vec_t<6, wp::float32>({var_19, var_20, var_21, var_22, var_23, var_24});
        // wp::array_store(var_body_acc_w, var_3, var_6, var_25);
        //---------
        // reverse
        wp::adj_array_store(var_body_acc_w, var_3, var_6, var_25, adj_body_acc_w, adj_3, adj_6, adj_25);
        wp::adj_vec_t({var_19, var_20, var_21, var_22, var_23, var_24}, {&adj_19, &adj_20, &adj_21, &adj_22, &adj_23, &adj_24}, adj_25);
        // adj: body_acc_w[env_id, body_id] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)    <L 49>
        wp::adj_array_store(var_body_com_velocity_w, var_3, var_6, var_15, adj_body_com_velocity_w, adj_3, adj_6, adj_15);
        // adj: body_com_velocity_w[env_id, body_id] = get_link_velocity_in_com_frame_func(       <L 43>
        adj_get_link_velocity_in_com_frame_func_0(var_16, var_17, var_18, adj_12, adj_13, adj_14, adj_15);
        wp::adj_address(var_body_com_pos_b, var_3, var_6, adj_body_com_pos_b, adj_3, adj_6, adj_14);
        // adj: body_com_pos_b[env_id, body_id],                                                  <L 46>
        wp::adj_address(var_body_link_pose_w, var_3, var_6, adj_body_link_pose_w, adj_3, adj_6, adj_13);
        // adj: body_link_pose_w[env_id, body_id],                                                <L 45>
        wp::adj_address(var_body_link_velocity_w, var_3, var_6, adj_body_link_velocity_w, adj_3, adj_6, adj_12);
        // adj: body_link_velocity_w[env_id, body_id],                                            <L 44>
        // adj: body_com_velocity_w[env_id, body_id] = get_link_velocity_in_com_frame_func(       <L 43>
        if (!var_from_mask) {
            wp::adj_array_store(var_body_link_velocity_w, var_3, var_6, var_11, adj_body_link_velocity_w, adj_3, adj_6, adj_10);
            wp::adj_address(var_data, var_0, var_1, adj_data, adj_0, adj_1, adj_10);
            // adj: body_link_velocity_w[env_id, body_id] = data[i, j]                            <L 41>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_body_link_velocity_w, var_3, var_6, var_9, adj_body_link_velocity_w, adj_3, adj_6, adj_8);
            wp::adj_address(var_data, var_3, var_6, adj_data, adj_3, adj_6, adj_8);
            // adj: body_link_velocity_w[env_id, body_id] = data[env_id, body_id]                 <L 39>
        }
        // adj: if from_mask:                                                                     <L 38>
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_5);
        // adj: body_id = wp.int32(body_ids[j])                                                   <L 37>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 36>
        // adj: i, j = wp.tid()                                                                   <L 35>
        // adj: def set_body_link_velocity_to_sim(                                                <L 1>
        continue;
    }
}



extern "C" __global__ void write_body_com_position_to_buffer_mask_bc6d859d_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_in_data,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_body_mask,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_data)
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
        bool var_2;
        bool* var_3;
        bool var_4;
        bool* var_5;
        bool var_6;
        wp::vec_t<3, wp::float32>* var_7;
        wp::vec_t<3, wp::float32> var_8;
        //---------
        // forward
        // def write_body_com_position_to_buffer_mask(                                            <L 1328>
        // i, j = wp.tid()                                                                        <L 1345>
        builtin_tid2d(var_0, var_1);
        // if env_mask[i] and body_mask[j]:                                                       <L 1346>
        var_3 = wp::address(var_env_mask, var_0);
        var_4 = wp::load(var_3);
        var_2 = var_4;
        if (var_2) {
            var_5 = wp::address(var_body_mask, var_1);
            var_6 = wp::load(var_5);
            var_2 = var_2 && var_6;
        }
        if (var_2) {
            // out_data[i, j] = in_data[i, j]                                                     <L 1347>
            var_7 = wp::address(var_in_data, var_0, var_1);
            var_8 = wp::load(var_7);
            wp::array_store(var_out_data, var_0, var_1, var_8);
        }
    }
}



extern "C" __global__ void write_body_com_position_to_buffer_mask_bc6d859d_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_in_data,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_body_mask,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_data,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_in_data,
    wp::array_t<bool> adj_env_mask,
    wp::array_t<bool> adj_body_mask,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_out_data)
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
        bool var_2;
        bool* var_3;
        bool var_4;
        bool* var_5;
        bool var_6;
        wp::vec_t<3, wp::float32>* var_7;
        wp::vec_t<3, wp::float32> var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        bool adj_6 = {};
        wp::vec_t<3, wp::float32> adj_7 = {};
        wp::vec_t<3, wp::float32> adj_8 = {};
        //---------
        // forward
        // def write_body_com_position_to_buffer_mask(                                            <L 1328>
        // i, j = wp.tid()                                                                        <L 1345>
        builtin_tid2d(var_0, var_1);
        // if env_mask[i] and body_mask[j]:                                                       <L 1346>
        var_3 = wp::address(var_env_mask, var_0);
        var_4 = wp::load(var_3);
        var_2 = var_4;
        if (var_2) {
            var_5 = wp::address(var_body_mask, var_1);
            var_6 = wp::load(var_5);
            var_2 = var_2 && var_6;
        }
        if (var_2) {
            // out_data[i, j] = in_data[i, j]                                                     <L 1347>
            var_7 = wp::address(var_in_data, var_0, var_1);
            var_8 = wp::load(var_7);
            // wp::array_store(var_out_data, var_0, var_1, var_8);
        }
        //---------
        // reverse
        if (var_2) {
            wp::adj_array_store(var_out_data, var_0, var_1, var_8, adj_out_data, adj_0, adj_1, adj_7);
            wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_7);
            // adj: out_data[i, j] = in_data[i, j]                                                <L 1347>
        }
        if (var_2) {
            wp::adj_address(var_body_mask, var_1, adj_body_mask, adj_1, adj_5);
        }
        wp::adj_address(var_env_mask, var_0, adj_env_mask, adj_0, adj_3);
        // adj: if env_mask[i] and body_mask[j]:                                                  <L 1346>
        // adj: i, j = wp.tid()                                                                   <L 1345>
        // adj: def write_body_com_position_to_buffer_mask(                                       <L 1328>
        continue;
    }
}



extern "C" __global__ void set_root_link_pose_to_sim_index_bd271fcb_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::transform_t<wp::float32>> var_root_link_pose_w)
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
        wp::int64* var_1;
        wp::int32 var_2;
        wp::int64 var_3;
        wp::transform_t<wp::float32>* var_4;
        wp::transform_t<wp::float32> var_5;
        //---------
        // forward
        // def set_root_link_pose_to_sim_index(                                                   <L 1>
        // i = wp.tid()                                                                           <L 15>
        var_0 = builtin_tid1d();
        // env_id = wp.int32(env_ids[i])                                                          <L 16>
        var_1 = wp::address(var_env_ids, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::int32(var_3);
        // root_link_pose_w[env_id] = data[i]                                                     <L 17>
        var_4 = wp::address(var_data, var_0);
        var_5 = wp::load(var_4);
        wp::array_store(var_root_link_pose_w, var_2, var_5);
    }
}



extern "C" __global__ void set_root_link_pose_to_sim_index_bd271fcb_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::transform_t<wp::float32>> var_root_link_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_data,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::transform_t<wp::float32>> adj_root_link_pose_w)
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
        wp::int64* var_1;
        wp::int32 var_2;
        wp::int64 var_3;
        wp::transform_t<wp::float32>* var_4;
        wp::transform_t<wp::float32> var_5;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int64 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int64 adj_3 = {};
        wp::transform_t<wp::float32> adj_4 = {};
        wp::transform_t<wp::float32> adj_5 = {};
        //---------
        // forward
        // def set_root_link_pose_to_sim_index(                                                   <L 1>
        // i = wp.tid()                                                                           <L 15>
        var_0 = builtin_tid1d();
        // env_id = wp.int32(env_ids[i])                                                          <L 16>
        var_1 = wp::address(var_env_ids, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::int32(var_3);
        // root_link_pose_w[env_id] = data[i]                                                     <L 17>
        var_4 = wp::address(var_data, var_0);
        var_5 = wp::load(var_4);
        // wp::array_store(var_root_link_pose_w, var_2, var_5);
        //---------
        // reverse
        wp::adj_array_store(var_root_link_pose_w, var_2, var_5, adj_root_link_pose_w, adj_2, adj_4);
        wp::adj_address(var_data, var_0, adj_data, adj_0, adj_4);
        // adj: root_link_pose_w[env_id] = data[i]                                                <L 17>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_1);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 16>
        // adj: i = wp.tid()                                                                      <L 15>
        // adj: def set_root_link_pose_to_sim_index(                                              <L 1>
        continue;
    }
}



extern "C" __global__ void set_root_link_velocity_to_sim_index_bd8b1a27_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> var_link_pose_w,
    wp::array_t<wp::int64> var_env_ids,
    wp::int32 var_num_bodies,
    wp::array_t<wp::vec_t<6, wp::float32>> var_root_link_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_root_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w)
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
        wp::int64* var_1;
        wp::int32 var_2;
        wp::int64 var_3;
        wp::vec_t<6, wp::float32>* var_4;
        wp::vec_t<6, wp::float32> var_5;
        wp::vec_t<6, wp::float32>* var_6;
        wp::transform_t<wp::float32>* var_7;
        const wp::int32 var_8 = 0;
        wp::vec_t<3, wp::float32>* var_9;
        wp::vec_t<6, wp::float32> var_10;
        wp::vec_t<6, wp::float32> var_11;
        wp::transform_t<wp::float32> var_12;
        wp::vec_t<3, wp::float32> var_13;
        wp::range_t var_14;
        wp::int32 var_15;
        const wp::float32 var_16 = 0.0;
        const wp::float32 var_17 = 0.0;
        const wp::float32 var_18 = 0.0;
        const wp::float32 var_19 = 0.0;
        const wp::float32 var_20 = 0.0;
        const wp::float32 var_21 = 0.0;
        wp::vec_t<6, wp::float32> var_22;
        //---------
        // forward
        // def set_root_link_velocity_to_sim_index(                                               <L 1>
        // i = wp.tid()                                                                           <L 31>
        var_0 = builtin_tid1d();
        // env_id = wp.int32(env_ids[i])                                                          <L 32>
        var_1 = wp::address(var_env_ids, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::int32(var_3);
        // root_link_velocity_w[env_id] = data[i]                                                 <L 33>
        var_4 = wp::address(var_data, var_0);
        var_5 = wp::load(var_4);
        wp::array_store(var_root_link_velocity_w, var_2, var_5);
        // root_com_velocity_w[env_id] = get_link_velocity_in_com_frame_func(                     <L 35>
        // root_link_velocity_w[env_id], link_pose_w[env_id], body_com_pos_b[env_id, 0]           <L 36>
        var_6 = wp::address(var_root_link_velocity_w, var_2);
        var_7 = wp::address(var_link_pose_w, var_2);
        var_9 = wp::address(var_body_com_pos_b, var_2, var_8);
        var_11 = wp::load(var_6);
        var_12 = wp::load(var_7);
        var_13 = wp::load(var_9);
        var_10 = get_link_velocity_in_com_frame_func_0(var_11, var_12, var_13);
        // root_com_velocity_w[env_id] = get_link_velocity_in_com_frame_func(                     <L 35>
        wp::array_store(var_root_com_velocity_w, var_2, var_10);
        // for j in range(num_bodies):                                                            <L 39>
        var_14 = wp::range(var_num_bodies);
        start_for_0:;
            if (iter_cmp(var_14) == 0) goto end_for_0;
            var_15 = wp::iter_next(var_14);
            // body_acc_w[env_id, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)           <L 40>
            var_22 = wp::vec_t<6, wp::float32>({var_16, var_17, var_18, var_19, var_20, var_21});
            wp::array_store(var_body_acc_w, var_2, var_15, var_22);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void set_root_link_velocity_to_sim_index_bd8b1a27_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> var_link_pose_w,
    wp::array_t<wp::int64> var_env_ids,
    wp::int32 var_num_bodies,
    wp::array_t<wp::vec_t<6, wp::float32>> var_root_link_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_root_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_data,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> adj_link_pose_w,
    wp::array_t<wp::int64> adj_env_ids,
    wp::int32 adj_num_bodies,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_root_link_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_root_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_acc_w)
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
        wp::int64* var_1;
        wp::int32 var_2;
        wp::int64 var_3;
        wp::vec_t<6, wp::float32>* var_4;
        wp::vec_t<6, wp::float32> var_5;
        wp::vec_t<6, wp::float32>* var_6;
        wp::transform_t<wp::float32>* var_7;
        const wp::int32 var_8 = 0;
        wp::vec_t<3, wp::float32>* var_9;
        wp::vec_t<6, wp::float32> var_10;
        wp::vec_t<6, wp::float32> var_11;
        wp::transform_t<wp::float32> var_12;
        wp::vec_t<3, wp::float32> var_13;
        wp::range_t var_14;
        wp::int32 var_15;
        const wp::float32 var_16 = 0.0;
        const wp::float32 var_17 = 0.0;
        const wp::float32 var_18 = 0.0;
        const wp::float32 var_19 = 0.0;
        const wp::float32 var_20 = 0.0;
        const wp::float32 var_21 = 0.0;
        wp::vec_t<6, wp::float32> var_22;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int64 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int64 adj_3 = {};
        wp::vec_t<6, wp::float32> adj_4 = {};
        wp::vec_t<6, wp::float32> adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        wp::transform_t<wp::float32> adj_7 = {};
        wp::int32 adj_8 = {};
        wp::vec_t<3, wp::float32> adj_9 = {};
        wp::vec_t<6, wp::float32> adj_10 = {};
        wp::vec_t<6, wp::float32> adj_11 = {};
        wp::transform_t<wp::float32> adj_12 = {};
        wp::vec_t<3, wp::float32> adj_13 = {};
        wp::range_t adj_14 = {};
        wp::int32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::vec_t<6, wp::float32> adj_22 = {};
        //---------
        // forward
        // def set_root_link_velocity_to_sim_index(                                               <L 1>
        // i = wp.tid()                                                                           <L 31>
        var_0 = builtin_tid1d();
        // env_id = wp.int32(env_ids[i])                                                          <L 32>
        var_1 = wp::address(var_env_ids, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::int32(var_3);
        // root_link_velocity_w[env_id] = data[i]                                                 <L 33>
        var_4 = wp::address(var_data, var_0);
        var_5 = wp::load(var_4);
        // wp::array_store(var_root_link_velocity_w, var_2, var_5);
        // root_com_velocity_w[env_id] = get_link_velocity_in_com_frame_func(                     <L 35>
        // root_link_velocity_w[env_id], link_pose_w[env_id], body_com_pos_b[env_id, 0]           <L 36>
        var_6 = wp::address(var_root_link_velocity_w, var_2);
        var_7 = wp::address(var_link_pose_w, var_2);
        var_9 = wp::address(var_body_com_pos_b, var_2, var_8);
        var_11 = wp::load(var_6);
        var_12 = wp::load(var_7);
        var_13 = wp::load(var_9);
        var_10 = get_link_velocity_in_com_frame_func_0(var_11, var_12, var_13);
        // root_com_velocity_w[env_id] = get_link_velocity_in_com_frame_func(                     <L 35>
        // wp::array_store(var_root_com_velocity_w, var_2, var_10);
        // for j in range(num_bodies):                                                            <L 39>
        var_14 = wp::range(var_num_bodies);
        //---------
        // reverse
        var_14 = wp::iter_reverse(var_14);
        start_for_0:;
            if (iter_cmp(var_14) == 0) goto end_for_0;
            var_15 = wp::iter_next(var_14);
        	adj_16 = {};
        	adj_17 = {};
        	adj_18 = {};
        	adj_19 = {};
        	adj_20 = {};
        	adj_21 = {};
        	adj_22 = {};
            // body_acc_w[env_id, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)           <L 40>
            var_22 = wp::vec_t<6, wp::float32>({var_16, var_17, var_18, var_19, var_20, var_21});
            // wp::array_store(var_body_acc_w, var_2, var_15, var_22);
            wp::adj_array_store(var_body_acc_w, var_2, var_15, var_22, adj_body_acc_w, adj_2, adj_15, adj_22);
            wp::adj_vec_t({var_16, var_17, var_18, var_19, var_20, var_21}, {&adj_16, &adj_17, &adj_18, &adj_19, &adj_20, &adj_21}, adj_22);
            // adj: body_acc_w[env_id, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)      <L 40>
        	goto start_for_0;
        end_for_0:;
        // adj: for j in range(num_bodies):                                                       <L 39>
        wp::adj_array_store(var_root_com_velocity_w, var_2, var_10, adj_root_com_velocity_w, adj_2, adj_10);
        // adj: root_com_velocity_w[env_id] = get_link_velocity_in_com_frame_func(                <L 35>
        adj_get_link_velocity_in_com_frame_func_0(var_11, var_12, var_13, adj_6, adj_7, adj_9, adj_10);
        wp::adj_address(var_body_com_pos_b, var_2, var_8, adj_body_com_pos_b, adj_2, adj_8, adj_9);
        wp::adj_address(var_link_pose_w, var_2, adj_link_pose_w, adj_2, adj_7);
        wp::adj_address(var_root_link_velocity_w, var_2, adj_root_link_velocity_w, adj_2, adj_6);
        // adj: root_link_velocity_w[env_id], link_pose_w[env_id], body_com_pos_b[env_id, 0]      <L 36>
        // adj: root_com_velocity_w[env_id] = get_link_velocity_in_com_frame_func(                <L 35>
        wp::adj_array_store(var_root_link_velocity_w, var_2, var_5, adj_root_link_velocity_w, adj_2, adj_4);
        wp::adj_address(var_data, var_0, adj_data, adj_0, adj_4);
        // adj: root_link_velocity_w[env_id] = data[i]                                            <L 33>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_1);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 32>
        // adj: i = wp.tid()                                                                      <L 31>
        // adj: def set_root_link_velocity_to_sim_index(                                          <L 1>
        continue;
    }
}



extern "C" __global__ void concat_root_pose_and_vel_to_state_bdcd9ba9_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_pose,
    wp::array_t<wp::vec_t<6, wp::float32>> var_vel,
    wp::array_t<wp::vec_t<13, wp::float32>> var_state)
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
        wp::transform_t<wp::float32>* var_1;
        wp::vec_t<6, wp::float32>* var_2;
        wp::vec_t<13, wp::float32> var_3;
        wp::transform_t<wp::float32> var_4;
        wp::vec_t<6, wp::float32> var_5;
        //---------
        // forward
        // def concat_root_pose_and_vel_to_state(                                                 <L 277>
        // i = wp.tid()                                                                           <L 292>
        var_0 = builtin_tid1d();
        // state[i] = concat_pose_and_vel_to_state_func(pose[i], vel[i])                          <L 293>
        var_1 = wp::address(var_pose, var_0);
        var_2 = wp::address(var_vel, var_0);
        var_4 = wp::load(var_1);
        var_5 = wp::load(var_2);
        var_3 = concat_pose_and_vel_to_state_func_0(var_4, var_5);
        wp::array_store(var_state, var_0, var_3);
    }
}



extern "C" __global__ void concat_root_pose_and_vel_to_state_bdcd9ba9_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_pose,
    wp::array_t<wp::vec_t<6, wp::float32>> var_vel,
    wp::array_t<wp::vec_t<13, wp::float32>> var_state,
    wp::array_t<wp::transform_t<wp::float32>> adj_pose,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_vel,
    wp::array_t<wp::vec_t<13, wp::float32>> adj_state)
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
        wp::transform_t<wp::float32>* var_1;
        wp::vec_t<6, wp::float32>* var_2;
        wp::vec_t<13, wp::float32> var_3;
        wp::transform_t<wp::float32> var_4;
        wp::vec_t<6, wp::float32> var_5;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::transform_t<wp::float32> adj_1 = {};
        wp::vec_t<6, wp::float32> adj_2 = {};
        wp::vec_t<13, wp::float32> adj_3 = {};
        wp::transform_t<wp::float32> adj_4 = {};
        wp::vec_t<6, wp::float32> adj_5 = {};
        //---------
        // forward
        // def concat_root_pose_and_vel_to_state(                                                 <L 277>
        // i = wp.tid()                                                                           <L 292>
        var_0 = builtin_tid1d();
        // state[i] = concat_pose_and_vel_to_state_func(pose[i], vel[i])                          <L 293>
        var_1 = wp::address(var_pose, var_0);
        var_2 = wp::address(var_vel, var_0);
        var_4 = wp::load(var_1);
        var_5 = wp::load(var_2);
        var_3 = concat_pose_and_vel_to_state_func_0(var_4, var_5);
        // wp::array_store(var_state, var_0, var_3);
        //---------
        // reverse
        wp::adj_array_store(var_state, var_0, var_3, adj_state, adj_0, adj_3);
        adj_concat_pose_and_vel_to_state_func_0(var_4, var_5, adj_1, adj_2, adj_3);
        wp::adj_address(var_vel, var_0, adj_vel, adj_0, adj_2);
        wp::adj_address(var_pose, var_0, adj_pose, adj_0, adj_1);
        // adj: state[i] = concat_pose_and_vel_to_state_func(pose[i], vel[i])                     <L 293>
        // adj: i = wp.tid()                                                                      <L 292>
        // adj: def concat_root_pose_and_vel_to_state(                                            <L 277>
        continue;
    }
}



extern "C" __global__ void write_single_body_inertia_to_buffer_bde34dff_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int64> var_env_ids,
    bool var_from_mask,
    wp::array_t<wp::float32> var_out_data)
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
        wp::int64* var_1;
        wp::int32 var_2;
        wp::int64 var_3;
        const wp::int32 var_4 = 0;
        wp::float32* var_5;
        wp::float32 var_6;
        const wp::int32 var_7 = 1;
        wp::float32* var_8;
        wp::float32 var_9;
        const wp::int32 var_10 = 2;
        wp::float32* var_11;
        wp::float32 var_12;
        const wp::int32 var_13 = 3;
        wp::float32* var_14;
        wp::float32 var_15;
        const wp::int32 var_16 = 4;
        wp::float32* var_17;
        wp::float32 var_18;
        const wp::int32 var_19 = 5;
        wp::float32* var_20;
        wp::float32 var_21;
        const wp::int32 var_22 = 6;
        wp::float32* var_23;
        wp::float32 var_24;
        const wp::int32 var_25 = 7;
        wp::float32* var_26;
        wp::float32 var_27;
        const wp::int32 var_28 = 8;
        wp::float32* var_29;
        wp::float32 var_30;
        const wp::int32 var_31 = 0;
        wp::float32* var_32;
        wp::float32 var_33;
        const wp::int32 var_34 = 1;
        wp::float32* var_35;
        wp::float32 var_36;
        const wp::int32 var_37 = 2;
        wp::float32* var_38;
        wp::float32 var_39;
        const wp::int32 var_40 = 3;
        wp::float32* var_41;
        wp::float32 var_42;
        const wp::int32 var_43 = 4;
        wp::float32* var_44;
        wp::float32 var_45;
        const wp::int32 var_46 = 5;
        wp::float32* var_47;
        wp::float32 var_48;
        const wp::int32 var_49 = 6;
        wp::float32* var_50;
        wp::float32 var_51;
        const wp::int32 var_52 = 7;
        wp::float32* var_53;
        wp::float32 var_54;
        const wp::int32 var_55 = 8;
        wp::float32* var_56;
        wp::float32 var_57;
        wp::int32 var_58;
        //---------
        // forward
        // def write_single_body_inertia_to_buffer(                                               <L 1>
        // i = wp.tid()                                                                           <L 19>
        var_0 = builtin_tid1d();
        // env_id = wp.int32(env_ids[i])                                                          <L 20>
        var_1 = wp::address(var_env_ids, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::int32(var_3);
        // if from_mask:                                                                          <L 21>
        if (var_from_mask) {
            // for k in range(9):                                                                 <L 22>
            // out_data[env_id, k] = in_data[env_id, k]                                           <L 23>
            var_5 = wp::address(var_in_data, var_2, var_4);
            var_6 = wp::load(var_5);
            wp::array_store(var_out_data, var_2, var_4, var_6);
            var_8 = wp::address(var_in_data, var_2, var_7);
            var_9 = wp::load(var_8);
            wp::array_store(var_out_data, var_2, var_7, var_9);
            var_11 = wp::address(var_in_data, var_2, var_10);
            var_12 = wp::load(var_11);
            wp::array_store(var_out_data, var_2, var_10, var_12);
            var_14 = wp::address(var_in_data, var_2, var_13);
            var_15 = wp::load(var_14);
            wp::array_store(var_out_data, var_2, var_13, var_15);
            var_17 = wp::address(var_in_data, var_2, var_16);
            var_18 = wp::load(var_17);
            wp::array_store(var_out_data, var_2, var_16, var_18);
            var_20 = wp::address(var_in_data, var_2, var_19);
            var_21 = wp::load(var_20);
            wp::array_store(var_out_data, var_2, var_19, var_21);
            var_23 = wp::address(var_in_data, var_2, var_22);
            var_24 = wp::load(var_23);
            wp::array_store(var_out_data, var_2, var_22, var_24);
            var_26 = wp::address(var_in_data, var_2, var_25);
            var_27 = wp::load(var_26);
            wp::array_store(var_out_data, var_2, var_25, var_27);
            var_29 = wp::address(var_in_data, var_2, var_28);
            var_30 = wp::load(var_29);
            wp::array_store(var_out_data, var_2, var_28, var_30);
        }
        if (!var_from_mask) {
            // for k in range(9):                                                                 <L 25>
            // out_data[env_id, k] = in_data[i, k]                                                <L 26>
            var_32 = wp::address(var_in_data, var_0, var_31);
            var_33 = wp::load(var_32);
            wp::array_store(var_out_data, var_2, var_31, var_33);
            var_35 = wp::address(var_in_data, var_0, var_34);
            var_36 = wp::load(var_35);
            wp::array_store(var_out_data, var_2, var_34, var_36);
            var_38 = wp::address(var_in_data, var_0, var_37);
            var_39 = wp::load(var_38);
            wp::array_store(var_out_data, var_2, var_37, var_39);
            var_41 = wp::address(var_in_data, var_0, var_40);
            var_42 = wp::load(var_41);
            wp::array_store(var_out_data, var_2, var_40, var_42);
            var_44 = wp::address(var_in_data, var_0, var_43);
            var_45 = wp::load(var_44);
            wp::array_store(var_out_data, var_2, var_43, var_45);
            var_47 = wp::address(var_in_data, var_0, var_46);
            var_48 = wp::load(var_47);
            wp::array_store(var_out_data, var_2, var_46, var_48);
            var_50 = wp::address(var_in_data, var_0, var_49);
            var_51 = wp::load(var_50);
            wp::array_store(var_out_data, var_2, var_49, var_51);
            var_53 = wp::address(var_in_data, var_0, var_52);
            var_54 = wp::load(var_53);
            wp::array_store(var_out_data, var_2, var_52, var_54);
            var_56 = wp::address(var_in_data, var_0, var_55);
            var_57 = wp::load(var_56);
            wp::array_store(var_out_data, var_2, var_55, var_57);
        }
        var_58 = wp::where(var_from_mask, var_28, var_55);
    }
}



extern "C" __global__ void write_single_body_inertia_to_buffer_bde34dff_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int64> var_env_ids,
    bool var_from_mask,
    wp::array_t<wp::float32> var_out_data,
    wp::array_t<wp::float32> adj_in_data,
    wp::array_t<wp::int64> adj_env_ids,
    bool adj_from_mask,
    wp::array_t<wp::float32> adj_out_data)
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
        wp::int64* var_1;
        wp::int32 var_2;
        wp::int64 var_3;
        const wp::int32 var_4 = 0;
        wp::float32* var_5;
        wp::float32 var_6;
        const wp::int32 var_7 = 1;
        wp::float32* var_8;
        wp::float32 var_9;
        const wp::int32 var_10 = 2;
        wp::float32* var_11;
        wp::float32 var_12;
        const wp::int32 var_13 = 3;
        wp::float32* var_14;
        wp::float32 var_15;
        const wp::int32 var_16 = 4;
        wp::float32* var_17;
        wp::float32 var_18;
        const wp::int32 var_19 = 5;
        wp::float32* var_20;
        wp::float32 var_21;
        const wp::int32 var_22 = 6;
        wp::float32* var_23;
        wp::float32 var_24;
        const wp::int32 var_25 = 7;
        wp::float32* var_26;
        wp::float32 var_27;
        const wp::int32 var_28 = 8;
        wp::float32* var_29;
        wp::float32 var_30;
        const wp::int32 var_31 = 0;
        wp::float32* var_32;
        wp::float32 var_33;
        const wp::int32 var_34 = 1;
        wp::float32* var_35;
        wp::float32 var_36;
        const wp::int32 var_37 = 2;
        wp::float32* var_38;
        wp::float32 var_39;
        const wp::int32 var_40 = 3;
        wp::float32* var_41;
        wp::float32 var_42;
        const wp::int32 var_43 = 4;
        wp::float32* var_44;
        wp::float32 var_45;
        const wp::int32 var_46 = 5;
        wp::float32* var_47;
        wp::float32 var_48;
        const wp::int32 var_49 = 6;
        wp::float32* var_50;
        wp::float32 var_51;
        const wp::int32 var_52 = 7;
        wp::float32* var_53;
        wp::float32 var_54;
        const wp::int32 var_55 = 8;
        wp::float32* var_56;
        wp::float32 var_57;
        wp::int32 var_58;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int64 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int64 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::float32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::float32 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::float32 adj_41 = {};
        wp::float32 adj_42 = {};
        wp::int32 adj_43 = {};
        wp::float32 adj_44 = {};
        wp::float32 adj_45 = {};
        wp::int32 adj_46 = {};
        wp::float32 adj_47 = {};
        wp::float32 adj_48 = {};
        wp::int32 adj_49 = {};
        wp::float32 adj_50 = {};
        wp::float32 adj_51 = {};
        wp::int32 adj_52 = {};
        wp::float32 adj_53 = {};
        wp::float32 adj_54 = {};
        wp::int32 adj_55 = {};
        wp::float32 adj_56 = {};
        wp::float32 adj_57 = {};
        wp::int32 adj_58 = {};
        //---------
        // forward
        // def write_single_body_inertia_to_buffer(                                               <L 1>
        // i = wp.tid()                                                                           <L 19>
        var_0 = builtin_tid1d();
        // env_id = wp.int32(env_ids[i])                                                          <L 20>
        var_1 = wp::address(var_env_ids, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::int32(var_3);
        // if from_mask:                                                                          <L 21>
        if (var_from_mask) {
            // for k in range(9):                                                                 <L 22>
            // out_data[env_id, k] = in_data[env_id, k]                                           <L 23>
            var_5 = wp::address(var_in_data, var_2, var_4);
            var_6 = wp::load(var_5);
            // wp::array_store(var_out_data, var_2, var_4, var_6);
            var_8 = wp::address(var_in_data, var_2, var_7);
            var_9 = wp::load(var_8);
            // wp::array_store(var_out_data, var_2, var_7, var_9);
            var_11 = wp::address(var_in_data, var_2, var_10);
            var_12 = wp::load(var_11);
            // wp::array_store(var_out_data, var_2, var_10, var_12);
            var_14 = wp::address(var_in_data, var_2, var_13);
            var_15 = wp::load(var_14);
            // wp::array_store(var_out_data, var_2, var_13, var_15);
            var_17 = wp::address(var_in_data, var_2, var_16);
            var_18 = wp::load(var_17);
            // wp::array_store(var_out_data, var_2, var_16, var_18);
            var_20 = wp::address(var_in_data, var_2, var_19);
            var_21 = wp::load(var_20);
            // wp::array_store(var_out_data, var_2, var_19, var_21);
            var_23 = wp::address(var_in_data, var_2, var_22);
            var_24 = wp::load(var_23);
            // wp::array_store(var_out_data, var_2, var_22, var_24);
            var_26 = wp::address(var_in_data, var_2, var_25);
            var_27 = wp::load(var_26);
            // wp::array_store(var_out_data, var_2, var_25, var_27);
            var_29 = wp::address(var_in_data, var_2, var_28);
            var_30 = wp::load(var_29);
            // wp::array_store(var_out_data, var_2, var_28, var_30);
        }
        if (!var_from_mask) {
            // for k in range(9):                                                                 <L 25>
            // out_data[env_id, k] = in_data[i, k]                                                <L 26>
            var_32 = wp::address(var_in_data, var_0, var_31);
            var_33 = wp::load(var_32);
            // wp::array_store(var_out_data, var_2, var_31, var_33);
            var_35 = wp::address(var_in_data, var_0, var_34);
            var_36 = wp::load(var_35);
            // wp::array_store(var_out_data, var_2, var_34, var_36);
            var_38 = wp::address(var_in_data, var_0, var_37);
            var_39 = wp::load(var_38);
            // wp::array_store(var_out_data, var_2, var_37, var_39);
            var_41 = wp::address(var_in_data, var_0, var_40);
            var_42 = wp::load(var_41);
            // wp::array_store(var_out_data, var_2, var_40, var_42);
            var_44 = wp::address(var_in_data, var_0, var_43);
            var_45 = wp::load(var_44);
            // wp::array_store(var_out_data, var_2, var_43, var_45);
            var_47 = wp::address(var_in_data, var_0, var_46);
            var_48 = wp::load(var_47);
            // wp::array_store(var_out_data, var_2, var_46, var_48);
            var_50 = wp::address(var_in_data, var_0, var_49);
            var_51 = wp::load(var_50);
            // wp::array_store(var_out_data, var_2, var_49, var_51);
            var_53 = wp::address(var_in_data, var_0, var_52);
            var_54 = wp::load(var_53);
            // wp::array_store(var_out_data, var_2, var_52, var_54);
            var_56 = wp::address(var_in_data, var_0, var_55);
            var_57 = wp::load(var_56);
            // wp::array_store(var_out_data, var_2, var_55, var_57);
        }
        var_58 = wp::where(var_from_mask, var_28, var_55);
        //---------
        // reverse
        wp::adj_where(var_from_mask, var_28, var_55, adj_from_mask, adj_28, adj_55, adj_58);
        if (!var_from_mask) {
            wp::adj_array_store(var_out_data, var_2, var_55, var_57, adj_out_data, adj_2, adj_55, adj_56);
            wp::adj_address(var_in_data, var_0, var_55, adj_in_data, adj_0, adj_55, adj_56);
            wp::adj_array_store(var_out_data, var_2, var_52, var_54, adj_out_data, adj_2, adj_52, adj_53);
            wp::adj_address(var_in_data, var_0, var_52, adj_in_data, adj_0, adj_52, adj_53);
            wp::adj_array_store(var_out_data, var_2, var_49, var_51, adj_out_data, adj_2, adj_49, adj_50);
            wp::adj_address(var_in_data, var_0, var_49, adj_in_data, adj_0, adj_49, adj_50);
            wp::adj_array_store(var_out_data, var_2, var_46, var_48, adj_out_data, adj_2, adj_46, adj_47);
            wp::adj_address(var_in_data, var_0, var_46, adj_in_data, adj_0, adj_46, adj_47);
            wp::adj_array_store(var_out_data, var_2, var_43, var_45, adj_out_data, adj_2, adj_43, adj_44);
            wp::adj_address(var_in_data, var_0, var_43, adj_in_data, adj_0, adj_43, adj_44);
            wp::adj_array_store(var_out_data, var_2, var_40, var_42, adj_out_data, adj_2, adj_40, adj_41);
            wp::adj_address(var_in_data, var_0, var_40, adj_in_data, adj_0, adj_40, adj_41);
            wp::adj_array_store(var_out_data, var_2, var_37, var_39, adj_out_data, adj_2, adj_37, adj_38);
            wp::adj_address(var_in_data, var_0, var_37, adj_in_data, adj_0, adj_37, adj_38);
            wp::adj_array_store(var_out_data, var_2, var_34, var_36, adj_out_data, adj_2, adj_34, adj_35);
            wp::adj_address(var_in_data, var_0, var_34, adj_in_data, adj_0, adj_34, adj_35);
            wp::adj_array_store(var_out_data, var_2, var_31, var_33, adj_out_data, adj_2, adj_31, adj_32);
            wp::adj_address(var_in_data, var_0, var_31, adj_in_data, adj_0, adj_31, adj_32);
            // adj: out_data[env_id, k] = in_data[i, k]                                           <L 26>
            // adj: for k in range(9):                                                            <L 25>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_out_data, var_2, var_28, var_30, adj_out_data, adj_2, adj_28, adj_29);
            wp::adj_address(var_in_data, var_2, var_28, adj_in_data, adj_2, adj_28, adj_29);
            wp::adj_array_store(var_out_data, var_2, var_25, var_27, adj_out_data, adj_2, adj_25, adj_26);
            wp::adj_address(var_in_data, var_2, var_25, adj_in_data, adj_2, adj_25, adj_26);
            wp::adj_array_store(var_out_data, var_2, var_22, var_24, adj_out_data, adj_2, adj_22, adj_23);
            wp::adj_address(var_in_data, var_2, var_22, adj_in_data, adj_2, adj_22, adj_23);
            wp::adj_array_store(var_out_data, var_2, var_19, var_21, adj_out_data, adj_2, adj_19, adj_20);
            wp::adj_address(var_in_data, var_2, var_19, adj_in_data, adj_2, adj_19, adj_20);
            wp::adj_array_store(var_out_data, var_2, var_16, var_18, adj_out_data, adj_2, adj_16, adj_17);
            wp::adj_address(var_in_data, var_2, var_16, adj_in_data, adj_2, adj_16, adj_17);
            wp::adj_array_store(var_out_data, var_2, var_13, var_15, adj_out_data, adj_2, adj_13, adj_14);
            wp::adj_address(var_in_data, var_2, var_13, adj_in_data, adj_2, adj_13, adj_14);
            wp::adj_array_store(var_out_data, var_2, var_10, var_12, adj_out_data, adj_2, adj_10, adj_11);
            wp::adj_address(var_in_data, var_2, var_10, adj_in_data, adj_2, adj_10, adj_11);
            wp::adj_array_store(var_out_data, var_2, var_7, var_9, adj_out_data, adj_2, adj_7, adj_8);
            wp::adj_address(var_in_data, var_2, var_7, adj_in_data, adj_2, adj_7, adj_8);
            wp::adj_array_store(var_out_data, var_2, var_4, var_6, adj_out_data, adj_2, adj_4, adj_5);
            wp::adj_address(var_in_data, var_2, var_4, adj_in_data, adj_2, adj_4, adj_5);
            // adj: out_data[env_id, k] = in_data[env_id, k]                                      <L 23>
            // adj: for k in range(9):                                                            <L 22>
        }
        // adj: if from_mask:                                                                     <L 21>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_1);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 20>
        // adj: i = wp.tid()                                                                      <L 19>
        // adj: def write_single_body_inertia_to_buffer(                                          <L 1>
        continue;
    }
}



extern "C" __global__ void set_body_link_velocity_to_sim_c0dfe98a_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_link_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w)
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
        wp::vec_t<6, wp::float32>* var_8;
        wp::vec_t<6, wp::float32> var_9;
        wp::vec_t<6, wp::float32>* var_10;
        wp::vec_t<6, wp::float32> var_11;
        wp::vec_t<6, wp::float32>* var_12;
        wp::transform_t<wp::float32>* var_13;
        wp::vec_t<3, wp::float32>* var_14;
        wp::vec_t<6, wp::float32> var_15;
        wp::vec_t<6, wp::float32> var_16;
        wp::transform_t<wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        const wp::float32 var_19 = 0.0;
        const wp::float32 var_20 = 0.0;
        const wp::float32 var_21 = 0.0;
        const wp::float32 var_22 = 0.0;
        const wp::float32 var_23 = 0.0;
        const wp::float32 var_24 = 0.0;
        wp::vec_t<6, wp::float32> var_25;
        //---------
        // forward
        // def set_body_link_velocity_to_sim(                                                     <L 1>
        // i, j = wp.tid()                                                                        <L 35>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 36>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 37>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 38>
        if (var_from_mask) {
            // body_link_velocity_w[env_id, body_id] = data[env_id, body_id]                      <L 39>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_body_link_velocity_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_link_velocity_w[env_id, body_id] = data[i, j]                                 <L 41>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            wp::array_store(var_body_link_velocity_w, var_3, var_6, var_11);
        }
        // body_com_velocity_w[env_id, body_id] = get_link_velocity_in_com_frame_func(            <L 43>
        // body_link_velocity_w[env_id, body_id],                                                 <L 44>
        var_12 = wp::address(var_body_link_velocity_w, var_3, var_6);
        // body_link_pose_w[env_id, body_id],                                                     <L 45>
        var_13 = wp::address(var_body_link_pose_w, var_3, var_6);
        // body_com_pos_b[env_id, body_id],                                                       <L 46>
        var_14 = wp::address(var_body_com_pos_b, var_3, var_6);
        var_16 = wp::load(var_12);
        var_17 = wp::load(var_13);
        var_18 = wp::load(var_14);
        var_15 = get_link_velocity_in_com_frame_func_0(var_16, var_17, var_18);
        // body_com_velocity_w[env_id, body_id] = get_link_velocity_in_com_frame_func(            <L 43>
        wp::array_store(var_body_com_velocity_w, var_3, var_6, var_15);
        // body_acc_w[env_id, body_id] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)         <L 49>
        var_25 = wp::vec_t<6, wp::float32>({var_19, var_20, var_21, var_22, var_23, var_24});
        wp::array_store(var_body_acc_w, var_3, var_6, var_25);
    }
}



extern "C" __global__ void set_body_link_velocity_to_sim_c0dfe98a_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_link_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_data,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_link_pose_w,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_body_ids,
    bool adj_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_link_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_acc_w)
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
        wp::vec_t<6, wp::float32>* var_8;
        wp::vec_t<6, wp::float32> var_9;
        wp::vec_t<6, wp::float32>* var_10;
        wp::vec_t<6, wp::float32> var_11;
        wp::vec_t<6, wp::float32>* var_12;
        wp::transform_t<wp::float32>* var_13;
        wp::vec_t<3, wp::float32>* var_14;
        wp::vec_t<6, wp::float32> var_15;
        wp::vec_t<6, wp::float32> var_16;
        wp::transform_t<wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        const wp::float32 var_19 = 0.0;
        const wp::float32 var_20 = 0.0;
        const wp::float32 var_21 = 0.0;
        const wp::float32 var_22 = 0.0;
        const wp::float32 var_23 = 0.0;
        const wp::float32 var_24 = 0.0;
        wp::vec_t<6, wp::float32> var_25;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::vec_t<6, wp::float32> adj_8 = {};
        wp::vec_t<6, wp::float32> adj_9 = {};
        wp::vec_t<6, wp::float32> adj_10 = {};
        wp::vec_t<6, wp::float32> adj_11 = {};
        wp::vec_t<6, wp::float32> adj_12 = {};
        wp::transform_t<wp::float32> adj_13 = {};
        wp::vec_t<3, wp::float32> adj_14 = {};
        wp::vec_t<6, wp::float32> adj_15 = {};
        wp::vec_t<6, wp::float32> adj_16 = {};
        wp::transform_t<wp::float32> adj_17 = {};
        wp::vec_t<3, wp::float32> adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::vec_t<6, wp::float32> adj_25 = {};
        //---------
        // forward
        // def set_body_link_velocity_to_sim(                                                     <L 1>
        // i, j = wp.tid()                                                                        <L 35>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 36>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 37>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 38>
        if (var_from_mask) {
            // body_link_velocity_w[env_id, body_id] = data[env_id, body_id]                      <L 39>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            // wp::array_store(var_body_link_velocity_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_link_velocity_w[env_id, body_id] = data[i, j]                                 <L 41>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            // wp::array_store(var_body_link_velocity_w, var_3, var_6, var_11);
        }
        // body_com_velocity_w[env_id, body_id] = get_link_velocity_in_com_frame_func(            <L 43>
        // body_link_velocity_w[env_id, body_id],                                                 <L 44>
        var_12 = wp::address(var_body_link_velocity_w, var_3, var_6);
        // body_link_pose_w[env_id, body_id],                                                     <L 45>
        var_13 = wp::address(var_body_link_pose_w, var_3, var_6);
        // body_com_pos_b[env_id, body_id],                                                       <L 46>
        var_14 = wp::address(var_body_com_pos_b, var_3, var_6);
        var_16 = wp::load(var_12);
        var_17 = wp::load(var_13);
        var_18 = wp::load(var_14);
        var_15 = get_link_velocity_in_com_frame_func_0(var_16, var_17, var_18);
        // body_com_velocity_w[env_id, body_id] = get_link_velocity_in_com_frame_func(            <L 43>
        // wp::array_store(var_body_com_velocity_w, var_3, var_6, var_15);
        // body_acc_w[env_id, body_id] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)         <L 49>
        var_25 = wp::vec_t<6, wp::float32>({var_19, var_20, var_21, var_22, var_23, var_24});
        // wp::array_store(var_body_acc_w, var_3, var_6, var_25);
        //---------
        // reverse
        wp::adj_array_store(var_body_acc_w, var_3, var_6, var_25, adj_body_acc_w, adj_3, adj_6, adj_25);
        wp::adj_vec_t({var_19, var_20, var_21, var_22, var_23, var_24}, {&adj_19, &adj_20, &adj_21, &adj_22, &adj_23, &adj_24}, adj_25);
        // adj: body_acc_w[env_id, body_id] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)    <L 49>
        wp::adj_array_store(var_body_com_velocity_w, var_3, var_6, var_15, adj_body_com_velocity_w, adj_3, adj_6, adj_15);
        // adj: body_com_velocity_w[env_id, body_id] = get_link_velocity_in_com_frame_func(       <L 43>
        adj_get_link_velocity_in_com_frame_func_0(var_16, var_17, var_18, adj_12, adj_13, adj_14, adj_15);
        wp::adj_address(var_body_com_pos_b, var_3, var_6, adj_body_com_pos_b, adj_3, adj_6, adj_14);
        // adj: body_com_pos_b[env_id, body_id],                                                  <L 46>
        wp::adj_address(var_body_link_pose_w, var_3, var_6, adj_body_link_pose_w, adj_3, adj_6, adj_13);
        // adj: body_link_pose_w[env_id, body_id],                                                <L 45>
        wp::adj_address(var_body_link_velocity_w, var_3, var_6, adj_body_link_velocity_w, adj_3, adj_6, adj_12);
        // adj: body_link_velocity_w[env_id, body_id],                                            <L 44>
        // adj: body_com_velocity_w[env_id, body_id] = get_link_velocity_in_com_frame_func(       <L 43>
        if (!var_from_mask) {
            wp::adj_array_store(var_body_link_velocity_w, var_3, var_6, var_11, adj_body_link_velocity_w, adj_3, adj_6, adj_10);
            wp::adj_address(var_data, var_0, var_1, adj_data, adj_0, adj_1, adj_10);
            // adj: body_link_velocity_w[env_id, body_id] = data[i, j]                            <L 41>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_body_link_velocity_w, var_3, var_6, var_9, adj_body_link_velocity_w, adj_3, adj_6, adj_8);
            wp::adj_address(var_data, var_3, var_6, adj_data, adj_3, adj_6, adj_8);
            // adj: body_link_velocity_w[env_id, body_id] = data[env_id, body_id]                 <L 39>
        }
        // adj: if from_mask:                                                                     <L 38>
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_5);
        // adj: body_id = wp.int32(body_ids[j])                                                   <L 37>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 36>
        // adj: i, j = wp.tid()                                                                   <L 35>
        // adj: def set_body_link_velocity_to_sim(                                                <L 1>
        continue;
    }
}



extern "C" __global__ void write_body_inertia_and_inverse_mask_c11d5b99_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_inertias,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_body_mask,
    wp::array_t<wp::int32> var_body_user_to_backend,
    bool var_has_body_ordering,
    wp::array_t<wp::float32> var_body_mass_backend,
    wp::array_t<wp::float32> var_body_inertia_user,
    wp::array_t<wp::float32> var_body_inertia_backend,
    wp::array_t<wp::float32> var_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_body_inv_inertia)
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
        bool var_2;
        bool* var_3;
        bool var_4;
        bool var_5;
        bool* var_6;
        bool var_7;
        bool var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        const wp::int32 var_14 = 0;
        wp::float32* var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        const wp::int32 var_18 = 1;
        wp::float32* var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        const wp::int32 var_22 = 2;
        wp::float32* var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        const wp::int32 var_26 = 3;
        wp::float32* var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        const wp::int32 var_30 = 4;
        wp::float32* var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        const wp::int32 var_34 = 5;
        wp::float32* var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        const wp::int32 var_38 = 6;
        wp::float32* var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        const wp::int32 var_42 = 7;
        wp::float32* var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        const wp::int32 var_46 = 8;
        wp::float32* var_47;
        wp::float32 var_48;
        wp::float32 var_49;
        const bool var_50 = false;
        //---------
        // forward
        // def write_body_inertia_and_inverse_mask(                                               <L 1207>
        // env_id, user_body_id = wp.tid()                                                        <L 1220>
        builtin_tid2d(var_0, var_1);
        // if not env_mask[env_id] or not body_mask[user_body_id]:                                <L 1221>
        var_3 = wp::address(var_env_mask, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::unot(var_5);
        var_2 = var_4;
        if (!var_2) {
            var_6 = wp::address(var_body_mask, var_1);
            var_8 = wp::load(var_6);
            var_7 = wp::unot(var_8);
            var_2 = var_2 || var_7;
        }
        if (var_2) {
            // return                                                                             <L 1222>
            continue;
        }
        // backend_body_id = user_body_id                                                         <L 1223>
        var_9 = wp::copy(var_1);
        // if has_body_ordering:                                                                  <L 1224>
        if (var_has_body_ordering) {
            // backend_body_id = body_user_to_backend[user_body_id]                               <L 1225>
            var_10 = wp::address(var_body_user_to_backend, var_1);
            var_12 = wp::load(var_10);
            var_11 = wp::copy(var_12);
        }
        var_13 = wp::where(var_has_body_ordering, var_11, var_9);
        // for k in range(9):                                                                     <L 1226>
        // value = inertias[env_id, user_body_id, k]                                              <L 1227>
        var_15 = wp::address(var_inertias, var_0, var_1, var_14);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 1228>
        wp::array_store(var_body_inertia_user, var_0, var_1, var_14, var_16);
        // if has_body_ordering:                                                                  <L 1229>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 1230>
            wp::array_store(var_body_inertia_backend, var_0, var_13, var_14, var_16);
        }
        // value = inertias[env_id, user_body_id, k]                                              <L 1227>
        var_19 = wp::address(var_inertias, var_0, var_1, var_18);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 1228>
        wp::array_store(var_body_inertia_user, var_0, var_1, var_18, var_20);
        // if has_body_ordering:                                                                  <L 1229>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 1230>
            wp::array_store(var_body_inertia_backend, var_0, var_13, var_18, var_20);
        }
        // value = inertias[env_id, user_body_id, k]                                              <L 1227>
        var_23 = wp::address(var_inertias, var_0, var_1, var_22);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 1228>
        wp::array_store(var_body_inertia_user, var_0, var_1, var_22, var_24);
        // if has_body_ordering:                                                                  <L 1229>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 1230>
            wp::array_store(var_body_inertia_backend, var_0, var_13, var_22, var_24);
        }
        // value = inertias[env_id, user_body_id, k]                                              <L 1227>
        var_27 = wp::address(var_inertias, var_0, var_1, var_26);
        var_29 = wp::load(var_27);
        var_28 = wp::copy(var_29);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 1228>
        wp::array_store(var_body_inertia_user, var_0, var_1, var_26, var_28);
        // if has_body_ordering:                                                                  <L 1229>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 1230>
            wp::array_store(var_body_inertia_backend, var_0, var_13, var_26, var_28);
        }
        // value = inertias[env_id, user_body_id, k]                                              <L 1227>
        var_31 = wp::address(var_inertias, var_0, var_1, var_30);
        var_33 = wp::load(var_31);
        var_32 = wp::copy(var_33);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 1228>
        wp::array_store(var_body_inertia_user, var_0, var_1, var_30, var_32);
        // if has_body_ordering:                                                                  <L 1229>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 1230>
            wp::array_store(var_body_inertia_backend, var_0, var_13, var_30, var_32);
        }
        // value = inertias[env_id, user_body_id, k]                                              <L 1227>
        var_35 = wp::address(var_inertias, var_0, var_1, var_34);
        var_37 = wp::load(var_35);
        var_36 = wp::copy(var_37);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 1228>
        wp::array_store(var_body_inertia_user, var_0, var_1, var_34, var_36);
        // if has_body_ordering:                                                                  <L 1229>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 1230>
            wp::array_store(var_body_inertia_backend, var_0, var_13, var_34, var_36);
        }
        // value = inertias[env_id, user_body_id, k]                                              <L 1227>
        var_39 = wp::address(var_inertias, var_0, var_1, var_38);
        var_41 = wp::load(var_39);
        var_40 = wp::copy(var_41);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 1228>
        wp::array_store(var_body_inertia_user, var_0, var_1, var_38, var_40);
        // if has_body_ordering:                                                                  <L 1229>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 1230>
            wp::array_store(var_body_inertia_backend, var_0, var_13, var_38, var_40);
        }
        // value = inertias[env_id, user_body_id, k]                                              <L 1227>
        var_43 = wp::address(var_inertias, var_0, var_1, var_42);
        var_45 = wp::load(var_43);
        var_44 = wp::copy(var_45);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 1228>
        wp::array_store(var_body_inertia_user, var_0, var_1, var_42, var_44);
        // if has_body_ordering:                                                                  <L 1229>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 1230>
            wp::array_store(var_body_inertia_backend, var_0, var_13, var_42, var_44);
        }
        // value = inertias[env_id, user_body_id, k]                                              <L 1227>
        var_47 = wp::address(var_inertias, var_0, var_1, var_46);
        var_49 = wp::load(var_47);
        var_48 = wp::copy(var_49);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 1228>
        wp::array_store(var_body_inertia_user, var_0, var_1, var_46, var_48);
        // if has_body_ordering:                                                                  <L 1229>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 1230>
            wp::array_store(var_body_inertia_backend, var_0, var_13, var_46, var_48);
        }
        // _update_body_inertial_properties_inverse(                                              <L 1231>
        // env_id, backend_body_id, body_mass_backend, body_inertia_backend, False, body_inv_mass, body_inv_inertia       <L 1232>
        _update_body_inertial_properties_inverse_0(var_0, var_13, var_body_mass_backend, var_body_inertia_backend, var_50, var_body_inv_mass, var_body_inv_inertia);
    }
}



extern "C" __global__ void write_body_inertia_and_inverse_mask_c11d5b99_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_inertias,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_body_mask,
    wp::array_t<wp::int32> var_body_user_to_backend,
    bool var_has_body_ordering,
    wp::array_t<wp::float32> var_body_mass_backend,
    wp::array_t<wp::float32> var_body_inertia_user,
    wp::array_t<wp::float32> var_body_inertia_backend,
    wp::array_t<wp::float32> var_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_body_inv_inertia,
    wp::array_t<wp::float32> adj_inertias,
    wp::array_t<bool> adj_env_mask,
    wp::array_t<bool> adj_body_mask,
    wp::array_t<wp::int32> adj_body_user_to_backend,
    bool adj_has_body_ordering,
    wp::array_t<wp::float32> adj_body_mass_backend,
    wp::array_t<wp::float32> adj_body_inertia_user,
    wp::array_t<wp::float32> adj_body_inertia_backend,
    wp::array_t<wp::float32> adj_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> adj_body_inv_inertia)
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
        bool var_2;
        bool* var_3;
        bool var_4;
        bool var_5;
        bool* var_6;
        bool var_7;
        bool var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        const wp::int32 var_14 = 0;
        wp::float32* var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        const wp::int32 var_18 = 1;
        wp::float32* var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        const wp::int32 var_22 = 2;
        wp::float32* var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        const wp::int32 var_26 = 3;
        wp::float32* var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        const wp::int32 var_30 = 4;
        wp::float32* var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        const wp::int32 var_34 = 5;
        wp::float32* var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        const wp::int32 var_38 = 6;
        wp::float32* var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        const wp::int32 var_42 = 7;
        wp::float32* var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        const wp::int32 var_46 = 8;
        wp::float32* var_47;
        wp::float32 var_48;
        wp::float32 var_49;
        const bool var_50 = false;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        bool adj_6 = {};
        bool adj_7 = {};
        bool adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::float32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::float32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::float32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::float32 adj_39 = {};
        wp::float32 adj_40 = {};
        wp::float32 adj_41 = {};
        wp::int32 adj_42 = {};
        wp::float32 adj_43 = {};
        wp::float32 adj_44 = {};
        wp::float32 adj_45 = {};
        wp::int32 adj_46 = {};
        wp::float32 adj_47 = {};
        wp::float32 adj_48 = {};
        wp::float32 adj_49 = {};
        bool adj_50 = {};
        //---------
        // forward
        // def write_body_inertia_and_inverse_mask(                                               <L 1207>
        // env_id, user_body_id = wp.tid()                                                        <L 1220>
        builtin_tid2d(var_0, var_1);
        // if not env_mask[env_id] or not body_mask[user_body_id]:                                <L 1221>
        var_3 = wp::address(var_env_mask, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::unot(var_5);
        var_2 = var_4;
        if (!var_2) {
            var_6 = wp::address(var_body_mask, var_1);
            var_8 = wp::load(var_6);
            var_7 = wp::unot(var_8);
            var_2 = var_2 || var_7;
        }
        if (var_2) {
            // return                                                                             <L 1222>
            goto label0;
        }
        // backend_body_id = user_body_id                                                         <L 1223>
        var_9 = wp::copy(var_1);
        // if has_body_ordering:                                                                  <L 1224>
        if (var_has_body_ordering) {
            // backend_body_id = body_user_to_backend[user_body_id]                               <L 1225>
            var_10 = wp::address(var_body_user_to_backend, var_1);
            var_12 = wp::load(var_10);
            var_11 = wp::copy(var_12);
        }
        var_13 = wp::where(var_has_body_ordering, var_11, var_9);
        // for k in range(9):                                                                     <L 1226>
        // value = inertias[env_id, user_body_id, k]                                              <L 1227>
        var_15 = wp::address(var_inertias, var_0, var_1, var_14);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 1228>
        // wp::array_store(var_body_inertia_user, var_0, var_1, var_14, var_16);
        // if has_body_ordering:                                                                  <L 1229>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 1230>
            // wp::array_store(var_body_inertia_backend, var_0, var_13, var_14, var_16);
        }
        // value = inertias[env_id, user_body_id, k]                                              <L 1227>
        var_19 = wp::address(var_inertias, var_0, var_1, var_18);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 1228>
        // wp::array_store(var_body_inertia_user, var_0, var_1, var_18, var_20);
        // if has_body_ordering:                                                                  <L 1229>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 1230>
            // wp::array_store(var_body_inertia_backend, var_0, var_13, var_18, var_20);
        }
        // value = inertias[env_id, user_body_id, k]                                              <L 1227>
        var_23 = wp::address(var_inertias, var_0, var_1, var_22);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 1228>
        // wp::array_store(var_body_inertia_user, var_0, var_1, var_22, var_24);
        // if has_body_ordering:                                                                  <L 1229>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 1230>
            // wp::array_store(var_body_inertia_backend, var_0, var_13, var_22, var_24);
        }
        // value = inertias[env_id, user_body_id, k]                                              <L 1227>
        var_27 = wp::address(var_inertias, var_0, var_1, var_26);
        var_29 = wp::load(var_27);
        var_28 = wp::copy(var_29);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 1228>
        // wp::array_store(var_body_inertia_user, var_0, var_1, var_26, var_28);
        // if has_body_ordering:                                                                  <L 1229>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 1230>
            // wp::array_store(var_body_inertia_backend, var_0, var_13, var_26, var_28);
        }
        // value = inertias[env_id, user_body_id, k]                                              <L 1227>
        var_31 = wp::address(var_inertias, var_0, var_1, var_30);
        var_33 = wp::load(var_31);
        var_32 = wp::copy(var_33);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 1228>
        // wp::array_store(var_body_inertia_user, var_0, var_1, var_30, var_32);
        // if has_body_ordering:                                                                  <L 1229>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 1230>
            // wp::array_store(var_body_inertia_backend, var_0, var_13, var_30, var_32);
        }
        // value = inertias[env_id, user_body_id, k]                                              <L 1227>
        var_35 = wp::address(var_inertias, var_0, var_1, var_34);
        var_37 = wp::load(var_35);
        var_36 = wp::copy(var_37);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 1228>
        // wp::array_store(var_body_inertia_user, var_0, var_1, var_34, var_36);
        // if has_body_ordering:                                                                  <L 1229>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 1230>
            // wp::array_store(var_body_inertia_backend, var_0, var_13, var_34, var_36);
        }
        // value = inertias[env_id, user_body_id, k]                                              <L 1227>
        var_39 = wp::address(var_inertias, var_0, var_1, var_38);
        var_41 = wp::load(var_39);
        var_40 = wp::copy(var_41);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 1228>
        // wp::array_store(var_body_inertia_user, var_0, var_1, var_38, var_40);
        // if has_body_ordering:                                                                  <L 1229>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 1230>
            // wp::array_store(var_body_inertia_backend, var_0, var_13, var_38, var_40);
        }
        // value = inertias[env_id, user_body_id, k]                                              <L 1227>
        var_43 = wp::address(var_inertias, var_0, var_1, var_42);
        var_45 = wp::load(var_43);
        var_44 = wp::copy(var_45);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 1228>
        // wp::array_store(var_body_inertia_user, var_0, var_1, var_42, var_44);
        // if has_body_ordering:                                                                  <L 1229>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 1230>
            // wp::array_store(var_body_inertia_backend, var_0, var_13, var_42, var_44);
        }
        // value = inertias[env_id, user_body_id, k]                                              <L 1227>
        var_47 = wp::address(var_inertias, var_0, var_1, var_46);
        var_49 = wp::load(var_47);
        var_48 = wp::copy(var_49);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 1228>
        // wp::array_store(var_body_inertia_user, var_0, var_1, var_46, var_48);
        // if has_body_ordering:                                                                  <L 1229>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 1230>
            // wp::array_store(var_body_inertia_backend, var_0, var_13, var_46, var_48);
        }
        // _update_body_inertial_properties_inverse(                                              <L 1231>
        // env_id, backend_body_id, body_mass_backend, body_inertia_backend, False, body_inv_mass, body_inv_inertia       <L 1232>
        _update_body_inertial_properties_inverse_0(var_0, var_13, var_body_mass_backend, var_body_inertia_backend, var_50, var_body_inv_mass, var_body_inv_inertia);
        //---------
        // reverse
        adj__update_body_inertial_properties_inverse_0(var_0, var_13, var_body_mass_backend, var_body_inertia_backend, var_50, var_body_inv_mass, var_body_inv_inertia, adj_0, adj_13, adj_body_mass_backend, adj_body_inertia_backend, adj_50, adj_body_inv_mass, adj_body_inv_inertia);
        // adj: env_id, backend_body_id, body_mass_backend, body_inertia_backend, False, body_inv_mass, body_inv_inertia  <L 1232>
        // adj: _update_body_inertial_properties_inverse(                                         <L 1231>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_0, var_13, var_46, var_48, adj_body_inertia_backend, adj_0, adj_13, adj_46, adj_48);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 1230>
        }
        // adj: if has_body_ordering:                                                             <L 1229>
        wp::adj_array_store(var_body_inertia_user, var_0, var_1, var_46, var_48, adj_body_inertia_user, adj_0, adj_1, adj_46, adj_48);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 1228>
        wp::adj_copy(var_49, adj_47, adj_48);
        wp::adj_address(var_inertias, var_0, var_1, var_46, adj_inertias, adj_0, adj_1, adj_46, adj_47);
        // adj: value = inertias[env_id, user_body_id, k]                                         <L 1227>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_0, var_13, var_42, var_44, adj_body_inertia_backend, adj_0, adj_13, adj_42, adj_44);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 1230>
        }
        // adj: if has_body_ordering:                                                             <L 1229>
        wp::adj_array_store(var_body_inertia_user, var_0, var_1, var_42, var_44, adj_body_inertia_user, adj_0, adj_1, adj_42, adj_44);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 1228>
        wp::adj_copy(var_45, adj_43, adj_44);
        wp::adj_address(var_inertias, var_0, var_1, var_42, adj_inertias, adj_0, adj_1, adj_42, adj_43);
        // adj: value = inertias[env_id, user_body_id, k]                                         <L 1227>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_0, var_13, var_38, var_40, adj_body_inertia_backend, adj_0, adj_13, adj_38, adj_40);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 1230>
        }
        // adj: if has_body_ordering:                                                             <L 1229>
        wp::adj_array_store(var_body_inertia_user, var_0, var_1, var_38, var_40, adj_body_inertia_user, adj_0, adj_1, adj_38, adj_40);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 1228>
        wp::adj_copy(var_41, adj_39, adj_40);
        wp::adj_address(var_inertias, var_0, var_1, var_38, adj_inertias, adj_0, adj_1, adj_38, adj_39);
        // adj: value = inertias[env_id, user_body_id, k]                                         <L 1227>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_0, var_13, var_34, var_36, adj_body_inertia_backend, adj_0, adj_13, adj_34, adj_36);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 1230>
        }
        // adj: if has_body_ordering:                                                             <L 1229>
        wp::adj_array_store(var_body_inertia_user, var_0, var_1, var_34, var_36, adj_body_inertia_user, adj_0, adj_1, adj_34, adj_36);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 1228>
        wp::adj_copy(var_37, adj_35, adj_36);
        wp::adj_address(var_inertias, var_0, var_1, var_34, adj_inertias, adj_0, adj_1, adj_34, adj_35);
        // adj: value = inertias[env_id, user_body_id, k]                                         <L 1227>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_0, var_13, var_30, var_32, adj_body_inertia_backend, adj_0, adj_13, adj_30, adj_32);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 1230>
        }
        // adj: if has_body_ordering:                                                             <L 1229>
        wp::adj_array_store(var_body_inertia_user, var_0, var_1, var_30, var_32, adj_body_inertia_user, adj_0, adj_1, adj_30, adj_32);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 1228>
        wp::adj_copy(var_33, adj_31, adj_32);
        wp::adj_address(var_inertias, var_0, var_1, var_30, adj_inertias, adj_0, adj_1, adj_30, adj_31);
        // adj: value = inertias[env_id, user_body_id, k]                                         <L 1227>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_0, var_13, var_26, var_28, adj_body_inertia_backend, adj_0, adj_13, adj_26, adj_28);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 1230>
        }
        // adj: if has_body_ordering:                                                             <L 1229>
        wp::adj_array_store(var_body_inertia_user, var_0, var_1, var_26, var_28, adj_body_inertia_user, adj_0, adj_1, adj_26, adj_28);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 1228>
        wp::adj_copy(var_29, adj_27, adj_28);
        wp::adj_address(var_inertias, var_0, var_1, var_26, adj_inertias, adj_0, adj_1, adj_26, adj_27);
        // adj: value = inertias[env_id, user_body_id, k]                                         <L 1227>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_0, var_13, var_22, var_24, adj_body_inertia_backend, adj_0, adj_13, adj_22, adj_24);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 1230>
        }
        // adj: if has_body_ordering:                                                             <L 1229>
        wp::adj_array_store(var_body_inertia_user, var_0, var_1, var_22, var_24, adj_body_inertia_user, adj_0, adj_1, adj_22, adj_24);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 1228>
        wp::adj_copy(var_25, adj_23, adj_24);
        wp::adj_address(var_inertias, var_0, var_1, var_22, adj_inertias, adj_0, adj_1, adj_22, adj_23);
        // adj: value = inertias[env_id, user_body_id, k]                                         <L 1227>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_0, var_13, var_18, var_20, adj_body_inertia_backend, adj_0, adj_13, adj_18, adj_20);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 1230>
        }
        // adj: if has_body_ordering:                                                             <L 1229>
        wp::adj_array_store(var_body_inertia_user, var_0, var_1, var_18, var_20, adj_body_inertia_user, adj_0, adj_1, adj_18, adj_20);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 1228>
        wp::adj_copy(var_21, adj_19, adj_20);
        wp::adj_address(var_inertias, var_0, var_1, var_18, adj_inertias, adj_0, adj_1, adj_18, adj_19);
        // adj: value = inertias[env_id, user_body_id, k]                                         <L 1227>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_0, var_13, var_14, var_16, adj_body_inertia_backend, adj_0, adj_13, adj_14, adj_16);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 1230>
        }
        // adj: if has_body_ordering:                                                             <L 1229>
        wp::adj_array_store(var_body_inertia_user, var_0, var_1, var_14, var_16, adj_body_inertia_user, adj_0, adj_1, adj_14, adj_16);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 1228>
        wp::adj_copy(var_17, adj_15, adj_16);
        wp::adj_address(var_inertias, var_0, var_1, var_14, adj_inertias, adj_0, adj_1, adj_14, adj_15);
        // adj: value = inertias[env_id, user_body_id, k]                                         <L 1227>
        // adj: for k in range(9):                                                                <L 1226>
        wp::adj_where(var_has_body_ordering, var_11, var_9, adj_has_body_ordering, adj_11, adj_9, adj_13);
        if (var_has_body_ordering) {
            wp::adj_copy(var_12, adj_10, adj_11);
            wp::adj_address(var_body_user_to_backend, var_1, adj_body_user_to_backend, adj_1, adj_10);
            // adj: backend_body_id = body_user_to_backend[user_body_id]                          <L 1225>
        }
        // adj: if has_body_ordering:                                                             <L 1224>
        wp::adj_copy(var_1, adj_1, adj_9);
        // adj: backend_body_id = user_body_id                                                    <L 1223>
        if (var_2) {
            label0:;
            // adj: return                                                                        <L 1222>
        }
        if (!var_2) {
            wp::adj_address(var_body_mask, var_1, adj_body_mask, adj_1, adj_6);
        }
        wp::adj_address(var_env_mask, var_0, adj_env_mask, adj_0, adj_3);
        // adj: if not env_mask[env_id] or not body_mask[user_body_id]:                           <L 1221>
        // adj: env_id, user_body_id = wp.tid()                                                   <L 1220>
        // adj: def write_body_inertia_and_inverse_mask(                                          <L 1207>
        continue;
    }
}



extern "C" __global__ void set_body_link_velocity_to_sim_c3600bf8_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_link_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::vec_t<6, wp::float32>* var_8;
        wp::vec_t<6, wp::float32> var_9;
        wp::vec_t<6, wp::float32>* var_10;
        wp::vec_t<6, wp::float32> var_11;
        wp::vec_t<6, wp::float32>* var_12;
        wp::transform_t<wp::float32>* var_13;
        wp::vec_t<3, wp::float32>* var_14;
        wp::vec_t<6, wp::float32> var_15;
        wp::vec_t<6, wp::float32> var_16;
        wp::transform_t<wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        const wp::float32 var_19 = 0.0;
        const wp::float32 var_20 = 0.0;
        const wp::float32 var_21 = 0.0;
        const wp::float32 var_22 = 0.0;
        const wp::float32 var_23 = 0.0;
        const wp::float32 var_24 = 0.0;
        wp::vec_t<6, wp::float32> var_25;
        //---------
        // forward
        // def set_body_link_velocity_to_sim(                                                     <L 1>
        // i, j = wp.tid()                                                                        <L 35>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 36>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 37>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 38>
        if (var_from_mask) {
            // body_link_velocity_w[env_id, body_id] = data[env_id, body_id]                      <L 39>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_body_link_velocity_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_link_velocity_w[env_id, body_id] = data[i, j]                                 <L 41>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            wp::array_store(var_body_link_velocity_w, var_3, var_6, var_11);
        }
        // body_com_velocity_w[env_id, body_id] = get_link_velocity_in_com_frame_func(            <L 43>
        // body_link_velocity_w[env_id, body_id],                                                 <L 44>
        var_12 = wp::address(var_body_link_velocity_w, var_3, var_6);
        // body_link_pose_w[env_id, body_id],                                                     <L 45>
        var_13 = wp::address(var_body_link_pose_w, var_3, var_6);
        // body_com_pos_b[env_id, body_id],                                                       <L 46>
        var_14 = wp::address(var_body_com_pos_b, var_3, var_6);
        var_16 = wp::load(var_12);
        var_17 = wp::load(var_13);
        var_18 = wp::load(var_14);
        var_15 = get_link_velocity_in_com_frame_func_0(var_16, var_17, var_18);
        // body_com_velocity_w[env_id, body_id] = get_link_velocity_in_com_frame_func(            <L 43>
        wp::array_store(var_body_com_velocity_w, var_3, var_6, var_15);
        // body_acc_w[env_id, body_id] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)         <L 49>
        var_25 = wp::vec_t<6, wp::float32>({var_19, var_20, var_21, var_22, var_23, var_24});
        wp::array_store(var_body_acc_w, var_3, var_6, var_25);
    }
}



extern "C" __global__ void set_body_link_velocity_to_sim_c3600bf8_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_link_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_data,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_link_pose_w,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int32> adj_body_ids,
    bool adj_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_link_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_acc_w)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::vec_t<6, wp::float32>* var_8;
        wp::vec_t<6, wp::float32> var_9;
        wp::vec_t<6, wp::float32>* var_10;
        wp::vec_t<6, wp::float32> var_11;
        wp::vec_t<6, wp::float32>* var_12;
        wp::transform_t<wp::float32>* var_13;
        wp::vec_t<3, wp::float32>* var_14;
        wp::vec_t<6, wp::float32> var_15;
        wp::vec_t<6, wp::float32> var_16;
        wp::transform_t<wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        const wp::float32 var_19 = 0.0;
        const wp::float32 var_20 = 0.0;
        const wp::float32 var_21 = 0.0;
        const wp::float32 var_22 = 0.0;
        const wp::float32 var_23 = 0.0;
        const wp::float32 var_24 = 0.0;
        wp::vec_t<6, wp::float32> var_25;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int64 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int64 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::vec_t<6, wp::float32> adj_8 = {};
        wp::vec_t<6, wp::float32> adj_9 = {};
        wp::vec_t<6, wp::float32> adj_10 = {};
        wp::vec_t<6, wp::float32> adj_11 = {};
        wp::vec_t<6, wp::float32> adj_12 = {};
        wp::transform_t<wp::float32> adj_13 = {};
        wp::vec_t<3, wp::float32> adj_14 = {};
        wp::vec_t<6, wp::float32> adj_15 = {};
        wp::vec_t<6, wp::float32> adj_16 = {};
        wp::transform_t<wp::float32> adj_17 = {};
        wp::vec_t<3, wp::float32> adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::vec_t<6, wp::float32> adj_25 = {};
        //---------
        // forward
        // def set_body_link_velocity_to_sim(                                                     <L 1>
        // i, j = wp.tid()                                                                        <L 35>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 36>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 37>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 38>
        if (var_from_mask) {
            // body_link_velocity_w[env_id, body_id] = data[env_id, body_id]                      <L 39>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            // wp::array_store(var_body_link_velocity_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_link_velocity_w[env_id, body_id] = data[i, j]                                 <L 41>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            // wp::array_store(var_body_link_velocity_w, var_3, var_6, var_11);
        }
        // body_com_velocity_w[env_id, body_id] = get_link_velocity_in_com_frame_func(            <L 43>
        // body_link_velocity_w[env_id, body_id],                                                 <L 44>
        var_12 = wp::address(var_body_link_velocity_w, var_3, var_6);
        // body_link_pose_w[env_id, body_id],                                                     <L 45>
        var_13 = wp::address(var_body_link_pose_w, var_3, var_6);
        // body_com_pos_b[env_id, body_id],                                                       <L 46>
        var_14 = wp::address(var_body_com_pos_b, var_3, var_6);
        var_16 = wp::load(var_12);
        var_17 = wp::load(var_13);
        var_18 = wp::load(var_14);
        var_15 = get_link_velocity_in_com_frame_func_0(var_16, var_17, var_18);
        // body_com_velocity_w[env_id, body_id] = get_link_velocity_in_com_frame_func(            <L 43>
        // wp::array_store(var_body_com_velocity_w, var_3, var_6, var_15);
        // body_acc_w[env_id, body_id] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)         <L 49>
        var_25 = wp::vec_t<6, wp::float32>({var_19, var_20, var_21, var_22, var_23, var_24});
        // wp::array_store(var_body_acc_w, var_3, var_6, var_25);
        //---------
        // reverse
        wp::adj_array_store(var_body_acc_w, var_3, var_6, var_25, adj_body_acc_w, adj_3, adj_6, adj_25);
        wp::adj_vec_t({var_19, var_20, var_21, var_22, var_23, var_24}, {&adj_19, &adj_20, &adj_21, &adj_22, &adj_23, &adj_24}, adj_25);
        // adj: body_acc_w[env_id, body_id] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)    <L 49>
        wp::adj_array_store(var_body_com_velocity_w, var_3, var_6, var_15, adj_body_com_velocity_w, adj_3, adj_6, adj_15);
        // adj: body_com_velocity_w[env_id, body_id] = get_link_velocity_in_com_frame_func(       <L 43>
        adj_get_link_velocity_in_com_frame_func_0(var_16, var_17, var_18, adj_12, adj_13, adj_14, adj_15);
        wp::adj_address(var_body_com_pos_b, var_3, var_6, adj_body_com_pos_b, adj_3, adj_6, adj_14);
        // adj: body_com_pos_b[env_id, body_id],                                                  <L 46>
        wp::adj_address(var_body_link_pose_w, var_3, var_6, adj_body_link_pose_w, adj_3, adj_6, adj_13);
        // adj: body_link_pose_w[env_id, body_id],                                                <L 45>
        wp::adj_address(var_body_link_velocity_w, var_3, var_6, adj_body_link_velocity_w, adj_3, adj_6, adj_12);
        // adj: body_link_velocity_w[env_id, body_id],                                            <L 44>
        // adj: body_com_velocity_w[env_id, body_id] = get_link_velocity_in_com_frame_func(       <L 43>
        if (!var_from_mask) {
            wp::adj_array_store(var_body_link_velocity_w, var_3, var_6, var_11, adj_body_link_velocity_w, adj_3, adj_6, adj_10);
            wp::adj_address(var_data, var_0, var_1, adj_data, adj_0, adj_1, adj_10);
            // adj: body_link_velocity_w[env_id, body_id] = data[i, j]                            <L 41>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_body_link_velocity_w, var_3, var_6, var_9, adj_body_link_velocity_w, adj_3, adj_6, adj_8);
            wp::adj_address(var_data, var_3, var_6, adj_data, adj_3, adj_6, adj_8);
            // adj: body_link_velocity_w[env_id, body_id] = data[env_id, body_id]                 <L 39>
        }
        // adj: if from_mask:                                                                     <L 38>
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_5);
        // adj: body_id = wp.int32(body_ids[j])                                                   <L 37>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 36>
        // adj: i, j = wp.tid()                                                                   <L 35>
        // adj: def set_body_link_velocity_to_sim(                                                <L 1>
        continue;
    }
}



extern "C" __global__ void split_state_to_root_pose_and_vel_c8744b4b_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float32> var_state,
    wp::array_t<wp::transform_t<wp::float32>> var_pose,
    wp::array_t<wp::vec_t<6, wp::float32>> var_vel)
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
        wp::float32* var_2;
        const wp::int32 var_3 = 1;
        wp::float32* var_4;
        const wp::int32 var_5 = 2;
        wp::float32* var_6;
        wp::vec_t<3, wp::float32> var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::int32 var_11 = 3;
        wp::float32* var_12;
        const wp::int32 var_13 = 4;
        wp::float32* var_14;
        const wp::int32 var_15 = 5;
        wp::float32* var_16;
        const wp::int32 var_17 = 6;
        wp::float32* var_18;
        wp::quat_t<wp::float32> var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::transform_t<wp::float32> var_24;
        const wp::int32 var_25 = 7;
        wp::float32* var_26;
        const wp::int32 var_27 = 8;
        wp::float32* var_28;
        const wp::int32 var_29 = 9;
        wp::float32* var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        const wp::int32 var_35 = 10;
        wp::float32* var_36;
        const wp::int32 var_37 = 11;
        wp::float32* var_38;
        const wp::int32 var_39 = 12;
        wp::float32* var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::vec_t<6, wp::float32> var_45;
        //---------
        // forward
        // def split_state_to_root_pose_and_vel(                                                  <L 297>
        // i = wp.tid()                                                                           <L 312>
        var_0 = builtin_tid1d();
        // pose[i] = wp.transform(                                                                <L 314>
        // wp.vec3f(state[i, 0], state[i, 1], state[i, 2]), wp.quatf(state[i, 3], state[i, 4], state[i, 5], state[i, 6])       <L 315>
        var_2 = wp::address(var_state, var_0, var_1);
        var_4 = wp::address(var_state, var_0, var_3);
        var_6 = wp::address(var_state, var_0, var_5);
        var_8 = wp::load(var_2);
        var_9 = wp::load(var_4);
        var_10 = wp::load(var_6);
        var_7 = wp::vec_t<3, wp::float32>(var_8, var_9, var_10);
        var_12 = wp::address(var_state, var_0, var_11);
        var_14 = wp::address(var_state, var_0, var_13);
        var_16 = wp::address(var_state, var_0, var_15);
        var_18 = wp::address(var_state, var_0, var_17);
        var_20 = wp::load(var_12);
        var_21 = wp::load(var_14);
        var_22 = wp::load(var_16);
        var_23 = wp::load(var_18);
        var_19 = wp::quat_t<wp::float32>(var_20, var_21, var_22, var_23);
        var_24 = wp::transform_t<wp::float32>(var_7, var_19);
        // pose[i] = wp.transform(                                                                <L 314>
        wp::array_store(var_pose, var_0, var_24);
        // vel[i] = wp.spatial_vector(                                                            <L 318>
        // wp.vec3f(state[i, 7], state[i, 8], state[i, 9]),  # angular velocity                   <L 319>
        var_26 = wp::address(var_state, var_0, var_25);
        var_28 = wp::address(var_state, var_0, var_27);
        var_30 = wp::address(var_state, var_0, var_29);
        var_32 = wp::load(var_26);
        var_33 = wp::load(var_28);
        var_34 = wp::load(var_30);
        var_31 = wp::vec_t<3, wp::float32>(var_32, var_33, var_34);
        // wp.vec3f(state[i, 10], state[i, 11], state[i, 12]),  # linear velocity                 <L 320>
        var_36 = wp::address(var_state, var_0, var_35);
        var_38 = wp::address(var_state, var_0, var_37);
        var_40 = wp::address(var_state, var_0, var_39);
        var_42 = wp::load(var_36);
        var_43 = wp::load(var_38);
        var_44 = wp::load(var_40);
        var_41 = wp::vec_t<3, wp::float32>(var_42, var_43, var_44);
        var_45 = wp::vec_t<6, wp::float32>(var_31, var_41);
        // vel[i] = wp.spatial_vector(                                                            <L 318>
        wp::array_store(var_vel, var_0, var_45);
    }
}



extern "C" __global__ void split_state_to_root_pose_and_vel_c8744b4b_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float32> var_state,
    wp::array_t<wp::transform_t<wp::float32>> var_pose,
    wp::array_t<wp::vec_t<6, wp::float32>> var_vel,
    wp::array_t<wp::float32> adj_state,
    wp::array_t<wp::transform_t<wp::float32>> adj_pose,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_vel)
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
        wp::float32* var_2;
        const wp::int32 var_3 = 1;
        wp::float32* var_4;
        const wp::int32 var_5 = 2;
        wp::float32* var_6;
        wp::vec_t<3, wp::float32> var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::int32 var_11 = 3;
        wp::float32* var_12;
        const wp::int32 var_13 = 4;
        wp::float32* var_14;
        const wp::int32 var_15 = 5;
        wp::float32* var_16;
        const wp::int32 var_17 = 6;
        wp::float32* var_18;
        wp::quat_t<wp::float32> var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::transform_t<wp::float32> var_24;
        const wp::int32 var_25 = 7;
        wp::float32* var_26;
        const wp::int32 var_27 = 8;
        wp::float32* var_28;
        const wp::int32 var_29 = 9;
        wp::float32* var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        const wp::int32 var_35 = 10;
        wp::float32* var_36;
        const wp::int32 var_37 = 11;
        wp::float32* var_38;
        const wp::int32 var_39 = 12;
        wp::float32* var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::vec_t<6, wp::float32> var_45;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::float32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::float32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::vec_t<3, wp::float32> adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::quat_t<wp::float32> adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::transform_t<wp::float32> adj_24 = {};
        wp::int32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::vec_t<3, wp::float32> adj_31 = {};
        wp::float32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::int32 adj_39 = {};
        wp::float32 adj_40 = {};
        wp::vec_t<3, wp::float32> adj_41 = {};
        wp::float32 adj_42 = {};
        wp::float32 adj_43 = {};
        wp::float32 adj_44 = {};
        wp::vec_t<6, wp::float32> adj_45 = {};
        //---------
        // forward
        // def split_state_to_root_pose_and_vel(                                                  <L 297>
        // i = wp.tid()                                                                           <L 312>
        var_0 = builtin_tid1d();
        // pose[i] = wp.transform(                                                                <L 314>
        // wp.vec3f(state[i, 0], state[i, 1], state[i, 2]), wp.quatf(state[i, 3], state[i, 4], state[i, 5], state[i, 6])       <L 315>
        var_2 = wp::address(var_state, var_0, var_1);
        var_4 = wp::address(var_state, var_0, var_3);
        var_6 = wp::address(var_state, var_0, var_5);
        var_8 = wp::load(var_2);
        var_9 = wp::load(var_4);
        var_10 = wp::load(var_6);
        var_7 = wp::vec_t<3, wp::float32>(var_8, var_9, var_10);
        var_12 = wp::address(var_state, var_0, var_11);
        var_14 = wp::address(var_state, var_0, var_13);
        var_16 = wp::address(var_state, var_0, var_15);
        var_18 = wp::address(var_state, var_0, var_17);
        var_20 = wp::load(var_12);
        var_21 = wp::load(var_14);
        var_22 = wp::load(var_16);
        var_23 = wp::load(var_18);
        var_19 = wp::quat_t<wp::float32>(var_20, var_21, var_22, var_23);
        var_24 = wp::transform_t<wp::float32>(var_7, var_19);
        // pose[i] = wp.transform(                                                                <L 314>
        // wp::array_store(var_pose, var_0, var_24);
        // vel[i] = wp.spatial_vector(                                                            <L 318>
        // wp.vec3f(state[i, 7], state[i, 8], state[i, 9]),  # angular velocity                   <L 319>
        var_26 = wp::address(var_state, var_0, var_25);
        var_28 = wp::address(var_state, var_0, var_27);
        var_30 = wp::address(var_state, var_0, var_29);
        var_32 = wp::load(var_26);
        var_33 = wp::load(var_28);
        var_34 = wp::load(var_30);
        var_31 = wp::vec_t<3, wp::float32>(var_32, var_33, var_34);
        // wp.vec3f(state[i, 10], state[i, 11], state[i, 12]),  # linear velocity                 <L 320>
        var_36 = wp::address(var_state, var_0, var_35);
        var_38 = wp::address(var_state, var_0, var_37);
        var_40 = wp::address(var_state, var_0, var_39);
        var_42 = wp::load(var_36);
        var_43 = wp::load(var_38);
        var_44 = wp::load(var_40);
        var_41 = wp::vec_t<3, wp::float32>(var_42, var_43, var_44);
        var_45 = wp::vec_t<6, wp::float32>(var_31, var_41);
        // vel[i] = wp.spatial_vector(                                                            <L 318>
        // wp::array_store(var_vel, var_0, var_45);
        //---------
        // reverse
        wp::adj_array_store(var_vel, var_0, var_45, adj_vel, adj_0, adj_45);
        // adj: vel[i] = wp.spatial_vector(                                                       <L 318>
        wp::adj_vec_t(var_31, var_41, adj_31, adj_41, adj_45);
        wp::adj_vec_t(var_42, var_43, var_44, adj_36, adj_38, adj_40, adj_41);
        wp::adj_address(var_state, var_0, var_39, adj_state, adj_0, adj_39, adj_40);
        wp::adj_address(var_state, var_0, var_37, adj_state, adj_0, adj_37, adj_38);
        wp::adj_address(var_state, var_0, var_35, adj_state, adj_0, adj_35, adj_36);
        // adj: wp.vec3f(state[i, 10], state[i, 11], state[i, 12]),  # linear velocity            <L 320>
        wp::adj_vec_t(var_32, var_33, var_34, adj_26, adj_28, adj_30, adj_31);
        wp::adj_address(var_state, var_0, var_29, adj_state, adj_0, adj_29, adj_30);
        wp::adj_address(var_state, var_0, var_27, adj_state, adj_0, adj_27, adj_28);
        wp::adj_address(var_state, var_0, var_25, adj_state, adj_0, adj_25, adj_26);
        // adj: wp.vec3f(state[i, 7], state[i, 8], state[i, 9]),  # angular velocity              <L 319>
        // adj: vel[i] = wp.spatial_vector(                                                       <L 318>
        wp::adj_array_store(var_pose, var_0, var_24, adj_pose, adj_0, adj_24);
        // adj: pose[i] = wp.transform(                                                           <L 314>
        wp::adj_transform_t(var_7, var_19, adj_7, adj_19, adj_24);
        wp::adj_quat_t(var_20, var_21, var_22, var_23, adj_12, adj_14, adj_16, adj_18, adj_19);
        wp::adj_address(var_state, var_0, var_17, adj_state, adj_0, adj_17, adj_18);
        wp::adj_address(var_state, var_0, var_15, adj_state, adj_0, adj_15, adj_16);
        wp::adj_address(var_state, var_0, var_13, adj_state, adj_0, adj_13, adj_14);
        wp::adj_address(var_state, var_0, var_11, adj_state, adj_0, adj_11, adj_12);
        wp::adj_vec_t(var_8, var_9, var_10, adj_2, adj_4, adj_6, adj_7);
        wp::adj_address(var_state, var_0, var_5, adj_state, adj_0, adj_5, adj_6);
        wp::adj_address(var_state, var_0, var_3, adj_state, adj_0, adj_3, adj_4);
        wp::adj_address(var_state, var_0, var_1, adj_state, adj_0, adj_1, adj_2);
        // adj: wp.vec3f(state[i, 0], state[i, 1], state[i, 2]), wp.quatf(state[i, 3], state[i, 4], state[i, 5], state[i, 6])  <L 315>
        // adj: pose[i] = wp.transform(                                                           <L 314>
        // adj: i = wp.tid()                                                                      <L 312>
        // adj: def split_state_to_root_pose_and_vel(                                             <L 297>
        continue;
    }
}



extern "C" __global__ void write_body_com_position_to_buffer_index_c88c5a3f_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_data)
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
        wp::vec_t<3, wp::float32>* var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::vec_t<3, wp::float32> var_9;
        //---------
        // forward
        // def write_body_com_position_to_buffer_index(                                           <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // out_data[wp.int32(env_ids[i]), wp.int32(body_ids[j])] = in_data[i, j]                  <L 19>
        var_2 = wp::address(var_in_data, var_0, var_1);
        var_3 = wp::address(var_env_ids, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::int32(var_5);
        var_6 = wp::address(var_body_ids, var_1);
        var_8 = wp::load(var_6);
        var_7 = wp::int32(var_8);
        var_9 = wp::load(var_2);
        wp::array_store(var_out_data, var_4, var_7, var_9);
    }
}



extern "C" __global__ void write_body_com_position_to_buffer_index_c88c5a3f_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_data,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_in_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_body_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_out_data)
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
        wp::vec_t<3, wp::float32>* var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::vec_t<3, wp::float32> var_9;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::vec_t<3, wp::float32> adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::vec_t<3, wp::float32> adj_9 = {};
        //---------
        // forward
        // def write_body_com_position_to_buffer_index(                                           <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // out_data[wp.int32(env_ids[i]), wp.int32(body_ids[j])] = in_data[i, j]                  <L 19>
        var_2 = wp::address(var_in_data, var_0, var_1);
        var_3 = wp::address(var_env_ids, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::int32(var_5);
        var_6 = wp::address(var_body_ids, var_1);
        var_8 = wp::load(var_6);
        var_7 = wp::int32(var_8);
        var_9 = wp::load(var_2);
        // wp::array_store(var_out_data, var_4, var_7, var_9);
        //---------
        // reverse
        wp::adj_array_store(var_out_data, var_4, var_7, var_9, adj_out_data, adj_4, adj_7, adj_2);
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_6);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_3);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_2);
        // adj: out_data[wp.int32(env_ids[i]), wp.int32(body_ids[j])] = in_data[i, j]             <L 19>
        // adj: i, j = wp.tid()                                                                   <L 18>
        // adj: def write_body_com_position_to_buffer_index(                                      <L 1>
        continue;
    }
}



extern "C" __global__ void set_body_com_pose_to_sim_cd0040aa_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::transform_t<wp::float32>* var_8;
        wp::transform_t<wp::float32> var_9;
        wp::transform_t<wp::float32>* var_10;
        wp::transform_t<wp::float32> var_11;
        wp::transform_t<wp::float32>* var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::transform_t<wp::float32> var_14;
        wp::transform_t<wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        //---------
        // forward
        // def set_body_com_pose_to_sim(                                                          <L 1>
        // i, j = wp.tid()                                                                        <L 29>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 30>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 31>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 32>
        if (var_from_mask) {
            // body_com_pose_w[env_id, body_id] = data[env_id, body_id]                           <L 33>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_body_com_pose_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_com_pose_w[env_id, body_id] = data[i, j]                                      <L 35>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            wp::array_store(var_body_com_pose_w, var_3, var_6, var_11);
        }
        // body_link_pose_w[env_id, body_id] = get_com_pose_in_link_frame_func(                   <L 37>
        // body_com_pose_w[env_id, body_id], body_com_pos_b[env_id, body_id]                      <L 38>
        var_12 = wp::address(var_body_com_pose_w, var_3, var_6);
        var_13 = wp::address(var_body_com_pos_b, var_3, var_6);
        var_15 = wp::load(var_12);
        var_16 = wp::load(var_13);
        var_14 = get_com_pose_in_link_frame_func_0(var_15, var_16);
        // body_link_pose_w[env_id, body_id] = get_com_pose_in_link_frame_func(                   <L 37>
        wp::array_store(var_body_link_pose_w, var_3, var_6, var_14);
    }
}



extern "C" __global__ void set_body_com_pose_to_sim_cd0040aa_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_data,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_body_com_pos_b,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int64> adj_body_ids,
    bool adj_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_link_pose_w)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::transform_t<wp::float32>* var_8;
        wp::transform_t<wp::float32> var_9;
        wp::transform_t<wp::float32>* var_10;
        wp::transform_t<wp::float32> var_11;
        wp::transform_t<wp::float32>* var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::transform_t<wp::float32> var_14;
        wp::transform_t<wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int64 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int64 adj_4 = {};
        wp::int64 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int64 adj_7 = {};
        wp::transform_t<wp::float32> adj_8 = {};
        wp::transform_t<wp::float32> adj_9 = {};
        wp::transform_t<wp::float32> adj_10 = {};
        wp::transform_t<wp::float32> adj_11 = {};
        wp::transform_t<wp::float32> adj_12 = {};
        wp::vec_t<3, wp::float32> adj_13 = {};
        wp::transform_t<wp::float32> adj_14 = {};
        wp::transform_t<wp::float32> adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        //---------
        // forward
        // def set_body_com_pose_to_sim(                                                          <L 1>
        // i, j = wp.tid()                                                                        <L 29>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 30>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 31>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 32>
        if (var_from_mask) {
            // body_com_pose_w[env_id, body_id] = data[env_id, body_id]                           <L 33>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            // wp::array_store(var_body_com_pose_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_com_pose_w[env_id, body_id] = data[i, j]                                      <L 35>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            // wp::array_store(var_body_com_pose_w, var_3, var_6, var_11);
        }
        // body_link_pose_w[env_id, body_id] = get_com_pose_in_link_frame_func(                   <L 37>
        // body_com_pose_w[env_id, body_id], body_com_pos_b[env_id, body_id]                      <L 38>
        var_12 = wp::address(var_body_com_pose_w, var_3, var_6);
        var_13 = wp::address(var_body_com_pos_b, var_3, var_6);
        var_15 = wp::load(var_12);
        var_16 = wp::load(var_13);
        var_14 = get_com_pose_in_link_frame_func_0(var_15, var_16);
        // body_link_pose_w[env_id, body_id] = get_com_pose_in_link_frame_func(                   <L 37>
        // wp::array_store(var_body_link_pose_w, var_3, var_6, var_14);
        //---------
        // reverse
        wp::adj_array_store(var_body_link_pose_w, var_3, var_6, var_14, adj_body_link_pose_w, adj_3, adj_6, adj_14);
        // adj: body_link_pose_w[env_id, body_id] = get_com_pose_in_link_frame_func(              <L 37>
        adj_get_com_pose_in_link_frame_func_0(var_15, var_16, adj_12, adj_13, adj_14);
        wp::adj_address(var_body_com_pos_b, var_3, var_6, adj_body_com_pos_b, adj_3, adj_6, adj_13);
        wp::adj_address(var_body_com_pose_w, var_3, var_6, adj_body_com_pose_w, adj_3, adj_6, adj_12);
        // adj: body_com_pose_w[env_id, body_id], body_com_pos_b[env_id, body_id]                 <L 38>
        // adj: body_link_pose_w[env_id, body_id] = get_com_pose_in_link_frame_func(              <L 37>
        if (!var_from_mask) {
            wp::adj_array_store(var_body_com_pose_w, var_3, var_6, var_11, adj_body_com_pose_w, adj_3, adj_6, adj_10);
            wp::adj_address(var_data, var_0, var_1, adj_data, adj_0, adj_1, adj_10);
            // adj: body_com_pose_w[env_id, body_id] = data[i, j]                                 <L 35>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_body_com_pose_w, var_3, var_6, var_9, adj_body_com_pose_w, adj_3, adj_6, adj_8);
            wp::adj_address(var_data, var_3, var_6, adj_data, adj_3, adj_6, adj_8);
            // adj: body_com_pose_w[env_id, body_id] = data[env_id, body_id]                      <L 33>
        }
        // adj: if from_mask:                                                                     <L 32>
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_5);
        // adj: body_id = wp.int32(body_ids[j])                                                   <L 31>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 30>
        // adj: i, j = wp.tid()                                                                   <L 29>
        // adj: def set_body_com_pose_to_sim(                                                     <L 1>
        continue;
    }
}



extern "C" __global__ void make_dummy_body_com_pose_b_cd11cdda_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_b)
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
        wp::vec_t<3, wp::float32>* var_2;
        const wp::float32 var_3 = 0.0;
        const wp::float32 var_4 = 0.0;
        const wp::float32 var_5 = 0.0;
        const wp::float32 var_6 = 1.0;
        wp::quat_t<wp::float32> var_7;
        wp::transform_t<wp::float32> var_8;
        wp::vec_t<3, wp::float32> var_9;
        //---------
        // forward
        // def make_dummy_body_com_pose_b(                                                        <L 1351>
        // i, j = wp.tid()                                                                        <L 1364>
        builtin_tid2d(var_0, var_1);
        // body_com_pose_b[i, j] = wp.transformf(body_com_pos_b[i, j], wp.quatf(0.0, 0.0, 0.0, 1.0))       <L 1365>
        var_2 = wp::address(var_body_com_pos_b, var_0, var_1);
        var_7 = wp::quat_t<wp::float32>(var_3, var_4, var_5, var_6);
        var_9 = wp::load(var_2);
        var_8 = wp::transform_t<wp::float32>(var_9, var_7);
        wp::array_store(var_body_com_pose_b, var_0, var_1, var_8);
    }
}



extern "C" __global__ void make_dummy_body_com_pose_b_cd11cdda_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_b,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_body_com_pos_b,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_com_pose_b)
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
        wp::vec_t<3, wp::float32>* var_2;
        const wp::float32 var_3 = 0.0;
        const wp::float32 var_4 = 0.0;
        const wp::float32 var_5 = 0.0;
        const wp::float32 var_6 = 1.0;
        wp::quat_t<wp::float32> var_7;
        wp::transform_t<wp::float32> var_8;
        wp::vec_t<3, wp::float32> var_9;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::vec_t<3, wp::float32> adj_2 = {};
        wp::float32 adj_3 = {};
        wp::float32 adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::quat_t<wp::float32> adj_7 = {};
        wp::transform_t<wp::float32> adj_8 = {};
        wp::vec_t<3, wp::float32> adj_9 = {};
        //---------
        // forward
        // def make_dummy_body_com_pose_b(                                                        <L 1351>
        // i, j = wp.tid()                                                                        <L 1364>
        builtin_tid2d(var_0, var_1);
        // body_com_pose_b[i, j] = wp.transformf(body_com_pos_b[i, j], wp.quatf(0.0, 0.0, 0.0, 1.0))       <L 1365>
        var_2 = wp::address(var_body_com_pos_b, var_0, var_1);
        var_7 = wp::quat_t<wp::float32>(var_3, var_4, var_5, var_6);
        var_9 = wp::load(var_2);
        var_8 = wp::transform_t<wp::float32>(var_9, var_7);
        // wp::array_store(var_body_com_pose_b, var_0, var_1, var_8);
        //---------
        // reverse
        wp::adj_array_store(var_body_com_pose_b, var_0, var_1, var_8, adj_body_com_pose_b, adj_0, adj_1, adj_8);
        wp::adj_transform_t(var_9, var_7, adj_2, adj_7, adj_8);
        wp::adj_quat_t(var_3, var_4, var_5, var_6, adj_3, adj_4, adj_5, adj_6, adj_7);
        wp::adj_address(var_body_com_pos_b, var_0, var_1, adj_body_com_pos_b, adj_0, adj_1, adj_2);
        // adj: body_com_pose_b[i, j] = wp.transformf(body_com_pos_b[i, j], wp.quatf(0.0, 0.0, 0.0, 1.0))  <L 1365>
        // adj: i, j = wp.tid()                                                                   <L 1364>
        // adj: def make_dummy_body_com_pose_b(                                                   <L 1351>
        continue;
    }
}



extern "C" __global__ void set_body_link_pose_to_sim_d2e142f9_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w)
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
        wp::transform_t<wp::float32>* var_8;
        wp::transform_t<wp::float32> var_9;
        wp::transform_t<wp::float32>* var_10;
        wp::transform_t<wp::float32> var_11;
        //---------
        // forward
        // def set_body_link_pose_to_sim(                                                         <L 1>
        // i, j = wp.tid()                                                                        <L 22>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 23>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 24>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 25>
        if (var_from_mask) {
            // body_link_pose_w[env_id, body_id] = data[env_id, body_id]                          <L 26>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_body_link_pose_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_link_pose_w[env_id, body_id] = data[i, j]                                     <L 28>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            wp::array_store(var_body_link_pose_w, var_3, var_6, var_11);
        }
    }
}



extern "C" __global__ void set_body_link_pose_to_sim_d2e142f9_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_body_ids,
    bool adj_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_link_pose_w)
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
        wp::transform_t<wp::float32>* var_8;
        wp::transform_t<wp::float32> var_9;
        wp::transform_t<wp::float32>* var_10;
        wp::transform_t<wp::float32> var_11;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::transform_t<wp::float32> adj_8 = {};
        wp::transform_t<wp::float32> adj_9 = {};
        wp::transform_t<wp::float32> adj_10 = {};
        wp::transform_t<wp::float32> adj_11 = {};
        //---------
        // forward
        // def set_body_link_pose_to_sim(                                                         <L 1>
        // i, j = wp.tid()                                                                        <L 22>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 23>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 24>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 25>
        if (var_from_mask) {
            // body_link_pose_w[env_id, body_id] = data[env_id, body_id]                          <L 26>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            // wp::array_store(var_body_link_pose_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_link_pose_w[env_id, body_id] = data[i, j]                                     <L 28>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            // wp::array_store(var_body_link_pose_w, var_3, var_6, var_11);
        }
        //---------
        // reverse
        if (!var_from_mask) {
            wp::adj_array_store(var_body_link_pose_w, var_3, var_6, var_11, adj_body_link_pose_w, adj_3, adj_6, adj_10);
            wp::adj_address(var_data, var_0, var_1, adj_data, adj_0, adj_1, adj_10);
            // adj: body_link_pose_w[env_id, body_id] = data[i, j]                                <L 28>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_body_link_pose_w, var_3, var_6, var_9, adj_body_link_pose_w, adj_3, adj_6, adj_8);
            wp::adj_address(var_data, var_3, var_6, adj_data, adj_3, adj_6, adj_8);
            // adj: body_link_pose_w[env_id, body_id] = data[env_id, body_id]                     <L 26>
        }
        // adj: if from_mask:                                                                     <L 25>
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_5);
        // adj: body_id = wp.int32(body_ids[j])                                                   <L 24>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 23>
        // adj: i, j = wp.tid()                                                                   <L 22>
        // adj: def set_body_link_pose_to_sim(                                                    <L 1>
        continue;
    }
}



extern "C" __global__ void derive_body_acceleration_from_body_com_velocities_d7161d69_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_vel,
    wp::float32 var_dt,
    wp::array_t<wp::vec_t<6, wp::float32>> var_prev_body_com_vel,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc)
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
        wp::vec_t<6, wp::float32>* var_2;
        wp::vec_t<6, wp::float32>* var_3;
        wp::vec_t<6, wp::float32> var_4;
        wp::vec_t<6, wp::float32> var_5;
        wp::vec_t<6, wp::float32> var_6;
        wp::vec_t<6, wp::float32> var_7;
        wp::vec_t<6, wp::float32>* var_8;
        wp::vec_t<6, wp::float32> var_9;
        //---------
        // forward
        // def derive_body_acceleration_from_body_com_velocities(                                 <L 1369>
        // i, j = wp.tid()                                                                        <L 1385>
        builtin_tid2d(var_0, var_1);
        // body_acc[i, j] = (body_com_vel[i, j] - prev_body_com_vel[i, j]) / dt                   <L 1387>
        var_2 = wp::address(var_body_com_vel, var_0, var_1);
        var_3 = wp::address(var_prev_body_com_vel, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = wp::sub(var_5, var_6);
        var_7 = wp::div(var_4, var_dt);
        wp::array_store(var_body_acc, var_0, var_1, var_7);
        // prev_body_com_vel[i, j] = body_com_vel[i, j]                                           <L 1389>
        var_8 = wp::address(var_body_com_vel, var_0, var_1);
        var_9 = wp::load(var_8);
        wp::array_store(var_prev_body_com_vel, var_0, var_1, var_9);
    }
}



extern "C" __global__ void derive_body_acceleration_from_body_com_velocities_d7161d69_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_vel,
    wp::float32 var_dt,
    wp::array_t<wp::vec_t<6, wp::float32>> var_prev_body_com_vel,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_com_vel,
    wp::float32 adj_dt,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_prev_body_com_vel,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_acc)
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
        wp::vec_t<6, wp::float32>* var_2;
        wp::vec_t<6, wp::float32>* var_3;
        wp::vec_t<6, wp::float32> var_4;
        wp::vec_t<6, wp::float32> var_5;
        wp::vec_t<6, wp::float32> var_6;
        wp::vec_t<6, wp::float32> var_7;
        wp::vec_t<6, wp::float32>* var_8;
        wp::vec_t<6, wp::float32> var_9;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::vec_t<6, wp::float32> adj_2 = {};
        wp::vec_t<6, wp::float32> adj_3 = {};
        wp::vec_t<6, wp::float32> adj_4 = {};
        wp::vec_t<6, wp::float32> adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        wp::vec_t<6, wp::float32> adj_7 = {};
        wp::vec_t<6, wp::float32> adj_8 = {};
        wp::vec_t<6, wp::float32> adj_9 = {};
        //---------
        // forward
        // def derive_body_acceleration_from_body_com_velocities(                                 <L 1369>
        // i, j = wp.tid()                                                                        <L 1385>
        builtin_tid2d(var_0, var_1);
        // body_acc[i, j] = (body_com_vel[i, j] - prev_body_com_vel[i, j]) / dt                   <L 1387>
        var_2 = wp::address(var_body_com_vel, var_0, var_1);
        var_3 = wp::address(var_prev_body_com_vel, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = wp::sub(var_5, var_6);
        var_7 = wp::div(var_4, var_dt);
        // wp::array_store(var_body_acc, var_0, var_1, var_7);
        // prev_body_com_vel[i, j] = body_com_vel[i, j]                                           <L 1389>
        var_8 = wp::address(var_body_com_vel, var_0, var_1);
        var_9 = wp::load(var_8);
        // wp::array_store(var_prev_body_com_vel, var_0, var_1, var_9);
        //---------
        // reverse
        wp::adj_array_store(var_prev_body_com_vel, var_0, var_1, var_9, adj_prev_body_com_vel, adj_0, adj_1, adj_8);
        wp::adj_address(var_body_com_vel, var_0, var_1, adj_body_com_vel, adj_0, adj_1, adj_8);
        // adj: prev_body_com_vel[i, j] = body_com_vel[i, j]                                      <L 1389>
        wp::adj_array_store(var_body_acc, var_0, var_1, var_7, adj_body_acc, adj_0, adj_1, adj_7);
        wp::adj_div(var_4, var_dt, adj_4, adj_dt, adj_7);
        wp::adj_sub(var_5, var_6, adj_2, adj_3, adj_4);
        wp::adj_address(var_prev_body_com_vel, var_0, var_1, adj_prev_body_com_vel, adj_0, adj_1, adj_3);
        wp::adj_address(var_body_com_vel, var_0, var_1, adj_body_com_vel, adj_0, adj_1, adj_2);
        // adj: body_acc[i, j] = (body_com_vel[i, j] - prev_body_com_vel[i, j]) / dt              <L 1387>
        // adj: i, j = wp.tid()                                                                   <L 1385>
        // adj: def derive_body_acceleration_from_body_com_velocities(                            <L 1369>
        continue;
    }
}



extern "C" __global__ void set_root_link_pose_to_sim_index_da5c542e_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::transform_t<wp::float32>> var_root_link_pose_w)
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
        wp::transform_t<wp::float32>* var_4;
        wp::transform_t<wp::float32> var_5;
        //---------
        // forward
        // def set_root_link_pose_to_sim_index(                                                   <L 1>
        // i = wp.tid()                                                                           <L 15>
        var_0 = builtin_tid1d();
        // env_id = wp.int32(env_ids[i])                                                          <L 16>
        var_1 = wp::address(var_env_ids, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::int32(var_3);
        // root_link_pose_w[env_id] = data[i]                                                     <L 17>
        var_4 = wp::address(var_data, var_0);
        var_5 = wp::load(var_4);
        wp::array_store(var_root_link_pose_w, var_2, var_5);
    }
}



extern "C" __global__ void set_root_link_pose_to_sim_index_da5c542e_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::transform_t<wp::float32>> var_root_link_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::transform_t<wp::float32>> adj_root_link_pose_w)
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
        wp::transform_t<wp::float32>* var_4;
        wp::transform_t<wp::float32> var_5;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::transform_t<wp::float32> adj_4 = {};
        wp::transform_t<wp::float32> adj_5 = {};
        //---------
        // forward
        // def set_root_link_pose_to_sim_index(                                                   <L 1>
        // i = wp.tid()                                                                           <L 15>
        var_0 = builtin_tid1d();
        // env_id = wp.int32(env_ids[i])                                                          <L 16>
        var_1 = wp::address(var_env_ids, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::int32(var_3);
        // root_link_pose_w[env_id] = data[i]                                                     <L 17>
        var_4 = wp::address(var_data, var_0);
        var_5 = wp::load(var_4);
        // wp::array_store(var_root_link_pose_w, var_2, var_5);
        //---------
        // reverse
        wp::adj_array_store(var_root_link_pose_w, var_2, var_5, adj_root_link_pose_w, adj_2, adj_4);
        wp::adj_address(var_data, var_0, adj_data, adj_0, adj_4);
        // adj: root_link_pose_w[env_id] = data[i]                                                <L 17>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_1);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 16>
        // adj: i = wp.tid()                                                                      <L 15>
        // adj: def set_root_link_pose_to_sim_index(                                              <L 1>
        continue;
    }
}



extern "C" __global__ void write_body_mass_and_inverse_index_dbd59ca6_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_masses,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    wp::array_t<wp::int32> var_body_user_to_backend,
    bool var_has_body_ordering,
    wp::array_t<wp::float32> var_body_inertia_backend,
    wp::array_t<wp::float32> var_body_mass_user,
    wp::array_t<wp::float32> var_body_mass_backend,
    wp::array_t<wp::float32> var_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_body_inv_inertia)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        const bool var_16 = true;
        //---------
        // forward
        // def write_body_mass_and_inverse_index(                                                 <L 1>
        // i, j = wp.tid()                                                                        <L 27>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 28>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_body_id = wp.int32(body_ids[j])                                                   <L 29>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // backend_body_id = user_body_id                                                         <L 30>
        var_8 = wp::copy(var_6);
        // if has_body_ordering:                                                                  <L 31>
        if (var_has_body_ordering) {
            // backend_body_id = body_user_to_backend[user_body_id]                               <L 32>
            var_9 = wp::address(var_body_user_to_backend, var_6);
            var_11 = wp::load(var_9);
            var_10 = wp::copy(var_11);
        }
        var_12 = wp::where(var_has_body_ordering, var_10, var_8);
        // mass = masses[i, j]                                                                    <L 33>
        var_13 = wp::address(var_masses, var_0, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // body_mass_user[env_id, user_body_id] = mass                                            <L 34>
        wp::array_store(var_body_mass_user, var_3, var_6, var_14);
        // if has_body_ordering:                                                                  <L 35>
        if (var_has_body_ordering) {
            // body_mass_backend[env_id, backend_body_id] = mass                                  <L 36>
            wp::array_store(var_body_mass_backend, var_3, var_12, var_14);
        }
        // _update_body_inertial_properties_inverse(                                              <L 37>
        // env_id, backend_body_id, body_mass_backend, body_inertia_backend, True, body_inv_mass, body_inv_inertia       <L 38>
        _update_body_inertial_properties_inverse_0(var_3, var_12, var_body_mass_backend, var_body_inertia_backend, var_16, var_body_inv_mass, var_body_inv_inertia);
    }
}



extern "C" __global__ void write_body_mass_and_inverse_index_dbd59ca6_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_masses,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    wp::array_t<wp::int32> var_body_user_to_backend,
    bool var_has_body_ordering,
    wp::array_t<wp::float32> var_body_inertia_backend,
    wp::array_t<wp::float32> var_body_mass_user,
    wp::array_t<wp::float32> var_body_mass_backend,
    wp::array_t<wp::float32> var_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_body_inv_inertia,
    wp::array_t<wp::float32> adj_masses,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int32> adj_body_ids,
    wp::array_t<wp::int32> adj_body_user_to_backend,
    bool adj_has_body_ordering,
    wp::array_t<wp::float32> adj_body_inertia_backend,
    wp::array_t<wp::float32> adj_body_mass_user,
    wp::array_t<wp::float32> adj_body_mass_backend,
    wp::array_t<wp::float32> adj_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> adj_body_inv_inertia)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        const bool var_16 = true;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int64 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int64 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        bool adj_16 = {};
        //---------
        // forward
        // def write_body_mass_and_inverse_index(                                                 <L 1>
        // i, j = wp.tid()                                                                        <L 27>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 28>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_body_id = wp.int32(body_ids[j])                                                   <L 29>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // backend_body_id = user_body_id                                                         <L 30>
        var_8 = wp::copy(var_6);
        // if has_body_ordering:                                                                  <L 31>
        if (var_has_body_ordering) {
            // backend_body_id = body_user_to_backend[user_body_id]                               <L 32>
            var_9 = wp::address(var_body_user_to_backend, var_6);
            var_11 = wp::load(var_9);
            var_10 = wp::copy(var_11);
        }
        var_12 = wp::where(var_has_body_ordering, var_10, var_8);
        // mass = masses[i, j]                                                                    <L 33>
        var_13 = wp::address(var_masses, var_0, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // body_mass_user[env_id, user_body_id] = mass                                            <L 34>
        // wp::array_store(var_body_mass_user, var_3, var_6, var_14);
        // if has_body_ordering:                                                                  <L 35>
        if (var_has_body_ordering) {
            // body_mass_backend[env_id, backend_body_id] = mass                                  <L 36>
            // wp::array_store(var_body_mass_backend, var_3, var_12, var_14);
        }
        // _update_body_inertial_properties_inverse(                                              <L 37>
        // env_id, backend_body_id, body_mass_backend, body_inertia_backend, True, body_inv_mass, body_inv_inertia       <L 38>
        _update_body_inertial_properties_inverse_0(var_3, var_12, var_body_mass_backend, var_body_inertia_backend, var_16, var_body_inv_mass, var_body_inv_inertia);
        //---------
        // reverse
        adj__update_body_inertial_properties_inverse_0(var_3, var_12, var_body_mass_backend, var_body_inertia_backend, var_16, var_body_inv_mass, var_body_inv_inertia, adj_3, adj_12, adj_body_mass_backend, adj_body_inertia_backend, adj_16, adj_body_inv_mass, adj_body_inv_inertia);
        // adj: env_id, backend_body_id, body_mass_backend, body_inertia_backend, True, body_inv_mass, body_inv_inertia  <L 38>
        // adj: _update_body_inertial_properties_inverse(                                         <L 37>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_mass_backend, var_3, var_12, var_14, adj_body_mass_backend, adj_3, adj_12, adj_14);
            // adj: body_mass_backend[env_id, backend_body_id] = mass                             <L 36>
        }
        // adj: if has_body_ordering:                                                             <L 35>
        wp::adj_array_store(var_body_mass_user, var_3, var_6, var_14, adj_body_mass_user, adj_3, adj_6, adj_14);
        // adj: body_mass_user[env_id, user_body_id] = mass                                       <L 34>
        wp::adj_copy(var_15, adj_13, adj_14);
        wp::adj_address(var_masses, var_0, var_1, adj_masses, adj_0, adj_1, adj_13);
        // adj: mass = masses[i, j]                                                               <L 33>
        wp::adj_where(var_has_body_ordering, var_10, var_8, adj_has_body_ordering, adj_10, adj_8, adj_12);
        if (var_has_body_ordering) {
            wp::adj_copy(var_11, adj_9, adj_10);
            wp::adj_address(var_body_user_to_backend, var_6, adj_body_user_to_backend, adj_6, adj_9);
            // adj: backend_body_id = body_user_to_backend[user_body_id]                          <L 32>
        }
        // adj: if has_body_ordering:                                                             <L 31>
        wp::adj_copy(var_6, adj_6, adj_8);
        // adj: backend_body_id = user_body_id                                                    <L 30>
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_5);
        // adj: user_body_id = wp.int32(body_ids[j])                                              <L 29>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 28>
        // adj: i, j = wp.tid()                                                                   <L 27>
        // adj: def write_body_mass_and_inverse_index(                                            <L 1>
        continue;
    }
}



extern "C" __global__ void write_body_mass_and_inverse_index_dd516e68_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_masses,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    wp::array_t<wp::int32> var_body_user_to_backend,
    bool var_has_body_ordering,
    wp::array_t<wp::float32> var_body_inertia_backend,
    wp::array_t<wp::float32> var_body_mass_user,
    wp::array_t<wp::float32> var_body_mass_backend,
    wp::array_t<wp::float32> var_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_body_inv_inertia)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        const bool var_16 = true;
        //---------
        // forward
        // def write_body_mass_and_inverse_index(                                                 <L 1>
        // i, j = wp.tid()                                                                        <L 27>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 28>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_body_id = wp.int32(body_ids[j])                                                   <L 29>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // backend_body_id = user_body_id                                                         <L 30>
        var_8 = wp::copy(var_6);
        // if has_body_ordering:                                                                  <L 31>
        if (var_has_body_ordering) {
            // backend_body_id = body_user_to_backend[user_body_id]                               <L 32>
            var_9 = wp::address(var_body_user_to_backend, var_6);
            var_11 = wp::load(var_9);
            var_10 = wp::copy(var_11);
        }
        var_12 = wp::where(var_has_body_ordering, var_10, var_8);
        // mass = masses[i, j]                                                                    <L 33>
        var_13 = wp::address(var_masses, var_0, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // body_mass_user[env_id, user_body_id] = mass                                            <L 34>
        wp::array_store(var_body_mass_user, var_3, var_6, var_14);
        // if has_body_ordering:                                                                  <L 35>
        if (var_has_body_ordering) {
            // body_mass_backend[env_id, backend_body_id] = mass                                  <L 36>
            wp::array_store(var_body_mass_backend, var_3, var_12, var_14);
        }
        // _update_body_inertial_properties_inverse(                                              <L 37>
        // env_id, backend_body_id, body_mass_backend, body_inertia_backend, True, body_inv_mass, body_inv_inertia       <L 38>
        _update_body_inertial_properties_inverse_0(var_3, var_12, var_body_mass_backend, var_body_inertia_backend, var_16, var_body_inv_mass, var_body_inv_inertia);
    }
}



extern "C" __global__ void write_body_mass_and_inverse_index_dd516e68_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_masses,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    wp::array_t<wp::int32> var_body_user_to_backend,
    bool var_has_body_ordering,
    wp::array_t<wp::float32> var_body_inertia_backend,
    wp::array_t<wp::float32> var_body_mass_user,
    wp::array_t<wp::float32> var_body_mass_backend,
    wp::array_t<wp::float32> var_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_body_inv_inertia,
    wp::array_t<wp::float32> adj_masses,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int64> adj_body_ids,
    wp::array_t<wp::int32> adj_body_user_to_backend,
    bool adj_has_body_ordering,
    wp::array_t<wp::float32> adj_body_inertia_backend,
    wp::array_t<wp::float32> adj_body_mass_user,
    wp::array_t<wp::float32> adj_body_mass_backend,
    wp::array_t<wp::float32> adj_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> adj_body_inv_inertia)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int64* var_5;
        wp::int32 var_6;
        wp::int64 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        const bool var_16 = true;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int64 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int64 adj_4 = {};
        wp::int64 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int64 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        bool adj_16 = {};
        //---------
        // forward
        // def write_body_mass_and_inverse_index(                                                 <L 1>
        // i, j = wp.tid()                                                                        <L 27>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 28>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_body_id = wp.int32(body_ids[j])                                                   <L 29>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // backend_body_id = user_body_id                                                         <L 30>
        var_8 = wp::copy(var_6);
        // if has_body_ordering:                                                                  <L 31>
        if (var_has_body_ordering) {
            // backend_body_id = body_user_to_backend[user_body_id]                               <L 32>
            var_9 = wp::address(var_body_user_to_backend, var_6);
            var_11 = wp::load(var_9);
            var_10 = wp::copy(var_11);
        }
        var_12 = wp::where(var_has_body_ordering, var_10, var_8);
        // mass = masses[i, j]                                                                    <L 33>
        var_13 = wp::address(var_masses, var_0, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // body_mass_user[env_id, user_body_id] = mass                                            <L 34>
        // wp::array_store(var_body_mass_user, var_3, var_6, var_14);
        // if has_body_ordering:                                                                  <L 35>
        if (var_has_body_ordering) {
            // body_mass_backend[env_id, backend_body_id] = mass                                  <L 36>
            // wp::array_store(var_body_mass_backend, var_3, var_12, var_14);
        }
        // _update_body_inertial_properties_inverse(                                              <L 37>
        // env_id, backend_body_id, body_mass_backend, body_inertia_backend, True, body_inv_mass, body_inv_inertia       <L 38>
        _update_body_inertial_properties_inverse_0(var_3, var_12, var_body_mass_backend, var_body_inertia_backend, var_16, var_body_inv_mass, var_body_inv_inertia);
        //---------
        // reverse
        adj__update_body_inertial_properties_inverse_0(var_3, var_12, var_body_mass_backend, var_body_inertia_backend, var_16, var_body_inv_mass, var_body_inv_inertia, adj_3, adj_12, adj_body_mass_backend, adj_body_inertia_backend, adj_16, adj_body_inv_mass, adj_body_inv_inertia);
        // adj: env_id, backend_body_id, body_mass_backend, body_inertia_backend, True, body_inv_mass, body_inv_inertia  <L 38>
        // adj: _update_body_inertial_properties_inverse(                                         <L 37>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_mass_backend, var_3, var_12, var_14, adj_body_mass_backend, adj_3, adj_12, adj_14);
            // adj: body_mass_backend[env_id, backend_body_id] = mass                             <L 36>
        }
        // adj: if has_body_ordering:                                                             <L 35>
        wp::adj_array_store(var_body_mass_user, var_3, var_6, var_14, adj_body_mass_user, adj_3, adj_6, adj_14);
        // adj: body_mass_user[env_id, user_body_id] = mass                                       <L 34>
        wp::adj_copy(var_15, adj_13, adj_14);
        wp::adj_address(var_masses, var_0, var_1, adj_masses, adj_0, adj_1, adj_13);
        // adj: mass = masses[i, j]                                                               <L 33>
        wp::adj_where(var_has_body_ordering, var_10, var_8, adj_has_body_ordering, adj_10, adj_8, adj_12);
        if (var_has_body_ordering) {
            wp::adj_copy(var_11, adj_9, adj_10);
            wp::adj_address(var_body_user_to_backend, var_6, adj_body_user_to_backend, adj_6, adj_9);
            // adj: backend_body_id = body_user_to_backend[user_body_id]                          <L 32>
        }
        // adj: if has_body_ordering:                                                             <L 31>
        wp::adj_copy(var_6, adj_6, adj_8);
        // adj: backend_body_id = user_body_id                                                    <L 30>
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_5);
        // adj: user_body_id = wp.int32(body_ids[j])                                              <L 29>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 28>
        // adj: i, j = wp.tid()                                                                   <L 27>
        // adj: def write_body_mass_and_inverse_index(                                            <L 1>
        continue;
    }
}



extern "C" __global__ void write_body_com_position_to_buffer_index_e29176cf_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_data)
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
        wp::vec_t<3, wp::float32>* var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int64* var_6;
        wp::int32 var_7;
        wp::int64 var_8;
        wp::vec_t<3, wp::float32> var_9;
        //---------
        // forward
        // def write_body_com_position_to_buffer_index(                                           <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // out_data[wp.int32(env_ids[i]), wp.int32(body_ids[j])] = in_data[i, j]                  <L 19>
        var_2 = wp::address(var_in_data, var_0, var_1);
        var_3 = wp::address(var_env_ids, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::int32(var_5);
        var_6 = wp::address(var_body_ids, var_1);
        var_8 = wp::load(var_6);
        var_7 = wp::int32(var_8);
        var_9 = wp::load(var_2);
        wp::array_store(var_out_data, var_4, var_7, var_9);
    }
}



extern "C" __global__ void write_body_com_position_to_buffer_index_e29176cf_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int64> var_body_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_data,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_in_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int64> adj_body_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_out_data)
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
        wp::vec_t<3, wp::float32>* var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int64* var_6;
        wp::int32 var_7;
        wp::int64 var_8;
        wp::vec_t<3, wp::float32> var_9;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::vec_t<3, wp::float32> adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int64 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int64 adj_8 = {};
        wp::vec_t<3, wp::float32> adj_9 = {};
        //---------
        // forward
        // def write_body_com_position_to_buffer_index(                                           <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // out_data[wp.int32(env_ids[i]), wp.int32(body_ids[j])] = in_data[i, j]                  <L 19>
        var_2 = wp::address(var_in_data, var_0, var_1);
        var_3 = wp::address(var_env_ids, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::int32(var_5);
        var_6 = wp::address(var_body_ids, var_1);
        var_8 = wp::load(var_6);
        var_7 = wp::int32(var_8);
        var_9 = wp::load(var_2);
        // wp::array_store(var_out_data, var_4, var_7, var_9);
        //---------
        // reverse
        wp::adj_array_store(var_out_data, var_4, var_7, var_9, adj_out_data, adj_4, adj_7, adj_2);
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_6);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_3);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_2);
        // adj: out_data[wp.int32(env_ids[i]), wp.int32(body_ids[j])] = in_data[i, j]             <L 19>
        // adj: i, j = wp.tid()                                                                   <L 18>
        // adj: def write_body_com_position_to_buffer_index(                                      <L 1>
        continue;
    }
}



extern "C" __global__ void write_body_inertia_and_inverse_index_e4ddbaa1_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_inertias,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    wp::array_t<wp::int32> var_body_user_to_backend,
    bool var_has_body_ordering,
    wp::array_t<wp::float32> var_body_mass_backend,
    wp::array_t<wp::float32> var_body_inertia_user,
    wp::array_t<wp::float32> var_body_inertia_backend,
    wp::array_t<wp::float32> var_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_body_inv_inertia)
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
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 0;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        const wp::int32 var_17 = 1;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        const wp::int32 var_21 = 2;
        wp::float32* var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        const wp::int32 var_25 = 3;
        wp::float32* var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        const wp::int32 var_29 = 4;
        wp::float32* var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        const wp::int32 var_33 = 5;
        wp::float32* var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        const wp::int32 var_37 = 6;
        wp::float32* var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        const wp::int32 var_41 = 7;
        wp::float32* var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        const wp::int32 var_45 = 8;
        wp::float32* var_46;
        wp::float32 var_47;
        wp::float32 var_48;
        const bool var_49 = false;
        //---------
        // forward
        // def write_body_inertia_and_inverse_index(                                              <L 1>
        // i, j = wp.tid()                                                                        <L 14>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 15>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_body_id = wp.int32(body_ids[j])                                                   <L 16>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // backend_body_id = user_body_id                                                         <L 17>
        var_8 = wp::copy(var_6);
        // if has_body_ordering:                                                                  <L 18>
        if (var_has_body_ordering) {
            // backend_body_id = body_user_to_backend[user_body_id]                               <L 19>
            var_9 = wp::address(var_body_user_to_backend, var_6);
            var_11 = wp::load(var_9);
            var_10 = wp::copy(var_11);
        }
        var_12 = wp::where(var_has_body_ordering, var_10, var_8);
        // for k in range(9):                                                                     <L 20>
        // value = inertias[i, j, k]                                                              <L 21>
        var_14 = wp::address(var_inertias, var_0, var_1, var_13);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_13, var_15);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_13, var_15);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_18 = wp::address(var_inertias, var_0, var_1, var_17);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_17, var_19);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_17, var_19);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_22 = wp::address(var_inertias, var_0, var_1, var_21);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_21, var_23);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_21, var_23);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_26 = wp::address(var_inertias, var_0, var_1, var_25);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_25, var_27);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_25, var_27);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_30 = wp::address(var_inertias, var_0, var_1, var_29);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_29, var_31);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_29, var_31);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_34 = wp::address(var_inertias, var_0, var_1, var_33);
        var_36 = wp::load(var_34);
        var_35 = wp::copy(var_36);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_33, var_35);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_33, var_35);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_38 = wp::address(var_inertias, var_0, var_1, var_37);
        var_40 = wp::load(var_38);
        var_39 = wp::copy(var_40);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_37, var_39);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_37, var_39);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_42 = wp::address(var_inertias, var_0, var_1, var_41);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_41, var_43);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_41, var_43);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_46 = wp::address(var_inertias, var_0, var_1, var_45);
        var_48 = wp::load(var_46);
        var_47 = wp::copy(var_48);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        wp::array_store(var_body_inertia_user, var_3, var_6, var_45, var_47);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            wp::array_store(var_body_inertia_backend, var_3, var_12, var_45, var_47);
        }
        // _update_body_inertial_properties_inverse(                                              <L 25>
        // env_id, backend_body_id, body_mass_backend, body_inertia_backend, False, body_inv_mass, body_inv_inertia       <L 26>
        _update_body_inertial_properties_inverse_0(var_3, var_12, var_body_mass_backend, var_body_inertia_backend, var_49, var_body_inv_mass, var_body_inv_inertia);
    }
}



extern "C" __global__ void write_body_inertia_and_inverse_index_e4ddbaa1_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_inertias,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    wp::array_t<wp::int32> var_body_user_to_backend,
    bool var_has_body_ordering,
    wp::array_t<wp::float32> var_body_mass_backend,
    wp::array_t<wp::float32> var_body_inertia_user,
    wp::array_t<wp::float32> var_body_inertia_backend,
    wp::array_t<wp::float32> var_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_body_inv_inertia,
    wp::array_t<wp::float32> adj_inertias,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_body_ids,
    wp::array_t<wp::int32> adj_body_user_to_backend,
    bool adj_has_body_ordering,
    wp::array_t<wp::float32> adj_body_mass_backend,
    wp::array_t<wp::float32> adj_body_inertia_user,
    wp::array_t<wp::float32> adj_body_inertia_backend,
    wp::array_t<wp::float32> adj_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> adj_body_inv_inertia)
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
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 0;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        const wp::int32 var_17 = 1;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        const wp::int32 var_21 = 2;
        wp::float32* var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        const wp::int32 var_25 = 3;
        wp::float32* var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        const wp::int32 var_29 = 4;
        wp::float32* var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        const wp::int32 var_33 = 5;
        wp::float32* var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        const wp::int32 var_37 = 6;
        wp::float32* var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        const wp::int32 var_41 = 7;
        wp::float32* var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        const wp::int32 var_45 = 8;
        wp::float32* var_46;
        wp::float32 var_47;
        wp::float32 var_48;
        const bool var_49 = false;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::float32 adj_39 = {};
        wp::float32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::float32 adj_42 = {};
        wp::float32 adj_43 = {};
        wp::float32 adj_44 = {};
        wp::int32 adj_45 = {};
        wp::float32 adj_46 = {};
        wp::float32 adj_47 = {};
        wp::float32 adj_48 = {};
        bool adj_49 = {};
        //---------
        // forward
        // def write_body_inertia_and_inverse_index(                                              <L 1>
        // i, j = wp.tid()                                                                        <L 14>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 15>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // user_body_id = wp.int32(body_ids[j])                                                   <L 16>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // backend_body_id = user_body_id                                                         <L 17>
        var_8 = wp::copy(var_6);
        // if has_body_ordering:                                                                  <L 18>
        if (var_has_body_ordering) {
            // backend_body_id = body_user_to_backend[user_body_id]                               <L 19>
            var_9 = wp::address(var_body_user_to_backend, var_6);
            var_11 = wp::load(var_9);
            var_10 = wp::copy(var_11);
        }
        var_12 = wp::where(var_has_body_ordering, var_10, var_8);
        // for k in range(9):                                                                     <L 20>
        // value = inertias[i, j, k]                                                              <L 21>
        var_14 = wp::address(var_inertias, var_0, var_1, var_13);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_13, var_15);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_13, var_15);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_18 = wp::address(var_inertias, var_0, var_1, var_17);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_17, var_19);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_17, var_19);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_22 = wp::address(var_inertias, var_0, var_1, var_21);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_21, var_23);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_21, var_23);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_26 = wp::address(var_inertias, var_0, var_1, var_25);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_25, var_27);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_25, var_27);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_30 = wp::address(var_inertias, var_0, var_1, var_29);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_29, var_31);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_29, var_31);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_34 = wp::address(var_inertias, var_0, var_1, var_33);
        var_36 = wp::load(var_34);
        var_35 = wp::copy(var_36);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_33, var_35);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_33, var_35);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_38 = wp::address(var_inertias, var_0, var_1, var_37);
        var_40 = wp::load(var_38);
        var_39 = wp::copy(var_40);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_37, var_39);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_37, var_39);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_42 = wp::address(var_inertias, var_0, var_1, var_41);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_41, var_43);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_41, var_43);
        }
        // value = inertias[i, j, k]                                                              <L 21>
        var_46 = wp::address(var_inertias, var_0, var_1, var_45);
        var_48 = wp::load(var_46);
        var_47 = wp::copy(var_48);
        // body_inertia_user[env_id, user_body_id, k] = value                                     <L 22>
        // wp::array_store(var_body_inertia_user, var_3, var_6, var_45, var_47);
        // if has_body_ordering:                                                                  <L 23>
        if (var_has_body_ordering) {
            // body_inertia_backend[env_id, backend_body_id, k] = value                           <L 24>
            // wp::array_store(var_body_inertia_backend, var_3, var_12, var_45, var_47);
        }
        // _update_body_inertial_properties_inverse(                                              <L 25>
        // env_id, backend_body_id, body_mass_backend, body_inertia_backend, False, body_inv_mass, body_inv_inertia       <L 26>
        _update_body_inertial_properties_inverse_0(var_3, var_12, var_body_mass_backend, var_body_inertia_backend, var_49, var_body_inv_mass, var_body_inv_inertia);
        //---------
        // reverse
        adj__update_body_inertial_properties_inverse_0(var_3, var_12, var_body_mass_backend, var_body_inertia_backend, var_49, var_body_inv_mass, var_body_inv_inertia, adj_3, adj_12, adj_body_mass_backend, adj_body_inertia_backend, adj_49, adj_body_inv_mass, adj_body_inv_inertia);
        // adj: env_id, backend_body_id, body_mass_backend, body_inertia_backend, False, body_inv_mass, body_inv_inertia  <L 26>
        // adj: _update_body_inertial_properties_inverse(                                         <L 25>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_45, var_47, adj_body_inertia_backend, adj_3, adj_12, adj_45, adj_47);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_45, var_47, adj_body_inertia_user, adj_3, adj_6, adj_45, adj_47);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_48, adj_46, adj_47);
        wp::adj_address(var_inertias, var_0, var_1, var_45, adj_inertias, adj_0, adj_1, adj_45, adj_46);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_41, var_43, adj_body_inertia_backend, adj_3, adj_12, adj_41, adj_43);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_41, var_43, adj_body_inertia_user, adj_3, adj_6, adj_41, adj_43);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_44, adj_42, adj_43);
        wp::adj_address(var_inertias, var_0, var_1, var_41, adj_inertias, adj_0, adj_1, adj_41, adj_42);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_37, var_39, adj_body_inertia_backend, adj_3, adj_12, adj_37, adj_39);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_37, var_39, adj_body_inertia_user, adj_3, adj_6, adj_37, adj_39);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_40, adj_38, adj_39);
        wp::adj_address(var_inertias, var_0, var_1, var_37, adj_inertias, adj_0, adj_1, adj_37, adj_38);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_33, var_35, adj_body_inertia_backend, adj_3, adj_12, adj_33, adj_35);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_33, var_35, adj_body_inertia_user, adj_3, adj_6, adj_33, adj_35);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_36, adj_34, adj_35);
        wp::adj_address(var_inertias, var_0, var_1, var_33, adj_inertias, adj_0, adj_1, adj_33, adj_34);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_29, var_31, adj_body_inertia_backend, adj_3, adj_12, adj_29, adj_31);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_29, var_31, adj_body_inertia_user, adj_3, adj_6, adj_29, adj_31);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_32, adj_30, adj_31);
        wp::adj_address(var_inertias, var_0, var_1, var_29, adj_inertias, adj_0, adj_1, adj_29, adj_30);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_25, var_27, adj_body_inertia_backend, adj_3, adj_12, adj_25, adj_27);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_25, var_27, adj_body_inertia_user, adj_3, adj_6, adj_25, adj_27);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_28, adj_26, adj_27);
        wp::adj_address(var_inertias, var_0, var_1, var_25, adj_inertias, adj_0, adj_1, adj_25, adj_26);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_21, var_23, adj_body_inertia_backend, adj_3, adj_12, adj_21, adj_23);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_21, var_23, adj_body_inertia_user, adj_3, adj_6, adj_21, adj_23);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_24, adj_22, adj_23);
        wp::adj_address(var_inertias, var_0, var_1, var_21, adj_inertias, adj_0, adj_1, adj_21, adj_22);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_17, var_19, adj_body_inertia_backend, adj_3, adj_12, adj_17, adj_19);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_17, var_19, adj_body_inertia_user, adj_3, adj_6, adj_17, adj_19);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_20, adj_18, adj_19);
        wp::adj_address(var_inertias, var_0, var_1, var_17, adj_inertias, adj_0, adj_1, adj_17, adj_18);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        if (var_has_body_ordering) {
            wp::adj_array_store(var_body_inertia_backend, var_3, var_12, var_13, var_15, adj_body_inertia_backend, adj_3, adj_12, adj_13, adj_15);
            // adj: body_inertia_backend[env_id, backend_body_id, k] = value                      <L 24>
        }
        // adj: if has_body_ordering:                                                             <L 23>
        wp::adj_array_store(var_body_inertia_user, var_3, var_6, var_13, var_15, adj_body_inertia_user, adj_3, adj_6, adj_13, adj_15);
        // adj: body_inertia_user[env_id, user_body_id, k] = value                                <L 22>
        wp::adj_copy(var_16, adj_14, adj_15);
        wp::adj_address(var_inertias, var_0, var_1, var_13, adj_inertias, adj_0, adj_1, adj_13, adj_14);
        // adj: value = inertias[i, j, k]                                                         <L 21>
        // adj: for k in range(9):                                                                <L 20>
        wp::adj_where(var_has_body_ordering, var_10, var_8, adj_has_body_ordering, adj_10, adj_8, adj_12);
        if (var_has_body_ordering) {
            wp::adj_copy(var_11, adj_9, adj_10);
            wp::adj_address(var_body_user_to_backend, var_6, adj_body_user_to_backend, adj_6, adj_9);
            // adj: backend_body_id = body_user_to_backend[user_body_id]                          <L 19>
        }
        // adj: if has_body_ordering:                                                             <L 18>
        wp::adj_copy(var_6, adj_6, adj_8);
        // adj: backend_body_id = user_body_id                                                    <L 17>
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_5);
        // adj: user_body_id = wp.int32(body_ids[j])                                              <L 16>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 15>
        // adj: i, j = wp.tid()                                                                   <L 14>
        // adj: def write_body_inertia_and_inverse_index(                                         <L 1>
        continue;
    }
}



extern "C" __global__ void projected_gravity_b_2D_kernel_e7284f70_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_gravity_w,
    wp::array_t<wp::quat_t<wp::float32>> var_quat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_projected_gravity_b)
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
        wp::quat_t<wp::float32>* var_2;
        wp::vec_t<3, wp::float32>* var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::vec_t<3, wp::float32> var_5;
        wp::vec_t<3, wp::float32> var_6;
        wp::quat_t<wp::float32> var_7;
        //---------
        // forward
        // def projected_gravity_b_2D_kernel(                                                     <L 482>
        // i, j = wp.tid()                                                                        <L 496>
        builtin_tid2d(var_0, var_1);
        // projected_gravity_b[i, j] = wp.quat_rotate_inv(quat[i, j], wp.normalize(gravity_w[i]))       <L 497>
        var_2 = wp::address(var_quat, var_0, var_1);
        var_3 = wp::address(var_gravity_w, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::normalize(var_5);
        var_7 = wp::load(var_2);
        var_6 = wp::quat_rotate_inv(var_7, var_4);
        wp::array_store(var_projected_gravity_b, var_0, var_1, var_6);
    }
}



extern "C" __global__ void projected_gravity_b_2D_kernel_e7284f70_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_gravity_w,
    wp::array_t<wp::quat_t<wp::float32>> var_quat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_projected_gravity_b,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_gravity_w,
    wp::array_t<wp::quat_t<wp::float32>> adj_quat,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_projected_gravity_b)
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
        wp::quat_t<wp::float32>* var_2;
        wp::vec_t<3, wp::float32>* var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::vec_t<3, wp::float32> var_5;
        wp::vec_t<3, wp::float32> var_6;
        wp::quat_t<wp::float32> var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::quat_t<wp::float32> adj_2 = {};
        wp::vec_t<3, wp::float32> adj_3 = {};
        wp::vec_t<3, wp::float32> adj_4 = {};
        wp::vec_t<3, wp::float32> adj_5 = {};
        wp::vec_t<3, wp::float32> adj_6 = {};
        wp::quat_t<wp::float32> adj_7 = {};
        //---------
        // forward
        // def projected_gravity_b_2D_kernel(                                                     <L 482>
        // i, j = wp.tid()                                                                        <L 496>
        builtin_tid2d(var_0, var_1);
        // projected_gravity_b[i, j] = wp.quat_rotate_inv(quat[i, j], wp.normalize(gravity_w[i]))       <L 497>
        var_2 = wp::address(var_quat, var_0, var_1);
        var_3 = wp::address(var_gravity_w, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::normalize(var_5);
        var_7 = wp::load(var_2);
        var_6 = wp::quat_rotate_inv(var_7, var_4);
        // wp::array_store(var_projected_gravity_b, var_0, var_1, var_6);
        //---------
        // reverse
        wp::adj_array_store(var_projected_gravity_b, var_0, var_1, var_6, adj_projected_gravity_b, adj_0, adj_1, adj_6);
        wp::adj_quat_rotate_inv(var_7, var_4, adj_2, adj_4, adj_6);
        wp::adj_normalize(var_5, var_4, adj_3, adj_4);
        wp::adj_address(var_gravity_w, var_0, adj_gravity_w, adj_0, adj_3);
        wp::adj_address(var_quat, var_0, var_1, adj_quat, adj_0, adj_1, adj_2);
        // adj: projected_gravity_b[i, j] = wp.quat_rotate_inv(quat[i, j], wp.normalize(gravity_w[i]))  <L 497>
        // adj: i, j = wp.tid()                                                                   <L 496>
        // adj: def projected_gravity_b_2D_kernel(                                                <L 482>
        continue;
    }
}



extern "C" __global__ void write_body_com_position_to_buffer_index_ee5e1dff_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_in_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_data)
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
        wp::vec_t<3, wp::float32>* var_2;
        wp::int64* var_3;
        wp::int32 var_4;
        wp::int64 var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::vec_t<3, wp::float32> var_9;
        //---------
        // forward
        // def write_body_com_position_to_buffer_index(                                           <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // out_data[wp.int32(env_ids[i]), wp.int32(body_ids[j])] = in_data[i, j]                  <L 19>
        var_2 = wp::address(var_in_data, var_0, var_1);
        var_3 = wp::address(var_env_ids, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::int32(var_5);
        var_6 = wp::address(var_body_ids, var_1);
        var_8 = wp::load(var_6);
        var_7 = wp::int32(var_8);
        var_9 = wp::load(var_2);
        wp::array_store(var_out_data, var_4, var_7, var_9);
    }
}



extern "C" __global__ void write_body_com_position_to_buffer_index_ee5e1dff_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_in_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_data,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_in_data,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int32> adj_body_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_out_data)
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
        wp::vec_t<3, wp::float32>* var_2;
        wp::int64* var_3;
        wp::int32 var_4;
        wp::int64 var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::vec_t<3, wp::float32> var_9;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::vec_t<3, wp::float32> adj_2 = {};
        wp::int64 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int64 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::vec_t<3, wp::float32> adj_9 = {};
        //---------
        // forward
        // def write_body_com_position_to_buffer_index(                                           <L 1>
        // i, j = wp.tid()                                                                        <L 18>
        builtin_tid2d(var_0, var_1);
        // out_data[wp.int32(env_ids[i]), wp.int32(body_ids[j])] = in_data[i, j]                  <L 19>
        var_2 = wp::address(var_in_data, var_0, var_1);
        var_3 = wp::address(var_env_ids, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::int32(var_5);
        var_6 = wp::address(var_body_ids, var_1);
        var_8 = wp::load(var_6);
        var_7 = wp::int32(var_8);
        var_9 = wp::load(var_2);
        // wp::array_store(var_out_data, var_4, var_7, var_9);
        //---------
        // reverse
        wp::adj_array_store(var_out_data, var_4, var_7, var_9, adj_out_data, adj_4, adj_7, adj_2);
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_6);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_3);
        wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_2);
        // adj: out_data[wp.int32(env_ids[i]), wp.int32(body_ids[j])] = in_data[i, j]             <L 19>
        // adj: i, j = wp.tid()                                                                   <L 18>
        // adj: def write_body_com_position_to_buffer_index(                                      <L 1>
        continue;
    }
}



extern "C" __global__ void set_body_link_pose_to_sim_f01c4946_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::transform_t<wp::float32>* var_8;
        wp::transform_t<wp::float32> var_9;
        wp::transform_t<wp::float32>* var_10;
        wp::transform_t<wp::float32> var_11;
        //---------
        // forward
        // def set_body_link_pose_to_sim(                                                         <L 1>
        // i, j = wp.tid()                                                                        <L 22>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 23>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 24>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 25>
        if (var_from_mask) {
            // body_link_pose_w[env_id, body_id] = data[env_id, body_id]                          <L 26>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            wp::array_store(var_body_link_pose_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_link_pose_w[env_id, body_id] = data[i, j]                                     <L 28>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            wp::array_store(var_body_link_pose_w, var_3, var_6, var_11);
        }
    }
}



extern "C" __global__ void set_body_link_pose_to_sim_f01c4946_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::int64> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_data,
    wp::array_t<wp::int64> adj_env_ids,
    wp::array_t<wp::int32> adj_body_ids,
    bool adj_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_link_pose_w)
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
        wp::int64* var_2;
        wp::int32 var_3;
        wp::int64 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::transform_t<wp::float32>* var_8;
        wp::transform_t<wp::float32> var_9;
        wp::transform_t<wp::float32>* var_10;
        wp::transform_t<wp::float32> var_11;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int64 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int64 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::transform_t<wp::float32> adj_8 = {};
        wp::transform_t<wp::float32> adj_9 = {};
        wp::transform_t<wp::float32> adj_10 = {};
        wp::transform_t<wp::float32> adj_11 = {};
        //---------
        // forward
        // def set_body_link_pose_to_sim(                                                         <L 1>
        // i, j = wp.tid()                                                                        <L 22>
        builtin_tid2d(var_0, var_1);
        // env_id = wp.int32(env_ids[i])                                                          <L 23>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::int32(var_4);
        // body_id = wp.int32(body_ids[j])                                                        <L 24>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::int32(var_7);
        // if from_mask:                                                                          <L 25>
        if (var_from_mask) {
            // body_link_pose_w[env_id, body_id] = data[env_id, body_id]                          <L 26>
            var_8 = wp::address(var_data, var_3, var_6);
            var_9 = wp::load(var_8);
            // wp::array_store(var_body_link_pose_w, var_3, var_6, var_9);
        }
        if (!var_from_mask) {
            // body_link_pose_w[env_id, body_id] = data[i, j]                                     <L 28>
            var_10 = wp::address(var_data, var_0, var_1);
            var_11 = wp::load(var_10);
            // wp::array_store(var_body_link_pose_w, var_3, var_6, var_11);
        }
        //---------
        // reverse
        if (!var_from_mask) {
            wp::adj_array_store(var_body_link_pose_w, var_3, var_6, var_11, adj_body_link_pose_w, adj_3, adj_6, adj_10);
            wp::adj_address(var_data, var_0, var_1, adj_data, adj_0, adj_1, adj_10);
            // adj: body_link_pose_w[env_id, body_id] = data[i, j]                                <L 28>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_body_link_pose_w, var_3, var_6, var_9, adj_body_link_pose_w, adj_3, adj_6, adj_8);
            wp::adj_address(var_data, var_3, var_6, adj_data, adj_3, adj_6, adj_8);
            // adj: body_link_pose_w[env_id, body_id] = data[env_id, body_id]                     <L 26>
        }
        // adj: if from_mask:                                                                     <L 25>
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_5);
        // adj: body_id = wp.int32(body_ids[j])                                                   <L 24>
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: env_id = wp.int32(env_ids[i])                                                     <L 23>
        // adj: i, j = wp.tid()                                                                   <L 22>
        // adj: def set_body_link_pose_to_sim(                                                    <L 1>
        continue;
    }
}



extern "C" __global__ void write_2d_data_to_buffer_with_mask_f7df63f0_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_joint_mask,
    wp::array_t<wp::float32> var_out_data)
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
        bool var_2;
        bool* var_3;
        bool var_4;
        bool* var_5;
        bool var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        //---------
        // forward
        // def write_2d_data_to_buffer_with_mask(                                                 <L 1037>
        // i, j = wp.tid()                                                                        <L 1054>
        builtin_tid2d(var_0, var_1);
        // if env_mask[i] and joint_mask[j]:                                                      <L 1055>
        var_3 = wp::address(var_env_mask, var_0);
        var_4 = wp::load(var_3);
        var_2 = var_4;
        if (var_2) {
            var_5 = wp::address(var_joint_mask, var_1);
            var_6 = wp::load(var_5);
            var_2 = var_2 && var_6;
        }
        if (var_2) {
            // out_data[i, j] = in_data[i, j]                                                     <L 1056>
            var_7 = wp::address(var_in_data, var_0, var_1);
            var_8 = wp::load(var_7);
            wp::array_store(var_out_data, var_0, var_1, var_8);
        }
    }
}



extern "C" __global__ void write_2d_data_to_buffer_with_mask_f7df63f0_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_joint_mask,
    wp::array_t<wp::float32> var_out_data,
    wp::array_t<wp::float32> adj_in_data,
    wp::array_t<bool> adj_env_mask,
    wp::array_t<bool> adj_joint_mask,
    wp::array_t<wp::float32> adj_out_data)
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
        bool var_2;
        bool* var_3;
        bool var_4;
        bool* var_5;
        bool var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        bool adj_6 = {};
        wp::float32 adj_7 = {};
        wp::float32 adj_8 = {};
        //---------
        // forward
        // def write_2d_data_to_buffer_with_mask(                                                 <L 1037>
        // i, j = wp.tid()                                                                        <L 1054>
        builtin_tid2d(var_0, var_1);
        // if env_mask[i] and joint_mask[j]:                                                      <L 1055>
        var_3 = wp::address(var_env_mask, var_0);
        var_4 = wp::load(var_3);
        var_2 = var_4;
        if (var_2) {
            var_5 = wp::address(var_joint_mask, var_1);
            var_6 = wp::load(var_5);
            var_2 = var_2 && var_6;
        }
        if (var_2) {
            // out_data[i, j] = in_data[i, j]                                                     <L 1056>
            var_7 = wp::address(var_in_data, var_0, var_1);
            var_8 = wp::load(var_7);
            // wp::array_store(var_out_data, var_0, var_1, var_8);
        }
        //---------
        // reverse
        if (var_2) {
            wp::adj_array_store(var_out_data, var_0, var_1, var_8, adj_out_data, adj_0, adj_1, adj_7);
            wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_7);
            // adj: out_data[i, j] = in_data[i, j]                                                <L 1056>
        }
        if (var_2) {
            wp::adj_address(var_joint_mask, var_1, adj_joint_mask, adj_1, adj_5);
        }
        wp::adj_address(var_env_mask, var_0, adj_env_mask, adj_0, adj_3);
        // adj: if env_mask[i] and joint_mask[j]:                                                 <L 1055>
        // adj: i, j = wp.tid()                                                                   <L 1054>
        // adj: def write_2d_data_to_buffer_with_mask(                                            <L 1037>
        continue;
    }
}



extern "C" __global__ void root_heading_w_fad61337_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_forward_vec,
    wp::array_t<wp::quat_t<wp::float32>> var_quat,
    wp::array_t<wp::float32> var_heading_w)
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
        wp::vec_t<3, wp::float32>* var_1;
        wp::quat_t<wp::float32>* var_2;
        wp::float32 var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::quat_t<wp::float32> var_5;
        //---------
        // forward
        // def root_heading_w(                                                                    <L 421>
        // i = wp.tid()                                                                           <L 436>
        var_0 = builtin_tid1d();
        // heading_w[i] = compute_heading_w_func(forward_vec[i], quat[i])                         <L 437>
        var_1 = wp::address(var_forward_vec, var_0);
        var_2 = wp::address(var_quat, var_0);
        var_4 = wp::load(var_1);
        var_5 = wp::load(var_2);
        var_3 = compute_heading_w_func_0(var_4, var_5);
        wp::array_store(var_heading_w, var_0, var_3);
    }
}



extern "C" __global__ void root_heading_w_fad61337_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_forward_vec,
    wp::array_t<wp::quat_t<wp::float32>> var_quat,
    wp::array_t<wp::float32> var_heading_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_forward_vec,
    wp::array_t<wp::quat_t<wp::float32>> adj_quat,
    wp::array_t<wp::float32> adj_heading_w)
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
        wp::vec_t<3, wp::float32>* var_1;
        wp::quat_t<wp::float32>* var_2;
        wp::float32 var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::quat_t<wp::float32> var_5;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::vec_t<3, wp::float32> adj_1 = {};
        wp::quat_t<wp::float32> adj_2 = {};
        wp::float32 adj_3 = {};
        wp::vec_t<3, wp::float32> adj_4 = {};
        wp::quat_t<wp::float32> adj_5 = {};
        //---------
        // forward
        // def root_heading_w(                                                                    <L 421>
        // i = wp.tid()                                                                           <L 436>
        var_0 = builtin_tid1d();
        // heading_w[i] = compute_heading_w_func(forward_vec[i], quat[i])                         <L 437>
        var_1 = wp::address(var_forward_vec, var_0);
        var_2 = wp::address(var_quat, var_0);
        var_4 = wp::load(var_1);
        var_5 = wp::load(var_2);
        var_3 = compute_heading_w_func_0(var_4, var_5);
        // wp::array_store(var_heading_w, var_0, var_3);
        //---------
        // reverse
        wp::adj_array_store(var_heading_w, var_0, var_3, adj_heading_w, adj_0, adj_3);
        adj_compute_heading_w_func_0(var_4, var_5, adj_1, adj_2, adj_3);
        wp::adj_address(var_quat, var_0, adj_quat, adj_0, adj_2);
        wp::adj_address(var_forward_vec, var_0, adj_forward_vec, adj_0, adj_1);
        // adj: heading_w[i] = compute_heading_w_func(forward_vec[i], quat[i])                    <L 437>
        // adj: i = wp.tid()                                                                      <L 436>
        // adj: def root_heading_w(                                                               <L 421>
        continue;
    }
}



extern "C" __global__ void quat_apply_inverse_2D_kernel_fdaabbf2_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vec,
    wp::array_t<wp::quat_t<wp::float32>> var_quat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_result)
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
        wp::quat_t<wp::float32>* var_2;
        wp::vec_t<3, wp::float32>* var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::quat_t<wp::float32> var_5;
        wp::vec_t<3, wp::float32> var_6;
        //---------
        // forward
        // def quat_apply_inverse_2D_kernel(                                                      <L 441>
        // i, j = wp.tid()                                                                        <L 456>
        builtin_tid2d(var_0, var_1);
        // result[i, j] = wp.quat_rotate_inv(quat[i, j], vec[i, j])                               <L 457>
        var_2 = wp::address(var_quat, var_0, var_1);
        var_3 = wp::address(var_vec, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = wp::quat_rotate_inv(var_5, var_6);
        wp::array_store(var_result, var_0, var_1, var_4);
    }
}



extern "C" __global__ void quat_apply_inverse_2D_kernel_fdaabbf2_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vec,
    wp::array_t<wp::quat_t<wp::float32>> var_quat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_result,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_vec,
    wp::array_t<wp::quat_t<wp::float32>> adj_quat,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_result)
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
        wp::quat_t<wp::float32>* var_2;
        wp::vec_t<3, wp::float32>* var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::quat_t<wp::float32> var_5;
        wp::vec_t<3, wp::float32> var_6;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::quat_t<wp::float32> adj_2 = {};
        wp::vec_t<3, wp::float32> adj_3 = {};
        wp::vec_t<3, wp::float32> adj_4 = {};
        wp::quat_t<wp::float32> adj_5 = {};
        wp::vec_t<3, wp::float32> adj_6 = {};
        //---------
        // forward
        // def quat_apply_inverse_2D_kernel(                                                      <L 441>
        // i, j = wp.tid()                                                                        <L 456>
        builtin_tid2d(var_0, var_1);
        // result[i, j] = wp.quat_rotate_inv(quat[i, j], vec[i, j])                               <L 457>
        var_2 = wp::address(var_quat, var_0, var_1);
        var_3 = wp::address(var_vec, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = wp::quat_rotate_inv(var_5, var_6);
        // wp::array_store(var_result, var_0, var_1, var_4);
        //---------
        // reverse
        wp::adj_array_store(var_result, var_0, var_1, var_4, adj_result, adj_0, adj_1, adj_4);
        wp::adj_quat_rotate_inv(var_5, var_6, adj_2, adj_3, adj_4);
        wp::adj_address(var_vec, var_0, var_1, adj_vec, adj_0, adj_1, adj_3);
        wp::adj_address(var_quat, var_0, var_1, adj_quat, adj_0, adj_1, adj_2);
        // adj: result[i, j] = wp.quat_rotate_inv(quat[i, j], vec[i, j])                          <L 457>
        // adj: i, j = wp.tid()                                                                   <L 456>
        // adj: def quat_apply_inverse_2D_kernel(                                                 <L 441>
        continue;
    }
}

