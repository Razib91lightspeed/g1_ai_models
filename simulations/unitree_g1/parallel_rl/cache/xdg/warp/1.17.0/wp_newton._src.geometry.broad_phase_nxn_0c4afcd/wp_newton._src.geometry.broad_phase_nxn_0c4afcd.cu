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


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/broad_phase_common.py:208
static CUDA_CALLABLE bool is_shape_pair_same_body_filtered_0(
    wp::int32 var_shape_a,
    wp::int32 var_shape_b,
    wp::array_t<wp::int32> var_shape_body)
{
    //---------
    // primal vars
    wp::shape_t* var_0;
    const wp::int32 var_1 = 0;
    wp::int32 var_2;
    wp::shape_t var_3;
    const wp::int32 var_4 = 0;
    bool var_5;
    const bool var_6 = false;
    wp::int32* var_7;
    wp::int32 var_8;
    wp::int32 var_9;
    bool var_10;
    const wp::int32 var_11 = 0;
    bool var_12;
    wp::int32* var_13;
    bool var_14;
    wp::int32 var_15;
    //---------
    // forward
    // def is_shape_pair_same_body_filtered(shape_a: int, shape_b: int, shape_body: wp.array[int]) -> bool:       <L 209>
    // if shape_body.shape[0] == 0:                                                           <L 211>
    var_0 = &(var_shape_body.shape);
    var_3 = wp::load(var_0);
    var_2 = wp::extract(var_3, var_1);
    var_5 = (var_2 == var_4);
    if (var_5) {
        // return False                                                                       <L 212>
        return var_6;
    }
    // body_a = shape_body[shape_a]                                                           <L 214>
    var_7 = wp::address(var_shape_body, var_shape_a);
    var_9 = wp::load(var_7);
    var_8 = wp::copy(var_9);
    // return body_a >= 0 and body_a == shape_body[shape_b]                                   <L 215>
    var_12 = (var_8 >= var_11);
    var_10 = var_12;
    if (var_10) {
        var_13 = wp::address(var_shape_body, var_shape_b);
        var_15 = wp::load(var_13);
        var_14 = (var_8 == var_15);
        var_10 = var_10 && var_14;
    }
    return var_10;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/broad_phase_common.py:165
static CUDA_CALLABLE bool is_shape_pair_immovable_filtered_0(
    wp::int32 var_shape_a,
    wp::int32 var_shape_b,
    wp::array_t<wp::int32> var_shape_body,
    wp::array_t<wp::int32> var_body_flags,
    bool var_include_static_kinematic_pairs)
{
    //---------
    // primal vars
    wp::shape_t* var_0;
    const wp::int32 var_1 = 0;
    wp::int32 var_2;
    wp::shape_t var_3;
    const wp::int32 var_4 = 0;
    bool var_5;
    const bool var_6 = false;
    wp::int32* var_7;
    wp::int32 var_8;
    wp::int32 var_9;
    wp::int32* var_10;
    wp::int32 var_11;
    wp::int32 var_12;
    const wp::int32 var_13 = 0;
    bool var_14;
    const wp::int32 var_15 = 0;
    bool var_16;
    bool var_17;
    const bool var_18 = true;
    const bool var_19 = false;
    wp::shape_t* var_20;
    const wp::int32 var_21 = 0;
    wp::int32 var_22;
    wp::shape_t var_23;
    const wp::int32 var_24 = 0;
    bool var_25;
    const bool var_26 = false;
    const bool var_27 = false;
    const bool var_28 = false;
    bool var_29;
    wp::int32* var_30;
    const wp::int32 var_31 = 2;
    wp::int32 var_32;
    wp::int32 var_33;
    const wp::int32 var_34 = 0;
    bool var_35;
    bool var_36;
    bool var_37;
    wp::int32* var_38;
    wp::int32 var_39;
    wp::int32 var_40;
    const wp::int32 var_41 = 0;
    bool var_42;
    bool var_43;
    bool var_44;
    bool var_45;
    bool var_46;
    //---------
    // forward
    // def is_shape_pair_immovable_filtered(                                                  <L 166>
    // if shape_body.shape[0] == 0:                                                           <L 176>
    var_0 = &(var_shape_body.shape);
    var_3 = wp::load(var_0);
    var_2 = wp::extract(var_3, var_1);
    var_5 = (var_2 == var_4);
    if (var_5) {
        // return False                                                                       <L 177>
        return var_6;
    }
    // body_a = shape_body[shape_a]                                                           <L 179>
    var_7 = wp::address(var_shape_body, var_shape_a);
    var_9 = wp::load(var_7);
    var_8 = wp::copy(var_9);
    // body_b = shape_body[shape_b]                                                           <L 180>
    var_10 = wp::address(var_shape_body, var_shape_b);
    var_12 = wp::load(var_10);
    var_11 = wp::copy(var_12);
    // static_a = body_a < 0                                                                  <L 182>
    var_14 = (var_8 < var_13);
    // static_b = body_b < 0                                                                  <L 183>
    var_16 = (var_11 < var_15);
    // if static_a and static_b:                                                              <L 186>
    var_17 = var_14;
    if (var_17) {
        var_17 = var_17 && var_16;
    }
    if (var_17) {
        // return True                                                                        <L 187>
        return var_18;
    }
    // if include_static_kinematic_pairs:                                                     <L 189>
    if (var_include_static_kinematic_pairs) {
        // return False                                                                       <L 190>
        return var_19;
    }
    // if body_flags.shape[0] == 0:                                                           <L 193>
    var_20 = &(var_body_flags.shape);
    var_23 = wp::load(var_20);
    var_22 = wp::extract(var_23, var_21);
    var_25 = (var_22 == var_24);
    if (var_25) {
        // return False                                                                       <L 194>
        return var_26;
    }
    // kinematic_a = False                                                                    <L 196>
    // kinematic_b = False                                                                    <L 197>
    // if not static_a:                                                                       <L 198>
    var_29 = wp::unot(var_14);
    if (var_29) {
        // kinematic_a = (body_flags[body_a] & BODY_FLAG_KINEMATIC) != 0                      <L 199>
        var_30 = wp::address(var_body_flags, var_8);
        var_33 = wp::load(var_30);
        var_32 = wp::bit_and(var_33, var_31);
        var_35 = (var_32 != var_34);
    }
    var_36 = wp::where(var_29, var_35, var_27);
    // if not static_b:                                                                       <L 200>
    var_37 = wp::unot(var_16);
    if (var_37) {
        // kinematic_b = (body_flags[body_b] & BODY_FLAG_KINEMATIC) != 0                      <L 201>
        var_38 = wp::address(var_body_flags, var_11);
        var_40 = wp::load(var_38);
        var_39 = wp::bit_and(var_40, var_31);
        var_42 = (var_39 != var_41);
    }
    var_43 = wp::where(var_37, var_42, var_28);
    // immovable_a = static_a or kinematic_a                                                  <L 203>
    var_44 = var_14;
    if (!var_44) {
        var_44 = var_44 || var_36;
    }
    // immovable_b = static_b or kinematic_b                                                  <L 204>
    var_45 = var_16;
    if (!var_45) {
        var_45 = var_45 || var_43;
    }
    // return immovable_a and immovable_b                                                     <L 205>
    var_46 = var_44;
    if (var_46) {
        var_46 = var_46 && var_45;
    }
    return var_46;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/broad_phase_common.py:20
static CUDA_CALLABLE bool check_aabb_overlap_0(
    wp::vec_t<3, wp::float32> var_box1_lower,
    wp::vec_t<3, wp::float32> var_box1_upper,
    wp::float32 var_box1_cutoff,
    wp::vec_t<3, wp::float32> var_box2_lower,
    wp::vec_t<3, wp::float32> var_box2_upper,
    wp::float32 var_box2_cutoff)
{
    //---------
    // primal vars
    wp::float32 var_0;
    bool var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    const wp::int32 var_4 = 0;
    wp::float32 var_5;
    wp::float32 var_6;
    bool var_7;
    const wp::int32 var_8 = 0;
    wp::float32 var_9;
    const wp::int32 var_10 = 0;
    wp::float32 var_11;
    wp::float32 var_12;
    bool var_13;
    const wp::int32 var_14 = 1;
    wp::float32 var_15;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    wp::float32 var_18;
    bool var_19;
    const wp::int32 var_20 = 1;
    wp::float32 var_21;
    const wp::int32 var_22 = 1;
    wp::float32 var_23;
    wp::float32 var_24;
    bool var_25;
    const wp::int32 var_26 = 2;
    wp::float32 var_27;
    const wp::int32 var_28 = 2;
    wp::float32 var_29;
    wp::float32 var_30;
    bool var_31;
    const wp::int32 var_32 = 2;
    wp::float32 var_33;
    const wp::int32 var_34 = 2;
    wp::float32 var_35;
    wp::float32 var_36;
    bool var_37;
    //---------
    // forward
    // def check_aabb_overlap(                                                                <L 21>
    // cutoff_combined = box1_cutoff + box2_cutoff                                            <L 29>
    var_0 = wp::add(var_box1_cutoff, var_box2_cutoff);
    // return (                                                                               <L 30>
    // box1_lower[0] <= box2_upper[0] + cutoff_combined                                       <L 31>
    var_3 = wp::extract(var_box1_lower, var_2);
    var_5 = wp::extract(var_box2_upper, var_4);
    var_6 = wp::add(var_5, var_0);
    var_7 = (var_3 <= var_6);
    var_1 = var_7;
    if (var_1) {
        // and box1_upper[0] >= box2_lower[0] - cutoff_combined                               <L 32>
        var_9 = wp::extract(var_box1_upper, var_8);
        var_11 = wp::extract(var_box2_lower, var_10);
        var_12 = wp::sub(var_11, var_0);
        var_13 = (var_9 >= var_12);
        var_1 = var_1 && var_13;
    }
    if (var_1) {
        // and box1_lower[1] <= box2_upper[1] + cutoff_combined                               <L 33>
        var_15 = wp::extract(var_box1_lower, var_14);
        var_17 = wp::extract(var_box2_upper, var_16);
        var_18 = wp::add(var_17, var_0);
        var_19 = (var_15 <= var_18);
        var_1 = var_1 && var_19;
    }
    if (var_1) {
        // and box1_upper[1] >= box2_lower[1] - cutoff_combined                               <L 34>
        var_21 = wp::extract(var_box1_upper, var_20);
        var_23 = wp::extract(var_box2_lower, var_22);
        var_24 = wp::sub(var_23, var_0);
        var_25 = (var_21 >= var_24);
        var_1 = var_1 && var_25;
    }
    if (var_1) {
        // and box1_lower[2] <= box2_upper[2] + cutoff_combined                               <L 35>
        var_27 = wp::extract(var_box1_lower, var_26);
        var_29 = wp::extract(var_box2_upper, var_28);
        var_30 = wp::add(var_29, var_0);
        var_31 = (var_27 <= var_30);
        var_1 = var_1 && var_31;
    }
    if (var_1) {
        // and box1_upper[2] >= box2_lower[2] - cutoff_combined                               <L 36>
        var_33 = wp::extract(var_box1_upper, var_32);
        var_35 = wp::extract(var_box2_lower, var_34);
        var_36 = wp::sub(var_35, var_0);
        var_37 = (var_33 >= var_36);
        var_1 = var_1 && var_37;
    }
    return var_1;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/broad_phase_common.py:40
static CUDA_CALLABLE bool check_aabb_overlap_moving_0(
    wp::int32 var_shape1,
    wp::int32 var_shape2,
    wp::array_t<wp::vec_t<3, wp::float32>> var_box_lower,
    wp::array_t<wp::vec_t<3, wp::float32>> var_box_upper,
    wp::float32 var_cutoff1,
    wp::float32 var_cutoff2,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_displacement)
{
    //---------
    // primal vars
    wp::shape_t* var_0;
    const wp::int32 var_1 = 0;
    wp::int32 var_2;
    wp::shape_t var_3;
    const wp::int32 var_4 = 0;
    bool var_5;
    wp::vec_t<3, wp::float32>* var_6;
    wp::vec_t<3, wp::float32>* var_7;
    wp::vec_t<3, wp::float32>* var_8;
    wp::vec_t<3, wp::float32>* var_9;
    bool var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::float32 var_15;
    wp::vec_t<3, wp::float32>* var_16;
    wp::vec_t<3, wp::float32>* var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32>* var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32>* var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32>* var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32>* var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    const wp::float32 var_33 = 0.0;
    wp::float32 var_34;
    const wp::float32 var_35 = 1.0;
    wp::float32 var_36;
    const wp::int32 var_37 = 0;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    const wp::float32 var_45 = 0.0;
    bool var_46;
    bool var_47;
    bool var_48;
    bool var_49;
    const bool var_50 = false;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    bool var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    bool var_63;
    const bool var_64 = false;
    wp::float32 var_65;
    wp::float32 var_66;
    const wp::int32 var_67 = 1;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    const wp::float32 var_75 = 0.0;
    bool var_76;
    bool var_77;
    bool var_78;
    bool var_79;
    const bool var_80 = false;
    wp::float32 var_81;
    wp::float32 var_82;
    wp::float32 var_83;
    wp::float32 var_84;
    bool var_85;
    wp::float32 var_86;
    wp::float32 var_87;
    wp::float32 var_88;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    wp::float32 var_92;
    wp::float32 var_93;
    bool var_94;
    const bool var_95 = false;
    wp::float32 var_96;
    wp::float32 var_97;
    wp::float32 var_98;
    wp::float32 var_99;
    wp::float32 var_100;
    const wp::int32 var_101 = 2;
    wp::float32 var_102;
    wp::float32 var_103;
    wp::float32 var_104;
    wp::float32 var_105;
    wp::float32 var_106;
    wp::float32 var_107;
    wp::float32 var_108;
    const wp::float32 var_109 = 0.0;
    bool var_110;
    bool var_111;
    bool var_112;
    bool var_113;
    const bool var_114 = false;
    wp::float32 var_115;
    wp::float32 var_116;
    wp::float32 var_117;
    wp::float32 var_118;
    bool var_119;
    wp::float32 var_120;
    wp::float32 var_121;
    wp::float32 var_122;
    wp::float32 var_123;
    wp::float32 var_124;
    wp::float32 var_125;
    wp::float32 var_126;
    wp::float32 var_127;
    bool var_128;
    const bool var_129 = false;
    wp::float32 var_130;
    wp::float32 var_131;
    wp::float32 var_132;
    wp::float32 var_133;
    wp::float32 var_134;
    const bool var_135 = true;
    //---------
    // forward
    // def check_aabb_overlap_moving(                                                         <L 41>
    // if shape_displacement.shape[0] == 0:                                                   <L 51>
    var_0 = &(var_shape_displacement.shape);
    var_3 = wp::load(var_0);
    var_2 = wp::extract(var_3, var_1);
    var_5 = (var_2 == var_4);
    if (var_5) {
        // return check_aabb_overlap(                                                         <L 52>
        // box_lower[shape1], box_upper[shape1], cutoff1, box_lower[shape2], box_upper[shape2], cutoff2       <L 53>
        var_6 = wp::address(var_box_lower, var_shape1);
        var_7 = wp::address(var_box_upper, var_shape1);
        var_8 = wp::address(var_box_lower, var_shape2);
        var_9 = wp::address(var_box_upper, var_shape2);
        var_11 = wp::load(var_6);
        var_12 = wp::load(var_7);
        var_13 = wp::load(var_8);
        var_14 = wp::load(var_9);
        var_10 = check_aabb_overlap_0(var_11, var_12, var_cutoff1, var_13, var_14, var_cutoff2);
        return var_10;
    }
    // cutoff_combined = cutoff1 + cutoff2                                                    <L 56>
    var_15 = wp::add(var_cutoff1, var_cutoff2);
    // relative_displacement = shape_displacement[shape1] - shape_displacement[shape2]        <L 57>
    var_16 = wp::address(var_shape_displacement, var_shape1);
    var_17 = wp::address(var_shape_displacement, var_shape2);
    var_19 = wp::load(var_16);
    var_20 = wp::load(var_17);
    var_18 = wp::sub(var_19, var_20);
    // lower_box1 = box_lower[shape1]                                                         <L 58>
    var_21 = wp::address(var_box_lower, var_shape1);
    var_23 = wp::load(var_21);
    var_22 = wp::copy(var_23);
    // upper_box1 = box_upper[shape1]                                                         <L 59>
    var_24 = wp::address(var_box_upper, var_shape1);
    var_26 = wp::load(var_24);
    var_25 = wp::copy(var_26);
    // lower_box2 = box_lower[shape2]                                                         <L 60>
    var_27 = wp::address(var_box_lower, var_shape2);
    var_29 = wp::load(var_27);
    var_28 = wp::copy(var_29);
    // upper_box2 = box_upper[shape2]                                                         <L 61>
    var_30 = wp::address(var_box_upper, var_shape2);
    var_32 = wp::load(var_30);
    var_31 = wp::copy(var_32);
    // enter = float(0.0)                                                                     <L 62>
    var_34 = wp::float(var_33);
    // exit_time = float(1.0)                                                                 <L 63>
    var_36 = wp::float(var_35);
    // for axis in range(3):                                                                  <L 64>
    // lower1 = lower_box1[axis]                                                              <L 65>
    var_38 = wp::extract(var_22, var_37);
    // upper1 = upper_box1[axis]                                                              <L 66>
    var_39 = wp::extract(var_25, var_37);
    // lower2 = lower_box2[axis] - cutoff_combined                                            <L 67>
    var_40 = wp::extract(var_28, var_37);
    var_41 = wp::sub(var_40, var_15);
    // upper2 = upper_box2[axis] + cutoff_combined                                            <L 68>
    var_42 = wp::extract(var_31, var_37);
    var_43 = wp::add(var_42, var_15);
    // delta = relative_displacement[axis]                                                    <L 69>
    var_44 = wp::extract(var_18, var_37);
    // if delta == 0.0:                                                                       <L 70>
    var_46 = (var_44 == var_45);
    if (var_46) {
        // if lower1 > upper2 or upper1 < lower2:                                             <L 71>
        var_48 = (var_38 > var_43);
        var_47 = var_48;
        if (!var_47) {
            var_49 = (var_39 < var_41);
            var_47 = var_47 || var_49;
        }
        if (var_47) {
            // return False                                                                   <L 72>
            return var_50;
        }
    }
    if (!var_46) {
        // axis_enter = (lower2 - upper1) / delta                                             <L 74>
        var_51 = wp::sub(var_41, var_39);
        var_52 = wp::div(var_51, var_44);
        // axis_exit = (upper2 - lower1) / delta                                              <L 75>
        var_53 = wp::sub(var_43, var_38);
        var_54 = wp::div(var_53, var_44);
        // if axis_enter > axis_exit:                                                         <L 76>
        var_55 = (var_52 > var_54);
        if (var_55) {
            // tmp = axis_enter                                                               <L 77>
            var_56 = wp::copy(var_52);
            // axis_enter = axis_exit                                                         <L 78>
            var_57 = wp::copy(var_54);
            // axis_exit = tmp                                                                <L 79>
            var_58 = wp::copy(var_56);
        }
        var_59 = wp::where(var_55, var_57, var_52);
        var_60 = wp::where(var_55, var_58, var_54);
        // enter = wp.max(enter, axis_enter)                                                  <L 80>
        var_61 = wp::max(var_34, var_59);
        // exit_time = wp.min(exit_time, axis_exit)                                           <L 81>
        var_62 = wp::min(var_36, var_60);
        // if enter > exit_time:                                                              <L 82>
        var_63 = (var_61 > var_62);
        if (var_63) {
            // return False                                                                   <L 83>
            return var_64;
        }
    }
    var_65 = wp::where(var_46, var_34, var_61);
    var_66 = wp::where(var_46, var_36, var_62);
    // lower1 = lower_box1[axis]                                                              <L 65>
    var_68 = wp::extract(var_22, var_67);
    // upper1 = upper_box1[axis]                                                              <L 66>
    var_69 = wp::extract(var_25, var_67);
    // lower2 = lower_box2[axis] - cutoff_combined                                            <L 67>
    var_70 = wp::extract(var_28, var_67);
    var_71 = wp::sub(var_70, var_15);
    // upper2 = upper_box2[axis] + cutoff_combined                                            <L 68>
    var_72 = wp::extract(var_31, var_67);
    var_73 = wp::add(var_72, var_15);
    // delta = relative_displacement[axis]                                                    <L 69>
    var_74 = wp::extract(var_18, var_67);
    // if delta == 0.0:                                                                       <L 70>
    var_76 = (var_74 == var_75);
    if (var_76) {
        // if lower1 > upper2 or upper1 < lower2:                                             <L 71>
        var_78 = (var_68 > var_73);
        var_77 = var_78;
        if (!var_77) {
            var_79 = (var_69 < var_71);
            var_77 = var_77 || var_79;
        }
        if (var_77) {
            // return False                                                                   <L 72>
            return var_80;
        }
    }
    if (!var_76) {
        // axis_enter = (lower2 - upper1) / delta                                             <L 74>
        var_81 = wp::sub(var_71, var_69);
        var_82 = wp::div(var_81, var_74);
        // axis_exit = (upper2 - lower1) / delta                                              <L 75>
        var_83 = wp::sub(var_73, var_68);
        var_84 = wp::div(var_83, var_74);
        // if axis_enter > axis_exit:                                                         <L 76>
        var_85 = (var_82 > var_84);
        if (var_85) {
            // tmp = axis_enter                                                               <L 77>
            var_86 = wp::copy(var_82);
            // axis_enter = axis_exit                                                         <L 78>
            var_87 = wp::copy(var_84);
            // axis_exit = tmp                                                                <L 79>
            var_88 = wp::copy(var_86);
        }
        var_89 = wp::where(var_85, var_87, var_82);
        var_90 = wp::where(var_85, var_88, var_84);
        var_91 = wp::where(var_85, var_86, var_56);
        // enter = wp.max(enter, axis_enter)                                                  <L 80>
        var_92 = wp::max(var_65, var_89);
        // exit_time = wp.min(exit_time, axis_exit)                                           <L 81>
        var_93 = wp::min(var_66, var_90);
        // if enter > exit_time:                                                              <L 82>
        var_94 = (var_92 > var_93);
        if (var_94) {
            // return False                                                                   <L 83>
            return var_95;
        }
    }
    var_96 = wp::where(var_76, var_65, var_92);
    var_97 = wp::where(var_76, var_66, var_93);
    var_98 = wp::where(var_76, var_59, var_89);
    var_99 = wp::where(var_76, var_60, var_90);
    var_100 = wp::where(var_76, var_56, var_91);
    // lower1 = lower_box1[axis]                                                              <L 65>
    var_102 = wp::extract(var_22, var_101);
    // upper1 = upper_box1[axis]                                                              <L 66>
    var_103 = wp::extract(var_25, var_101);
    // lower2 = lower_box2[axis] - cutoff_combined                                            <L 67>
    var_104 = wp::extract(var_28, var_101);
    var_105 = wp::sub(var_104, var_15);
    // upper2 = upper_box2[axis] + cutoff_combined                                            <L 68>
    var_106 = wp::extract(var_31, var_101);
    var_107 = wp::add(var_106, var_15);
    // delta = relative_displacement[axis]                                                    <L 69>
    var_108 = wp::extract(var_18, var_101);
    // if delta == 0.0:                                                                       <L 70>
    var_110 = (var_108 == var_109);
    if (var_110) {
        // if lower1 > upper2 or upper1 < lower2:                                             <L 71>
        var_112 = (var_102 > var_107);
        var_111 = var_112;
        if (!var_111) {
            var_113 = (var_103 < var_105);
            var_111 = var_111 || var_113;
        }
        if (var_111) {
            // return False                                                                   <L 72>
            return var_114;
        }
    }
    if (!var_110) {
        // axis_enter = (lower2 - upper1) / delta                                             <L 74>
        var_115 = wp::sub(var_105, var_103);
        var_116 = wp::div(var_115, var_108);
        // axis_exit = (upper2 - lower1) / delta                                              <L 75>
        var_117 = wp::sub(var_107, var_102);
        var_118 = wp::div(var_117, var_108);
        // if axis_enter > axis_exit:                                                         <L 76>
        var_119 = (var_116 > var_118);
        if (var_119) {
            // tmp = axis_enter                                                               <L 77>
            var_120 = wp::copy(var_116);
            // axis_enter = axis_exit                                                         <L 78>
            var_121 = wp::copy(var_118);
            // axis_exit = tmp                                                                <L 79>
            var_122 = wp::copy(var_120);
        }
        var_123 = wp::where(var_119, var_121, var_116);
        var_124 = wp::where(var_119, var_122, var_118);
        var_125 = wp::where(var_119, var_120, var_100);
        // enter = wp.max(enter, axis_enter)                                                  <L 80>
        var_126 = wp::max(var_96, var_123);
        // exit_time = wp.min(exit_time, axis_exit)                                           <L 81>
        var_127 = wp::min(var_97, var_124);
        // if enter > exit_time:                                                              <L 82>
        var_128 = (var_126 > var_127);
        if (var_128) {
            // return False                                                                   <L 83>
            return var_129;
        }
    }
    var_130 = wp::where(var_110, var_96, var_126);
    var_131 = wp::where(var_110, var_97, var_127);
    var_132 = wp::where(var_110, var_98, var_123);
    var_133 = wp::where(var_110, var_99, var_124);
    var_134 = wp::where(var_110, var_100, var_125);
    // return True                                                                            <L 84>
    return var_135;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/broad_phase_common.py:218
static CUDA_CALLABLE void write_pair_0(
    wp::vec_t<2, wp::int32> var_pair,
    wp::array_t<wp::vec_t<2, wp::int32>> var_candidate_pair,
    wp::array_t<wp::int32> var_candidate_pair_count,
    wp::int32 var_max_candidate_pair)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    const wp::int32 var_1 = 1;
    wp::int32 var_2;
    bool var_3;
    //---------
    // forward
    // def write_pair(                                                                        <L 219>
    // pairid = wp.atomic_add(candidate_pair_count, 0, 1)                                     <L 225>
    var_2 = wp::atomic_add(var_candidate_pair_count, var_0, var_1);
    // if pairid >= max_candidate_pair:                                                       <L 227>
    var_3 = (var_2 >= var_max_candidate_pair);
    if (var_3) {
        // return                                                                             <L 228>
        return;
    }
    // candidate_pair[pairid] = pair                                                          <L 230>
    wp::array_store(var_candidate_pair, var_2, var_pair);
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/broad_phase_nxn.py:98
static CUDA_CALLABLE void _find_world_and_local_id_0(
    wp::int32 var_tid,
    wp::array_t<wp::int32> var_world_cumsum_lower_tri,
    wp::int32 & ret_0,
    wp::int32 & ret_1)
{
    //---------
    // primal vars
    wp::shape_t* var_0;
    const wp::int32 var_1 = 0;
    wp::int32 var_2;
    wp::shape_t var_3;
    const wp::int32 var_4 = 0;
    wp::int32 var_5;
    const wp::int32 var_6 = 1;
    wp::int32 var_7;
    wp::int32 var_8;
    const wp::int32 var_9 = 0;
    wp::int32 var_10;
    bool var_11;
    wp::int32 var_12;
    const wp::int32 var_13 = 1;
    wp::int32 var_14;
    wp::int32* var_15;
    bool var_16;
    wp::int32 var_17;
    const wp::int32 var_18 = 1;
    wp::int32 var_19;
    wp::int32 var_20;
    const wp::int32 var_21 = 1;
    wp::int32 var_22;
    wp::int32 var_23;
    wp::int32 var_24;
    wp::int32 var_25;
    wp::int32 var_26;
    const wp::int32 var_27 = 0;
    bool var_28;
    const wp::int32 var_29 = 1;
    wp::int32 var_30;
    wp::int32* var_31;
    wp::int32 var_32;
    wp::int32 var_33;
    wp::int32 var_34;
    //---------
    // forward
    // def _find_world_and_local_id(                                                          <L 99>
    // world_count = world_cumsum_lower_tri.shape[0]                                          <L 112>
    var_0 = &(var_world_cumsum_lower_tri.shape);
    var_3 = wp::load(var_0);
    var_2 = wp::extract(var_3, var_1);
    // low = int(0)                                                                           <L 116>
    var_5 = wp::int(var_4);
    // high = int(world_count - 1)                                                            <L 117>
    var_7 = wp::sub(var_2, var_6);
    var_8 = wp::int(var_7);
    // world_id = int(0)                                                                      <L 118>
    var_10 = wp::int(var_9);
    // while low <= high:                                                                     <L 120>
    start_while_0:;
    var_11 = (var_5 <= var_8);
    if ((var_11) == false) goto end_while_0;
        // mid = (low + high) >> 1                                                            <L 121>
        var_12 = wp::add(var_5, var_8);
        var_14 = wp::rshift(var_12, var_13);
        // if tid < world_cumsum_lower_tri[mid]:                                              <L 122>
        var_15 = wp::address(var_world_cumsum_lower_tri, var_14);
        var_17 = wp::load(var_15);
        var_16 = (var_tid < var_17);
        if (var_16) {
            // high = mid - 1                                                                 <L 123>
            var_19 = wp::sub(var_14, var_18);
            // world_id = mid                                                                 <L 124>
            var_20 = wp::copy(var_14);
        }
        if (!var_16) {
            // low = mid + 1                                                                  <L 126>
            var_22 = wp::add(var_14, var_21);
        }
        var_23 = wp::where(var_16, var_5, var_22);
        var_24 = wp::where(var_16, var_19, var_8);
        var_25 = wp::where(var_16, var_20, var_10);
        wp::assign(var_5, var_23);
        wp::assign(var_8, var_24);
        wp::assign(var_10, var_25);
    goto start_while_0;
    end_while_0:;
    // local_id = tid                                                                         <L 129>
    var_26 = wp::copy(var_tid);
    // if world_id > 0:                                                                       <L 130>
    var_28 = (var_10 > var_27);
    if (var_28) {
        // local_id = tid - world_cumsum_lower_tri[world_id - 1]                              <L 131>
        var_30 = wp::sub(var_10, var_29);
        var_31 = wp::address(var_world_cumsum_lower_tri, var_30);
        var_33 = wp::load(var_31);
        var_32 = wp::sub(var_tid, var_33);
    }
    var_34 = wp::where(var_28, var_32, var_26);
    // return world_id, local_id                                                              <L 133>
    ret_0 = var_10;
    ret_1 = var_34;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/broad_phase_nxn.py:76
static CUDA_CALLABLE void _get_lower_triangular_indices_0(
    wp::int32 var_index,
    wp::int32 var_matrix_size,
    wp::int32 & ret_0,
    wp::int32 & ret_1)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::int32 var_1;
    wp::int32 var_2;
    const wp::int32 var_3 = 1;
    wp::int32 var_4;
    bool var_5;
    const wp::int32 var_6 = -1;
    const wp::int32 var_7 = -1;
    const wp::int32 var_8 = 0;
    wp::int32 var_9;
    const wp::int32 var_10 = 1;
    wp::int32 var_11;
    bool var_12;
    wp::int32 var_13;
    const wp::int32 var_14 = 1;
    wp::int32 var_15;
    const wp::int32 var_16 = 2;
    wp::int32 var_17;
    wp::int32 var_18;
    const wp::int32 var_19 = 1;
    wp::int32 var_20;
    wp::int32 var_21;
    const wp::int32 var_22 = 1;
    wp::int32 var_23;
    bool var_24;
    const wp::int32 var_25 = 1;
    wp::int32 var_26;
    wp::int32 var_27;
    wp::int32 var_28;
    wp::int32 var_29;
    const wp::int32 var_30 = 1;
    wp::int32 var_31;
    const wp::int32 var_32 = 2;
    wp::int32 var_33;
    wp::int32 var_34;
    const wp::int32 var_35 = 1;
    wp::int32 var_36;
    wp::int32 var_37;
    const wp::int32 var_38 = 1;
    wp::int32 var_39;
    wp::int32 var_40;
    wp::int32 var_41;
    const wp::int32 var_42 = 1;
    wp::int32 var_43;
    //---------
    // forward
    // def _get_lower_triangular_indices(index: int, matrix_size: int) -> tuple[int, int]:       <L 77>
    // total = (matrix_size * (matrix_size - 1)) >> 1                                         <L 78>
    var_1 = wp::sub(var_matrix_size, var_0);
    var_2 = wp::mul(var_matrix_size, var_1);
    var_4 = wp::rshift(var_2, var_3);
    // if index >= total:                                                                     <L 79>
    var_5 = (var_index >= var_4);
    if (var_5) {
        // return -1, -1                                                                      <L 81>
        ret_0 = var_6;
        ret_1 = var_7;
        return;
    }
    // low = int(0)                                                                           <L 83>
    var_9 = wp::int(var_8);
    // high = matrix_size - 1                                                                 <L 84>
    var_11 = wp::sub(var_matrix_size, var_10);
    // while low < high:                                                                      <L 85>
    start_while_1:;
    var_12 = (var_9 < var_11);
    if ((var_12) == false) goto end_while_1;
        // mid = (low + high) >> 1                                                            <L 86>
        var_13 = wp::add(var_9, var_11);
        var_15 = wp::rshift(var_13, var_14);
        // count = (mid * (2 * matrix_size - mid - 1)) >> 1                                   <L 87>
        var_17 = wp::mul(var_16, var_matrix_size);
        var_18 = wp::sub(var_17, var_15);
        var_20 = wp::sub(var_18, var_19);
        var_21 = wp::mul(var_15, var_20);
        var_23 = wp::rshift(var_21, var_22);
        // if count <= index:                                                                 <L 88>
        var_24 = (var_23 <= var_index);
        if (var_24) {
            // low = mid + 1                                                                  <L 89>
            var_26 = wp::add(var_15, var_25);
        }
        if (!var_24) {
            // high = mid                                                                     <L 91>
            var_27 = wp::copy(var_15);
        }
        var_28 = wp::where(var_24, var_26, var_9);
        var_29 = wp::where(var_24, var_11, var_27);
        wp::assign(var_9, var_28);
        wp::assign(var_11, var_29);
    goto start_while_1;
    end_while_1:;
    // r = low - 1                                                                            <L 92>
    var_31 = wp::sub(var_9, var_30);
    // f = (r * (2 * matrix_size - r - 1)) >> 1                                               <L 93>
    var_33 = wp::mul(var_32, var_matrix_size);
    var_34 = wp::sub(var_33, var_31);
    var_36 = wp::sub(var_34, var_35);
    var_37 = wp::mul(var_31, var_36);
    var_39 = wp::rshift(var_37, var_38);
    // c = (index - f) + r + 1                                                                <L 94>
    var_40 = wp::sub(var_index, var_39);
    var_41 = wp::add(var_40, var_31);
    var_43 = wp::add(var_41, var_42);
    // return r, c                                                                            <L 95>
    ret_0 = var_31;
    ret_1 = var_43;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/broad_phase_common.py:234
static CUDA_CALLABLE bool test_group_pair_0(
    wp::int32 var_group_a,
    wp::int32 var_group_b)
{
    //---------
    // primal vars
    bool var_0;
    const wp::int32 var_1 = 0;
    bool var_2;
    const wp::int32 var_3 = 0;
    bool var_4;
    const bool var_5 = false;
    const wp::int32 var_6 = 0;
    bool var_7;
    bool var_8;
    bool var_9;
    const wp::int32 var_10 = 0;
    bool var_11;
    const wp::int32 var_12 = 0;
    bool var_13;
    bool var_14;
    //---------
    // forward
    // def test_group_pair(group_a: int, group_b: int) -> bool:                               <L 235>
    // if group_a == 0 or group_b == 0:                                                       <L 247>
    var_2 = (var_group_a == var_1);
    var_0 = var_2;
    if (!var_0) {
        var_4 = (var_group_b == var_3);
        var_0 = var_0 || var_4;
    }
    if (var_0) {
        // return False                                                                       <L 248>
        return var_5;
    }
    // if group_a > 0:                                                                        <L 249>
    var_7 = (var_group_a > var_6);
    if (var_7) {
        // return group_a == group_b or group_b < 0                                           <L 250>
        var_9 = (var_group_a == var_group_b);
        var_8 = var_9;
        if (!var_8) {
            var_11 = (var_group_b < var_10);
            var_8 = var_8 || var_11;
        }
        return var_8;
    }
    // if group_a < 0:                                                                        <L 251>
    var_13 = (var_group_a < var_12);
    if (var_13) {
        // return group_a != group_b                                                          <L 252>
        var_14 = (var_group_a != var_group_b);
        return var_14;
    }
    return {};
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/broad_phase_common.py:255
static CUDA_CALLABLE bool test_world_and_group_pair_0(
    wp::int32 var_world_a,
    wp::int32 var_world_b,
    wp::int32 var_collision_group_a,
    wp::int32 var_collision_group_b)
{
    //---------
    // primal vars
    bool var_0;
    const wp::int32 var_1 = -1;
    bool var_2;
    const wp::int32 var_3 = -1;
    bool var_4;
    bool var_5;
    const bool var_6 = false;
    bool var_7;
    //---------
    // forward
    // def test_world_and_group_pair(world_a: int, world_b: int, collision_group_a: int, collision_group_b: int) -> bool:       <L 256>
    // if world_a != -1 and world_b != -1 and world_a != world_b:                             <L 278>
    var_2 = (var_world_a != var_1);
    var_0 = var_2;
    if (var_0) {
        var_4 = (var_world_b != var_3);
        var_0 = var_0 && var_4;
    }
    if (var_0) {
        var_5 = (var_world_a != var_world_b);
        var_0 = var_0 && var_5;
    }
    if (var_0) {
        // return False                                                                       <L 279>
        return var_6;
    }
    // return test_group_pair(collision_group_a, collision_group_b)                           <L 282>
    var_7 = test_group_pair_0(var_collision_group_a, var_collision_group_b);
    return var_7;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/broad_phase_common.py:117
static CUDA_CALLABLE bool _vec2i_equal_0(
    wp::vec_t<2, wp::int32> var_p,
    wp::vec_t<2, wp::int32> var_q)
{
    //---------
    // primal vars
    bool var_0;
    const wp::int32 var_1 = 0;
    wp::int32 var_2;
    const wp::int32 var_3 = 0;
    wp::int32 var_4;
    bool var_5;
    const wp::int32 var_6 = 1;
    wp::int32 var_7;
    const wp::int32 var_8 = 1;
    wp::int32 var_9;
    bool var_10;
    //---------
    // forward
    // def _vec2i_equal(p: wp.vec2i, q: wp.vec2i) -> bool:                                    <L 118>
    // return p[0] == q[0] and p[1] == q[1]                                                   <L 128>
    var_2 = wp::extract(var_p, var_1);
    var_4 = wp::extract(var_q, var_3);
    var_5 = (var_2 == var_4);
    var_0 = var_5;
    if (var_0) {
        var_7 = wp::extract(var_p, var_6);
        var_9 = wp::extract(var_q, var_8);
        var_10 = (var_7 == var_9);
        var_0 = var_0 && var_10;
    }
    return var_0;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/broad_phase_common.py:99
static CUDA_CALLABLE bool _vec2i_less_0(
    wp::vec_t<2, wp::int32> var_p,
    wp::vec_t<2, wp::int32> var_q)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::int32 var_1;
    const wp::int32 var_2 = 0;
    wp::int32 var_3;
    bool var_4;
    const bool var_5 = true;
    const wp::int32 var_6 = 0;
    wp::int32 var_7;
    const wp::int32 var_8 = 0;
    wp::int32 var_9;
    bool var_10;
    const bool var_11 = false;
    const wp::int32 var_12 = 1;
    wp::int32 var_13;
    const wp::int32 var_14 = 1;
    wp::int32 var_15;
    bool var_16;
    //---------
    // forward
    // def _vec2i_less(p: wp.vec2i, q: wp.vec2i) -> bool:                                     <L 100>
    // if p[0] < q[0]:                                                                        <L 110>
    var_1 = wp::extract(var_p, var_0);
    var_3 = wp::extract(var_q, var_2);
    var_4 = (var_1 < var_3);
    if (var_4) {
        // return True                                                                        <L 111>
        return var_5;
    }
    // if p[0] > q[0]:                                                                        <L 112>
    var_7 = wp::extract(var_p, var_6);
    var_9 = wp::extract(var_q, var_8);
    var_10 = (var_7 > var_9);
    if (var_10) {
        // return False                                                                       <L 113>
        return var_11;
    }
    // return p[1] < q[1]                                                                     <L 114>
    var_13 = wp::extract(var_p, var_12);
    var_15 = wp::extract(var_q, var_14);
    var_16 = (var_13 < var_15);
    return var_16;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/broad_phase_common.py:131
static CUDA_CALLABLE bool is_pair_excluded_0(
    wp::vec_t<2, wp::int32> var_pair,
    wp::array_t<wp::vec_t<2, wp::int32>> var_filter_pairs,
    wp::int32 var_num_filter_pairs)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    bool var_1;
    const bool var_2 = false;
    const wp::int32 var_3 = 0;
    wp::int32 var_4;
    const wp::int32 var_5 = 1;
    wp::int32 var_6;
    bool var_7;
    wp::int32 var_8;
    const wp::int32 var_9 = 1;
    wp::int32 var_10;
    wp::vec_t<2, wp::int32>* var_11;
    wp::vec_t<2, wp::int32> var_12;
    wp::vec_t<2, wp::int32> var_13;
    bool var_14;
    const bool var_15 = true;
    bool var_16;
    const wp::int32 var_17 = 1;
    wp::int32 var_18;
    const wp::int32 var_19 = 1;
    wp::int32 var_20;
    wp::int32 var_21;
    wp::int32 var_22;
    const bool var_23 = false;
    //---------
    // forward
    // def is_pair_excluded(                                                                  <L 132>
    // if num_filter_pairs <= 0:                                                              <L 149>
    var_1 = (var_num_filter_pairs <= var_0);
    if (var_1) {
        // return False                                                                       <L 150>
        return var_2;
    }
    // low = int(0)                                                                           <L 151>
    var_4 = wp::int(var_3);
    // high = num_filter_pairs - 1                                                            <L 152>
    var_6 = wp::sub(var_num_filter_pairs, var_5);
    // while low <= high:                                                                     <L 153>
    start_while_1:;
    var_7 = (var_4 <= var_6);
    if ((var_7) == false) goto end_while_1;
        // mid = (low + high) >> 1                                                            <L 154>
        var_8 = wp::add(var_4, var_6);
        var_10 = wp::rshift(var_8, var_9);
        // m = filter_pairs[mid]                                                              <L 155>
        var_11 = wp::address(var_filter_pairs, var_10);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // if _vec2i_equal(pair, m):                                                          <L 156>
        var_14 = _vec2i_equal_0(var_pair, var_12);
        if (var_14) {
            // return True                                                                    <L 157>
            return var_15;
        }
        // if _vec2i_less(pair, m):                                                           <L 158>
        var_16 = _vec2i_less_0(var_pair, var_12);
        if (var_16) {
            // high = mid - 1                                                                 <L 159>
            var_18 = wp::sub(var_10, var_17);
        }
        if (!var_16) {
            // low = mid + 1                                                                  <L 161>
            var_20 = wp::add(var_10, var_19);
        }
        var_21 = wp::where(var_16, var_4, var_20);
        var_22 = wp::where(var_16, var_18, var_6);
        wp::assign(var_4, var_21);
        wp::assign(var_6, var_22);
    goto start_while_1;
    end_while_1:;
    // return False                                                                           <L 162>
    return var_23;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/broad_phase_common.py:208
static CUDA_CALLABLE void adj_is_shape_pair_same_body_filtered_0(
    wp::int32 var_shape_a,
    wp::int32 var_shape_b,
    wp::array_t<wp::int32> var_shape_body,
    wp::int32 & adj_shape_a,
    wp::int32 & adj_shape_b,
    wp::array_t<wp::int32> & adj_shape_body,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/broad_phase_common.py:165
static CUDA_CALLABLE void adj_is_shape_pair_immovable_filtered_0(
    wp::int32 var_shape_a,
    wp::int32 var_shape_b,
    wp::array_t<wp::int32> var_shape_body,
    wp::array_t<wp::int32> var_body_flags,
    bool var_include_static_kinematic_pairs,
    wp::int32 & adj_shape_a,
    wp::int32 & adj_shape_b,
    wp::array_t<wp::int32> & adj_shape_body,
    wp::array_t<wp::int32> & adj_body_flags,
    bool & adj_include_static_kinematic_pairs,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/broad_phase_common.py:20
static CUDA_CALLABLE void adj_check_aabb_overlap_0(
    wp::vec_t<3, wp::float32> var_box1_lower,
    wp::vec_t<3, wp::float32> var_box1_upper,
    wp::float32 var_box1_cutoff,
    wp::vec_t<3, wp::float32> var_box2_lower,
    wp::vec_t<3, wp::float32> var_box2_upper,
    wp::float32 var_box2_cutoff,
    wp::vec_t<3, wp::float32> & adj_box1_lower,
    wp::vec_t<3, wp::float32> & adj_box1_upper,
    wp::float32 & adj_box1_cutoff,
    wp::vec_t<3, wp::float32> & adj_box2_lower,
    wp::vec_t<3, wp::float32> & adj_box2_upper,
    wp::float32 & adj_box2_cutoff,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/broad_phase_common.py:40
static CUDA_CALLABLE void adj_check_aabb_overlap_moving_0(
    wp::int32 var_shape1,
    wp::int32 var_shape2,
    wp::array_t<wp::vec_t<3, wp::float32>> var_box_lower,
    wp::array_t<wp::vec_t<3, wp::float32>> var_box_upper,
    wp::float32 var_cutoff1,
    wp::float32 var_cutoff2,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_displacement,
    wp::int32 & adj_shape1,
    wp::int32 & adj_shape2,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_box_lower,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_box_upper,
    wp::float32 & adj_cutoff1,
    wp::float32 & adj_cutoff2,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_shape_displacement,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/broad_phase_common.py:218
static CUDA_CALLABLE void adj_write_pair_0(
    wp::vec_t<2, wp::int32> var_pair,
    wp::array_t<wp::vec_t<2, wp::int32>> var_candidate_pair,
    wp::array_t<wp::int32> var_candidate_pair_count,
    wp::int32 var_max_candidate_pair,
    wp::vec_t<2, wp::int32> & adj_pair,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_candidate_pair,
    wp::array_t<wp::int32> & adj_candidate_pair_count,
    wp::int32 & adj_max_candidate_pair)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/broad_phase_nxn.py:98
static CUDA_CALLABLE void adj__find_world_and_local_id_0(
    wp::int32 var_tid,
    wp::array_t<wp::int32> var_world_cumsum_lower_tri,
    wp::int32 & ret_0,
    wp::int32 & ret_1,
    wp::int32 & adj_tid,
    wp::array_t<wp::int32> & adj_world_cumsum_lower_tri,
    wp::int32 & adj_ret_0,
    wp::int32 & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/broad_phase_nxn.py:76
static CUDA_CALLABLE void adj__get_lower_triangular_indices_0(
    wp::int32 var_index,
    wp::int32 var_matrix_size,
    wp::int32 & ret_0,
    wp::int32 & ret_1,
    wp::int32 & adj_index,
    wp::int32 & adj_matrix_size,
    wp::int32 & adj_ret_0,
    wp::int32 & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/broad_phase_common.py:234
static CUDA_CALLABLE void adj_test_group_pair_0(
    wp::int32 var_group_a,
    wp::int32 var_group_b,
    wp::int32 & adj_group_a,
    wp::int32 & adj_group_b,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/broad_phase_common.py:255
static CUDA_CALLABLE void adj_test_world_and_group_pair_0(
    wp::int32 var_world_a,
    wp::int32 var_world_b,
    wp::int32 var_collision_group_a,
    wp::int32 var_collision_group_b,
    wp::int32 & adj_world_a,
    wp::int32 & adj_world_b,
    wp::int32 & adj_collision_group_a,
    wp::int32 & adj_collision_group_b,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/broad_phase_common.py:117
static CUDA_CALLABLE void adj__vec2i_equal_0(
    wp::vec_t<2, wp::int32> var_p,
    wp::vec_t<2, wp::int32> var_q,
    wp::vec_t<2, wp::int32> & adj_p,
    wp::vec_t<2, wp::int32> & adj_q,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/broad_phase_common.py:99
static CUDA_CALLABLE void adj__vec2i_less_0(
    wp::vec_t<2, wp::int32> var_p,
    wp::vec_t<2, wp::int32> var_q,
    wp::vec_t<2, wp::int32> & adj_p,
    wp::vec_t<2, wp::int32> & adj_q,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/broad_phase_common.py:131
static CUDA_CALLABLE void adj_is_pair_excluded_0(
    wp::vec_t<2, wp::int32> var_pair,
    wp::array_t<wp::vec_t<2, wp::int32>> var_filter_pairs,
    wp::int32 var_num_filter_pairs,
    wp::vec_t<2, wp::int32> & adj_pair,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_filter_pairs,
    wp::int32 & adj_num_filter_pairs,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void _nxn_broadphase_precomputed_pairs_287f638a_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_bounding_box_lower,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_bounding_box_upper,
    wp::array_t<wp::float32> var_shape_gap,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_displacement,
    wp::array_t<wp::vec_t<2, wp::int32>> var_nxn_shape_pair,
    wp::array_t<wp::int32> var_shape_body,
    wp::array_t<wp::int32> var_body_flags,
    bool var_include_static_kinematic_pairs,
    wp::array_t<wp::vec_t<2, wp::int32>> var_candidate_pair,
    wp::array_t<wp::int32> var_candidate_pair_count,
    wp::int32 var_max_candidate_pair)
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
        wp::vec_t<2, wp::int32>* var_1;
        wp::vec_t<2, wp::int32> var_2;
        wp::vec_t<2, wp::int32> var_3;
        const wp::int32 var_4 = 0;
        wp::int32 var_5;
        const wp::int32 var_6 = 1;
        wp::int32 var_7;
        bool var_8;
        bool var_9;
        const wp::float32 var_10 = 0.0;
        const wp::float32 var_11 = 0.0;
        wp::shape_t* var_12;
        const wp::int32 var_13 = 0;
        wp::int32 var_14;
        wp::shape_t var_15;
        const wp::int32 var_16 = 0;
        bool var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        bool var_26;
        //---------
        // forward
        // def _nxn_broadphase_precomputed_pairs(                                                 <L 31>
        // elementid = wp.tid()                                                                   <L 46>
        var_0 = builtin_tid1d();
        // pair = nxn_shape_pair[elementid]                                                       <L 48>
        var_1 = wp::address(var_nxn_shape_pair, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // shape1 = pair[0]                                                                       <L 49>
        var_5 = wp::extract(var_2, var_4);
        // shape2 = pair[1]                                                                       <L 50>
        var_7 = wp::extract(var_2, var_6);
        // if is_shape_pair_same_body_filtered(shape1, shape2, shape_body):                       <L 52>
        var_8 = is_shape_pair_same_body_filtered_0(var_5, var_7, var_shape_body);
        if (var_8) {
            // return                                                                             <L 53>
            continue;
        }
        // if is_shape_pair_immovable_filtered(shape1, shape2, shape_body, body_flags, include_static_kinematic_pairs):       <L 55>
        var_9 = is_shape_pair_immovable_filtered_0(var_5, var_7, var_shape_body, var_body_flags, var_include_static_kinematic_pairs);
        if (var_9) {
            // return                                                                             <L 56>
            continue;
        }
        // gap1 = 0.0                                                                             <L 59>
        // gap2 = 0.0                                                                             <L 60>
        // if shape_gap.shape[0] > 0:                                                             <L 61>
        var_12 = &(var_shape_gap.shape);
        var_15 = wp::load(var_12);
        var_14 = wp::extract(var_15, var_13);
        var_17 = (var_14 > var_16);
        if (var_17) {
            // gap1 = shape_gap[shape1]                                                           <L 62>
            var_18 = wp::address(var_shape_gap, var_5);
            var_20 = wp::load(var_18);
            var_19 = wp::copy(var_20);
            // gap2 = shape_gap[shape2]                                                           <L 63>
            var_21 = wp::address(var_shape_gap, var_7);
            var_23 = wp::load(var_21);
            var_22 = wp::copy(var_23);
        }
        var_24 = wp::where(var_17, var_19, var_10);
        var_25 = wp::where(var_17, var_22, var_11);
        // if check_aabb_overlap_moving(                                                          <L 65>
        // shape1, shape2, shape_bounding_box_lower, shape_bounding_box_upper, gap1, gap2, shape_displacement       <L 66>
        var_26 = check_aabb_overlap_moving_0(var_5, var_7, var_shape_bounding_box_lower, var_shape_bounding_box_upper, var_24, var_25, var_shape_displacement);
        if (var_26) {
            // write_pair(                                                                        <L 68>
            // pair,                                                                              <L 69>
            // candidate_pair,                                                                    <L 70>
            // candidate_pair_count,                                                              <L 71>
            // max_candidate_pair,                                                                <L 72>
            write_pair_0(var_2, var_candidate_pair, var_candidate_pair_count, var_max_candidate_pair);
        }
    }
}



extern "C" __global__ void _nxn_broadphase_kernel_4f801a9e_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_bounding_box_lower,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_bounding_box_upper,
    wp::array_t<wp::float32> var_shape_gap,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_displacement,
    wp::array_t<wp::int32> var_collision_group,
    wp::array_t<wp::int32> var_shape_world,
    wp::array_t<wp::int32> var_shape_body,
    wp::array_t<wp::int32> var_body_flags,
    bool var_include_static_kinematic_pairs,
    wp::array_t<wp::int32> var_world_cumsum_lower_tri,
    wp::array_t<wp::int32> var_world_slice_ends,
    wp::array_t<wp::int32> var_world_index_map,
    wp::int32 var_num_regular_worlds,
    wp::array_t<wp::vec_t<2, wp::int32>> var_filter_pairs,
    wp::int32 var_num_filter_pairs,
    wp::array_t<wp::vec_t<2, wp::int32>> var_candidate_pair,
    wp::array_t<wp::int32> var_candidate_pair_count,
    wp::int32 var_max_candidate_pair)
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
        const wp::int32 var_3 = 0;
        const wp::int32 var_4 = 0;
        bool var_5;
        const wp::int32 var_6 = 1;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::int32* var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::int32* var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        wp::int32* var_31;
        wp::int32 var_32;
        wp::int32 var_33;
        wp::int32* var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::int32* var_37;
        wp::int32 var_38;
        wp::int32 var_39;
        bool var_40;
        bool var_41;
        const wp::int32 var_42 = -1;
        bool var_43;
        const wp::int32 var_44 = -1;
        bool var_45;
        bool var_46;
        bool var_47;
        bool var_48;
        bool var_49;
        bool var_50;
        const wp::float32 var_51 = 0.0;
        const wp::float32 var_52 = 0.0;
        wp::shape_t* var_53;
        const wp::int32 var_54 = 0;
        wp::int32 var_55;
        wp::shape_t var_56;
        const wp::int32 var_57 = 0;
        bool var_58;
        wp::float32* var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::float32* var_62;
        wp::float32 var_63;
        wp::float32 var_64;
        wp::float32 var_65;
        wp::float32 var_66;
        bool var_67;
        bool var_68;
        const wp::int32 var_69 = 0;
        bool var_70;
        wp::vec_t<2, wp::int32> var_71;
        bool var_72;
        wp::vec_t<2, wp::int32> var_73;
        //---------
        // forward
        // def _nxn_broadphase_kernel(                                                            <L 137>
        // tid = wp.tid()                                                                         <L 159>
        var_0 = builtin_tid1d();
        // world_id, local_id = _find_world_and_local_id(tid, world_cumsum_lower_tri)             <L 162>
        _find_world_and_local_id_0(var_0, var_world_cumsum_lower_tri, var_1, var_2);
        // world_slice_start = 0                                                                  <L 165>
        // if world_id > 0:                                                                       <L 166>
        var_5 = (var_1 > var_4);
        if (var_5) {
            // world_slice_start = world_slice_ends[world_id - 1]                                 <L 167>
            var_7 = wp::sub(var_1, var_6);
            var_8 = wp::address(var_world_slice_ends, var_7);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
        }
        var_11 = wp::where(var_5, var_9, var_3);
        // world_slice_end = world_slice_ends[world_id]                                           <L 168>
        var_12 = wp::address(var_world_slice_ends, var_1);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // num_shapes_in_world = world_slice_end - world_slice_start                              <L 171>
        var_15 = wp::sub(var_13, var_11);
        // local_shape1, local_shape2 = _get_lower_triangular_indices(local_id, num_shapes_in_world)       <L 174>
        _get_lower_triangular_indices_0(var_2, var_15, var_16, var_17);
        // shape1_tmp = world_index_map[world_slice_start + local_shape1]                         <L 177>
        var_18 = wp::add(var_11, var_16);
        var_19 = wp::address(var_world_index_map, var_18);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // shape2_tmp = world_index_map[world_slice_start + local_shape2]                         <L 178>
        var_22 = wp::add(var_11, var_17);
        var_23 = wp::address(var_world_index_map, var_22);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // shape1 = wp.min(shape1_tmp, shape2_tmp)                                                <L 182>
        var_26 = wp::min(var_20, var_24);
        // shape2 = wp.max(shape1_tmp, shape2_tmp)                                                <L 183>
        var_27 = wp::max(var_20, var_24);
        // world1 = shape_world[shape1]                                                           <L 186>
        var_28 = wp::address(var_shape_world, var_26);
        var_30 = wp::load(var_28);
        var_29 = wp::copy(var_30);
        // world2 = shape_world[shape2]                                                           <L 187>
        var_31 = wp::address(var_shape_world, var_27);
        var_33 = wp::load(var_31);
        var_32 = wp::copy(var_33);
        // collision_group1 = collision_group[shape1]                                             <L 188>
        var_34 = wp::address(var_collision_group, var_26);
        var_36 = wp::load(var_34);
        var_35 = wp::copy(var_36);
        // collision_group2 = collision_group[shape2]                                             <L 189>
        var_37 = wp::address(var_collision_group, var_27);
        var_39 = wp::load(var_37);
        var_38 = wp::copy(var_39);
        // is_dedicated_minus_one_segment = world_id >= num_regular_worlds                        <L 193>
        var_40 = (var_1 >= var_num_regular_worlds);
        // if world1 == -1 and world2 == -1 and not is_dedicated_minus_one_segment:               <L 194>
        var_43 = (var_29 == var_42);
        var_41 = var_43;
        if (var_41) {
            var_45 = (var_32 == var_44);
            var_41 = var_41 && var_45;
        }
        if (var_41) {
            var_46 = wp::unot(var_40);
            var_41 = var_41 && var_46;
        }
        if (var_41) {
            // return                                                                             <L 195>
            continue;
        }
        // if not test_world_and_group_pair(world1, world2, collision_group1, collision_group2):       <L 198>
        var_47 = test_world_and_group_pair_0(var_29, var_32, var_35, var_38);
        var_48 = wp::unot(var_47);
        if (var_48) {
            // return                                                                             <L 199>
            continue;
        }
        // if is_shape_pair_same_body_filtered(shape1, shape2, shape_body):                       <L 201>
        var_49 = is_shape_pair_same_body_filtered_0(var_26, var_27, var_shape_body);
        if (var_49) {
            // return                                                                             <L 202>
            continue;
        }
        // if is_shape_pair_immovable_filtered(shape1, shape2, shape_body, body_flags, include_static_kinematic_pairs):       <L 204>
        var_50 = is_shape_pair_immovable_filtered_0(var_26, var_27, var_shape_body, var_body_flags, var_include_static_kinematic_pairs);
        if (var_50) {
            // return                                                                             <L 205>
            continue;
        }
        // gap1 = 0.0                                                                             <L 208>
        // gap2 = 0.0                                                                             <L 209>
        // if shape_gap.shape[0] > 0:                                                             <L 210>
        var_53 = &(var_shape_gap.shape);
        var_56 = wp::load(var_53);
        var_55 = wp::extract(var_56, var_54);
        var_58 = (var_55 > var_57);
        if (var_58) {
            // gap1 = shape_gap[shape1]                                                           <L 211>
            var_59 = wp::address(var_shape_gap, var_26);
            var_61 = wp::load(var_59);
            var_60 = wp::copy(var_61);
            // gap2 = shape_gap[shape2]                                                           <L 212>
            var_62 = wp::address(var_shape_gap, var_27);
            var_64 = wp::load(var_62);
            var_63 = wp::copy(var_64);
        }
        var_65 = wp::where(var_58, var_60, var_51);
        var_66 = wp::where(var_58, var_63, var_52);
        // if check_aabb_overlap_moving(                                                          <L 214>
        // shape1, shape2, shape_bounding_box_lower, shape_bounding_box_upper, gap1, gap2, shape_displacement       <L 215>
        var_67 = check_aabb_overlap_moving_0(var_26, var_27, var_shape_bounding_box_lower, var_shape_bounding_box_upper, var_65, var_66, var_shape_displacement);
        if (var_67) {
            // if num_filter_pairs > 0 and is_pair_excluded(wp.vec2i(shape1, shape2), filter_pairs, num_filter_pairs):       <L 218>
            var_70 = (var_num_filter_pairs > var_69);
            var_68 = var_70;
            if (var_68) {
                var_71 = wp::vec_t<2, wp::int32>(var_26, var_27);
                var_72 = is_pair_excluded_0(var_71, var_filter_pairs, var_num_filter_pairs);
                var_68 = var_68 && var_72;
            }
            if (var_68) {
                // return                                                                         <L 219>
                continue;
            }
            // write_pair(                                                                        <L 220>
            // wp.vec2i(shape1, shape2),                                                          <L 221>
            var_73 = wp::vec_t<2, wp::int32>(var_26, var_27);
            // candidate_pair,                                                                    <L 222>
            // candidate_pair_count,                                                              <L 223>
            // max_candidate_pair,                                                                <L 224>
            write_pair_0(var_73, var_candidate_pair, var_candidate_pair_count, var_max_candidate_pair);
        }
    }
}

