#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 128
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


struct ContactData_40360d7c
{
    wp::vec_t<3, wp::float32> contact_point_center;
    wp::vec_t<3, wp::float32> contact_normal_a_to_b;
    wp::float32 contact_distance;
    wp::float32 radius_eff_a;
    wp::float32 radius_eff_b;
    wp::float32 margin_a;
    wp::float32 margin_b;
    wp::int32 shape_a;
    wp::int32 shape_b;
    wp::float32 gap_sum;
    wp::float32 contact_stiffness;
    wp::float32 contact_damping;
    wp::float32 contact_friction_scale;
    wp::int32 sort_sub_key;


    ContactData_40360d7c() = default;
    CUDA_CALLABLE ContactData_40360d7c(wp::vec_t<3, wp::float32> const& contact_point_center,
    wp::vec_t<3, wp::float32> const& contact_normal_a_to_b = {},
    wp::float32 const& contact_distance = {},
    wp::float32 const& radius_eff_a = {},
    wp::float32 const& radius_eff_b = {},
    wp::float32 const& margin_a = {},
    wp::float32 const& margin_b = {},
    wp::int32 const& shape_a = {},
    wp::int32 const& shape_b = {},
    wp::float32 const& gap_sum = {},
    wp::float32 const& contact_stiffness = {},
    wp::float32 const& contact_damping = {},
    wp::float32 const& contact_friction_scale = {},
    wp::int32 const& sort_sub_key = {})
        : contact_point_center{contact_point_center}
        , contact_normal_a_to_b{contact_normal_a_to_b}
        , contact_distance{contact_distance}
        , radius_eff_a{radius_eff_a}
        , radius_eff_b{radius_eff_b}
        , margin_a{margin_a}
        , margin_b{margin_b}
        , shape_a{shape_a}
        , shape_b{shape_b}
        , gap_sum{gap_sum}
        , contact_stiffness{contact_stiffness}
        , contact_damping{contact_damping}
        , contact_friction_scale{contact_friction_scale}
        , sort_sub_key{sort_sub_key}

    {
    }

    CUDA_CALLABLE ContactData_40360d7c& operator += (const ContactData_40360d7c& rhs)
    {    contact_point_center += rhs.contact_point_center;
    contact_normal_a_to_b += rhs.contact_normal_a_to_b;
    contact_distance += rhs.contact_distance;
    radius_eff_a += rhs.radius_eff_a;
    radius_eff_b += rhs.radius_eff_b;
    margin_a += rhs.margin_a;
    margin_b += rhs.margin_b;
    shape_a += rhs.shape_a;
    shape_b += rhs.shape_b;
    gap_sum += rhs.gap_sum;
    contact_stiffness += rhs.contact_stiffness;
    contact_damping += rhs.contact_damping;
    contact_friction_scale += rhs.contact_friction_scale;
    sort_sub_key += rhs.sort_sub_key;

        return *this;}

};

static CUDA_CALLABLE void adj_ContactData_40360d7c(wp::vec_t<3, wp::float32> const&,
    wp::vec_t<3, wp::float32> const&,
    wp::float32 const&,
    wp::float32 const&,
    wp::float32 const&,
    wp::float32 const&,
    wp::float32 const&,
    wp::int32 const&,
    wp::int32 const&,
    wp::float32 const&,
    wp::float32 const&,
    wp::float32 const&,
    wp::float32 const&,
    wp::int32 const&,
    wp::vec_t<3, wp::float32> & adj_contact_point_center,
    wp::vec_t<3, wp::float32> & adj_contact_normal_a_to_b,
    wp::float32 & adj_contact_distance,
    wp::float32 & adj_radius_eff_a,
    wp::float32 & adj_radius_eff_b,
    wp::float32 & adj_margin_a,
    wp::float32 & adj_margin_b,
    wp::int32 & adj_shape_a,
    wp::int32 & adj_shape_b,
    wp::float32 & adj_gap_sum,
    wp::float32 & adj_contact_stiffness,
    wp::float32 & adj_contact_damping,
    wp::float32 & adj_contact_friction_scale,
    wp::int32 & adj_sort_sub_key,
    ContactData_40360d7c & adj_ret)
{
    adj_contact_point_center += adj_ret.contact_point_center;
    adj_contact_normal_a_to_b += adj_ret.contact_normal_a_to_b;
    adj_contact_distance += adj_ret.contact_distance;
    adj_radius_eff_a += adj_ret.radius_eff_a;
    adj_radius_eff_b += adj_ret.radius_eff_b;
    adj_margin_a += adj_ret.margin_a;
    adj_margin_b += adj_ret.margin_b;
    adj_shape_a += adj_ret.shape_a;
    adj_shape_b += adj_ret.shape_b;
    adj_gap_sum += adj_ret.gap_sum;
    adj_contact_stiffness += adj_ret.contact_stiffness;
    adj_contact_damping += adj_ret.contact_damping;
    adj_contact_friction_scale += adj_ret.contact_friction_scale;
    adj_sort_sub_key += adj_ret.sort_sub_key;
}

// Required when compiling adjoints.
CUDA_CALLABLE ContactData_40360d7c add(const ContactData_40360d7c& a, const ContactData_40360d7c& b)
{
    return ContactData_40360d7c();
}

CUDA_CALLABLE void adj_atomic_add(ContactData_40360d7c* p, ContactData_40360d7c t)
{
    wp::adj_atomic_add(&p->contact_point_center, t.contact_point_center);
    wp::adj_atomic_add(&p->contact_normal_a_to_b, t.contact_normal_a_to_b);
    wp::adj_atomic_add(&p->contact_distance, t.contact_distance);
    wp::adj_atomic_add(&p->radius_eff_a, t.radius_eff_a);
    wp::adj_atomic_add(&p->radius_eff_b, t.radius_eff_b);
    wp::adj_atomic_add(&p->margin_a, t.margin_a);
    wp::adj_atomic_add(&p->margin_b, t.margin_b);
    wp::adj_atomic_add(&p->shape_a, t.shape_a);
    wp::adj_atomic_add(&p->shape_b, t.shape_b);
    wp::adj_atomic_add(&p->gap_sum, t.gap_sum);
    wp::adj_atomic_add(&p->contact_stiffness, t.contact_stiffness);
    wp::adj_atomic_add(&p->contact_damping, t.contact_damping);
    wp::adj_atomic_add(&p->contact_friction_scale, t.contact_friction_scale);
    wp::adj_atomic_add(&p->sort_sub_key, t.sort_sub_key);
}




struct ContactWriterData_099b2624
{
    wp::int32 contact_max;
    wp::int32 shape_index_bits;
    wp::int32 sub_key_bits;
    wp::array_t<wp::transform_t<wp::float32>> body_q;
    wp::array_t<wp::int32> shape_body;
    wp::array_t<wp::float32> shape_gap;
    wp::array_t<wp::int32> contact_count;
    wp::array_t<wp::int32> out_shape0;
    wp::array_t<wp::int32> out_shape1;
    wp::array_t<wp::vec_t<3, wp::float32>> out_point0;
    wp::array_t<wp::vec_t<3, wp::float32>> out_point1;
    wp::array_t<wp::vec_t<3, wp::float32>> out_offset0;
    wp::array_t<wp::vec_t<3, wp::float32>> out_offset1;
    wp::array_t<wp::vec_t<3, wp::float32>> out_normal;
    wp::array_t<wp::float32> out_margin0;
    wp::array_t<wp::float32> out_margin1;
    wp::array_t<wp::int32> out_tids;
    wp::array_t<wp::float32> out_stiffness;
    wp::array_t<wp::float32> out_damping;
    wp::array_t<wp::float32> out_friction;
    wp::array_t<wp::int64> out_sort_key;
    wp::array_t<wp::transform_t<wp::float32>> shape_transform;
    wp::array_t<wp::vec_t<3, wp::float32>> shape_linear_velocity;
    wp::array_t<wp::vec_t<3, wp::float32>> shape_angular_velocity;
    wp::float32 collision_update_dt;
    wp::float32 max_speculative_extension;


    ContactWriterData_099b2624() = default;
    CUDA_CALLABLE ContactWriterData_099b2624(wp::int32 const& contact_max,
    wp::int32 const& shape_index_bits = {},
    wp::int32 const& sub_key_bits = {},
    wp::array_t<wp::transform_t<wp::float32>> const& body_q = {},
    wp::array_t<wp::int32> const& shape_body = {},
    wp::array_t<wp::float32> const& shape_gap = {},
    wp::array_t<wp::int32> const& contact_count = {},
    wp::array_t<wp::int32> const& out_shape0 = {},
    wp::array_t<wp::int32> const& out_shape1 = {},
    wp::array_t<wp::vec_t<3, wp::float32>> const& out_point0 = {},
    wp::array_t<wp::vec_t<3, wp::float32>> const& out_point1 = {},
    wp::array_t<wp::vec_t<3, wp::float32>> const& out_offset0 = {},
    wp::array_t<wp::vec_t<3, wp::float32>> const& out_offset1 = {},
    wp::array_t<wp::vec_t<3, wp::float32>> const& out_normal = {},
    wp::array_t<wp::float32> const& out_margin0 = {},
    wp::array_t<wp::float32> const& out_margin1 = {},
    wp::array_t<wp::int32> const& out_tids = {},
    wp::array_t<wp::float32> const& out_stiffness = {},
    wp::array_t<wp::float32> const& out_damping = {},
    wp::array_t<wp::float32> const& out_friction = {},
    wp::array_t<wp::int64> const& out_sort_key = {},
    wp::array_t<wp::transform_t<wp::float32>> const& shape_transform = {},
    wp::array_t<wp::vec_t<3, wp::float32>> const& shape_linear_velocity = {},
    wp::array_t<wp::vec_t<3, wp::float32>> const& shape_angular_velocity = {},
    wp::float32 const& collision_update_dt = {},
    wp::float32 const& max_speculative_extension = {})
        : contact_max{contact_max}
        , shape_index_bits{shape_index_bits}
        , sub_key_bits{sub_key_bits}
        , body_q{body_q}
        , shape_body{shape_body}
        , shape_gap{shape_gap}
        , contact_count{contact_count}
        , out_shape0{out_shape0}
        , out_shape1{out_shape1}
        , out_point0{out_point0}
        , out_point1{out_point1}
        , out_offset0{out_offset0}
        , out_offset1{out_offset1}
        , out_normal{out_normal}
        , out_margin0{out_margin0}
        , out_margin1{out_margin1}
        , out_tids{out_tids}
        , out_stiffness{out_stiffness}
        , out_damping{out_damping}
        , out_friction{out_friction}
        , out_sort_key{out_sort_key}
        , shape_transform{shape_transform}
        , shape_linear_velocity{shape_linear_velocity}
        , shape_angular_velocity{shape_angular_velocity}
        , collision_update_dt{collision_update_dt}
        , max_speculative_extension{max_speculative_extension}

    {
    }

    CUDA_CALLABLE ContactWriterData_099b2624& operator += (const ContactWriterData_099b2624& rhs)
    {    contact_max += rhs.contact_max;
    shape_index_bits += rhs.shape_index_bits;
    sub_key_bits += rhs.sub_key_bits;
    collision_update_dt += rhs.collision_update_dt;
    max_speculative_extension += rhs.max_speculative_extension;

        return *this;}

};

static CUDA_CALLABLE void adj_ContactWriterData_099b2624(wp::int32 const&,
    wp::int32 const&,
    wp::int32 const&,
    wp::array_t<wp::transform_t<wp::float32>> const&,
    wp::array_t<wp::int32> const&,
    wp::array_t<wp::float32> const&,
    wp::array_t<wp::int32> const&,
    wp::array_t<wp::int32> const&,
    wp::array_t<wp::int32> const&,
    wp::array_t<wp::vec_t<3, wp::float32>> const&,
    wp::array_t<wp::vec_t<3, wp::float32>> const&,
    wp::array_t<wp::vec_t<3, wp::float32>> const&,
    wp::array_t<wp::vec_t<3, wp::float32>> const&,
    wp::array_t<wp::vec_t<3, wp::float32>> const&,
    wp::array_t<wp::float32> const&,
    wp::array_t<wp::float32> const&,
    wp::array_t<wp::int32> const&,
    wp::array_t<wp::float32> const&,
    wp::array_t<wp::float32> const&,
    wp::array_t<wp::float32> const&,
    wp::array_t<wp::int64> const&,
    wp::array_t<wp::transform_t<wp::float32>> const&,
    wp::array_t<wp::vec_t<3, wp::float32>> const&,
    wp::array_t<wp::vec_t<3, wp::float32>> const&,
    wp::float32 const&,
    wp::float32 const&,
    wp::int32 & adj_contact_max,
    wp::int32 & adj_shape_index_bits,
    wp::int32 & adj_sub_key_bits,
    wp::array_t<wp::transform_t<wp::float32>> & adj_body_q,
    wp::array_t<wp::int32> & adj_shape_body,
    wp::array_t<wp::float32> & adj_shape_gap,
    wp::array_t<wp::int32> & adj_contact_count,
    wp::array_t<wp::int32> & adj_out_shape0,
    wp::array_t<wp::int32> & adj_out_shape1,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_out_point0,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_out_point1,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_out_offset0,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_out_offset1,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_out_normal,
    wp::array_t<wp::float32> & adj_out_margin0,
    wp::array_t<wp::float32> & adj_out_margin1,
    wp::array_t<wp::int32> & adj_out_tids,
    wp::array_t<wp::float32> & adj_out_stiffness,
    wp::array_t<wp::float32> & adj_out_damping,
    wp::array_t<wp::float32> & adj_out_friction,
    wp::array_t<wp::int64> & adj_out_sort_key,
    wp::array_t<wp::transform_t<wp::float32>> & adj_shape_transform,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_shape_linear_velocity,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_shape_angular_velocity,
    wp::float32 & adj_collision_update_dt,
    wp::float32 & adj_max_speculative_extension,
    ContactWriterData_099b2624 & adj_ret)
{
    adj_contact_max += adj_ret.contact_max;
    adj_shape_index_bits += adj_ret.shape_index_bits;
    adj_sub_key_bits += adj_ret.sub_key_bits;
    adj_body_q = adj_ret.body_q;
    adj_shape_body = adj_ret.shape_body;
    adj_shape_gap = adj_ret.shape_gap;
    adj_contact_count = adj_ret.contact_count;
    adj_out_shape0 = adj_ret.out_shape0;
    adj_out_shape1 = adj_ret.out_shape1;
    adj_out_point0 = adj_ret.out_point0;
    adj_out_point1 = adj_ret.out_point1;
    adj_out_offset0 = adj_ret.out_offset0;
    adj_out_offset1 = adj_ret.out_offset1;
    adj_out_normal = adj_ret.out_normal;
    adj_out_margin0 = adj_ret.out_margin0;
    adj_out_margin1 = adj_ret.out_margin1;
    adj_out_tids = adj_ret.out_tids;
    adj_out_stiffness = adj_ret.out_stiffness;
    adj_out_damping = adj_ret.out_damping;
    adj_out_friction = adj_ret.out_friction;
    adj_out_sort_key = adj_ret.out_sort_key;
    adj_shape_transform = adj_ret.shape_transform;
    adj_shape_linear_velocity = adj_ret.shape_linear_velocity;
    adj_shape_angular_velocity = adj_ret.shape_angular_velocity;
    adj_collision_update_dt += adj_ret.collision_update_dt;
    adj_max_speculative_extension += adj_ret.max_speculative_extension;
}

// Required when compiling adjoints.
CUDA_CALLABLE ContactWriterData_099b2624 add(const ContactWriterData_099b2624& a, const ContactWriterData_099b2624& b)
{
    return ContactWriterData_099b2624();
}

CUDA_CALLABLE void adj_atomic_add(ContactWriterData_099b2624* p, ContactWriterData_099b2624 t)
{
    wp::adj_atomic_add(&p->contact_max, t.contact_max);
    wp::adj_atomic_add(&p->shape_index_bits, t.shape_index_bits);
    wp::adj_atomic_add(&p->sub_key_bits, t.sub_key_bits);
    wp::adj_atomic_add(&p->body_q, t.body_q);
    wp::adj_atomic_add(&p->shape_body, t.shape_body);
    wp::adj_atomic_add(&p->shape_gap, t.shape_gap);
    wp::adj_atomic_add(&p->contact_count, t.contact_count);
    wp::adj_atomic_add(&p->out_shape0, t.out_shape0);
    wp::adj_atomic_add(&p->out_shape1, t.out_shape1);
    wp::adj_atomic_add(&p->out_point0, t.out_point0);
    wp::adj_atomic_add(&p->out_point1, t.out_point1);
    wp::adj_atomic_add(&p->out_offset0, t.out_offset0);
    wp::adj_atomic_add(&p->out_offset1, t.out_offset1);
    wp::adj_atomic_add(&p->out_normal, t.out_normal);
    wp::adj_atomic_add(&p->out_margin0, t.out_margin0);
    wp::adj_atomic_add(&p->out_margin1, t.out_margin1);
    wp::adj_atomic_add(&p->out_tids, t.out_tids);
    wp::adj_atomic_add(&p->out_stiffness, t.out_stiffness);
    wp::adj_atomic_add(&p->out_damping, t.out_damping);
    wp::adj_atomic_add(&p->out_friction, t.out_friction);
    wp::adj_atomic_add(&p->out_sort_key, t.out_sort_key);
    wp::adj_atomic_add(&p->shape_transform, t.shape_transform);
    wp::adj_atomic_add(&p->shape_linear_velocity, t.shape_linear_velocity);
    wp::adj_atomic_add(&p->shape_angular_velocity, t.shape_angular_velocity);
    wp::adj_atomic_add(&p->collision_update_dt, t.collision_update_dt);
    wp::adj_atomic_add(&p->max_speculative_extension, t.max_speculative_extension);
}




// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/narrow_phase.py:94
static CUDA_CALLABLE wp::int32 _reserve_compacted_slot_0(
    wp::int32 var_predicate,
    wp::array_t<wp::int32> var_work_count)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    bool var_1;
    const wp::int32 var_2 = -1;
    const wp::int32 var_3 = 0;
    const wp::int32 var_4 = 1;
    wp::int32 var_5;
    //---------
    // forward
    // def _reserve_compacted_slot(predicate: int, work_count: wp.array[int]) -> int:         <L 95>
    // if predicate == 0:                                                                     <L 97>
    var_1 = (var_predicate == var_0);
    if (var_1) {
        // return -1                                                                          <L 98>
        return var_2;
    }
    // return wp.atomic_add(work_count, 0, 1)                                                 <L 99>
    var_5 = wp::atomic_add(var_work_count, var_3, var_4);
    return var_5;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/narrow_phase.py:115
static CUDA_CALLABLE void _append_pair_compacted_0(
    bool var_predicate,
    wp::vec_t<2, wp::int32> var_value,
    wp::array_t<wp::vec_t<2, wp::int32>> var_work_items,
    wp::array_t<wp::int32> var_work_count)
{
    //---------
    // primal vars
    wp::int32 var_0;
    wp::int32 var_1;
    bool var_2;
    const wp::int32 var_3 = 0;
    bool var_4;
    wp::shape_t* var_5;
    const wp::int32 var_6 = 0;
    wp::int32 var_7;
    wp::shape_t var_8;
    bool var_9;
    //---------
    // forward
    // def _append_pair_compacted(                                                            <L 116>
    // index = _reserve_compacted_slot(int(predicate), work_count)                            <L 123>
    var_0 = wp::int(var_predicate);
    var_1 = _reserve_compacted_slot_0(var_0, var_work_count);
    // if index >= 0 and index < work_items.shape[0]:                                         <L 124>
    var_4 = (var_1 >= var_3);
    var_2 = var_4;
    if (var_2) {
        var_5 = &(var_work_items.shape);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        var_9 = (var_1 < var_7);
        var_2 = var_2 && var_9;
    }
    if (var_2) {
        // work_items[index] = value                                                          <L 125>
        wp::array_store(var_work_items, var_1, var_value);
    }
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_primitive.py:100
static CUDA_CALLABLE void collide_plane_sphere_0(
    wp::vec_t<3, wp::float32> var_plane_normal,
    wp::vec_t<3, wp::float32> var_plane_pos,
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::float32 var_3 = 0.5;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    //---------
    // forward
    // def collide_plane_sphere(                                                              <L 101>
    // dist = wp.dot(sphere_pos - plane_pos, plane_normal) - sphere_radius                    <L 105>
    var_0 = wp::sub(var_sphere_pos, var_plane_pos);
    var_1 = wp::dot(var_0, var_plane_normal);
    var_2 = wp::sub(var_1, var_sphere_radius);
    // pos = sphere_pos - plane_normal * (sphere_radius + 0.5 * dist)                         <L 106>
    var_4 = wp::mul(var_3, var_2);
    var_5 = wp::add(var_sphere_radius, var_4);
    var_6 = wp::mul(var_plane_normal, var_5);
    var_7 = wp::sub(var_sphere_pos, var_6);
    // return dist, pos                                                                       <L 107>
    ret_0 = var_2;
    ret_1 = var_7;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_primitive.py:352
static CUDA_CALLABLE void collide_plane_ellipsoid_0(
    wp::vec_t<3, wp::float32> var_plane_normal,
    wp::vec_t<3, wp::float32> var_plane_pos,
    wp::vec_t<3, wp::float32> var_ellipsoid_pos,
    wp::mat_t<3, 3, wp::float32> var_ellipsoid_rot,
    wp::vec_t<3, wp::float32> var_ellipsoid_size,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::mat_t<3, 3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::float32 var_9;
    wp::vec_t<3, wp::float32> var_10;
    const wp::float32 var_11 = 0.5;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    //---------
    // forward
    // def collide_plane_ellipsoid(                                                           <L 353>
    // sphere_support = -wp.normalize(wp.cw_mul(wp.transpose(ellipsoid_rot) @ plane_normal, ellipsoid_size))       <L 376>
    var_0 = wp::transpose(var_ellipsoid_rot);
    var_1 = wp::mul(var_0, var_plane_normal);
    var_2 = wp::cw_mul(var_1, var_ellipsoid_size);
    var_3 = wp::normalize(var_2);
    var_4 = wp::neg(var_3);
    // pos = ellipsoid_pos + ellipsoid_rot @ wp.cw_mul(sphere_support, ellipsoid_size)        <L 377>
    var_5 = wp::cw_mul(var_4, var_ellipsoid_size);
    var_6 = wp::mul(var_ellipsoid_rot, var_5);
    var_7 = wp::add(var_ellipsoid_pos, var_6);
    // dist = wp.dot(plane_normal, pos - plane_pos)                                           <L 378>
    var_8 = wp::sub(var_7, var_plane_pos);
    var_9 = wp::dot(var_plane_normal, var_8);
    // pos = pos - plane_normal * dist * 0.5                                                  <L 379>
    var_10 = wp::mul(var_plane_normal, var_9);
    var_12 = wp::mul(var_10, var_11);
    var_13 = wp::sub(var_7, var_12);
    // return dist, pos, plane_normal                                                         <L 381>
    ret_0 = var_9;
    ret_1 = var_13;
    ret_2 = var_plane_normal;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_primitive.py:384
static CUDA_CALLABLE void collide_plane_box_0(
    wp::vec_t<3, wp::float32> var_plane_normal,
    wp::vec_t<3, wp::float32> var_plane_pos,
    wp::vec_t<3, wp::float32> var_box_pos,
    wp::mat_t<3, 3, wp::float32> var_box_rot,
    wp::vec_t<3, wp::float32> var_box_size,
    wp::float32 var_margin,
    wp::vec_t<4, wp::float32> & ret_0,
    wp::mat_t<4, 3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::float32 var_2;
    const wp::float32 var_3 = 10000000000.0;
    wp::vec_t<4, wp::float32> var_4;
    wp::mat_t<4, 3, wp::float32> var_5;
    const wp::int32 var_6 = 0;
    wp::int32 var_7;
    const wp::int32 var_8 = 0;
    wp::int32 var_9;
    const wp::int32 var_10 = 8;
    wp::range_t var_11;
    wp::int32 var_12;
    const wp::int32 var_13 = 1;
    wp::int32 var_14;
    const wp::int32 var_15 = 0;
    bool var_16;
    const wp::int32 var_17 = 0;
    wp::float32 var_18;
    const wp::int32 var_19 = 0;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::int32 var_23 = 0;
    const wp::int32 var_24 = 2;
    wp::int32 var_25;
    const wp::int32 var_26 = 0;
    bool var_27;
    const wp::int32 var_28 = 1;
    wp::float32 var_29;
    const wp::int32 var_30 = 1;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::int32 var_34 = 1;
    const wp::int32 var_35 = 4;
    wp::int32 var_36;
    const wp::int32 var_37 = 0;
    bool var_38;
    const wp::int32 var_39 = 2;
    wp::float32 var_40;
    const wp::int32 var_41 = 2;
    wp::float32 var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    const wp::int32 var_45 = 2;
    wp::vec_t<3, wp::float32> var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    bool var_49;
    wp::vec_t<3, wp::float32> var_50;
    wp::vec_t<3, wp::float32> var_51;
    const wp::float32 var_52 = 0.5;
    wp::vec_t<3, wp::float32> var_53;
    wp::vec_t<3, wp::float32> var_54;
    wp::vec_t<3, wp::float32> var_55;
    const wp::int32 var_56 = 4;
    bool var_57;
    bool var_58;
    const wp::int32 var_59 = 0;
    bool var_60;
    wp::float32 var_61;
    bool var_62;
    wp::int32 var_63;
    wp::int32 var_64;
    const wp::int32 var_65 = 1;
    wp::int32 var_66;
    wp::float32 var_67;
    bool var_68;
    const wp::int32 var_69 = 0;
    const wp::int32 var_70 = 1;
    wp::float32 var_71;
    wp::float32 var_72;
    bool var_73;
    const wp::int32 var_74 = 1;
    wp::int32 var_75;
    const wp::int32 var_76 = 2;
    wp::float32 var_77;
    wp::float32 var_78;
    bool var_79;
    const wp::int32 var_80 = 2;
    wp::int32 var_81;
    const wp::int32 var_82 = 3;
    wp::float32 var_83;
    wp::float32 var_84;
    bool var_85;
    const wp::int32 var_86 = 3;
    wp::int32 var_87;
    wp::int32 var_88;
    wp::int32 var_89;
    wp::int32 var_90;
    //---------
    // forward
    // def collide_plane_box(                                                                 <L 385>
    // corner = wp.vec3()                                                                     <L 411>
    var_0 = wp::vec_t<3, wp::float32>();
    // center_dist = wp.dot(box_pos - plane_pos, plane_normal)                                <L 412>
    var_1 = wp::sub(var_box_pos, var_plane_pos);
    var_2 = wp::dot(var_1, var_plane_normal);
    // dist = wp.vec4(MAXVAL)                                                                 <L 414>
    var_4 = wp::vec_t<4, wp::float32>(var_3);
    // pos = _mat43f()                                                                        <L 415>
    var_5 = wp::mat_t<4, 3, wp::float32>();
    // ncontact = wp.int32(0)                                                                 <L 419>
    var_7 = wp::int32(var_6);
    // worst_idx = wp.int32(0)                                                                <L 420>
    var_9 = wp::int32(var_8);
    // for i in range(8):                                                                     <L 421>
    var_11 = wp::range(var_10);
    start_for_0:;
        if (iter_cmp(var_11) == 0) goto end_for_0;
        var_12 = wp::iter_next(var_11);
        // corner.x = wp.where((i & 1) != 0, box_size.x, -box_size.x)                         <L 423>
        var_14 = wp::bit_and(var_12, var_13);
        var_16 = (var_14 != var_15);
        var_18 = wp::extract(var_box_size, var_17);
        var_20 = wp::extract(var_box_size, var_19);
        var_21 = wp::neg(var_20);
        var_22 = wp::where(var_16, var_18, var_21);
        wp::assign_inplace(var_0, var_23, var_22);
        // corner.y = wp.where((i & 2) != 0, box_size.y, -box_size.y)                         <L 424>
        var_25 = wp::bit_and(var_12, var_24);
        var_27 = (var_25 != var_26);
        var_29 = wp::extract(var_box_size, var_28);
        var_31 = wp::extract(var_box_size, var_30);
        var_32 = wp::neg(var_31);
        var_33 = wp::where(var_27, var_29, var_32);
        wp::assign_inplace(var_0, var_34, var_33);
        // corner.z = wp.where((i & 4) != 0, box_size.z, -box_size.z)                         <L 425>
        var_36 = wp::bit_and(var_12, var_35);
        var_38 = (var_36 != var_37);
        var_40 = wp::extract(var_box_size, var_39);
        var_42 = wp::extract(var_box_size, var_41);
        var_43 = wp::neg(var_42);
        var_44 = wp::where(var_38, var_40, var_43);
        wp::assign_inplace(var_0, var_45, var_44);
        // corner = box_rot @ corner                                                          <L 428>
        var_46 = wp::mul(var_box_rot, var_0);
        // ldist = wp.dot(plane_normal, corner)                                               <L 431>
        var_47 = wp::dot(var_plane_normal, var_46);
        // cdist = center_dist + ldist                                                        <L 432>
        var_48 = wp::add(var_2, var_47);
        // if cdist > margin:                                                                 <L 433>
        var_49 = (var_48 > var_margin);
        if (var_49) {
            // continue                                                                       <L 434>
            wp::assign(var_0, var_46);
            goto start_for_0;
        }
        var_50 = wp::where(var_49, var_0, var_46);
        // cpos = corner + box_pos - 0.5 * plane_normal * cdist                               <L 436>
        var_51 = wp::add(var_50, var_box_pos);
        var_53 = wp::mul(var_52, var_plane_normal);
        var_54 = wp::mul(var_53, var_48);
        var_55 = wp::sub(var_51, var_54);
        // if ncontact < 4:                                                                   <L 438>
        var_57 = (var_7 < var_56);
        if (var_57) {
            // dist[ncontact] = cdist                                                         <L 439>
            wp::assign_inplace(var_4, var_7, var_48);
            // pos[ncontact] = cpos                                                           <L 440>
            wp::assign_inplace(var_5, var_7, var_55);
            // if ncontact == 0 or cdist > dist[worst_idx]:                                   <L 441>
            var_60 = (var_7 == var_59);
            var_58 = var_60;
            if (!var_58) {
                var_61 = wp::extract(var_4, var_9);
                var_62 = (var_48 > var_61);
                var_58 = var_58 || var_62;
            }
            if (var_58) {
                // worst_idx = ncontact                                                       <L 442>
                var_63 = wp::copy(var_7);
            }
            var_64 = wp::where(var_58, var_63, var_9);
            // ncontact += 1                                                                  <L 443>
            var_66 = wp::add(var_7, var_65);
        }
        if (!var_57) {
            // if cdist < dist[worst_idx]:                                                    <L 445>
            var_67 = wp::extract(var_4, var_9);
            var_68 = (var_48 < var_67);
            if (var_68) {
                // dist[worst_idx] = cdist                                                    <L 446>
                wp::assign_inplace(var_4, var_9, var_48);
                // pos[worst_idx] = cpos                                                      <L 447>
                wp::assign_inplace(var_5, var_9, var_55);
                // worst_idx = 0                                                              <L 450>
                // if dist[1] > dist[worst_idx]:                                              <L 451>
                var_71 = wp::extract(var_4, var_70);
                var_72 = wp::extract(var_4, var_69);
                var_73 = (var_71 > var_72);
                if (var_73) {
                    // worst_idx = 1                                                          <L 452>
                }
                var_75 = wp::where(var_73, var_74, var_69);
                // if dist[2] > dist[worst_idx]:                                              <L 453>
                var_77 = wp::extract(var_4, var_76);
                var_78 = wp::extract(var_4, var_75);
                var_79 = (var_77 > var_78);
                if (var_79) {
                    // worst_idx = 2                                                          <L 454>
                }
                var_81 = wp::where(var_79, var_80, var_75);
                // if dist[3] > dist[worst_idx]:                                              <L 455>
                var_83 = wp::extract(var_4, var_82);
                var_84 = wp::extract(var_4, var_81);
                var_85 = (var_83 > var_84);
                if (var_85) {
                    // worst_idx = 3                                                          <L 456>
                }
                var_87 = wp::where(var_85, var_86, var_81);
            }
            var_88 = wp::where(var_68, var_87, var_9);
        }
        var_89 = wp::where(var_57, var_66, var_7);
        var_90 = wp::where(var_57, var_64, var_88);
        wp::assign(var_0, var_50);
        wp::assign(var_7, var_89);
        wp::assign(var_9, var_90);
        goto start_for_0;
    end_for_0:;
    // return dist, pos, plane_normal                                                         <L 458>
    ret_0 = var_4;
    ret_1 = var_5;
    ret_2 = var_plane_normal;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_primitive.py:110
static CUDA_CALLABLE void collide_sphere_sphere_0(
    wp::vec_t<3, wp::float32> var_pos1,
    wp::float32 var_radius1,
    wp::vec_t<3, wp::float32> var_pos2,
    wp::float32 var_radius2,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    const wp::float32 var_2 = 0.0;
    bool var_3;
    const wp::float32 var_4 = 1.0;
    const wp::float32 var_5 = 0.0;
    const wp::float32 var_6 = 0.0;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    const wp::float32 var_12 = 0.5;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    //---------
    // forward
    // def collide_sphere_sphere(                                                             <L 111>
    // dir = pos2 - pos1                                                                      <L 132>
    var_0 = wp::sub(var_pos2, var_pos1);
    // dist = wp.length(dir)                                                                  <L 133>
    var_1 = wp::length(var_0);
    // if dist == 0.0:                                                                        <L 134>
    var_3 = (var_1 == var_2);
    if (var_3) {
        // n = wp.vec3(1.0, 0.0, 0.0)                                                         <L 135>
        var_7 = wp::vec_t<3, wp::float32>(var_4, var_5, var_6);
    }
    if (!var_3) {
        // n = dir / dist                                                                     <L 137>
        var_8 = wp::div(var_0, var_1);
    }
    var_9 = wp::where(var_3, var_7, var_8);
    // dist = dist - (radius1 + radius2)                                                      <L 138>
    var_10 = wp::add(var_radius1, var_radius2);
    var_11 = wp::sub(var_1, var_10);
    // pos = pos1 + n * (radius1 + 0.5 * dist)                                                <L 139>
    var_13 = wp::mul(var_12, var_11);
    var_14 = wp::add(var_radius1, var_13);
    var_15 = wp::mul(var_9, var_14);
    var_16 = wp::add(var_pos1, var_15);
    // return dist, pos, n                                                                    <L 140>
    ret_0 = var_11;
    ret_1 = var_16;
    ret_2 = var_9;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_primitive.py:278
static CUDA_CALLABLE void _collide_plane_capsule_contacts_0(
    wp::vec_t<3, wp::float32> var_plane_normal,
    wp::vec_t<3, wp::float32> var_plane_pos,
    wp::vec_t<3, wp::float32> var_capsule_pos,
    wp::vec_t<3, wp::float32> var_capsule_axis,
    wp::float32 var_capsule_radius,
    wp::float32 var_capsule_half_length,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::float32 & ret_2,
    wp::vec_t<3, wp::float32> & ret_3)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::float32 var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    //---------
    // forward
    // def _collide_plane_capsule_contacts(                                                   <L 279>
    // segment = capsule_axis * capsule_half_length                                           <L 288>
    var_0 = wp::mul(var_capsule_axis, var_capsule_half_length);
    // dist0, pos0 = collide_plane_sphere(plane_normal, plane_pos, capsule_pos + segment, capsule_radius)       <L 290>
    var_1 = wp::add(var_capsule_pos, var_0);
    collide_plane_sphere_0(var_plane_normal, var_plane_pos, var_1, var_capsule_radius, var_2, var_3);
    // dist1, pos1 = collide_plane_sphere(plane_normal, plane_pos, capsule_pos - segment, capsule_radius)       <L 291>
    var_4 = wp::sub(var_capsule_pos, var_0);
    collide_plane_sphere_0(var_plane_normal, var_plane_pos, var_4, var_capsule_radius, var_5, var_6);
    // return dist0, pos0, dist1, pos1                                                        <L 293>
    ret_0 = var_2;
    ret_1 = var_3;
    ret_2 = var_5;
    ret_3 = var_6;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_primitive.py:534
static CUDA_CALLABLE void collide_plane_cylinder_0(
    wp::vec_t<3, wp::float32> var_plane_normal,
    wp::vec_t<3, wp::float32> var_plane_pos,
    wp::vec_t<3, wp::float32> var_cylinder_pos,
    wp::vec_t<3, wp::float32> var_cylinder_axis,
    wp::float32 var_cylinder_radius,
    wp::float32 var_cylinder_half_height,
    wp::vec_t<4, wp::float32> & ret_0,
    wp::mat_t<4, 3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 10000000000.0;
    wp::vec_t<4, wp::float32> var_1;
    wp::mat_t<4, 3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::float32 var_5;
    const wp::float32 var_6 = 0.0;
    bool var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::float32 var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::float32 var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::float32 var_17;
    const wp::float32 var_18 = 1e-10;
    bool var_19;
    const wp::float32 var_20 = 1.0;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::float32 var_25;
    const wp::float32 var_26 = 0.9238795325112867;
    bool var_27;
    wp::vec_t<3, wp::float32> var_28;
    bool var_29;
    bool var_30;
    const wp::float32 var_31 = 1.0;
    const wp::float32 var_32 = 0.0;
    const wp::float32 var_33 = 0.0;
    wp::vec_t<3, wp::float32> var_34;
    wp::float32 var_35;
    wp::float32 var_36;
    const wp::float32 var_37 = 0.9;
    bool var_38;
    const wp::float32 var_39 = 0.0;
    const wp::float32 var_40 = 1.0;
    const wp::float32 var_41 = 0.0;
    wp::vec_t<3, wp::float32> var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::float32 var_44;
    wp::vec_t<3, wp::float32> var_45;
    wp::vec_t<3, wp::float32> var_46;
    wp::vec_t<3, wp::float32> var_47;
    wp::vec_t<3, wp::float32> var_48;
    wp::vec_t<3, wp::float32> var_49;
    wp::vec_t<3, wp::float32> var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::vec_t<3, wp::float32> var_52;
    wp::float32 var_53;
    const wp::float32 var_54 = 0.5;
    wp::float32 var_55;
    wp::vec_t<3, wp::float32> var_56;
    wp::vec_t<3, wp::float32> var_57;
    const wp::int32 var_58 = 0;
    const wp::int32 var_59 = 0;
    const wp::int32 var_60 = 1;
    wp::int32 var_61;
    const wp::float32 var_62 = 0.01;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::vec_t<3, wp::float32> var_68;
    const wp::float32 var_69 = -0.5;
    wp::float32 var_70;
    const wp::float32 var_71 = 0.8660254;
    wp::float32 var_72;
    wp::vec_t<3, wp::float32> var_73;
    wp::vec_t<3, wp::float32> var_74;
    wp::float32 var_75;
    const wp::float32 var_76 = 0.5;
    wp::float32 var_77;
    wp::vec_t<3, wp::float32> var_78;
    wp::vec_t<3, wp::float32> var_79;
    bool var_80;
    const wp::int32 var_81 = 4;
    bool var_82;
    wp::vec_t<3, wp::float32> var_83;
    wp::float32 var_84;
    bool var_85;
    const wp::int32 var_86 = 1;
    wp::int32 var_87;
    wp::int32 var_88;
    wp::vec_t<3, wp::float32> var_89;
    wp::vec_t<3, wp::float32> var_90;
    wp::vec_t<3, wp::float32> var_91;
    wp::vec_t<3, wp::float32> var_92;
    wp::vec_t<3, wp::float32> var_93;
    wp::float32 var_94;
    const wp::float32 var_95 = 0.5;
    wp::float32 var_96;
    wp::vec_t<3, wp::float32> var_97;
    wp::vec_t<3, wp::float32> var_98;
    bool var_99;
    const wp::int32 var_100 = 4;
    bool var_101;
    wp::vec_t<3, wp::float32> var_102;
    wp::float32 var_103;
    bool var_104;
    const wp::int32 var_105 = 1;
    wp::int32 var_106;
    wp::int32 var_107;
    wp::vec_t<3, wp::float32> var_108;
    wp::vec_t<3, wp::float32> var_109;
    wp::vec_t<3, wp::float32> var_110;
    wp::vec_t<3, wp::float32> var_111;
    wp::vec_t<3, wp::float32> var_112;
    wp::float32 var_113;
    const wp::float32 var_114 = 0.5;
    wp::float32 var_115;
    wp::vec_t<3, wp::float32> var_116;
    wp::vec_t<3, wp::float32> var_117;
    bool var_118;
    const wp::int32 var_119 = 4;
    bool var_120;
    wp::vec_t<3, wp::float32> var_121;
    wp::float32 var_122;
    bool var_123;
    const wp::int32 var_124 = 1;
    wp::int32 var_125;
    wp::int32 var_126;
    wp::vec_t<3, wp::float32> var_127;
    wp::vec_t<3, wp::float32> var_128;
    wp::vec_t<3, wp::float32> var_129;
    wp::vec_t<3, wp::float32> var_130;
    wp::vec_t<3, wp::float32> var_131;
    wp::vec_t<3, wp::float32> var_132;
    wp::vec_t<3, wp::float32> var_133;
    wp::vec_t<3, wp::float32> var_134;
    wp::float32 var_135;
    const wp::float32 var_136 = 0.5;
    wp::float32 var_137;
    wp::vec_t<3, wp::float32> var_138;
    wp::vec_t<3, wp::float32> var_139;
    bool var_140;
    const wp::int32 var_141 = 4;
    bool var_142;
    wp::vec_t<3, wp::float32> var_143;
    wp::float32 var_144;
    bool var_145;
    const wp::int32 var_146 = 1;
    wp::int32 var_147;
    wp::int32 var_148;
    wp::vec_t<3, wp::float32> var_149;
    wp::vec_t<3, wp::float32> var_150;
    wp::float32 var_151;
    wp::vec_t<3, wp::float32> var_152;
    wp::vec_t<3, wp::float32> var_153;
    wp::float32 var_154;
    bool var_155;
    wp::vec_t<3, wp::float32> var_156;
    wp::float32 var_157;
    const wp::float32 var_158 = 0.5;
    wp::float32 var_159;
    wp::vec_t<3, wp::float32> var_160;
    wp::vec_t<3, wp::float32> var_161;
    bool var_162;
    const wp::int32 var_163 = 4;
    bool var_164;
    wp::vec_t<3, wp::float32> var_165;
    wp::float32 var_166;
    bool var_167;
    const wp::int32 var_168 = 1;
    wp::int32 var_169;
    wp::int32 var_170;
    wp::int32 var_171;
    //---------
    // forward
    // def collide_plane_cylinder(                                                            <L 535>
    // contact_dist = wp.vec4(MAXVAL)                                                         <L 568>
    var_1 = wp::vec_t<4, wp::float32>(var_0);
    // contact_pos = _mat43f()                                                                <L 569>
    var_2 = wp::mat_t<4, 3, wp::float32>();
    // n = plane_normal                                                                       <L 571>
    var_3 = wp::copy(var_plane_normal);
    // axis = cylinder_axis                                                                   <L 572>
    var_4 = wp::copy(var_cylinder_axis);
    // dot_na = wp.dot(n, axis)                                                               <L 575>
    var_5 = wp::dot(var_3, var_4);
    // if dot_na > 0.0:                                                                       <L 576>
    var_7 = (var_5 > var_6);
    if (var_7) {
        // axis = -axis                                                                       <L 577>
        var_8 = wp::neg(var_4);
        // dot_na = -dot_na                                                                   <L 578>
        var_9 = wp::neg(var_5);
    }
    var_10 = wp::where(var_7, var_8, var_4);
    var_11 = wp::where(var_7, var_9, var_5);
    // cap_center = cylinder_pos + axis * cylinder_half_height                                <L 581>
    var_12 = wp::mul(var_10, var_cylinder_half_height);
    var_13 = wp::add(var_cylinder_pos, var_12);
    // perp_align = -n + axis * dot_na                                                        <L 585>
    var_14 = wp::neg(var_3);
    var_15 = wp::mul(var_10, var_11);
    var_16 = wp::add(var_14, var_15);
    // perp_align_len_sq = wp.dot(perp_align, perp_align)                                     <L 586>
    var_17 = wp::dot(var_16, var_16);
    // has_align = perp_align_len_sq > 1e-10                                                  <L 587>
    var_19 = (var_17 > var_18);
    // if has_align:                                                                          <L 588>
    if (var_19) {
        // perp_align = perp_align * (1.0 / wp.sqrt(perp_align_len_sq))                       <L 589>
        var_21 = wp::sqrt(var_17);
        var_22 = wp::div(var_20, var_21);
        var_23 = wp::mul(var_16, var_22);
    }
    var_24 = wp::where(var_19, var_23, var_16);
    // abs_dot = -dot_na  # in [0, 1], where 1 is upright                                     <L 591>
    var_25 = wp::neg(var_11);
    // flat_mode_cos = wp.static(CYLINDER_FLAT_MODE_COS)                                      <L 592>
    // in_flat_surface_mode = abs_dot >= flat_mode_cos                                        <L 593>
    var_27 = (var_25 >= var_26);
    // perp_fixed = wp.vec3()                                                                 <L 597>
    var_28 = wp::vec_t<3, wp::float32>();
    // if in_flat_surface_mode or not has_align:                                              <L 598>
    var_29 = var_27;
    if (!var_29) {
        var_30 = wp::unot(var_19);
        var_29 = var_29 || var_30;
    }
    if (var_29) {
        // ref = wp.vec3(1.0, 0.0, 0.0)                                                       <L 599>
        var_34 = wp::vec_t<3, wp::float32>(var_31, var_32, var_33);
        // if wp.abs(wp.dot(axis, ref)) > 0.9:                                                <L 600>
        var_35 = wp::dot(var_10, var_34);
        var_36 = wp::abs(var_35);
        var_38 = (var_36 > var_37);
        if (var_38) {
            // ref = wp.vec3(0.0, 1.0, 0.0)                                                   <L 601>
            var_42 = wp::vec_t<3, wp::float32>(var_39, var_40, var_41);
        }
        var_43 = wp::where(var_38, var_42, var_34);
        // perp_fixed = ref - axis * wp.dot(axis, ref)                                        <L 602>
        var_44 = wp::dot(var_10, var_43);
        var_45 = wp::mul(var_10, var_44);
        var_46 = wp::sub(var_43, var_45);
        // perp_fixed = wp.normalize(perp_fixed)                                              <L 603>
        var_47 = wp::normalize(var_46);
    }
    var_48 = wp::where(var_29, var_47, var_28);
    // deepest_perp = wp.where(has_align, perp_align, perp_fixed)                             <L 605>
    var_49 = wp::where(var_19, var_24, var_48);
    // deepest_pt = cap_center + deepest_perp * cylinder_radius                               <L 606>
    var_50 = wp::mul(var_49, var_cylinder_radius);
    var_51 = wp::add(var_13, var_50);
    // deepest_d = wp.dot(deepest_pt - plane_pos, n)                                          <L 607>
    var_52 = wp::sub(var_51, var_plane_pos);
    var_53 = wp::dot(var_52, var_3);
    // deepest_pos = deepest_pt - n * (deepest_d * 0.5)                                       <L 608>
    var_55 = wp::mul(var_53, var_54);
    var_56 = wp::mul(var_3, var_55);
    var_57 = wp::sub(var_51, var_56);
    // contact_dist[0] = deepest_d                                                            <L 612>
    wp::assign_inplace(var_1, var_58, var_53);
    // contact_pos[0] = deepest_pos                                                           <L 613>
    wp::assign_inplace(var_2, var_59, var_57);
    // ncontact = wp.int32(1)                                                                 <L 614>
    var_61 = wp::int32(var_60);
    // merge_threshold = 0.01 * wp.max(cylinder_radius, cylinder_half_height)                 <L 615>
    var_63 = wp::max(var_cylinder_radius, var_cylinder_half_height);
    var_64 = wp::mul(var_62, var_63);
    // merge_threshold_sq = merge_threshold * merge_threshold                                 <L 616>
    var_65 = wp::mul(var_64, var_64);
    // if in_flat_surface_mode:                                                               <L 621>
    if (var_27) {
        // u_fixed = perp_fixed * cylinder_radius                                             <L 622>
        var_66 = wp::mul(var_48, var_cylinder_radius);
        // v_fixed = wp.cross(axis, perp_fixed) * cylinder_radius                             <L 623>
        var_67 = wp::cross(var_10, var_48);
        var_68 = wp::mul(var_67, var_cylinder_radius);
        // c120 = float(-0.5)                                                                 <L 626>
        var_70 = wp::float(var_69);
        // s120 = float(0.8660254)                                                            <L 627>
        var_72 = wp::float(var_71);
        // pt0 = cap_center + u_fixed                                                         <L 629>
        var_73 = wp::add(var_13, var_66);
        // d0 = wp.dot(pt0 - plane_pos, n)                                                    <L 630>
        var_74 = wp::sub(var_73, var_plane_pos);
        var_75 = wp::dot(var_74, var_3);
        // pos0 = pt0 - n * (d0 * 0.5)                                                        <L 631>
        var_77 = wp::mul(var_75, var_76);
        var_78 = wp::mul(var_3, var_77);
        var_79 = wp::sub(var_73, var_78);
        // if ncontact < 4 and wp.length_sq(pos0 - deepest_pos) > merge_threshold_sq:         <L 632>
        var_82 = (var_61 < var_81);
        var_80 = var_82;
        if (var_80) {
            var_83 = wp::sub(var_79, var_57);
            var_84 = wp::length_sq(var_83);
            var_85 = (var_84 > var_65);
            var_80 = var_80 && var_85;
        }
        if (var_80) {
            // contact_dist[ncontact] = d0                                                    <L 633>
            wp::assign_inplace(var_1, var_61, var_75);
            // contact_pos[ncontact] = pos0                                                   <L 634>
            wp::assign_inplace(var_2, var_61, var_79);
            // ncontact += 1                                                                  <L 635>
            var_87 = wp::add(var_61, var_86);
        }
        var_88 = wp::where(var_80, var_87, var_61);
        // pt1 = cap_center + c120 * u_fixed + s120 * v_fixed                                 <L 637>
        var_89 = wp::mul(var_70, var_66);
        var_90 = wp::add(var_13, var_89);
        var_91 = wp::mul(var_72, var_68);
        var_92 = wp::add(var_90, var_91);
        // d1 = wp.dot(pt1 - plane_pos, n)                                                    <L 638>
        var_93 = wp::sub(var_92, var_plane_pos);
        var_94 = wp::dot(var_93, var_3);
        // pos1 = pt1 - n * (d1 * 0.5)                                                        <L 639>
        var_96 = wp::mul(var_94, var_95);
        var_97 = wp::mul(var_3, var_96);
        var_98 = wp::sub(var_92, var_97);
        // if ncontact < 4 and wp.length_sq(pos1 - deepest_pos) > merge_threshold_sq:         <L 640>
        var_101 = (var_88 < var_100);
        var_99 = var_101;
        if (var_99) {
            var_102 = wp::sub(var_98, var_57);
            var_103 = wp::length_sq(var_102);
            var_104 = (var_103 > var_65);
            var_99 = var_99 && var_104;
        }
        if (var_99) {
            // contact_dist[ncontact] = d1                                                    <L 641>
            wp::assign_inplace(var_1, var_88, var_94);
            // contact_pos[ncontact] = pos1                                                   <L 642>
            wp::assign_inplace(var_2, var_88, var_98);
            // ncontact += 1                                                                  <L 643>
            var_106 = wp::add(var_88, var_105);
        }
        var_107 = wp::where(var_99, var_106, var_88);
        // pt2 = cap_center + c120 * u_fixed - s120 * v_fixed                                 <L 645>
        var_108 = wp::mul(var_70, var_66);
        var_109 = wp::add(var_13, var_108);
        var_110 = wp::mul(var_72, var_68);
        var_111 = wp::sub(var_109, var_110);
        // d2 = wp.dot(pt2 - plane_pos, n)                                                    <L 646>
        var_112 = wp::sub(var_111, var_plane_pos);
        var_113 = wp::dot(var_112, var_3);
        // pos2 = pt2 - n * (d2 * 0.5)                                                        <L 647>
        var_115 = wp::mul(var_113, var_114);
        var_116 = wp::mul(var_3, var_115);
        var_117 = wp::sub(var_111, var_116);
        // if ncontact < 4 and wp.length_sq(pos2 - deepest_pos) > merge_threshold_sq:         <L 648>
        var_120 = (var_107 < var_119);
        var_118 = var_120;
        if (var_118) {
            var_121 = wp::sub(var_117, var_57);
            var_122 = wp::length_sq(var_121);
            var_123 = (var_122 > var_65);
            var_118 = var_118 && var_123;
        }
        if (var_118) {
            // contact_dist[ncontact] = d2                                                    <L 649>
            wp::assign_inplace(var_1, var_107, var_113);
            // contact_pos[ncontact] = pos2                                                   <L 650>
            wp::assign_inplace(var_2, var_107, var_117);
            // ncontact += 1                                                                  <L 651>
            var_125 = wp::add(var_107, var_124);
        }
        var_126 = wp::where(var_118, var_125, var_107);
    }
    if (!var_27) {
        // perp_roll = wp.where(has_align, perp_align, perp_fixed)                            <L 655>
        var_127 = wp::where(var_19, var_24, var_48);
        // u = perp_roll * cylinder_radius                                                    <L 656>
        var_128 = wp::mul(var_127, var_cylinder_radius);
        // v = wp.cross(axis, perp_roll) * cylinder_radius                                    <L 657>
        var_129 = wp::cross(var_10, var_127);
        var_130 = wp::mul(var_129, var_cylinder_radius);
        // pt = cylinder_pos - axis * cylinder_half_height + u                                <L 661>
        var_131 = wp::mul(var_10, var_cylinder_half_height);
        var_132 = wp::sub(var_cylinder_pos, var_131);
        var_133 = wp::add(var_132, var_128);
        // d = wp.dot(pt - plane_pos, n)                                                      <L 662>
        var_134 = wp::sub(var_133, var_plane_pos);
        var_135 = wp::dot(var_134, var_3);
        // pos = pt - n * (d * 0.5)                                                           <L 663>
        var_137 = wp::mul(var_135, var_136);
        var_138 = wp::mul(var_3, var_137);
        var_139 = wp::sub(var_133, var_138);
        // if ncontact < 4 and wp.length_sq(pos - deepest_pos) > merge_threshold_sq:          <L 664>
        var_142 = (var_61 < var_141);
        var_140 = var_142;
        if (var_140) {
            var_143 = wp::sub(var_139, var_57);
            var_144 = wp::length_sq(var_143);
            var_145 = (var_144 > var_65);
            var_140 = var_140 && var_145;
        }
        if (var_140) {
            // contact_dist[ncontact] = d                                                     <L 665>
            wp::assign_inplace(var_1, var_61, var_135);
            // contact_pos[ncontact] = pos                                                    <L 666>
            wp::assign_inplace(var_2, var_61, var_139);
            // ncontact += 1                                                                  <L 667>
            var_147 = wp::add(var_61, var_146);
        }
        var_148 = wp::where(var_140, var_147, var_61);
        // pt_pos_v = cap_center + v                                                          <L 670>
        var_149 = wp::add(var_13, var_130);
        // d_pos_v = wp.dot(pt_pos_v - plane_pos, n)                                          <L 671>
        var_150 = wp::sub(var_149, var_plane_pos);
        var_151 = wp::dot(var_150, var_3);
        // pt_neg_v = cap_center - v                                                          <L 672>
        var_152 = wp::sub(var_13, var_130);
        // d_neg_v = wp.dot(pt_neg_v - plane_pos, n)                                          <L 673>
        var_153 = wp::sub(var_152, var_plane_pos);
        var_154 = wp::dot(var_153, var_3);
        // use_pos_v = d_pos_v <= d_neg_v                                                     <L 674>
        var_155 = (var_151 <= var_154);
        // pt = wp.where(use_pos_v, pt_pos_v, pt_neg_v)                                       <L 675>
        var_156 = wp::where(var_155, var_149, var_152);
        // d = wp.where(use_pos_v, d_pos_v, d_neg_v)                                          <L 676>
        var_157 = wp::where(var_155, var_151, var_154);
        // pos = pt - n * (d * 0.5)                                                           <L 677>
        var_159 = wp::mul(var_157, var_158);
        var_160 = wp::mul(var_3, var_159);
        var_161 = wp::sub(var_156, var_160);
        // if ncontact < 4 and wp.length_sq(pos - deepest_pos) > merge_threshold_sq:          <L 678>
        var_164 = (var_148 < var_163);
        var_162 = var_164;
        if (var_162) {
            var_165 = wp::sub(var_161, var_57);
            var_166 = wp::length_sq(var_165);
            var_167 = (var_166 > var_65);
            var_162 = var_162 && var_167;
        }
        if (var_162) {
            // contact_dist[ncontact] = d                                                     <L 679>
            wp::assign_inplace(var_1, var_148, var_157);
            // contact_pos[ncontact] = pos                                                    <L 680>
            wp::assign_inplace(var_2, var_148, var_161);
            // ncontact += 1                                                                  <L 681>
            var_169 = wp::add(var_148, var_168);
        }
        var_170 = wp::where(var_162, var_169, var_148);
    }
    var_171 = wp::where(var_27, var_126, var_170);
    // return contact_dist, contact_pos, n                                                    <L 683>
    ret_0 = var_1;
    ret_1 = var_2;
    ret_2 = var_3;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_primitive.py:48
static CUDA_CALLABLE wp::vec_t<3, wp::float32> closest_segment_point_0(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_b,
    wp::vec_t<3, wp::float32> var_pt)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    const wp::float32 var_4 = 1e-06;
    wp::float32 var_5;
    wp::float32 var_6;
    const wp::float32 var_7 = 0.0;
    const wp::float32 var_8 = 1.0;
    wp::float32 var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    //---------
    // forward
    // def closest_segment_point(a: wp.vec3, b: wp.vec3, pt: wp.vec3) -> wp.vec3:             <L 49>
    // ab = b - a                                                                             <L 51>
    var_0 = wp::sub(var_b, var_a);
    // t = wp.dot(pt - a, ab) / (wp.dot(ab, ab) + 1e-6)                                       <L 52>
    var_1 = wp::sub(var_pt, var_a);
    var_2 = wp::dot(var_1, var_0);
    var_3 = wp::dot(var_0, var_0);
    var_5 = wp::add(var_3, var_4);
    var_6 = wp::div(var_2, var_5);
    // return a + wp.clamp(t, 0.0, 1.0) * ab                                                  <L 53>
    var_9 = wp::clamp(var_6, var_7, var_8);
    var_10 = wp::mul(var_9, var_0);
    var_11 = wp::add(var_a, var_10);
    return var_11;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_primitive.py:143
static CUDA_CALLABLE void collide_sphere_capsule_0(
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::vec_t<3, wp::float32> var_capsule_pos,
    wp::vec_t<3, wp::float32> var_capsule_axis,
    wp::float32 var_capsule_radius,
    wp::float32 var_capsule_half_length,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::float32 var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    //---------
    // forward
    // def collide_sphere_capsule(                                                            <L 144>
    // segment = capsule_axis * capsule_half_length                                           <L 171>
    var_0 = wp::mul(var_capsule_axis, var_capsule_half_length);
    // pt = closest_segment_point(capsule_pos - segment, capsule_pos + segment, sphere_pos)       <L 174>
    var_1 = wp::sub(var_capsule_pos, var_0);
    var_2 = wp::add(var_capsule_pos, var_0);
    var_3 = closest_segment_point_0(var_1, var_2, var_sphere_pos);
    // return collide_sphere_sphere(sphere_pos, sphere_radius, pt, capsule_radius)            <L 177>
    collide_sphere_sphere_0(var_sphere_pos, var_sphere_radius, var_3, var_capsule_radius, var_4, var_5, var_6);
    ret_0 = var_4;
    ret_1 = var_5;
    ret_2 = var_6;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_primitive.py:180
static CUDA_CALLABLE void collide_capsule_capsule_0(
    wp::vec_t<3, wp::float32> var_cap1_pos,
    wp::vec_t<3, wp::float32> var_cap1_axis,
    wp::float32 var_cap1_radius,
    wp::float32 var_cap1_half_length,
    wp::vec_t<3, wp::float32> var_cap2_pos,
    wp::vec_t<3, wp::float32> var_cap2_axis,
    wp::float32 var_cap2_radius,
    wp::float32 var_cap2_half_length,
    wp::vec_t<2, wp::float32> & ret_0,
    wp::mat_t<2, 3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 10000000000.0;
    wp::vec_t<2, wp::float32> var_1;
    wp::mat_t<2, 3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    const wp::float32 var_18 = 1e-15;
    bool var_19;
    const wp::float32 var_20 = 1.0;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    const wp::float32 var_30 = 1.0;
    bool var_31;
    const wp::float32 var_32 = 1.0;
    wp::float32 var_33;
    wp::float32 var_34;
    const wp::float32 var_35 = -1.0;
    bool var_36;
    const wp::float32 var_37 = -1.0;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    const wp::float32 var_44 = 1.0;
    bool var_45;
    const wp::float32 var_46 = 1.0;
    wp::float32 var_47;
    wp::float32 var_48;
    const wp::float32 var_49 = -1.0;
    const wp::float32 var_50 = 1.0;
    wp::float32 var_51;
    const wp::float32 var_52 = -1.0;
    bool var_53;
    const wp::float32 var_54 = -1.0;
    wp::float32 var_55;
    wp::float32 var_56;
    const wp::float32 var_57 = -1.0;
    const wp::float32 var_58 = 1.0;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::vec_t<3, wp::float32> var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::float32 var_68;
    wp::vec_t<3, wp::float32> var_69;
    wp::vec_t<3, wp::float32> var_70;
    const wp::int32 var_71 = 0;
    const wp::int32 var_72 = 0;
    wp::vec_t<3, wp::float32> var_73;
    wp::vec_t<3, wp::float32> var_74;
    wp::float32 var_75;
    wp::float32 var_76;
    const wp::float32 var_77 = -1.0;
    const wp::float32 var_78 = 1.0;
    wp::float32 var_79;
    wp::vec_t<3, wp::float32> var_80;
    wp::vec_t<3, wp::float32> var_81;
    wp::float32 var_82;
    wp::vec_t<3, wp::float32> var_83;
    wp::vec_t<3, wp::float32> var_84;
    const wp::int32 var_85 = 0;
    const wp::int32 var_86 = 0;
    wp::vec_t<3, wp::float32> var_87;
    wp::vec_t<3, wp::float32> var_88;
    wp::float32 var_89;
    wp::float32 var_90;
    const wp::float32 var_91 = -1.0;
    const wp::float32 var_92 = 1.0;
    wp::float32 var_93;
    wp::vec_t<3, wp::float32> var_94;
    wp::vec_t<3, wp::float32> var_95;
    wp::float32 var_96;
    wp::vec_t<3, wp::float32> var_97;
    wp::vec_t<3, wp::float32> var_98;
    const wp::int32 var_99 = 1;
    const wp::int32 var_100 = 1;
    wp::vec_t<3, wp::float32> var_101;
    wp::float32 var_102;
    wp::vec_t<3, wp::float32> var_103;
    wp::vec_t<3, wp::float32> var_104;
    wp::float32 var_105;
    wp::vec_t<3, wp::float32> var_106;
    wp::vec_t<3, wp::float32> var_107;
    //---------
    // forward
    // def collide_capsule_capsule(                                                           <L 181>
    // contact_dist = wp.vec2(MAXVAL, MAXVAL)                                                 <L 210>
    var_1 = wp::vec_t<2, wp::float32>(var_0, var_0);
    // contact_pos = _mat23f()                                                                <L 211>
    var_2 = wp::mat_t<2, 3, wp::float32>();
    // contact_normal = wp.vec3()                                                             <L 212>
    var_3 = wp::vec_t<3, wp::float32>();
    // axis1 = cap1_axis * cap1_half_length                                                   <L 215>
    var_4 = wp::mul(var_cap1_axis, var_cap1_half_length);
    // axis2 = cap2_axis * cap2_half_length                                                   <L 216>
    var_5 = wp::mul(var_cap2_axis, var_cap2_half_length);
    // dif = cap1_pos - cap2_pos                                                              <L 217>
    var_6 = wp::sub(var_cap1_pos, var_cap2_pos);
    // ma = wp.dot(axis1, axis1)                                                              <L 220>
    var_7 = wp::dot(var_4, var_4);
    // mb = -wp.dot(axis1, axis2)                                                             <L 221>
    var_8 = wp::dot(var_4, var_5);
    var_9 = wp::neg(var_8);
    // mc = wp.dot(axis2, axis2)                                                              <L 222>
    var_10 = wp::dot(var_5, var_5);
    // u = -wp.dot(axis1, dif)                                                                <L 223>
    var_11 = wp::dot(var_4, var_6);
    var_12 = wp::neg(var_11);
    // v = wp.dot(axis2, dif)                                                                 <L 224>
    var_13 = wp::dot(var_5, var_6);
    // det = ma * mc - mb * mb                                                                <L 225>
    var_14 = wp::mul(var_7, var_10);
    var_15 = wp::mul(var_9, var_9);
    var_16 = wp::sub(var_14, var_15);
    // if wp.abs(det) >= MINVAL:                                                              <L 228>
    var_17 = wp::abs(var_16);
    var_19 = (var_17 >= var_18);
    if (var_19) {
        // inv_det = 1.0 / det                                                                <L 229>
        var_21 = wp::div(var_20, var_16);
        // x1 = (mc * u - mb * v) * inv_det                                                   <L 230>
        var_22 = wp::mul(var_10, var_12);
        var_23 = wp::mul(var_9, var_13);
        var_24 = wp::sub(var_22, var_23);
        var_25 = wp::mul(var_24, var_21);
        // x2 = (ma * v - mb * u) * inv_det                                                   <L 231>
        var_26 = wp::mul(var_7, var_13);
        var_27 = wp::mul(var_9, var_12);
        var_28 = wp::sub(var_26, var_27);
        var_29 = wp::mul(var_28, var_21);
        // if x1 > 1.0:                                                                       <L 233>
        var_31 = (var_25 > var_30);
        if (var_31) {
            // x1 = 1.0                                                                       <L 234>
            // x2 = (v - mb) / mc                                                             <L 235>
            var_33 = wp::sub(var_13, var_9);
            var_34 = wp::div(var_33, var_10);
        }
        if (!var_31) {
            // elif x1 < -1.0:                                                                <L 236>
            var_36 = (var_25 < var_35);
            if (var_36) {
                // x1 = -1.0                                                                  <L 237>
                // x2 = (v + mb) / mc                                                         <L 238>
                var_38 = wp::add(var_13, var_9);
                var_39 = wp::div(var_38, var_10);
            }
            var_40 = wp::where(var_36, var_37, var_25);
            var_41 = wp::where(var_36, var_39, var_29);
        }
        var_42 = wp::where(var_31, var_32, var_40);
        var_43 = wp::where(var_31, var_34, var_41);
        // if x2 > 1.0:                                                                       <L 240>
        var_45 = (var_43 > var_44);
        if (var_45) {
            // x2 = 1.0                                                                       <L 241>
            // x1 = wp.clamp((u - mb) / ma, -1.0, 1.0)                                        <L 242>
            var_47 = wp::sub(var_12, var_9);
            var_48 = wp::div(var_47, var_7);
            var_51 = wp::clamp(var_48, var_49, var_50);
        }
        if (!var_45) {
            // elif x2 < -1.0:                                                                <L 243>
            var_53 = (var_43 < var_52);
            if (var_53) {
                // x2 = -1.0                                                                  <L 244>
                // x1 = wp.clamp((u + mb) / ma, -1.0, 1.0)                                    <L 245>
                var_55 = wp::add(var_12, var_9);
                var_56 = wp::div(var_55, var_7);
                var_59 = wp::clamp(var_56, var_57, var_58);
            }
            var_60 = wp::where(var_53, var_59, var_42);
            var_61 = wp::where(var_53, var_54, var_43);
        }
        var_62 = wp::where(var_45, var_51, var_60);
        var_63 = wp::where(var_45, var_46, var_61);
        // vec1 = cap1_pos + axis1 * x1                                                       <L 248>
        var_64 = wp::mul(var_4, var_62);
        var_65 = wp::add(var_cap1_pos, var_64);
        // vec2 = cap2_pos + axis2 * x2                                                       <L 249>
        var_66 = wp::mul(var_5, var_63);
        var_67 = wp::add(var_cap2_pos, var_66);
        // dist, pos, normal = collide_sphere_sphere(vec1, cap1_radius, vec2, cap2_radius)       <L 251>
        collide_sphere_sphere_0(var_65, var_cap1_radius, var_67, var_cap2_radius, var_68, var_69, var_70);
        // contact_dist[0] = dist                                                             <L 252>
        wp::assign_inplace(var_1, var_71, var_68);
        // contact_pos[0] = pos                                                               <L 253>
        wp::assign_inplace(var_2, var_72, var_69);
        // contact_normal = normal                                                            <L 254>
        var_73 = wp::copy(var_70);
    }
    if (!var_19) {
        // vec1 = cap1_pos + axis1                                                            <L 259>
        var_74 = wp::add(var_cap1_pos, var_4);
        // x2 = wp.clamp((v - mb) / mc, -1.0, 1.0)                                            <L 260>
        var_75 = wp::sub(var_13, var_9);
        var_76 = wp::div(var_75, var_10);
        var_79 = wp::clamp(var_76, var_77, var_78);
        // vec2 = cap2_pos + axis2 * x2                                                       <L 261>
        var_80 = wp::mul(var_5, var_79);
        var_81 = wp::add(var_cap2_pos, var_80);
        // dist, pos, normal = collide_sphere_sphere(vec1, cap1_radius, vec2, cap2_radius)       <L 262>
        collide_sphere_sphere_0(var_74, var_cap1_radius, var_81, var_cap2_radius, var_82, var_83, var_84);
        // contact_dist[0] = dist                                                             <L 263>
        wp::assign_inplace(var_1, var_85, var_82);
        // contact_pos[0] = pos                                                               <L 264>
        wp::assign_inplace(var_2, var_86, var_83);
        // contact_normal = normal  # Use first contact's normal for both                     <L 265>
        var_87 = wp::copy(var_84);
        // vec1 = cap1_pos - axis1                                                            <L 268>
        var_88 = wp::sub(var_cap1_pos, var_4);
        // x2 = wp.clamp((v + mb) / mc, -1.0, 1.0)                                            <L 269>
        var_89 = wp::add(var_13, var_9);
        var_90 = wp::div(var_89, var_10);
        var_93 = wp::clamp(var_90, var_91, var_92);
        // vec2 = cap2_pos + axis2 * x2                                                       <L 270>
        var_94 = wp::mul(var_5, var_93);
        var_95 = wp::add(var_cap2_pos, var_94);
        // dist, pos, _normal = collide_sphere_sphere(vec1, cap1_radius, vec2, cap2_radius)       <L 271>
        collide_sphere_sphere_0(var_88, var_cap1_radius, var_95, var_cap2_radius, var_96, var_97, var_98);
        // contact_dist[1] = dist                                                             <L 272>
        wp::assign_inplace(var_1, var_99, var_96);
        // contact_pos[1] = pos                                                               <L 273>
        wp::assign_inplace(var_2, var_100, var_97);
    }
    var_101 = wp::where(var_19, var_73, var_87);
    var_102 = wp::where(var_19, var_63, var_93);
    var_103 = wp::where(var_19, var_65, var_88);
    var_104 = wp::where(var_19, var_67, var_95);
    var_105 = wp::where(var_19, var_68, var_96);
    var_106 = wp::where(var_19, var_69, var_97);
    var_107 = wp::where(var_19, var_70, var_84);
    // return contact_dist, contact_pos, contact_normal                                       <L 275>
    ret_0 = var_1;
    ret_1 = var_2;
    ret_2 = var_101;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/math/__init__.py:0
static CUDA_CALLABLE wp::float32 safe_div_0(
    wp::float32 var_x,
    wp::float32 var_y,
    wp::float32 var_eps)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    bool var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    //---------
    // forward
    // def safe_div(x: Any, y: Any, eps: float = EPSILON) -> Any:                             <L 1>
    // return x / wp.where(y != 0.0, y, eps)                                                  <L 12>
    var_1 = (var_y != var_0);
    var_2 = wp::where(var_1, var_y, var_eps);
    var_3 = wp::div(var_x, var_2);
    return var_3;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_primitive.py:461
static CUDA_CALLABLE void collide_sphere_cylinder_0(
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::vec_t<3, wp::float32> var_cylinder_pos,
    wp::vec_t<3, wp::float32> var_cylinder_axis,
    wp::float32 var_cylinder_radius,
    wp::float32 var_cylinder_half_height,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    bool var_6;
    wp::float32 var_7;
    bool var_8;
    bool var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    bool var_14;
    const bool var_15 = false;
    const bool var_16 = false;
    bool var_17;
    bool var_18;
    bool var_19;
    bool var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::float32 var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    const wp::float32 var_25 = 0.0;
    bool var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::float32 var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    const wp::float32 var_38 = 1.0;
    wp::float32 var_39;
    wp::float32 var_40;
    const wp::float32 var_41 = 1e-15;
    wp::float32 var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::vec_t<3, wp::float32> var_46;
    wp::vec_t<3, wp::float32> var_47;
    wp::vec_t<3, wp::float32> var_48;
    const wp::float32 var_49 = 0.0;
    wp::float32 var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::vec_t<3, wp::float32> var_52;
    wp::vec_t<3, wp::float32> var_53;
    wp::vec_t<3, wp::float32> var_54;
    //---------
    // forward
    // def collide_sphere_cylinder(                                                           <L 462>
    // vec = sphere_pos - cylinder_pos                                                        <L 487>
    var_0 = wp::sub(var_sphere_pos, var_cylinder_pos);
    // x = wp.dot(vec, cylinder_axis)                                                         <L 488>
    var_1 = wp::dot(var_0, var_cylinder_axis);
    // a_proj = cylinder_axis * x                                                             <L 490>
    var_2 = wp::mul(var_cylinder_axis, var_1);
    // p_proj = vec - a_proj                                                                  <L 491>
    var_3 = wp::sub(var_0, var_2);
    // p_proj_sqr = wp.dot(p_proj, p_proj)                                                    <L 492>
    var_4 = wp::dot(var_3, var_3);
    // collide_side = wp.abs(x) < cylinder_half_height                                        <L 494>
    var_5 = wp::abs(var_1);
    var_6 = (var_5 < var_cylinder_half_height);
    // collide_cap = p_proj_sqr < (cylinder_radius * cylinder_radius)                         <L 495>
    var_7 = wp::mul(var_cylinder_radius, var_cylinder_radius);
    var_8 = (var_4 < var_7);
    // if collide_side and collide_cap:                                                       <L 497>
    var_9 = var_6;
    if (var_9) {
        var_9 = var_9 && var_8;
    }
    if (var_9) {
        // dist_cap = cylinder_half_height - wp.abs(x)                                        <L 498>
        var_10 = wp::abs(var_1);
        var_11 = wp::sub(var_cylinder_half_height, var_10);
        // dist_radius = cylinder_radius - wp.sqrt(p_proj_sqr)                                <L 499>
        var_12 = wp::sqrt(var_4);
        var_13 = wp::sub(var_cylinder_radius, var_12);
        // if dist_cap < dist_radius:                                                         <L 501>
        var_14 = (var_11 < var_13);
        if (var_14) {
            // collide_side = False                                                           <L 502>
        }
        if (!var_14) {
            // collide_cap = False                                                            <L 504>
        }
        var_17 = wp::where(var_14, var_15, var_6);
        var_18 = wp::where(var_14, var_8, var_16);
    }
    var_19 = wp::where(var_9, var_17, var_6);
    var_20 = wp::where(var_9, var_18, var_8);
    // if collide_side:                                                                       <L 507>
    if (var_19) {
        // pos_target = cylinder_pos + a_proj                                                 <L 508>
        var_21 = wp::add(var_cylinder_pos, var_2);
        // return collide_sphere_sphere(sphere_pos, sphere_radius, pos_target, cylinder_radius)       <L 509>
        collide_sphere_sphere_0(var_sphere_pos, var_sphere_radius, var_21, var_cylinder_radius, var_22, var_23, var_24);
        ret_0 = var_22;
        ret_1 = var_23;
        ret_2 = var_24;
        return;
    }
    if (!var_19) {
        // elif collide_cap:                                                                  <L 511>
        if (var_20) {
            // if x > 0.0:                                                                    <L 512>
            var_26 = (var_1 > var_25);
            if (var_26) {
                // pos_cap = cylinder_pos + cylinder_axis * cylinder_half_height              <L 514>
                var_27 = wp::mul(var_cylinder_axis, var_cylinder_half_height);
                var_28 = wp::add(var_cylinder_pos, var_27);
                // plane_normal = cylinder_axis                                               <L 515>
                var_29 = wp::copy(var_cylinder_axis);
            }
            if (!var_26) {
                // pos_cap = cylinder_pos - cylinder_axis * cylinder_half_height              <L 518>
                var_30 = wp::mul(var_cylinder_axis, var_cylinder_half_height);
                var_31 = wp::sub(var_cylinder_pos, var_30);
                // plane_normal = -cylinder_axis                                              <L 519>
                var_32 = wp::neg(var_cylinder_axis);
            }
            var_33 = wp::where(var_26, var_28, var_31);
            var_34 = wp::where(var_26, var_29, var_32);
            // dist, pos = collide_plane_sphere(plane_normal, pos_cap, sphere_pos, sphere_radius)       <L 521>
            collide_plane_sphere_0(var_34, var_33, var_sphere_pos, var_sphere_radius, var_35, var_36);
            // return dist, pos, -plane_normal  # flip normal after position calculation       <L 522>
            var_37 = wp::neg(var_34);
            ret_0 = var_35;
            ret_1 = var_36;
            ret_2 = var_37;
            return;
        }
        if (!var_20) {
            // inv_len = safe_div(1.0, wp.sqrt(p_proj_sqr))                                   <L 525>
            var_39 = wp::sqrt(var_4);
            var_40 = safe_div_0(var_38, var_39, var_41);
            // p_proj = p_proj * (cylinder_radius * inv_len)                                  <L 526>
            var_42 = wp::mul(var_cylinder_radius, var_40);
            var_43 = wp::mul(var_3, var_42);
            // cap_offset = cylinder_axis * (wp.sign(x) * cylinder_half_height)               <L 528>
            var_44 = wp::sign(var_1);
            var_45 = wp::mul(var_44, var_cylinder_half_height);
            var_46 = wp::mul(var_cylinder_axis, var_45);
            // pos_corner = cylinder_pos + cap_offset + p_proj                                <L 529>
            var_47 = wp::add(var_cylinder_pos, var_46);
            var_48 = wp::add(var_47, var_43);
            // return collide_sphere_sphere(sphere_pos, sphere_radius, pos_corner, 0.0)       <L 531>
            collide_sphere_sphere_0(var_sphere_pos, var_sphere_radius, var_48, var_49, var_50, var_51, var_52);
            ret_0 = var_50;
            ret_1 = var_51;
            ret_2 = var_52;
            return;
        }
        var_53 = wp::where(var_20, var_3, var_43);
    }
    var_54 = wp::where(var_19, var_3, var_53);
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/math/__init__.py:0
static CUDA_CALLABLE void normalize_with_norm_0(
    wp::vec_t<3, wp::float32> var_x,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::float32 & ret_1)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 0.0;
    bool var_2;
    const wp::float32 var_3 = 0.0;
    wp::vec_t<3, wp::float32> var_4;
    //---------
    // forward
    // def normalize_with_norm(x: Any):                                                       <L 1>
    // norm = wp.length(x)                                                                    <L 13>
    var_0 = wp::length(var_x);
    // if norm == 0.0:                                                                        <L 14>
    var_2 = (var_0 == var_1);
    if (var_2) {
        // return x, 0.0                                                                      <L 15>
        ret_0 = var_x;
        ret_1 = var_3;
        return;
    }
    // return x / norm, norm                                                                  <L 16>
    var_4 = wp::div(var_x, var_0);
    ret_0 = var_4;
    ret_1 = var_0;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_primitive.py:1176
static CUDA_CALLABLE void collide_sphere_box_0(
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::vec_t<3, wp::float32> var_box_pos,
    wp::mat_t<3, 3, wp::float32> var_box_rot,
    wp::vec_t<3, wp::float32> var_box_size,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::mat_t<3, 3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::float32 var_8;
    const wp::float32 var_9 = 1e-06;
    bool var_10;
    const wp::float32 var_11 = 2.0;
    const wp::int32 var_12 = 0;
    wp::float32 var_13;
    const wp::int32 var_14 = 1;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 2;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    const wp::int32 var_21 = 0;
    wp::int32 var_22;
    const wp::int32 var_23 = 0;
    const wp::int32 var_24 = 2;
    wp::int32 var_25;
    const wp::float32 var_26 = 1.0;
    const wp::float32 var_27 = -1.0;
    wp::float32 var_28;
    const wp::int32 var_29 = 2;
    wp::int32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    const wp::int32 var_33 = 2;
    wp::int32 var_34;
    wp::float32 var_35;
    wp::float32 var_36;
    wp::float32 var_37;
    bool var_38;
    wp::float32 var_39;
    wp::int32 var_40;
    wp::float32 var_41;
    wp::int32 var_42;
    const wp::int32 var_43 = 1;
    const wp::int32 var_44 = 2;
    wp::int32 var_45;
    const wp::float32 var_46 = 1.0;
    const wp::float32 var_47 = -1.0;
    wp::float32 var_48;
    const wp::int32 var_49 = 2;
    wp::int32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    const wp::int32 var_53 = 2;
    wp::int32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    bool var_58;
    wp::float32 var_59;
    wp::int32 var_60;
    wp::float32 var_61;
    wp::int32 var_62;
    const wp::int32 var_63 = 2;
    const wp::int32 var_64 = 2;
    wp::int32 var_65;
    const wp::float32 var_66 = 1.0;
    const wp::float32 var_67 = -1.0;
    wp::float32 var_68;
    const wp::int32 var_69 = 2;
    wp::int32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    const wp::int32 var_73 = 2;
    wp::int32 var_74;
    wp::float32 var_75;
    wp::float32 var_76;
    wp::float32 var_77;
    bool var_78;
    wp::float32 var_79;
    wp::int32 var_80;
    wp::float32 var_81;
    wp::int32 var_82;
    const wp::int32 var_83 = 3;
    const wp::int32 var_84 = 2;
    wp::int32 var_85;
    const wp::float32 var_86 = 1.0;
    const wp::float32 var_87 = -1.0;
    wp::float32 var_88;
    const wp::int32 var_89 = 2;
    wp::int32 var_90;
    wp::float32 var_91;
    wp::float32 var_92;
    const wp::int32 var_93 = 2;
    wp::int32 var_94;
    wp::float32 var_95;
    wp::float32 var_96;
    wp::float32 var_97;
    bool var_98;
    wp::float32 var_99;
    wp::int32 var_100;
    wp::float32 var_101;
    wp::int32 var_102;
    const wp::int32 var_103 = 4;
    const wp::int32 var_104 = 2;
    wp::int32 var_105;
    const wp::float32 var_106 = 1.0;
    const wp::float32 var_107 = -1.0;
    wp::float32 var_108;
    const wp::int32 var_109 = 2;
    wp::int32 var_110;
    wp::float32 var_111;
    wp::float32 var_112;
    const wp::int32 var_113 = 2;
    wp::int32 var_114;
    wp::float32 var_115;
    wp::float32 var_116;
    wp::float32 var_117;
    bool var_118;
    wp::float32 var_119;
    wp::int32 var_120;
    wp::float32 var_121;
    wp::int32 var_122;
    const wp::int32 var_123 = 5;
    const wp::int32 var_124 = 2;
    wp::int32 var_125;
    const wp::float32 var_126 = 1.0;
    const wp::float32 var_127 = -1.0;
    wp::float32 var_128;
    const wp::int32 var_129 = 2;
    wp::int32 var_130;
    wp::float32 var_131;
    wp::float32 var_132;
    const wp::int32 var_133 = 2;
    wp::int32 var_134;
    wp::float32 var_135;
    wp::float32 var_136;
    wp::float32 var_137;
    bool var_138;
    wp::float32 var_139;
    wp::int32 var_140;
    wp::float32 var_141;
    wp::int32 var_142;
    const wp::float32 var_143 = 0.0;
    wp::vec_t<3, wp::float32> var_144;
    const wp::int32 var_145 = 2;
    wp::int32 var_146;
    const wp::float32 var_147 = -1.0;
    const wp::float32 var_148 = 1.0;
    wp::float32 var_149;
    const wp::int32 var_150 = 2;
    wp::int32 var_151;
    wp::float32 var_152;
    wp::vec_t<3, wp::float32> var_153;
    const wp::float32 var_154 = 2.0;
    wp::vec_t<3, wp::float32> var_155;
    wp::vec_t<3, wp::float32> var_156;
    wp::vec_t<3, wp::float32> var_157;
    wp::float32 var_158;
    wp::float32 var_159;
    wp::vec_t<3, wp::float32> var_160;
    wp::vec_t<3, wp::float32> var_161;
    const wp::float32 var_162 = 0.5;
    wp::vec_t<3, wp::float32> var_163;
    wp::vec_t<3, wp::float32> var_164;
    wp::vec_t<3, wp::float32> var_165;
    wp::float32 var_166;
    wp::vec_t<3, wp::float32> var_167;
    wp::vec_t<3, wp::float32> var_168;
    wp::float32 var_169;
    wp::vec_t<3, wp::float32> var_170;
    wp::vec_t<3, wp::float32> var_171;
    //---------
    // forward
    // def collide_sphere_box(                                                                <L 1177>
    // center = wp.transpose(box_rot) @ (sphere_pos - box_pos)                                <L 1201>
    var_0 = wp::transpose(var_box_rot);
    var_1 = wp::sub(var_sphere_pos, var_box_pos);
    var_2 = wp::mul(var_0, var_1);
    // clamped = wp.max(-box_size, wp.min(box_size, center))                                  <L 1203>
    var_3 = wp::neg(var_box_size);
    var_4 = wp::min(var_box_size, var_2);
    var_5 = wp::max(var_3, var_4);
    // clamped_dir, dist = normalize_with_norm(clamped - center)                              <L 1204>
    var_6 = wp::sub(var_5, var_2);
    normalize_with_norm_0(var_6, var_7, var_8);
    // if dist <= 1e-6:                                                                       <L 1207>
    var_10 = (var_8 <= var_9);
    if (var_10) {
        // closest = 2.0 * (box_size[0] + box_size[1] + box_size[2])                          <L 1208>
        var_13 = wp::extract(var_box_size, var_12);
        var_15 = wp::extract(var_box_size, var_14);
        var_16 = wp::add(var_13, var_15);
        var_18 = wp::extract(var_box_size, var_17);
        var_19 = wp::add(var_16, var_18);
        var_20 = wp::mul(var_11, var_19);
        // k = wp.int32(0)                                                                    <L 1209>
        var_22 = wp::int32(var_21);
        // for i in range(6):                                                                 <L 1210>
        // face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i // 2] - center[i // 2])       <L 1211>
        var_25 = wp::mod(var_23, var_24);
        var_28 = wp::where(var_25, var_26, var_27);
        var_30 = wp::floordiv(var_23, var_29);
        var_31 = wp::extract(var_box_size, var_30);
        var_32 = wp::mul(var_28, var_31);
        var_34 = wp::floordiv(var_23, var_33);
        var_35 = wp::extract(var_2, var_34);
        var_36 = wp::sub(var_32, var_35);
        var_37 = wp::abs(var_36);
        // if closest > face_dist:                                                            <L 1212>
        var_38 = (var_20 > var_37);
        if (var_38) {
            // closest = face_dist                                                            <L 1213>
            var_39 = wp::copy(var_37);
            // k = i                                                                          <L 1214>
            var_40 = wp::copy(var_23);
        }
        var_41 = wp::where(var_38, var_39, var_20);
        var_42 = wp::where(var_38, var_40, var_22);
        // face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i // 2] - center[i // 2])       <L 1211>
        var_45 = wp::mod(var_43, var_44);
        var_48 = wp::where(var_45, var_46, var_47);
        var_50 = wp::floordiv(var_43, var_49);
        var_51 = wp::extract(var_box_size, var_50);
        var_52 = wp::mul(var_48, var_51);
        var_54 = wp::floordiv(var_43, var_53);
        var_55 = wp::extract(var_2, var_54);
        var_56 = wp::sub(var_52, var_55);
        var_57 = wp::abs(var_56);
        // if closest > face_dist:                                                            <L 1212>
        var_58 = (var_41 > var_57);
        if (var_58) {
            // closest = face_dist                                                            <L 1213>
            var_59 = wp::copy(var_57);
            // k = i                                                                          <L 1214>
            var_60 = wp::copy(var_43);
        }
        var_61 = wp::where(var_58, var_59, var_41);
        var_62 = wp::where(var_58, var_60, var_42);
        // face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i // 2] - center[i // 2])       <L 1211>
        var_65 = wp::mod(var_63, var_64);
        var_68 = wp::where(var_65, var_66, var_67);
        var_70 = wp::floordiv(var_63, var_69);
        var_71 = wp::extract(var_box_size, var_70);
        var_72 = wp::mul(var_68, var_71);
        var_74 = wp::floordiv(var_63, var_73);
        var_75 = wp::extract(var_2, var_74);
        var_76 = wp::sub(var_72, var_75);
        var_77 = wp::abs(var_76);
        // if closest > face_dist:                                                            <L 1212>
        var_78 = (var_61 > var_77);
        if (var_78) {
            // closest = face_dist                                                            <L 1213>
            var_79 = wp::copy(var_77);
            // k = i                                                                          <L 1214>
            var_80 = wp::copy(var_63);
        }
        var_81 = wp::where(var_78, var_79, var_61);
        var_82 = wp::where(var_78, var_80, var_62);
        // face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i // 2] - center[i // 2])       <L 1211>
        var_85 = wp::mod(var_83, var_84);
        var_88 = wp::where(var_85, var_86, var_87);
        var_90 = wp::floordiv(var_83, var_89);
        var_91 = wp::extract(var_box_size, var_90);
        var_92 = wp::mul(var_88, var_91);
        var_94 = wp::floordiv(var_83, var_93);
        var_95 = wp::extract(var_2, var_94);
        var_96 = wp::sub(var_92, var_95);
        var_97 = wp::abs(var_96);
        // if closest > face_dist:                                                            <L 1212>
        var_98 = (var_81 > var_97);
        if (var_98) {
            // closest = face_dist                                                            <L 1213>
            var_99 = wp::copy(var_97);
            // k = i                                                                          <L 1214>
            var_100 = wp::copy(var_83);
        }
        var_101 = wp::where(var_98, var_99, var_81);
        var_102 = wp::where(var_98, var_100, var_82);
        // face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i // 2] - center[i // 2])       <L 1211>
        var_105 = wp::mod(var_103, var_104);
        var_108 = wp::where(var_105, var_106, var_107);
        var_110 = wp::floordiv(var_103, var_109);
        var_111 = wp::extract(var_box_size, var_110);
        var_112 = wp::mul(var_108, var_111);
        var_114 = wp::floordiv(var_103, var_113);
        var_115 = wp::extract(var_2, var_114);
        var_116 = wp::sub(var_112, var_115);
        var_117 = wp::abs(var_116);
        // if closest > face_dist:                                                            <L 1212>
        var_118 = (var_101 > var_117);
        if (var_118) {
            // closest = face_dist                                                            <L 1213>
            var_119 = wp::copy(var_117);
            // k = i                                                                          <L 1214>
            var_120 = wp::copy(var_103);
        }
        var_121 = wp::where(var_118, var_119, var_101);
        var_122 = wp::where(var_118, var_120, var_102);
        // face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i // 2] - center[i // 2])       <L 1211>
        var_125 = wp::mod(var_123, var_124);
        var_128 = wp::where(var_125, var_126, var_127);
        var_130 = wp::floordiv(var_123, var_129);
        var_131 = wp::extract(var_box_size, var_130);
        var_132 = wp::mul(var_128, var_131);
        var_134 = wp::floordiv(var_123, var_133);
        var_135 = wp::extract(var_2, var_134);
        var_136 = wp::sub(var_132, var_135);
        var_137 = wp::abs(var_136);
        // if closest > face_dist:                                                            <L 1212>
        var_138 = (var_121 > var_137);
        if (var_138) {
            // closest = face_dist                                                            <L 1213>
            var_139 = wp::copy(var_137);
            // k = i                                                                          <L 1214>
            var_140 = wp::copy(var_123);
        }
        var_141 = wp::where(var_138, var_139, var_121);
        var_142 = wp::where(var_138, var_140, var_122);
        // nearest = wp.vec3(0.0)                                                             <L 1216>
        var_144 = wp::vec_t<3, wp::float32>(var_143);
        // nearest[k // 2] = wp.where(k % 2, -1.0, 1.0)                                       <L 1217>
        var_146 = wp::mod(var_142, var_145);
        var_149 = wp::where(var_146, var_147, var_148);
        var_151 = wp::floordiv(var_142, var_150);
        wp::assign_inplace(var_144, var_151, var_149);
        // pos = center + nearest * (sphere_radius - closest) / 2.0                           <L 1218>
        var_152 = wp::sub(var_sphere_radius, var_141);
        var_153 = wp::mul(var_144, var_152);
        var_155 = wp::div(var_153, var_154);
        var_156 = wp::add(var_2, var_155);
        // contact_normal = box_rot @ nearest                                                 <L 1219>
        var_157 = wp::mul(var_box_rot, var_144);
        // contact_distance = -closest - sphere_radius                                        <L 1220>
        var_158 = wp::neg(var_141);
        var_159 = wp::sub(var_158, var_sphere_radius);
    }
    if (!var_10) {
        // deepest = center + clamped_dir * sphere_radius                                     <L 1223>
        var_160 = wp::mul(var_7, var_sphere_radius);
        var_161 = wp::add(var_2, var_160);
        // pos = 0.5 * (clamped + deepest)                                                    <L 1224>
        var_163 = wp::add(var_5, var_161);
        var_164 = wp::mul(var_162, var_163);
        // contact_normal = box_rot @ clamped_dir                                             <L 1225>
        var_165 = wp::mul(var_box_rot, var_7);
        // contact_distance = dist - sphere_radius                                            <L 1226>
        var_166 = wp::sub(var_8, var_sphere_radius);
    }
    var_167 = wp::where(var_10, var_156, var_164);
    var_168 = wp::where(var_10, var_157, var_165);
    var_169 = wp::where(var_10, var_159, var_166);
    // contact_position = box_pos + box_rot @ pos                                             <L 1228>
    var_170 = wp::mul(var_box_rot, var_167);
    var_171 = wp::add(var_box_pos, var_170);
    // return contact_distance, contact_position, contact_normal                              <L 1230>
    ret_0 = var_169;
    ret_1 = var_171;
    ret_2 = var_168;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/contact_data.py:178
static CUDA_CALLABLE bool _contact_passes_gap_check_precomputed_0(
    ContactData_40360d7c var_contact_data,
    wp::vec_t<3, wp::float32> var_contact_normal_a_to_b,
    wp::float32 var_total_separation_needed)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32>* var_0;
    const wp::float32 var_1 = 0.5;
    wp::float32* var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32* var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32>* var_11;
    const wp::float32 var_12 = 0.5;
    wp::float32* var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32* var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32* var_25;
    bool var_26;
    wp::float32 var_27;
    //---------
    // forward
    // def _contact_passes_gap_check_precomputed(                                             <L 179>
    // a_contact_world = contact_data.contact_point_center - contact_normal_a_to_b * (        <L 185>
    var_0 = &((var_contact_data).contact_point_center);
    // 0.5 * contact_data.contact_distance + contact_data.radius_eff_a                        <L 186>
    var_2 = &((var_contact_data).contact_distance);
    var_4 = wp::load(var_2);
    var_3 = wp::mul(var_1, var_4);
    var_5 = &((var_contact_data).radius_eff_a);
    var_7 = wp::load(var_5);
    var_6 = wp::add(var_3, var_7);
    var_8 = wp::mul(var_contact_normal_a_to_b, var_6);
    var_10 = wp::load(var_0);
    var_9 = wp::sub(var_10, var_8);
    // b_contact_world = contact_data.contact_point_center + contact_normal_a_to_b * (        <L 188>
    var_11 = &((var_contact_data).contact_point_center);
    // 0.5 * contact_data.contact_distance + contact_data.radius_eff_b                        <L 189>
    var_13 = &((var_contact_data).contact_distance);
    var_15 = wp::load(var_13);
    var_14 = wp::mul(var_12, var_15);
    var_16 = &((var_contact_data).radius_eff_b);
    var_18 = wp::load(var_16);
    var_17 = wp::add(var_14, var_18);
    var_19 = wp::mul(var_contact_normal_a_to_b, var_17);
    var_21 = wp::load(var_11);
    var_20 = wp::add(var_21, var_19);
    // diff = b_contact_world - a_contact_world                                               <L 192>
    var_22 = wp::sub(var_20, var_9);
    // distance = wp.dot(diff, contact_normal_a_to_b)                                         <L 193>
    var_23 = wp::dot(var_22, var_contact_normal_a_to_b);
    // d = distance - total_separation_needed                                                 <L 194>
    var_24 = wp::sub(var_23, var_total_separation_needed);
    // return d <= contact_data.gap_sum                                                       <L 196>
    var_25 = &((var_contact_data).gap_sum);
    var_27 = wp::load(var_25);
    var_26 = (var_24 <= var_27);
    return var_26;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/narrow_phase.py:441
static CUDA_CALLABLE bool create_narrow_phase_primitive_kernel__locals___admit_0(
    ContactData_40360d7c var_contact_data,
    wp::vec_t<3, wp::float32> var_position,
    wp::float32 var_distance,
    wp::vec_t<3, wp::float32> var_normalized_contact_normal,
    wp::float32 var_total_separation_needed,
    wp::array_t<wp::transform_t<wp::float32>> var_shape_transform,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_linear_velocity,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_angular_velocity,
    wp::float32 var_collision_update_dt,
    wp::float32 var_max_speculative_extension)
{
    //---------
    // primal vars
    const bool var_0 = false;
    bool var_1;
    //---------
    // forward
    // def _admit(                                                                            <L 442>
    // contact_data.contact_point_center = position                                           <L 454>
    var_contact_data.contact_point_center = var_position;
    // contact_data.contact_distance = distance                                               <L 455>
    var_contact_data.contact_distance = var_distance;
    // if wp.static(speculative):                                                             <L 456>
    // return _contact_passes_gap_check_precomputed(                                          <L 465>
    // contact_data,                                                                          <L 466>
    // normalized_contact_normal,                                                             <L 467>
    // total_separation_needed,                                                               <L 468>
    var_1 = _contact_passes_gap_check_precomputed_0(var_contact_data, var_normalized_contact_normal, var_total_separation_needed);
    return var_1;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/contact_data.py:70
static CUDA_CALLABLE wp::int64 make_contact_sort_key_with_bits_0(
    wp::int32 var_shape_a,
    wp::int32 var_shape_b,
    wp::int32 var_sort_sub_key,
    wp::int32 var_shape_index_bits,
    wp::int32 var_sub_key_bit_count)
{
    //---------
    // primal vars
    wp::int64 var_0;
    wp::int64 var_1;
    wp::int64 var_2;
    wp::int64 var_3;
    wp::int64 var_4;
    wp::int64 var_5;
    wp::int64 var_6;
    wp::int64 var_7;
    wp::int64 var_8;
    wp::int64 var_9;
    wp::int64 var_10;
    wp::int64 var_11;
    wp::int64 var_12;
    wp::int64 var_13;
    wp::int64 var_14;
    wp::int64 var_15;
    wp::int64 var_16;
    wp::int64 var_17;
    wp::int64 var_18;
    wp::int64 var_19;
    wp::int64 var_20;
    //---------
    // forward
    // def make_contact_sort_key_with_bits(                                                   <L 71>
    // shape_bits = wp.int64(shape_index_bits)                                                <L 83>
    var_0 = wp::int64(var_shape_index_bits);
    // sub_key_bits = wp.int64(sub_key_bit_count)                                             <L 84>
    var_1 = wp::int64(var_sub_key_bit_count);
    // shape_mask = (wp.int64(1) << shape_bits) - wp.int64(1)                                 <L 85>
    var_2 = 1ll;
    var_3 = wp::lshift(var_2, var_0);
    var_4 = 1ll;
    var_5 = wp::sub(var_3, var_4);
    // sub_key_mask = (wp.int64(1) << sub_key_bits) - wp.int64(1)                             <L 86>
    var_6 = 1ll;
    var_7 = wp::lshift(var_6, var_1);
    var_8 = 1ll;
    var_9 = wp::sub(var_7, var_8);
    // return (                                                                               <L 87>
    // ((wp.int64(shape_a) & shape_mask) << (sub_key_bits + shape_bits))                      <L 88>
    var_10 = wp::int64(var_shape_a);
    var_11 = wp::bit_and(var_10, var_5);
    var_12 = wp::add(var_1, var_0);
    var_13 = wp::lshift(var_11, var_12);
    // | ((wp.int64(shape_b) & shape_mask) << sub_key_bits)                                   <L 89>
    var_14 = wp::int64(var_shape_b);
    var_15 = wp::bit_and(var_14, var_5);
    var_16 = wp::lshift(var_15, var_1);
    var_17 = wp::bit_or(var_13, var_16);
    // | (wp.int64(sort_sub_key) & sub_key_mask)                                              <L 90>
    var_18 = wp::int64(var_sort_sub_key);
    var_19 = wp::bit_and(var_18, var_9);
    var_20 = wp::bit_or(var_17, var_19);
    return var_20;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/sim/collide.py:132
static CUDA_CALLABLE void _write_contact_at_index_0(
    ContactData_40360d7c var_contact_data,
    ContactWriterData_099b2624 var_writer_data,
    wp::int32 var_index,
    wp::vec_t<3, wp::float32> var_point_a_world,
    wp::vec_t<3, wp::float32> var_point_b_world,
    wp::vec_t<3, wp::float32> var_normal_a_to_b)
{
    //---------
    // primal vars
    wp::int32* var_0;
    bool var_1;
    wp::int32 var_2;
    wp::int32* var_3;
    wp::array_t<wp::int32>* var_4;
    wp::array_t<wp::int32> var_5;
    wp::int32 var_6;
    wp::int32* var_7;
    wp::array_t<wp::int32>* var_8;
    wp::array_t<wp::int32> var_9;
    wp::int32 var_10;
    wp::array_t<wp::int32>* var_11;
    wp::int32* var_12;
    wp::int32* var_13;
    wp::array_t<wp::int32> var_14;
    wp::int32 var_15;
    wp::int32 var_16;
    wp::int32 var_17;
    wp::array_t<wp::int32>* var_18;
    wp::int32* var_19;
    wp::int32* var_20;
    wp::array_t<wp::int32> var_21;
    wp::int32 var_22;
    wp::int32 var_23;
    wp::int32 var_24;
    const wp::int32 var_25 = -1;
    bool var_26;
    wp::transform_t<wp::float32> var_27;
    wp::array_t<wp::transform_t<wp::float32>>* var_28;
    wp::transform_t<wp::float32>* var_29;
    wp::array_t<wp::transform_t<wp::float32>> var_30;
    wp::transform_t<wp::float32> var_31;
    wp::transform_t<wp::float32> var_32;
    wp::transform_t<wp::float32> var_33;
    const wp::int32 var_34 = -1;
    bool var_35;
    wp::transform_t<wp::float32> var_36;
    wp::array_t<wp::transform_t<wp::float32>>* var_37;
    wp::transform_t<wp::float32>* var_38;
    wp::array_t<wp::transform_t<wp::float32>> var_39;
    wp::transform_t<wp::float32> var_40;
    wp::transform_t<wp::float32> var_41;
    wp::transform_t<wp::float32> var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_44;
    wp::array_t<wp::vec_t<3, wp::float32>> var_45;
    wp::vec_t<3, wp::float32> var_46;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_47;
    wp::array_t<wp::vec_t<3, wp::float32>> var_48;
    wp::float32* var_49;
    wp::float32* var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32* var_54;
    wp::float32* var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::vec_t<3, wp::float32> var_59;
    wp::vec_t<3, wp::float32> var_60;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_61;
    wp::array_t<wp::vec_t<3, wp::float32>> var_62;
    wp::float32 var_63;
    wp::vec_t<3, wp::float32> var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_66;
    wp::array_t<wp::vec_t<3, wp::float32>> var_67;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_68;
    wp::array_t<wp::vec_t<3, wp::float32>> var_69;
    wp::array_t<wp::float32>* var_70;
    wp::array_t<wp::float32> var_71;
    wp::array_t<wp::float32>* var_72;
    wp::array_t<wp::float32> var_73;
    const wp::int32 var_74 = 0;
    wp::array_t<wp::int32>* var_75;
    wp::array_t<wp::int32> var_76;
    wp::array_t<wp::float32>* var_77;
    wp::shape_t* var_78;
    const wp::int32 var_79 = 0;
    wp::int32 var_80;
    wp::shape_t var_81;
    const wp::int32 var_82 = 0;
    bool var_83;
    wp::float32* var_84;
    wp::array_t<wp::float32>* var_85;
    wp::array_t<wp::float32> var_86;
    wp::float32 var_87;
    wp::float32* var_88;
    wp::array_t<wp::float32>* var_89;
    wp::array_t<wp::float32> var_90;
    wp::float32 var_91;
    wp::float32* var_92;
    wp::array_t<wp::float32>* var_93;
    wp::array_t<wp::float32> var_94;
    wp::float32 var_95;
    wp::array_t<wp::int64>* var_96;
    wp::shape_t* var_97;
    const wp::int32 var_98 = 0;
    wp::int32 var_99;
    wp::shape_t var_100;
    const wp::int32 var_101 = 0;
    bool var_102;
    wp::int32* var_103;
    wp::int32* var_104;
    wp::int32* var_105;
    wp::int32* var_106;
    wp::int32* var_107;
    wp::int64 var_108;
    wp::int32 var_109;
    wp::int32 var_110;
    wp::int32 var_111;
    wp::int32 var_112;
    wp::int32 var_113;
    wp::array_t<wp::int64>* var_114;
    wp::array_t<wp::int64> var_115;
    //---------
    // forward
    // def _write_contact_at_index(                                                           <L 133>
    // if index >= writer_data.contact_max:                                                   <L 142>
    var_0 = &((var_writer_data).contact_max);
    var_2 = wp::load(var_0);
    var_1 = (var_index >= var_2);
    if (var_1) {
        // return                                                                             <L 143>
        return;
    }
    // writer_data.out_shape0[index] = contact_data.shape_a                                   <L 145>
    var_3 = &((var_contact_data).shape_a);
    var_4 = &((var_writer_data).out_shape0);
    var_5 = wp::load(var_4);
    var_6 = wp::load(var_3);
    wp::array_store(var_5, var_index, var_6);
    // writer_data.out_shape1[index] = contact_data.shape_b                                   <L 146>
    var_7 = &((var_contact_data).shape_b);
    var_8 = &((var_writer_data).out_shape1);
    var_9 = wp::load(var_8);
    var_10 = wp::load(var_7);
    wp::array_store(var_9, var_index, var_10);
    // body0 = writer_data.shape_body[contact_data.shape_a]                                   <L 148>
    var_11 = &((var_writer_data).shape_body);
    var_12 = &((var_contact_data).shape_a);
    var_14 = wp::load(var_11);
    var_15 = wp::load(var_12);
    var_13 = wp::address(var_14, var_15);
    var_17 = wp::load(var_13);
    var_16 = wp::copy(var_17);
    // body1 = writer_data.shape_body[contact_data.shape_b]                                   <L 149>
    var_18 = &((var_writer_data).shape_body);
    var_19 = &((var_contact_data).shape_b);
    var_21 = wp::load(var_18);
    var_22 = wp::load(var_19);
    var_20 = wp::address(var_21, var_22);
    var_24 = wp::load(var_20);
    var_23 = wp::copy(var_24);
    // X_bw_a = wp.transform_identity() if body0 == -1 else wp.transform_inverse(writer_data.body_q[body0])       <L 150>
    var_26 = (var_16 == var_25);
    if (var_26) {
        var_27 = wp::transform_identity<wp::float32>();
    }
    if (!var_26) {
        var_28 = &((var_writer_data).body_q);
        var_30 = wp::load(var_28);
        var_29 = wp::address(var_30, var_16);
        var_32 = wp::load(var_29);
        var_31 = wp::transform_inverse(var_32);
    }
    var_33 = wp::where(var_26, var_27, var_31);
    // X_bw_b = wp.transform_identity() if body1 == -1 else wp.transform_inverse(writer_data.body_q[body1])       <L 151>
    var_35 = (var_23 == var_34);
    if (var_35) {
        var_36 = wp::transform_identity<wp::float32>();
    }
    if (!var_35) {
        var_37 = &((var_writer_data).body_q);
        var_39 = wp::load(var_37);
        var_38 = wp::address(var_39, var_23);
        var_41 = wp::load(var_38);
        var_40 = wp::transform_inverse(var_41);
    }
    var_42 = wp::where(var_35, var_36, var_40);
    // writer_data.out_point0[index] = wp.transform_point(X_bw_a, point_a_world)              <L 153>
    var_43 = wp::transform_point(var_33, var_point_a_world);
    var_44 = &((var_writer_data).out_point0);
    var_45 = wp::load(var_44);
    wp::array_store(var_45, var_index, var_43);
    // writer_data.out_point1[index] = wp.transform_point(X_bw_b, point_b_world)              <L 154>
    var_46 = wp::transform_point(var_42, var_point_b_world);
    var_47 = &((var_writer_data).out_point1);
    var_48 = wp::load(var_47);
    wp::array_store(var_48, var_index, var_46);
    // offset_mag_a = contact_data.radius_eff_a + contact_data.margin_a                       <L 156>
    var_49 = &((var_contact_data).radius_eff_a);
    var_50 = &((var_contact_data).margin_a);
    var_52 = wp::load(var_49);
    var_53 = wp::load(var_50);
    var_51 = wp::add(var_52, var_53);
    // offset_mag_b = contact_data.radius_eff_b + contact_data.margin_b                       <L 157>
    var_54 = &((var_contact_data).radius_eff_b);
    var_55 = &((var_contact_data).margin_b);
    var_57 = wp::load(var_54);
    var_58 = wp::load(var_55);
    var_56 = wp::add(var_57, var_58);
    // writer_data.out_offset0[index] = wp.transform_vector(X_bw_a, offset_mag_a * normal_a_to_b)       <L 158>
    var_59 = wp::mul(var_51, var_normal_a_to_b);
    var_60 = wp::transform_vector(var_33, var_59);
    var_61 = &((var_writer_data).out_offset0);
    var_62 = wp::load(var_61);
    wp::array_store(var_62, var_index, var_60);
    // writer_data.out_offset1[index] = wp.transform_vector(X_bw_b, -offset_mag_b * normal_a_to_b)       <L 159>
    var_63 = wp::neg(var_56);
    var_64 = wp::mul(var_63, var_normal_a_to_b);
    var_65 = wp::transform_vector(var_42, var_64);
    var_66 = &((var_writer_data).out_offset1);
    var_67 = wp::load(var_66);
    wp::array_store(var_67, var_index, var_65);
    // writer_data.out_normal[index] = normal_a_to_b                                          <L 160>
    var_68 = &((var_writer_data).out_normal);
    var_69 = wp::load(var_68);
    wp::array_store(var_69, var_index, var_normal_a_to_b);
    // writer_data.out_margin0[index] = offset_mag_a                                          <L 161>
    var_70 = &((var_writer_data).out_margin0);
    var_71 = wp::load(var_70);
    wp::array_store(var_71, var_index, var_51);
    // writer_data.out_margin1[index] = offset_mag_b                                          <L 162>
    var_72 = &((var_writer_data).out_margin1);
    var_73 = wp::load(var_72);
    wp::array_store(var_73, var_index, var_56);
    // writer_data.out_tids[index] = 0                                                        <L 163>
    var_75 = &((var_writer_data).out_tids);
    var_76 = wp::load(var_75);
    wp::array_store(var_76, var_index, var_74);
    // if writer_data.out_stiffness.shape[0] > 0:                                             <L 165>
    var_77 = &((var_writer_data).out_stiffness);
    var_78 = &(var_77->shape);
    var_81 = wp::load(var_78);
    var_80 = wp::extract(var_81, var_79);
    var_83 = (var_80 > var_82);
    if (var_83) {
        // writer_data.out_stiffness[index] = contact_data.contact_stiffness                  <L 166>
        var_84 = &((var_contact_data).contact_stiffness);
        var_85 = &((var_writer_data).out_stiffness);
        var_86 = wp::load(var_85);
        var_87 = wp::load(var_84);
        wp::array_store(var_86, var_index, var_87);
        // writer_data.out_damping[index] = contact_data.contact_damping                      <L 167>
        var_88 = &((var_contact_data).contact_damping);
        var_89 = &((var_writer_data).out_damping);
        var_90 = wp::load(var_89);
        var_91 = wp::load(var_88);
        wp::array_store(var_90, var_index, var_91);
        // writer_data.out_friction[index] = contact_data.contact_friction_scale              <L 168>
        var_92 = &((var_contact_data).contact_friction_scale);
        var_93 = &((var_writer_data).out_friction);
        var_94 = wp::load(var_93);
        var_95 = wp::load(var_92);
        wp::array_store(var_94, var_index, var_95);
    }
    // if writer_data.out_sort_key.shape[0] > 0:                                              <L 170>
    var_96 = &((var_writer_data).out_sort_key);
    var_97 = &(var_96->shape);
    var_100 = wp::load(var_97);
    var_99 = wp::extract(var_100, var_98);
    var_102 = (var_99 > var_101);
    if (var_102) {
        // writer_data.out_sort_key[index] = make_contact_sort_key_with_bits(                 <L 171>
        // contact_data.shape_a,                                                              <L 172>
        var_103 = &((var_contact_data).shape_a);
        // contact_data.shape_b,                                                              <L 173>
        var_104 = &((var_contact_data).shape_b);
        // contact_data.sort_sub_key,                                                         <L 174>
        var_105 = &((var_contact_data).sort_sub_key);
        // writer_data.shape_index_bits,                                                      <L 175>
        var_106 = &((var_writer_data).shape_index_bits);
        // writer_data.sub_key_bits,                                                          <L 176>
        var_107 = &((var_writer_data).sub_key_bits);
        var_109 = wp::load(var_103);
        var_110 = wp::load(var_104);
        var_111 = wp::load(var_105);
        var_112 = wp::load(var_106);
        var_113 = wp::load(var_107);
        var_108 = make_contact_sort_key_with_bits_0(var_109, var_110, var_111, var_112, var_113);
        // writer_data.out_sort_key[index] = make_contact_sort_key_with_bits(                 <L 171>
        var_114 = &((var_writer_data).out_sort_key);
        var_115 = wp::load(var_114);
        wp::array_store(var_115, var_index, var_108);
    }
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/sim/collide.py:180
static CUDA_CALLABLE void write_contact_0(
    ContactData_40360d7c var_contact_data,
    ContactWriterData_099b2624 var_writer_data,
    wp::int32 var_output_index)
{
    //---------
    // primal vars
    wp::float32* var_0;
    wp::float32* var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32* var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32* var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::vec_t<3, wp::float32>* var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32>* var_14;
    const wp::float32 var_15 = 0.5;
    wp::float32* var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32* var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32>* var_25;
    const wp::float32 var_26 = 0.5;
    wp::float32* var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32* var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::vec_t<3, wp::float32> var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::array_t<wp::float32>* var_39;
    wp::int32* var_40;
    wp::float32* var_41;
    wp::array_t<wp::float32> var_42;
    wp::int32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::array_t<wp::float32>* var_46;
    wp::int32* var_47;
    wp::float32* var_48;
    wp::array_t<wp::float32> var_49;
    wp::int32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::int32 var_54;
    const wp::int32 var_55 = 0;
    bool var_56;
    bool var_57;
    wp::array_t<wp::int32>* var_58;
    const wp::int32 var_59 = 0;
    const wp::int32 var_60 = 1;
    wp::int32 var_61;
    wp::array_t<wp::int32> var_62;
    wp::int32 var_63;
    //---------
    // forward
    // def write_contact(                                                                     <L 181>
    // total_separation_needed = (                                                            <L 194>
    // contact_data.radius_eff_a + contact_data.radius_eff_b + contact_data.margin_a + contact_data.margin_b       <L 195>
    var_0 = &((var_contact_data).radius_eff_a);
    var_1 = &((var_contact_data).radius_eff_b);
    var_3 = wp::load(var_0);
    var_4 = wp::load(var_1);
    var_2 = wp::add(var_3, var_4);
    var_5 = &((var_contact_data).margin_a);
    var_7 = wp::load(var_5);
    var_6 = wp::add(var_2, var_7);
    var_8 = &((var_contact_data).margin_b);
    var_10 = wp::load(var_8);
    var_9 = wp::add(var_6, var_10);
    // contact_normal_a_to_b = wp.normalize(contact_data.contact_normal_a_to_b)               <L 199>
    var_11 = &((var_contact_data).contact_normal_a_to_b);
    var_13 = wp::load(var_11);
    var_12 = wp::normalize(var_13);
    // a_contact_world = contact_data.contact_point_center - contact_normal_a_to_b * (        <L 201>
    var_14 = &((var_contact_data).contact_point_center);
    // 0.5 * contact_data.contact_distance + contact_data.radius_eff_a                        <L 202>
    var_16 = &((var_contact_data).contact_distance);
    var_18 = wp::load(var_16);
    var_17 = wp::mul(var_15, var_18);
    var_19 = &((var_contact_data).radius_eff_a);
    var_21 = wp::load(var_19);
    var_20 = wp::add(var_17, var_21);
    var_22 = wp::mul(var_12, var_20);
    var_24 = wp::load(var_14);
    var_23 = wp::sub(var_24, var_22);
    // b_contact_world = contact_data.contact_point_center + contact_normal_a_to_b * (        <L 204>
    var_25 = &((var_contact_data).contact_point_center);
    // 0.5 * contact_data.contact_distance + contact_data.radius_eff_b                        <L 205>
    var_27 = &((var_contact_data).contact_distance);
    var_29 = wp::load(var_27);
    var_28 = wp::mul(var_26, var_29);
    var_30 = &((var_contact_data).radius_eff_b);
    var_32 = wp::load(var_30);
    var_31 = wp::add(var_28, var_32);
    var_33 = wp::mul(var_12, var_31);
    var_35 = wp::load(var_25);
    var_34 = wp::add(var_35, var_33);
    // diff = b_contact_world - a_contact_world                                               <L 208>
    var_36 = wp::sub(var_34, var_23);
    // distance = wp.dot(diff, contact_normal_a_to_b)                                         <L 209>
    var_37 = wp::dot(var_36, var_12);
    // d = distance - total_separation_needed                                                 <L 210>
    var_38 = wp::sub(var_37, var_9);
    // gap_a = writer_data.shape_gap[contact_data.shape_a]                                    <L 213>
    var_39 = &((var_writer_data).shape_gap);
    var_40 = &((var_contact_data).shape_a);
    var_42 = wp::load(var_39);
    var_43 = wp::load(var_40);
    var_41 = wp::address(var_42, var_43);
    var_45 = wp::load(var_41);
    var_44 = wp::copy(var_45);
    // gap_b = writer_data.shape_gap[contact_data.shape_b]                                    <L 214>
    var_46 = &((var_writer_data).shape_gap);
    var_47 = &((var_contact_data).shape_b);
    var_49 = wp::load(var_46);
    var_50 = wp::load(var_47);
    var_48 = wp::address(var_49, var_50);
    var_52 = wp::load(var_48);
    var_51 = wp::copy(var_52);
    // contact_gap = gap_a + gap_b                                                            <L 215>
    var_53 = wp::add(var_44, var_51);
    // index = output_index                                                                   <L 217>
    var_54 = wp::copy(var_output_index);
    // if index < 0:                                                                          <L 219>
    var_56 = (var_54 < var_55);
    if (var_56) {
        // if d > contact_gap:                                                                <L 221>
        var_57 = (var_38 > var_53);
        if (var_57) {
            // return                                                                         <L 222>
            return;
        }
        // index = wp.atomic_add(writer_data.contact_count, 0, 1)                             <L 223>
        var_58 = &((var_writer_data).contact_count);
        var_62 = wp::load(var_58);
        var_61 = wp::atomic_add(var_62, var_59, var_60);
    }
    var_63 = wp::where(var_56, var_61, var_54);
    // _write_contact_at_index(contact_data, writer_data, index, a_contact_world, b_contact_world, contact_normal_a_to_b)       <L 224>
    _write_contact_at_index_0(var_contact_data, var_writer_data, var_63, var_23, var_34, var_12);
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/narrow_phase.py:94
static CUDA_CALLABLE void adj__reserve_compacted_slot_0(
    wp::int32 var_predicate,
    wp::array_t<wp::int32> var_work_count,
    wp::int32 & adj_predicate,
    wp::array_t<wp::int32> & adj_work_count,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/narrow_phase.py:115
static CUDA_CALLABLE void adj__append_pair_compacted_0(
    bool var_predicate,
    wp::vec_t<2, wp::int32> var_value,
    wp::array_t<wp::vec_t<2, wp::int32>> var_work_items,
    wp::array_t<wp::int32> var_work_count,
    bool & adj_predicate,
    wp::vec_t<2, wp::int32> & adj_value,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_work_items,
    wp::array_t<wp::int32> & adj_work_count)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_primitive.py:100
static CUDA_CALLABLE void adj_collide_plane_sphere_0(
    wp::vec_t<3, wp::float32> var_plane_normal,
    wp::vec_t<3, wp::float32> var_plane_pos,
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_plane_normal,
    wp::vec_t<3, wp::float32> & adj_plane_pos,
    wp::vec_t<3, wp::float32> & adj_sphere_pos,
    wp::float32 & adj_sphere_radius,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_primitive.py:352
static CUDA_CALLABLE void adj_collide_plane_ellipsoid_0(
    wp::vec_t<3, wp::float32> var_plane_normal,
    wp::vec_t<3, wp::float32> var_plane_pos,
    wp::vec_t<3, wp::float32> var_ellipsoid_pos,
    wp::mat_t<3, 3, wp::float32> var_ellipsoid_rot,
    wp::vec_t<3, wp::float32> var_ellipsoid_size,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_plane_normal,
    wp::vec_t<3, wp::float32> & adj_plane_pos,
    wp::vec_t<3, wp::float32> & adj_ellipsoid_pos,
    wp::mat_t<3, 3, wp::float32> & adj_ellipsoid_rot,
    wp::vec_t<3, wp::float32> & adj_ellipsoid_size,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_primitive.py:384
static CUDA_CALLABLE void adj_collide_plane_box_0(
    wp::vec_t<3, wp::float32> var_plane_normal,
    wp::vec_t<3, wp::float32> var_plane_pos,
    wp::vec_t<3, wp::float32> var_box_pos,
    wp::mat_t<3, 3, wp::float32> var_box_rot,
    wp::vec_t<3, wp::float32> var_box_size,
    wp::float32 var_margin,
    wp::vec_t<4, wp::float32> & ret_0,
    wp::mat_t<4, 3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_plane_normal,
    wp::vec_t<3, wp::float32> & adj_plane_pos,
    wp::vec_t<3, wp::float32> & adj_box_pos,
    wp::mat_t<3, 3, wp::float32> & adj_box_rot,
    wp::vec_t<3, wp::float32> & adj_box_size,
    wp::float32 & adj_margin,
    wp::vec_t<4, wp::float32> & adj_ret_0,
    wp::mat_t<4, 3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_primitive.py:110
static CUDA_CALLABLE void adj_collide_sphere_sphere_0(
    wp::vec_t<3, wp::float32> var_pos1,
    wp::float32 var_radius1,
    wp::vec_t<3, wp::float32> var_pos2,
    wp::float32 var_radius2,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_pos1,
    wp::float32 & adj_radius1,
    wp::vec_t<3, wp::float32> & adj_pos2,
    wp::float32 & adj_radius2,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_primitive.py:278
static CUDA_CALLABLE void adj__collide_plane_capsule_contacts_0(
    wp::vec_t<3, wp::float32> var_plane_normal,
    wp::vec_t<3, wp::float32> var_plane_pos,
    wp::vec_t<3, wp::float32> var_capsule_pos,
    wp::vec_t<3, wp::float32> var_capsule_axis,
    wp::float32 var_capsule_radius,
    wp::float32 var_capsule_half_length,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::float32 & ret_2,
    wp::vec_t<3, wp::float32> & ret_3,
    wp::vec_t<3, wp::float32> & adj_plane_normal,
    wp::vec_t<3, wp::float32> & adj_plane_pos,
    wp::vec_t<3, wp::float32> & adj_capsule_pos,
    wp::vec_t<3, wp::float32> & adj_capsule_axis,
    wp::float32 & adj_capsule_radius,
    wp::float32 & adj_capsule_half_length,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::float32 & adj_ret_2,
    wp::vec_t<3, wp::float32> & adj_ret_3)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_primitive.py:534
static CUDA_CALLABLE void adj_collide_plane_cylinder_0(
    wp::vec_t<3, wp::float32> var_plane_normal,
    wp::vec_t<3, wp::float32> var_plane_pos,
    wp::vec_t<3, wp::float32> var_cylinder_pos,
    wp::vec_t<3, wp::float32> var_cylinder_axis,
    wp::float32 var_cylinder_radius,
    wp::float32 var_cylinder_half_height,
    wp::vec_t<4, wp::float32> & ret_0,
    wp::mat_t<4, 3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_plane_normal,
    wp::vec_t<3, wp::float32> & adj_plane_pos,
    wp::vec_t<3, wp::float32> & adj_cylinder_pos,
    wp::vec_t<3, wp::float32> & adj_cylinder_axis,
    wp::float32 & adj_cylinder_radius,
    wp::float32 & adj_cylinder_half_height,
    wp::vec_t<4, wp::float32> & adj_ret_0,
    wp::mat_t<4, 3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_primitive.py:48
static CUDA_CALLABLE void adj_closest_segment_point_0(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_b,
    wp::vec_t<3, wp::float32> var_pt,
    wp::vec_t<3, wp::float32> & adj_a,
    wp::vec_t<3, wp::float32> & adj_b,
    wp::vec_t<3, wp::float32> & adj_pt,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_primitive.py:143
static CUDA_CALLABLE void adj_collide_sphere_capsule_0(
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::vec_t<3, wp::float32> var_capsule_pos,
    wp::vec_t<3, wp::float32> var_capsule_axis,
    wp::float32 var_capsule_radius,
    wp::float32 var_capsule_half_length,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_sphere_pos,
    wp::float32 & adj_sphere_radius,
    wp::vec_t<3, wp::float32> & adj_capsule_pos,
    wp::vec_t<3, wp::float32> & adj_capsule_axis,
    wp::float32 & adj_capsule_radius,
    wp::float32 & adj_capsule_half_length,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_primitive.py:180
static CUDA_CALLABLE void adj_collide_capsule_capsule_0(
    wp::vec_t<3, wp::float32> var_cap1_pos,
    wp::vec_t<3, wp::float32> var_cap1_axis,
    wp::float32 var_cap1_radius,
    wp::float32 var_cap1_half_length,
    wp::vec_t<3, wp::float32> var_cap2_pos,
    wp::vec_t<3, wp::float32> var_cap2_axis,
    wp::float32 var_cap2_radius,
    wp::float32 var_cap2_half_length,
    wp::vec_t<2, wp::float32> & ret_0,
    wp::mat_t<2, 3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_cap1_pos,
    wp::vec_t<3, wp::float32> & adj_cap1_axis,
    wp::float32 & adj_cap1_radius,
    wp::float32 & adj_cap1_half_length,
    wp::vec_t<3, wp::float32> & adj_cap2_pos,
    wp::vec_t<3, wp::float32> & adj_cap2_axis,
    wp::float32 & adj_cap2_radius,
    wp::float32 & adj_cap2_half_length,
    wp::vec_t<2, wp::float32> & adj_ret_0,
    wp::mat_t<2, 3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/math/__init__.py:0
static CUDA_CALLABLE void adj_safe_div_0(
    wp::float32 var_x,
    wp::float32 var_y,
    wp::float32 var_eps,
    wp::float32 & adj_x,
    wp::float32 & adj_y,
    wp::float32 & adj_eps,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_primitive.py:461
static CUDA_CALLABLE void adj_collide_sphere_cylinder_0(
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::vec_t<3, wp::float32> var_cylinder_pos,
    wp::vec_t<3, wp::float32> var_cylinder_axis,
    wp::float32 var_cylinder_radius,
    wp::float32 var_cylinder_half_height,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_sphere_pos,
    wp::float32 & adj_sphere_radius,
    wp::vec_t<3, wp::float32> & adj_cylinder_pos,
    wp::vec_t<3, wp::float32> & adj_cylinder_axis,
    wp::float32 & adj_cylinder_radius,
    wp::float32 & adj_cylinder_half_height,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/math/__init__.py:0
static CUDA_CALLABLE void adj_normalize_with_norm_0(
    wp::vec_t<3, wp::float32> var_x,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::float32 & ret_1,
    wp::vec_t<3, wp::float32> & adj_x,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::float32 & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_primitive.py:1176
static CUDA_CALLABLE void adj_collide_sphere_box_0(
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::vec_t<3, wp::float32> var_box_pos,
    wp::mat_t<3, 3, wp::float32> var_box_rot,
    wp::vec_t<3, wp::float32> var_box_size,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_sphere_pos,
    wp::float32 & adj_sphere_radius,
    wp::vec_t<3, wp::float32> & adj_box_pos,
    wp::mat_t<3, 3, wp::float32> & adj_box_rot,
    wp::vec_t<3, wp::float32> & adj_box_size,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/contact_data.py:178
static CUDA_CALLABLE void adj__contact_passes_gap_check_precomputed_0(
    ContactData_40360d7c var_contact_data,
    wp::vec_t<3, wp::float32> var_contact_normal_a_to_b,
    wp::float32 var_total_separation_needed,
    ContactData_40360d7c & adj_contact_data,
    wp::vec_t<3, wp::float32> & adj_contact_normal_a_to_b,
    wp::float32 & adj_total_separation_needed,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/narrow_phase.py:441
static CUDA_CALLABLE void adj_create_narrow_phase_primitive_kernel__locals___admit_0(
    ContactData_40360d7c var_contact_data,
    wp::vec_t<3, wp::float32> var_position,
    wp::float32 var_distance,
    wp::vec_t<3, wp::float32> var_normalized_contact_normal,
    wp::float32 var_total_separation_needed,
    wp::array_t<wp::transform_t<wp::float32>> var_shape_transform,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_linear_velocity,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_angular_velocity,
    wp::float32 var_collision_update_dt,
    wp::float32 var_max_speculative_extension,
    ContactData_40360d7c & adj_contact_data,
    wp::vec_t<3, wp::float32> & adj_position,
    wp::float32 & adj_distance,
    wp::vec_t<3, wp::float32> & adj_normalized_contact_normal,
    wp::float32 & adj_total_separation_needed,
    wp::array_t<wp::transform_t<wp::float32>> & adj_shape_transform,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_shape_linear_velocity,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_shape_angular_velocity,
    wp::float32 & adj_collision_update_dt,
    wp::float32 & adj_max_speculative_extension,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/contact_data.py:70
static CUDA_CALLABLE void adj_make_contact_sort_key_with_bits_0(
    wp::int32 var_shape_a,
    wp::int32 var_shape_b,
    wp::int32 var_sort_sub_key,
    wp::int32 var_shape_index_bits,
    wp::int32 var_sub_key_bit_count,
    wp::int32 & adj_shape_a,
    wp::int32 & adj_shape_b,
    wp::int32 & adj_sort_sub_key,
    wp::int32 & adj_shape_index_bits,
    wp::int32 & adj_sub_key_bit_count,
    wp::int64 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/sim/collide.py:132
static CUDA_CALLABLE void adj__write_contact_at_index_0(
    ContactData_40360d7c var_contact_data,
    ContactWriterData_099b2624 var_writer_data,
    wp::int32 var_index,
    wp::vec_t<3, wp::float32> var_point_a_world,
    wp::vec_t<3, wp::float32> var_point_b_world,
    wp::vec_t<3, wp::float32> var_normal_a_to_b,
    ContactData_40360d7c & adj_contact_data,
    ContactWriterData_099b2624 & adj_writer_data,
    wp::int32 & adj_index,
    wp::vec_t<3, wp::float32> & adj_point_a_world,
    wp::vec_t<3, wp::float32> & adj_point_b_world,
    wp::vec_t<3, wp::float32> & adj_normal_a_to_b)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/sim/collide.py:180
static CUDA_CALLABLE void adj_write_contact_0(
    ContactData_40360d7c var_contact_data,
    ContactWriterData_099b2624 var_writer_data,
    wp::int32 var_output_index,
    ContactData_40360d7c & adj_contact_data,
    ContactWriterData_099b2624 & adj_writer_data,
    wp::int32 & adj_output_index)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void create_narrow_phase_primitive_kernel__locals__narrow_phase_primitive_kernel_c547cfd6_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<2, wp::int32>> var_candidate_pair,
    wp::array_t<wp::int32> var_candidate_pair_count,
    wp::array_t<wp::int32> var_shape_types,
    wp::array_t<wp::vec_t<4, wp::float32>> var_shape_data,
    wp::array_t<wp::transform_t<wp::float32>> var_shape_transform,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_linear_velocity,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_angular_velocity,
    wp::float32 var_collision_update_dt,
    wp::float32 var_max_speculative_extension,
    wp::array_t<wp::uint64> var_shape_source,
    wp::array_t<wp::float32> var_shape_gap,
    wp::array_t<wp::int32> var_shape_flags,
    wp::array_t<wp::int32> var_shape_sdf_index,
    wp::array_t<wp::vec_t<2, wp::int32>> var_shape_edge_range,
    ContactWriterData_099b2624 var_writer_data,
    wp::int32 var_total_num_threads,
    wp::array_t<wp::vec_t<2, wp::int32>> var_gjk_candidate_pairs,
    wp::array_t<wp::int32> var_gjk_candidate_pairs_count,
    wp::array_t<wp::vec_t<2, wp::int32>> var_shape_pairs_mesh,
    wp::array_t<wp::int32> var_shape_pairs_mesh_count,
    wp::array_t<wp::vec_t<2, wp::int32>> var_shape_pairs_mesh_plane,
    wp::array_t<wp::int32> var_shape_pairs_mesh_plane_cumsum,
    wp::array_t<wp::int32> var_shape_pairs_mesh_plane_count,
    wp::array_t<wp::int32> var_mesh_plane_vertex_total_count,
    wp::array_t<wp::vec_t<2, wp::int32>> var_shape_pairs_mesh_mesh,
    wp::array_t<wp::int32> var_shape_pairs_mesh_mesh_count,
    wp::array_t<wp::vec_t<2, wp::int32>> var_shape_pairs_sdf_sdf,
    wp::array_t<wp::int32> var_shape_pairs_sdf_sdf_count)
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
        wp::shape_t* var_1;
        const wp::int32 var_2 = 0;
        wp::int32 var_3;
        wp::shape_t var_4;
        const wp::int32 var_5 = 0;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        const wp::int32 var_9 = 0;
        bool var_10;
        wp::range_t var_11;
        wp::int32 var_12;
        wp::vec_t<2, wp::int32>* var_13;
        wp::vec_t<2, wp::int32> var_14;
        wp::vec_t<2, wp::int32> var_15;
        bool var_16;
        const bool var_17 = false;
        wp::shape_t* var_18;
        const wp::int32 var_19 = 0;
        wp::int32 var_20;
        wp::shape_t var_21;
        bool var_22;
        const wp::int32 var_23 = -1;
        const wp::int32 var_24 = -1;
        wp::vec_t<2, wp::int32> var_25;
        const wp::int32 var_26 = 0;
        wp::int32 var_27;
        const wp::int32 var_28 = 1;
        wp::int32 var_29;
        bool var_30;
        bool var_31;
        const wp::int32 var_32 = 0;
        bool var_33;
        const wp::int32 var_34 = 0;
        bool var_35;
        wp::int32* var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        wp::int32* var_39;
        wp::int32 var_40;
        wp::int32 var_41;
        bool var_42;
        wp::int32 var_43;
        wp::int32 var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        const bool var_47 = false;
        wp::vec_t<4, wp::float32>* var_48;
        wp::vec_t<4, wp::float32> var_49;
        wp::vec_t<4, wp::float32> var_50;
        wp::vec_t<4, wp::float32>* var_51;
        wp::vec_t<4, wp::float32> var_52;
        wp::vec_t<4, wp::float32> var_53;
        const wp::int32 var_54 = 0;
        wp::float32 var_55;
        const wp::int32 var_56 = 1;
        wp::float32 var_57;
        const wp::int32 var_58 = 2;
        wp::float32 var_59;
        wp::vec_t<3, wp::float32> var_60;
        const wp::int32 var_61 = 0;
        wp::float32 var_62;
        const wp::int32 var_63 = 1;
        wp::float32 var_64;
        const wp::int32 var_65 = 2;
        wp::float32 var_66;
        wp::vec_t<3, wp::float32> var_67;
        const wp::int32 var_68 = 2;
        bool var_69;
        const wp::int32 var_70 = 2;
        bool var_71;
        bool var_72;
        bool var_73;
        const wp::int32 var_74 = 8;
        bool var_75;
        bool var_76;
        const wp::int32 var_77 = 8;
        bool var_78;
        bool var_79;
        bool var_80;
        const wp::int32 var_81 = 1073741824;
        wp::int32 var_82;
        wp::int32 var_83;
        const wp::int32 var_84 = 1073741824;
        wp::int32 var_85;
        wp::int32 var_86;
        wp::int32 var_87;
        wp::int32 var_88;
        wp::int32 var_89;
        wp::int32 var_90;
        wp::int32 var_91;
        wp::int32 var_92;
        wp::int32 var_93;
        const wp::int32 var_94 = 0;
        const wp::int32 var_95 = 1;
        wp::int32 var_96;
        wp::shape_t* var_97;
        const wp::int32 var_98 = 0;
        wp::int32 var_99;
        wp::shape_t var_100;
        bool var_101;
        wp::vec_t<2, wp::int32> var_102;
        wp::vec_t<2, wp::int32> var_103;
        wp::vec_t<2, wp::int32> var_104;
        wp::vec_t<2, wp::int32> var_105;
        const wp::int32 var_106 = 0;
        const wp::int32 var_107 = 1;
        wp::int32 var_108;
        wp::shape_t* var_109;
        const wp::int32 var_110 = 0;
        wp::int32 var_111;
        wp::shape_t var_112;
        bool var_113;
        const wp::int32 var_114 = 8;
        bool var_115;
        const wp::int32 var_116 = 8;
        bool var_117;
        const wp::int32 var_118 = 7;
        bool var_119;
        const wp::int32 var_120 = 7;
        bool var_121;
        const wp::int32 var_122 = 1;
        bool var_123;
        bool var_124;
        bool var_125;
        const wp::int32 var_126 = 0;
        wp::float32 var_127;
        const wp::float32 var_128 = 0.0;
        bool var_129;
        const wp::int32 var_130 = 1;
        wp::float32 var_131;
        const wp::float32 var_132 = 0.0;
        bool var_133;
        bool var_134;
        wp::int32* var_135;
        const wp::int32 var_136 = 0;
        bool var_137;
        wp::int32 var_138;
        wp::vec_t<2, wp::int32>* var_139;
        const wp::int32 var_140 = 1;
        wp::int32 var_141;
        wp::vec_t<2, wp::int32> var_142;
        const wp::int32 var_143 = 0;
        bool var_144;
        bool var_145;
        wp::int32* var_146;
        const wp::int32 var_147 = 0;
        bool var_148;
        wp::int32 var_149;
        wp::vec_t<2, wp::int32>* var_150;
        const wp::int32 var_151 = 1;
        wp::int32 var_152;
        wp::vec_t<2, wp::int32> var_153;
        const wp::int32 var_154 = 0;
        bool var_155;
        bool var_156;
        bool var_157;
        bool var_158;
        wp::shape_t* var_159;
        const wp::int32 var_160 = 0;
        wp::int32 var_161;
        wp::shape_t var_162;
        const wp::int32 var_163 = 0;
        bool var_164;
        bool var_165;
        bool var_166;
        const wp::int32 var_167 = 0;
        const wp::int32 var_168 = 1;
        wp::int32 var_169;
        wp::shape_t* var_170;
        const wp::int32 var_171 = 0;
        wp::int32 var_172;
        wp::shape_t var_173;
        bool var_174;
        wp::vec_t<2, wp::int32> var_175;
        wp::int32 var_176;
        bool var_177;
        wp::uint64* var_178;
        wp::uint64 var_179;
        wp::uint64 var_180;
        wp::uint64 var_181;
        bool var_182;
        wp::Mesh var_183;
        wp::array_t<wp::vec_t<3, wp::float32>>* var_184;
        wp::shape_t* var_185;
        const wp::int32 var_186 = 0;
        wp::int32 var_187;
        wp::shape_t var_188;
        const wp::int32 var_189 = 0;
        const wp::int32 var_190 = 1;
        wp::int32 var_191;
        wp::shape_t* var_192;
        const wp::int32 var_193 = 0;
        wp::int32 var_194;
        wp::shape_t var_195;
        bool var_196;
        wp::vec_t<2, wp::int32> var_197;
        const wp::int32 var_198 = 0;
        wp::int32 var_199;
        wp::int32 var_200;
        bool var_201;
        const wp::int32 var_202 = 0;
        const wp::int32 var_203 = 1;
        wp::int32 var_204;
        wp::shape_t* var_205;
        const wp::int32 var_206 = 0;
        wp::int32 var_207;
        wp::shape_t var_208;
        bool var_209;
        wp::vec_t<2, wp::int32> var_210;
        wp::int32 var_211;
        bool var_212;
        const wp::int32 var_213 = 5;
        bool var_214;
        const wp::int32 var_215 = 9;
        bool var_216;
        bool var_217;
        const wp::int32 var_218 = 4;
        bool var_219;
        const wp::int32 var_220 = 4;
        bool var_221;
        const bool var_222 = false;
        const bool var_223 = true;
        wp::vec_t<2, wp::int32> var_224;
        const wp::int32 var_225 = 3;
        wp::float32 var_226;
        const wp::int32 var_227 = 3;
        wp::float32 var_228;
        wp::transform_t<wp::float32>* var_229;
        wp::transform_t<wp::float32> var_230;
        wp::transform_t<wp::float32> var_231;
        wp::transform_t<wp::float32>* var_232;
        wp::transform_t<wp::float32> var_233;
        wp::transform_t<wp::float32> var_234;
        wp::vec_t<3, wp::float32> var_235;
        wp::vec_t<3, wp::float32> var_236;
        wp::quat_t<wp::float32> var_237;
        wp::quat_t<wp::float32> var_238;
        wp::float32* var_239;
        wp::float32 var_240;
        wp::float32 var_241;
        wp::float32* var_242;
        wp::float32 var_243;
        wp::float32 var_244;
        wp::float32 var_245;
        const wp::int32 var_246 = 3;
        bool var_247;
        const wp::int32 var_248 = 3;
        bool var_249;
        const wp::int32 var_250 = 4;
        bool var_251;
        const wp::int32 var_252 = 4;
        bool var_253;
        const wp::int32 var_254 = 5;
        bool var_255;
        const wp::int32 var_256 = 6;
        bool var_257;
        const wp::int32 var_258 = 7;
        bool var_259;
        bool var_260;
        bool var_261;
        const wp::int32 var_262 = 2;
        wp::float32 var_263;
        const wp::float32 var_264 = 0.0;
        bool var_265;
        const wp::float32 var_266 = 0.0;
        const wp::float32 var_267 = 0.0;
        const wp::float32 var_268 = 1.0;
        wp::vec_t<3, wp::float32> var_269;
        wp::vec_t<3, wp::float32> var_270;
        const wp::float32 var_271 = 0.0;
        const wp::float32 var_272 = 0.0;
        const wp::float32 var_273 = 1.0;
        wp::vec_t<3, wp::float32> var_274;
        wp::vec_t<3, wp::float32> var_275;
        wp::float32 var_276;
        wp::float32 var_277;
        const wp::int32 var_278 = 2;
        wp::float32 var_279;
        wp::float32 var_280;
        const wp::int32 var_281 = 1;
        wp::float32 var_282;
        bool var_283;
        bool var_284;
        const wp::float32 var_285 = 0.0;
        wp::float32 var_286;
        const wp::float32 var_287 = 0.0;
        wp::float32 var_288;
        bool var_289;
        const wp::int32 var_290 = 0;
        wp::float32 var_291;
        wp::float32 var_292;
        bool var_293;
        const wp::int32 var_294 = 0;
        wp::float32 var_295;
        wp::float32 var_296;
        const wp::float32 var_297 = 10000000000.0;
        wp::float32 var_298;
        wp::float32 var_299;
        wp::float32 var_300;
        wp::float32 var_301;
        wp::vec_t<3, wp::float32> var_302;
        wp::vec_t<3, wp::float32> var_303;
        wp::vec_t<3, wp::float32> var_304;
        wp::vec_t<3, wp::float32> var_305;
        wp::vec_t<3, wp::float32> var_306;
        bool var_307;
        const wp::float32 var_308 = 0.0;
        const wp::float32 var_309 = 0.0;
        const wp::float32 var_310 = 1.0;
        wp::vec_t<3, wp::float32> var_311;
        wp::vec_t<3, wp::float32> var_312;
        const wp::int32 var_313 = 0;
        wp::float32 var_314;
        wp::float32 var_315;
        wp::vec_t<3, wp::float32> var_316;
        wp::vec_t<3, wp::float32> var_317;
        bool var_318;
        const wp::float32 var_319 = 0.0;
        const wp::float32 var_320 = 0.0;
        const wp::float32 var_321 = 1.0;
        wp::vec_t<3, wp::float32> var_322;
        wp::vec_t<3, wp::float32> var_323;
        wp::mat_t<3, 3, wp::float32> var_324;
        wp::vec_t<3, wp::float32> var_325;
        wp::float32 var_326;
        wp::vec_t<3, wp::float32> var_327;
        wp::vec_t<3, wp::float32> var_328;
        bool var_329;
        const wp::float32 var_330 = 0.0;
        const wp::float32 var_331 = 0.0;
        const wp::float32 var_332 = 1.0;
        wp::vec_t<3, wp::float32> var_333;
        wp::vec_t<3, wp::float32> var_334;
        wp::mat_t<3, 3, wp::float32> var_335;
        wp::vec_t<3, wp::float32> var_336;
        wp::float32 var_337;
        wp::float32 var_338;
        wp::vec_t<4, wp::float32> var_339;
        wp::mat_t<4, 3, wp::float32> var_340;
        wp::vec_t<3, wp::float32> var_341;
        const wp::int32 var_342 = 0;
        wp::float32 var_343;
        const wp::int32 var_344 = 1;
        wp::float32 var_345;
        const wp::int32 var_346 = 2;
        wp::float32 var_347;
        const wp::int32 var_348 = 3;
        wp::float32 var_349;
        const wp::int32 var_350 = 0;
        const wp::int32 var_351 = 0;
        wp::float32 var_352;
        const wp::int32 var_353 = 0;
        const wp::int32 var_354 = 1;
        wp::float32 var_355;
        const wp::int32 var_356 = 0;
        const wp::int32 var_357 = 2;
        wp::float32 var_358;
        wp::vec_t<3, wp::float32> var_359;
        const wp::int32 var_360 = 1;
        const wp::int32 var_361 = 0;
        wp::float32 var_362;
        const wp::int32 var_363 = 1;
        const wp::int32 var_364 = 1;
        wp::float32 var_365;
        const wp::int32 var_366 = 1;
        const wp::int32 var_367 = 2;
        wp::float32 var_368;
        wp::vec_t<3, wp::float32> var_369;
        const wp::int32 var_370 = 2;
        const wp::int32 var_371 = 0;
        wp::float32 var_372;
        const wp::int32 var_373 = 2;
        const wp::int32 var_374 = 1;
        wp::float32 var_375;
        const wp::int32 var_376 = 2;
        const wp::int32 var_377 = 2;
        wp::float32 var_378;
        wp::vec_t<3, wp::float32> var_379;
        const wp::int32 var_380 = 3;
        const wp::int32 var_381 = 0;
        wp::float32 var_382;
        const wp::int32 var_383 = 3;
        const wp::int32 var_384 = 1;
        wp::float32 var_385;
        const wp::int32 var_386 = 3;
        const wp::int32 var_387 = 2;
        wp::float32 var_388;
        wp::vec_t<3, wp::float32> var_389;
        bool var_390;
        const wp::int32 var_391 = 0;
        wp::float32 var_392;
        const wp::int32 var_393 = 0;
        wp::float32 var_394;
        wp::float32 var_395;
        wp::vec_t<3, wp::float32> var_396;
        wp::vec_t<3, wp::float32> var_397;
        bool var_398;
        const wp::float32 var_399 = 0.0;
        const wp::float32 var_400 = 0.0;
        const wp::float32 var_401 = 1.0;
        wp::vec_t<3, wp::float32> var_402;
        wp::vec_t<3, wp::float32> var_403;
        const wp::float32 var_404 = 0.0;
        const wp::float32 var_405 = 0.0;
        const wp::float32 var_406 = 1.0;
        wp::vec_t<3, wp::float32> var_407;
        wp::vec_t<3, wp::float32> var_408;
        const wp::int32 var_409 = 0;
        wp::float32 var_410;
        const wp::int32 var_411 = 1;
        wp::float32 var_412;
        wp::float32 var_413;
        wp::vec_t<3, wp::float32> var_414;
        wp::float32 var_415;
        wp::vec_t<3, wp::float32> var_416;
        wp::vec_t<3, wp::float32> var_417;
        const wp::float32 var_418 = 0.0;
        const wp::float32 var_419 = 0.0;
        const wp::float32 var_420 = 1.0;
        wp::vec_t<3, wp::float32> var_421;
        wp::vec_t<3, wp::float32> var_422;
        const wp::float32 var_423 = 0.0;
        const wp::float32 var_424 = 0.0;
        const wp::float32 var_425 = 1.0;
        wp::vec_t<3, wp::float32> var_426;
        wp::vec_t<3, wp::float32> var_427;
        const wp::int32 var_428 = 0;
        wp::float32 var_429;
        const wp::int32 var_430 = 1;
        wp::float32 var_431;
        wp::vec_t<4, wp::float32> var_432;
        wp::mat_t<4, 3, wp::float32> var_433;
        wp::vec_t<3, wp::float32> var_434;
        const wp::int32 var_435 = 0;
        wp::float32 var_436;
        const wp::int32 var_437 = 1;
        wp::float32 var_438;
        const wp::int32 var_439 = 2;
        wp::float32 var_440;
        const wp::int32 var_441 = 3;
        wp::float32 var_442;
        const wp::int32 var_443 = 0;
        const wp::int32 var_444 = 0;
        wp::float32 var_445;
        const wp::int32 var_446 = 0;
        const wp::int32 var_447 = 1;
        wp::float32 var_448;
        const wp::int32 var_449 = 0;
        const wp::int32 var_450 = 2;
        wp::float32 var_451;
        wp::vec_t<3, wp::float32> var_452;
        const wp::int32 var_453 = 1;
        const wp::int32 var_454 = 0;
        wp::float32 var_455;
        const wp::int32 var_456 = 1;
        const wp::int32 var_457 = 1;
        wp::float32 var_458;
        const wp::int32 var_459 = 1;
        const wp::int32 var_460 = 2;
        wp::float32 var_461;
        wp::vec_t<3, wp::float32> var_462;
        const wp::int32 var_463 = 2;
        const wp::int32 var_464 = 0;
        wp::float32 var_465;
        const wp::int32 var_466 = 2;
        const wp::int32 var_467 = 1;
        wp::float32 var_468;
        const wp::int32 var_469 = 2;
        const wp::int32 var_470 = 2;
        wp::float32 var_471;
        wp::vec_t<3, wp::float32> var_472;
        const wp::int32 var_473 = 3;
        const wp::int32 var_474 = 0;
        wp::float32 var_475;
        const wp::int32 var_476 = 3;
        const wp::int32 var_477 = 1;
        wp::float32 var_478;
        const wp::int32 var_479 = 3;
        const wp::int32 var_480 = 2;
        wp::float32 var_481;
        wp::vec_t<3, wp::float32> var_482;
        bool var_483;
        const wp::int32 var_484 = 0;
        wp::float32 var_485;
        const wp::float32 var_486 = 0.0;
        const wp::float32 var_487 = 0.0;
        const wp::float32 var_488 = 1.0;
        wp::vec_t<3, wp::float32> var_489;
        wp::vec_t<3, wp::float32> var_490;
        const wp::int32 var_491 = 0;
        wp::float32 var_492;
        const wp::int32 var_493 = 1;
        wp::float32 var_494;
        wp::float32 var_495;
        wp::vec_t<3, wp::float32> var_496;
        wp::vec_t<3, wp::float32> var_497;
        bool var_498;
        const wp::float32 var_499 = 0.0;
        const wp::float32 var_500 = 0.0;
        const wp::float32 var_501 = 1.0;
        wp::vec_t<3, wp::float32> var_502;
        wp::vec_t<3, wp::float32> var_503;
        const wp::float32 var_504 = 0.0;
        const wp::float32 var_505 = 0.0;
        const wp::float32 var_506 = 1.0;
        wp::vec_t<3, wp::float32> var_507;
        wp::vec_t<3, wp::float32> var_508;
        const wp::int32 var_509 = 0;
        wp::float32 var_510;
        const wp::int32 var_511 = 1;
        wp::float32 var_512;
        const wp::int32 var_513 = 0;
        wp::float32 var_514;
        const wp::int32 var_515 = 1;
        wp::float32 var_516;
        wp::vec_t<2, wp::float32> var_517;
        wp::mat_t<2, 3, wp::float32> var_518;
        wp::vec_t<3, wp::float32> var_519;
        const wp::int32 var_520 = 0;
        wp::float32 var_521;
        const wp::int32 var_522 = 0;
        const wp::int32 var_523 = 0;
        wp::float32 var_524;
        const wp::int32 var_525 = 0;
        const wp::int32 var_526 = 1;
        wp::float32 var_527;
        const wp::int32 var_528 = 0;
        const wp::int32 var_529 = 2;
        wp::float32 var_530;
        wp::vec_t<3, wp::float32> var_531;
        const wp::int32 var_532 = 1;
        wp::float32 var_533;
        const wp::int32 var_534 = 1;
        const wp::int32 var_535 = 0;
        wp::float32 var_536;
        const wp::int32 var_537 = 1;
        const wp::int32 var_538 = 1;
        wp::float32 var_539;
        const wp::int32 var_540 = 1;
        const wp::int32 var_541 = 2;
        wp::float32 var_542;
        wp::vec_t<3, wp::float32> var_543;
        bool var_544;
        const wp::int32 var_545 = 2;
        wp::float32 var_546;
        const wp::float32 var_547 = 0.0;
        bool var_548;
        const wp::int32 var_549 = 0;
        wp::float32 var_550;
        const wp::float32 var_551 = 0.0;
        const wp::float32 var_552 = 0.0;
        const wp::float32 var_553 = 1.0;
        wp::vec_t<3, wp::float32> var_554;
        wp::vec_t<3, wp::float32> var_555;
        const wp::int32 var_556 = 0;
        wp::float32 var_557;
        const wp::int32 var_558 = 1;
        wp::float32 var_559;
        wp::float32 var_560;
        wp::vec_t<3, wp::float32> var_561;
        wp::vec_t<3, wp::float32> var_562;
        bool var_563;
        const wp::int32 var_564 = 0;
        wp::float32 var_565;
        wp::mat_t<3, 3, wp::float32> var_566;
        wp::vec_t<3, wp::float32> var_567;
        wp::float32 var_568;
        wp::vec_t<3, wp::float32> var_569;
        wp::vec_t<3, wp::float32> var_570;
        wp::float32 var_571;
        wp::vec_t<3, wp::float32> var_572;
        wp::vec_t<3, wp::float32> var_573;
        wp::vec_t<3, wp::float32> var_574;
        wp::float32 var_575;
        wp::vec_t<3, wp::float32> var_576;
        wp::vec_t<3, wp::float32> var_577;
        wp::float32 var_578;
        wp::vec_t<3, wp::float32> var_579;
        wp::float32 var_580;
        wp::float32 var_581;
        wp::vec_t<3, wp::float32> var_582;
        wp::vec_t<3, wp::float32> var_583;
        wp::vec_t<3, wp::float32> var_584;
        wp::vec_t<3, wp::float32> var_585;
        wp::float32 var_586;
        wp::float32 var_587;
        wp::vec_t<3, wp::float32> var_588;
        wp::vec_t<3, wp::float32> var_589;
        wp::vec_t<3, wp::float32> var_590;
        wp::float32 var_591;
        wp::vec_t<3, wp::float32> var_592;
        wp::vec_t<3, wp::float32> var_593;
        wp::float32 var_594;
        wp::float32 var_595;
        wp::float32 var_596;
        wp::float32 var_597;
        wp::vec_t<3, wp::float32> var_598;
        wp::vec_t<3, wp::float32> var_599;
        wp::vec_t<3, wp::float32> var_600;
        wp::vec_t<3, wp::float32> var_601;
        wp::vec_t<3, wp::float32> var_602;
        wp::float32 var_603;
        wp::float32 var_604;
        wp::vec_t<3, wp::float32> var_605;
        wp::vec_t<3, wp::float32> var_606;
        wp::float32 var_607;
        wp::float32 var_608;
        wp::float32 var_609;
        wp::float32 var_610;
        wp::vec_t<3, wp::float32> var_611;
        wp::vec_t<3, wp::float32> var_612;
        wp::vec_t<3, wp::float32> var_613;
        wp::vec_t<3, wp::float32> var_614;
        wp::vec_t<3, wp::float32> var_615;
        wp::vec_t<3, wp::float32> var_616;
        wp::float32 var_617;
        wp::float32 var_618;
        wp::vec_t<3, wp::float32> var_619;
        wp::vec_t<3, wp::float32> var_620;
        wp::float32 var_621;
        wp::float32 var_622;
        wp::float32 var_623;
        wp::float32 var_624;
        wp::vec_t<3, wp::float32> var_625;
        wp::vec_t<3, wp::float32> var_626;
        wp::vec_t<3, wp::float32> var_627;
        wp::vec_t<3, wp::float32> var_628;
        wp::vec_t<3, wp::float32> var_629;
        wp::float32 var_630;
        wp::float32 var_631;
        wp::vec_t<3, wp::float32> var_632;
        wp::vec_t<3, wp::float32> var_633;
        wp::float32 var_634;
        wp::float32 var_635;
        wp::float32 var_636;
        wp::float32 var_637;
        wp::vec_t<3, wp::float32> var_638;
        wp::vec_t<3, wp::float32> var_639;
        wp::vec_t<3, wp::float32> var_640;
        wp::vec_t<3, wp::float32> var_641;
        wp::vec_t<3, wp::float32> var_642;
        wp::mat_t<3, 3, wp::float32> var_643;
        wp::vec_t<3, wp::float32> var_644;
        wp::vec_t<3, wp::float32> var_645;
        wp::vec_t<3, wp::float32> var_646;
        wp::float32 var_647;
        wp::float32 var_648;
        wp::float32 var_649;
        wp::float32 var_650;
        wp::vec_t<3, wp::float32> var_651;
        wp::vec_t<3, wp::float32> var_652;
        wp::vec_t<3, wp::float32> var_653;
        wp::vec_t<3, wp::float32> var_654;
        wp::vec_t<3, wp::float32> var_655;
        wp::vec_t<3, wp::float32> var_656;
        wp::vec_t<3, wp::float32> var_657;
        wp::float32 var_658;
        wp::float32 var_659;
        wp::float32 var_660;
        wp::float32 var_661;
        wp::vec_t<3, wp::float32> var_662;
        wp::vec_t<3, wp::float32> var_663;
        wp::vec_t<3, wp::float32> var_664;
        wp::vec_t<3, wp::float32> var_665;
        wp::vec_t<3, wp::float32> var_666;
        wp::float32 var_667;
        bool var_668;
        wp::int32 var_669;
        bool var_670;
        wp::int32 var_671;
        wp::int32 var_672;
        bool var_673;
        wp::int32 var_674;
        wp::int32 var_675;
        bool var_676;
        wp::int32 var_677;
        wp::int32 var_678;
        const wp::int32 var_679 = 0;
        bool var_680;
        ContactData_40360d7c var_681;
        const bool var_682 = false;
        wp::float32 var_683;
        wp::float32 var_684;
        wp::float32 var_685;
        wp::vec_t<3, wp::float32> var_686;
        const bool var_687 = false;
        bool var_688;
        bool var_689;
        bool var_690;
        const bool var_691 = false;
        bool var_692;
        bool var_693;
        bool var_694;
        const bool var_695 = false;
        bool var_696;
        bool var_697;
        bool var_698;
        const bool var_699 = false;
        bool var_700;
        bool var_701;
        bool var_702;
        wp::int32 var_703;
        wp::int32 var_704;
        wp::int32 var_705;
        wp::int32 var_706;
        wp::int32 var_707;
        wp::int32 var_708;
        wp::int32 var_709;
        const wp::int32 var_710 = 0;
        bool var_711;
        wp::array_t<wp::int32>* var_712;
        const wp::int32 var_713 = 0;
        wp::int32 var_714;
        wp::array_t<wp::int32> var_715;
        wp::int32 var_716;
        wp::int32* var_717;
        bool var_718;
        wp::int32 var_719;
        const wp::int32 var_720 = 0;
        const wp::int32 var_721 = 1;
        wp::int32 var_722;
        wp::int32 var_723;
        const wp::int32 var_724 = 1;
        const wp::int32 var_725 = 1;
        wp::int32 var_726;
        wp::int32 var_727;
        const wp::int32 var_728 = 2;
        const wp::int32 var_729 = 1;
        wp::int32 var_730;
        wp::int32 var_731;
        const wp::int32 var_732 = 3;
        bool var_733;
        bool var_734;
        bool var_735;
        bool var_736;
        bool var_737;
        bool var_738;
        const wp::int32 var_739 = 2;
        wp::float32 var_740;
        const wp::float32 var_741 = 0.0;
        bool var_742;
        bool var_743;
        const bool var_744 = false;
        const bool var_745 = true;
        wp::vec_t<2, wp::int32> var_746;
        //---------
        // forward
        // def narrow_phase_primitive_kernel(                                                     <L 1>
        // tid = wp.tid()                                                                         <L 43>
        var_0 = builtin_tid1d();
        // num_work_items = wp.min(candidate_pair.shape[0], candidate_pair_count[0])              <L 45>
        var_1 = &(var_candidate_pair.shape);
        var_4 = wp::load(var_1);
        var_3 = wp::extract(var_4, var_2);
        var_6 = wp::address(var_candidate_pair_count, var_5);
        var_8 = wp::load(var_6);
        var_7 = wp::min(var_3, var_8);
        // if num_work_items == 0:                                                                <L 48>
        var_10 = (var_7 == var_9);
        if (var_10) {
            // return                                                                             <L 49>
            continue;
        }
        // for t in range(tid, num_work_items, total_num_threads):                                <L 51>
        var_11 = wp::range(var_0, var_7, var_total_num_threads);
        start_for_1:;
            if (iter_cmp(var_11) == 0) goto end_for_1;
            var_12 = wp::iter_next(var_11);
            // pair = candidate_pair[t]                                                           <L 53>
            var_13 = wp::address(var_candidate_pair, var_12);
            var_15 = wp::load(var_13);
            var_14 = wp::copy(var_15);
            // if wp.static(sparse_gjk_pairs) and t < gjk_candidate_pairs.shape[0]:               <L 54>
            var_16 = var_17;
            if (var_16) {
                var_18 = &(var_gjk_candidate_pairs.shape);
                var_21 = wp::load(var_18);
                var_20 = wp::extract(var_21, var_19);
                var_22 = (var_12 < var_20);
                var_16 = var_16 && var_22;
            }
            if (var_16) {
                // gjk_candidate_pairs[t] = wp.vec2i(-1, -1)                                      <L 55>
                var_25 = wp::vec_t<2, wp::int32>(var_23, var_24);
                wp::array_store(var_gjk_candidate_pairs, var_12, var_25);
            }
            // shape_a = pair[0]                                                                  <L 56>
            var_27 = wp::extract(var_14, var_26);
            // shape_b = pair[1]                                                                  <L 57>
            var_29 = wp::extract(var_14, var_28);
            // if shape_a == shape_b or shape_a < 0 or shape_b < 0:                               <L 60>
            var_31 = (var_27 == var_29);
            var_30 = var_31;
            if (!var_30) {
                var_33 = (var_27 < var_32);
                var_30 = var_30 || var_33;
            }
            if (!var_30) {
                var_35 = (var_29 < var_34);
                var_30 = var_30 || var_35;
            }
            if (var_30) {
                // continue                                                                       <L 61>
                goto start_for_1;
            }
            // type_a = shape_types[shape_a]                                                      <L 64>
            var_36 = wp::address(var_shape_types, var_27);
            var_38 = wp::load(var_36);
            var_37 = wp::copy(var_38);
            // type_b = shape_types[shape_b]                                                      <L 65>
            var_39 = wp::address(var_shape_types, var_29);
            var_41 = wp::load(var_39);
            var_40 = wp::copy(var_41);
            // if type_a > type_b:                                                                <L 68>
            var_42 = (var_37 > var_40);
            if (var_42) {
                // shape_a, shape_b = shape_b, shape_a                                            <L 69>
                // type_a, type_b = type_b, type_a                                                <L 70>
            }
            var_43 = wp::where(var_42, var_29, var_27);
            var_44 = wp::where(var_42, var_27, var_29);
            var_45 = wp::where(var_42, var_40, var_37);
            var_46 = wp::where(var_42, var_37, var_40);
            // if wp.static(hydroelastic_enabled):                                                <L 73>
            // data_a = shape_data[shape_a]                                                       <L 83>
            var_48 = wp::address(var_shape_data, var_43);
            var_50 = wp::load(var_48);
            var_49 = wp::copy(var_50);
            // data_b = shape_data[shape_b]                                                       <L 84>
            var_51 = wp::address(var_shape_data, var_44);
            var_53 = wp::load(var_51);
            var_52 = wp::copy(var_53);
            // scale_a = wp.vec3(data_a[0], data_a[1], data_a[2])                                 <L 85>
            var_55 = wp::extract(var_49, var_54);
            var_57 = wp::extract(var_49, var_56);
            var_59 = wp::extract(var_49, var_58);
            var_60 = wp::vec_t<3, wp::float32>(var_55, var_57, var_59);
            // scale_b = wp.vec3(data_b[0], data_b[1], data_b[2])                                 <L 86>
            var_62 = wp::extract(var_52, var_61);
            var_64 = wp::extract(var_52, var_63);
            var_66 = wp::extract(var_52, var_65);
            var_67 = wp::vec_t<3, wp::float32>(var_62, var_64, var_66);
            // is_hfield_a = type_a == GeoType.HFIELD                                             <L 95>
            var_69 = (var_45 == var_68);
            // is_hfield_b = type_b == GeoType.HFIELD                                             <L 96>
            var_71 = (var_46 == var_70);
            // if is_hfield_a or is_hfield_b:                                                     <L 98>
            var_72 = var_69;
            if (!var_72) {
                var_72 = var_72 || var_71;
            }
            if (var_72) {
                // is_mesh_like_a = type_a == GeoType.MESH or is_hfield_a                         <L 99>
                var_75 = (var_45 == var_74);
                var_73 = var_75;
                if (!var_73) {
                    var_73 = var_73 || var_69;
                }
                // is_mesh_like_b = type_b == GeoType.MESH or is_hfield_b                         <L 100>
                var_78 = (var_46 == var_77);
                var_76 = var_78;
                if (!var_76) {
                    var_76 = var_76 || var_71;
                }
                // if is_mesh_like_a and is_mesh_like_b:                                          <L 102>
                var_79 = var_73;
                if (var_79) {
                    var_79 = var_79 && var_76;
                }
                if (var_79) {
                    // if is_hfield_a and is_hfield_b:                                            <L 104>
                    var_80 = var_69;
                    if (var_80) {
                        var_80 = var_80 && var_71;
                    }
                    if (var_80) {
                        // continue                                                               <L 105>
                        goto start_for_1;
                    }
                    // if is_hfield_b:                                                            <L 108>
                    if (var_71) {
                        // encoded_a = shape_b | SHAPE_PAIR_HFIELD_BIT                            <L 109>
                        var_82 = wp::bit_or(var_44, var_81);
                        // encoded_b = shape_a                                                    <L 110>
                        var_83 = wp::copy(var_43);
                    }
                    if (!var_71) {
                        // elif is_hfield_a:                                                      <L 111>
                        if (var_69) {
                            // encoded_a = shape_a | SHAPE_PAIR_HFIELD_BIT                        <L 112>
                            var_85 = wp::bit_or(var_43, var_84);
                            // encoded_b = shape_b                                                <L 113>
                            var_86 = wp::copy(var_44);
                        }
                        if (!var_69) {
                            // encoded_a = shape_a                                                <L 115>
                            var_87 = wp::copy(var_43);
                            // encoded_b = shape_b                                                <L 116>
                            var_88 = wp::copy(var_44);
                        }
                        var_89 = wp::where(var_69, var_85, var_87);
                        var_90 = wp::where(var_69, var_86, var_88);
                    }
                    var_91 = wp::where(var_71, var_81, var_84);
                    var_92 = wp::where(var_71, var_82, var_89);
                    var_93 = wp::where(var_71, var_83, var_90);
                    // idx = wp.atomic_add(shape_pairs_mesh_mesh_count, 0, 1)                     <L 117>
                    var_96 = wp::atomic_add(var_shape_pairs_mesh_mesh_count, var_94, var_95);
                    // if idx < shape_pairs_mesh_mesh.shape[0]:                                   <L 118>
                    var_97 = &(var_shape_pairs_mesh_mesh.shape);
                    var_100 = wp::load(var_97);
                    var_99 = wp::extract(var_100, var_98);
                    var_101 = (var_96 < var_99);
                    if (var_101) {
                        // shape_pairs_mesh_mesh[idx] = wp.vec2i(encoded_a, encoded_b)            <L 119>
                        var_102 = wp::vec_t<2, wp::int32>(var_92, var_93);
                        wp::array_store(var_shape_pairs_mesh_mesh, var_96, var_102);
                    }
                    // continue                                                                   <L 120>
                    goto start_for_1;
                }
                // if is_hfield_a:                                                                <L 124>
                if (var_69) {
                    // hf_pair = wp.vec2i(shape_a, shape_b)                                       <L 125>
                    var_103 = wp::vec_t<2, wp::int32>(var_43, var_44);
                }
                if (!var_69) {
                    // hf_pair = wp.vec2i(shape_b, shape_a)                                       <L 127>
                    var_104 = wp::vec_t<2, wp::int32>(var_44, var_43);
                }
                var_105 = wp::where(var_69, var_103, var_104);
                // idx = wp.atomic_add(shape_pairs_mesh_count, 0, 1)                              <L 128>
                var_108 = wp::atomic_add(var_shape_pairs_mesh_count, var_106, var_107);
                // if idx < shape_pairs_mesh.shape[0]:                                            <L 129>
                var_109 = &(var_shape_pairs_mesh.shape);
                var_112 = wp::load(var_109);
                var_111 = wp::extract(var_112, var_110);
                var_113 = (var_108 < var_111);
                if (var_113) {
                    // shape_pairs_mesh[idx] = hf_pair                                            <L 130>
                    wp::array_store(var_shape_pairs_mesh, var_108, var_105);
                }
                // continue                                                                       <L 131>
                goto start_for_1;
            }
            // is_mesh_a = type_a == GeoType.MESH                                                 <L 136>
            var_115 = (var_45 == var_114);
            // is_mesh_b = type_b == GeoType.MESH                                                 <L 137>
            var_117 = (var_46 == var_116);
            // is_box_a = type_a == GeoType.BOX                                                   <L 138>
            var_119 = (var_45 == var_118);
            // is_box_b = type_b == GeoType.BOX                                                   <L 139>
            var_121 = (var_46 == var_120);
            // is_plane_a = type_a == GeoType.PLANE                                               <L 140>
            var_123 = (var_45 == var_122);
            // is_infinite_plane_a = is_plane_a and (scale_a[0] == 0.0 and scale_a[1] == 0.0)       <L 141>
            var_124 = var_123;
            if (var_124) {
                var_127 = wp::extract(var_60, var_126);
                var_129 = (var_127 == var_128);
                var_125 = var_129;
                if (var_125) {
                    var_131 = wp::extract(var_60, var_130);
                    var_133 = (var_131 == var_132);
                    var_125 = var_125 && var_133;
                }
                var_124 = var_124 && var_125;
            }
            // has_sdf_edges_a = shape_sdf_index[shape_a] >= 0 and shape_edge_range[shape_a][1] > 0       <L 142>
            var_135 = wp::address(var_shape_sdf_index, var_43);
            var_138 = wp::load(var_135);
            var_137 = (var_138 >= var_136);
            var_134 = var_137;
            if (var_134) {
                var_139 = wp::address(var_shape_edge_range, var_43);
                var_142 = wp::load(var_139);
                var_141 = wp::extract(var_142, var_140);
                var_144 = (var_141 > var_143);
                var_134 = var_134 && var_144;
            }
            // has_sdf_edges_b = shape_sdf_index[shape_b] >= 0 and shape_edge_range[shape_b][1] > 0       <L 143>
            var_146 = wp::address(var_shape_sdf_index, var_44);
            var_149 = wp::load(var_146);
            var_148 = (var_149 >= var_147);
            var_145 = var_148;
            if (var_145) {
                var_150 = wp::address(var_shape_edge_range, var_44);
                var_153 = wp::load(var_150);
                var_152 = wp::extract(var_153, var_151);
                var_155 = (var_152 > var_154);
                var_145 = var_145 && var_155;
            }
            // if (is_mesh_a and is_mesh_b) or (                                                  <L 150>
            var_157 = var_115;
            if (var_157) {
                var_157 = var_157 && var_117;
            }
            var_156 = var_157;
            if (!var_156) {
                // shape_pairs_mesh_mesh.shape[0] > 0                                             <L 151>
                var_159 = &(var_shape_pairs_mesh_mesh.shape);
                var_162 = wp::load(var_159);
                var_161 = wp::extract(var_162, var_160);
                var_164 = (var_161 > var_163);
                var_158 = var_164;
                if (var_158) {
                    // and has_sdf_edges_a                                                        <L 152>
                    var_158 = var_158 && var_134;
                }
                if (var_158) {
                    // and has_sdf_edges_b                                                        <L 153>
                    var_158 = var_158 && var_145;
                }
                if (var_158) {
                    // and not (is_box_a and is_box_b)                                            <L 154>
                    var_165 = var_119;
                    if (var_165) {
                        var_165 = var_165 && var_121;
                    }
                    var_166 = wp::unot(var_165);
                    var_158 = var_158 && var_166;
                }
                var_156 = var_156 || var_158;
            }
            if (var_156) {
                // idx = wp.atomic_add(shape_pairs_mesh_mesh_count, 0, 1)                         <L 156>
                var_169 = wp::atomic_add(var_shape_pairs_mesh_mesh_count, var_167, var_168);
                // if idx < shape_pairs_mesh_mesh.shape[0]:                                       <L 157>
                var_170 = &(var_shape_pairs_mesh_mesh.shape);
                var_173 = wp::load(var_170);
                var_172 = wp::extract(var_173, var_171);
                var_174 = (var_169 < var_172);
                if (var_174) {
                    // shape_pairs_mesh_mesh[idx] = wp.vec2i(shape_a, shape_b)                    <L 158>
                    var_175 = wp::vec_t<2, wp::int32>(var_43, var_44);
                    wp::array_store(var_shape_pairs_mesh_mesh, var_169, var_175);
                }
                // continue                                                                       <L 159>
                goto start_for_1;
            }
            var_176 = wp::where(var_156, var_169, var_108);
            // if is_infinite_plane_a and is_mesh_b:                                              <L 162>
            var_177 = var_124;
            if (var_177) {
                var_177 = var_177 && var_117;
            }
            if (var_177) {
                // mesh_id = shape_source[shape_b]                                                <L 163>
                var_178 = wp::address(var_shape_source, var_44);
                var_180 = wp::load(var_178);
                var_179 = wp::copy(var_180);
                // if mesh_id != wp.uint64(0):                                                    <L 164>
                var_181 = 0ull;
                var_182 = (var_179 != var_181);
                if (var_182) {
                    // mesh_obj = wp.mesh_get(mesh_id)                                            <L 165>
                    var_183 = wp::mesh_get(var_179);
                    // vertex_count = mesh_obj.points.shape[0]                                    <L 166>
                    var_184 = &((var_183).points);
                    var_185 = &(var_184->shape);
                    var_188 = wp::load(var_185);
                    var_187 = wp::extract(var_188, var_186);
                    // mesh_plane_idx = wp.atomic_add(shape_pairs_mesh_plane_count, 0, 1)         <L 167>
                    var_191 = wp::atomic_add(var_shape_pairs_mesh_plane_count, var_189, var_190);
                    // if mesh_plane_idx < shape_pairs_mesh_plane.shape[0]:                       <L 168>
                    var_192 = &(var_shape_pairs_mesh_plane.shape);
                    var_195 = wp::load(var_192);
                    var_194 = wp::extract(var_195, var_193);
                    var_196 = (var_191 < var_194);
                    if (var_196) {
                        // shape_pairs_mesh_plane[mesh_plane_idx] = wp.vec2i(shape_b, shape_a)       <L 170>
                        var_197 = wp::vec_t<2, wp::int32>(var_44, var_43);
                        wp::array_store(var_shape_pairs_mesh_plane, var_191, var_197);
                        // cumulative_count_before = wp.atomic_add(mesh_plane_vertex_total_count, 0, vertex_count)       <L 171>
                        var_199 = wp::atomic_add(var_mesh_plane_vertex_total_count, var_198, var_187);
                        // shape_pairs_mesh_plane_cumsum[mesh_plane_idx] = cumulative_count_before + vertex_count       <L 172>
                        var_200 = wp::add(var_199, var_187);
                        wp::array_store(var_shape_pairs_mesh_plane_cumsum, var_191, var_200);
                    }
                }
                // continue                                                                       <L 173>
                goto start_for_1;
            }
            // if is_mesh_a or is_mesh_b:                                                         <L 176>
            var_201 = var_115;
            if (!var_201) {
                var_201 = var_201 || var_117;
            }
            if (var_201) {
                // idx = wp.atomic_add(shape_pairs_mesh_count, 0, 1)                              <L 177>
                var_204 = wp::atomic_add(var_shape_pairs_mesh_count, var_202, var_203);
                // if idx < shape_pairs_mesh.shape[0]:                                            <L 178>
                var_205 = &(var_shape_pairs_mesh.shape);
                var_208 = wp::load(var_205);
                var_207 = wp::extract(var_208, var_206);
                var_209 = (var_204 < var_207);
                if (var_209) {
                    // shape_pairs_mesh[idx] = wp.vec2i(shape_a, shape_b)                         <L 179>
                    var_210 = wp::vec_t<2, wp::int32>(var_43, var_44);
                    wp::array_store(var_shape_pairs_mesh, var_204, var_210);
                }
                // continue                                                                       <L 180>
                goto start_for_1;
            }
            var_211 = wp::where(var_201, var_204, var_176);
            // if (                                                                               <L 184>
            // type_a >= GeoType.ELLIPSOID                                                        <L 185>
            var_214 = (var_45 >= var_213);
            var_212 = var_214;
            if (!var_212) {
                // or type_b == GeoType.CONE                                                      <L 186>
                var_216 = (var_46 == var_215);
                var_212 = var_212 || var_216;
            }
            if (!var_212) {
                // or (type_a == GeoType.CAPSULE and type_b > GeoType.CAPSULE)                    <L 187>
                var_219 = (var_45 == var_218);
                var_217 = var_219;
                if (var_217) {
                    var_221 = (var_46 > var_220);
                    var_217 = var_217 && var_221;
                }
                var_212 = var_212 || var_217;
            }
            if (var_212) {
                // if wp.static(sparse_gjk_pairs):                                                <L 189>
                // _append_pair_compacted(                                                        <L 194>
                // True, wp.vec2i(shape_a, shape_b), gjk_candidate_pairs, gjk_candidate_pairs_count       <L 195>
                var_224 = wp::vec_t<2, wp::int32>(var_43, var_44);
                _append_pair_compacted_0(var_223, var_224, var_gjk_candidate_pairs, var_gjk_candidate_pairs_count);
                // continue                                                                       <L 197>
                goto start_for_1;
            }
            // margin_offset_a = data_a[3]                                                        <L 199>
            var_226 = wp::extract(var_49, var_225);
            // margin_offset_b = data_b[3]                                                        <L 200>
            var_228 = wp::extract(var_52, var_227);
            // X_a = shape_transform[shape_a]                                                     <L 203>
            var_229 = wp::address(var_shape_transform, var_43);
            var_231 = wp::load(var_229);
            var_230 = wp::copy(var_231);
            // X_b = shape_transform[shape_b]                                                     <L 204>
            var_232 = wp::address(var_shape_transform, var_44);
            var_234 = wp::load(var_232);
            var_233 = wp::copy(var_234);
            // pos_a = wp.transform_get_translation(X_a)                                          <L 205>
            var_235 = wp::transform_get_translation(var_230);
            // pos_b = wp.transform_get_translation(X_b)                                          <L 206>
            var_236 = wp::transform_get_translation(var_233);
            // quat_a = wp.transform_get_rotation(X_a)                                            <L 207>
            var_237 = wp::transform_get_rotation(var_230);
            // quat_b = wp.transform_get_rotation(X_b)                                            <L 208>
            var_238 = wp::transform_get_rotation(var_233);
            // gap_a = shape_gap[shape_a]                                                         <L 209>
            var_239 = wp::address(var_shape_gap, var_43);
            var_241 = wp::load(var_239);
            var_240 = wp::copy(var_241);
            // gap_b = shape_gap[shape_b]                                                         <L 210>
            var_242 = wp::address(var_shape_gap, var_44);
            var_244 = wp::load(var_242);
            var_243 = wp::copy(var_244);
            // gap_sum = gap_a + gap_b                                                            <L 211>
            var_245 = wp::add(var_240, var_243);
            // is_sphere_a = type_a == GeoType.SPHERE                                             <L 216>
            var_247 = (var_45 == var_246);
            // is_sphere_b = type_b == GeoType.SPHERE                                             <L 217>
            var_249 = (var_46 == var_248);
            // is_capsule_a = type_a == GeoType.CAPSULE                                           <L 218>
            var_251 = (var_45 == var_250);
            // is_capsule_b = type_b == GeoType.CAPSULE                                           <L 219>
            var_253 = (var_46 == var_252);
            // is_ellipsoid_b = type_b == GeoType.ELLIPSOID                                       <L 220>
            var_255 = (var_46 == var_254);
            // is_cylinder_b = type_b == GeoType.CYLINDER                                         <L 221>
            var_257 = (var_46 == var_256);
            // is_box_b = type_b == GeoType.BOX                                                   <L 222>
            var_259 = (var_46 == var_258);
            // use_plane_cylinder = is_plane_a and is_cylinder_b                                  <L 224>
            var_260 = var_123;
            if (var_260) {
                var_260 = var_260 && var_257;
            }
            // if use_plane_cylinder and scale_b[2] > 0.0:                                        <L 225>
            var_261 = var_260;
            if (var_261) {
                var_263 = wp::extract(var_67, var_262);
                var_265 = (var_263 > var_264);
                var_261 = var_261 && var_265;
            }
            if (var_261) {
                // plane_normal = wp.quat_rotate(quat_a, wp.vec3(0.0, 0.0, 1.0))                  <L 226>
                var_269 = wp::vec_t<3, wp::float32>(var_266, var_267, var_268);
                var_270 = wp::quat_rotate(var_237, var_269);
                // cylinder_axis = wp.quat_rotate(quat_b, wp.vec3(0.0, 0.0, 1.0))                 <L 227>
                var_274 = wp::vec_t<3, wp::float32>(var_271, var_272, var_273);
                var_275 = wp::quat_rotate(var_238, var_274);
                // use_plane_cylinder = wp.abs(wp.dot(plane_normal, cylinder_axis)) * scale_b[2] >= scale_b[1]       <L 228>
                var_276 = wp::dot(var_270, var_275);
                var_277 = wp::abs(var_276);
                var_279 = wp::extract(var_67, var_278);
                var_280 = wp::mul(var_277, var_279);
                var_282 = wp::extract(var_67, var_281);
                var_283 = (var_280 >= var_282);
            }
            var_284 = wp::where(var_261, var_283, var_260);
            // radius_eff_a = float(0.0)                                                          <L 232>
            var_286 = wp::float(var_285);
            // radius_eff_b = float(0.0)                                                          <L 233>
            var_288 = wp::float(var_287);
            // if is_sphere_a or is_capsule_a:                                                    <L 234>
            var_289 = var_247;
            if (!var_289) {
                var_289 = var_289 || var_251;
            }
            if (var_289) {
                // radius_eff_a = scale_a[0]                                                      <L 235>
                var_291 = wp::extract(var_60, var_290);
            }
            var_292 = wp::where(var_289, var_291, var_286);
            // if is_sphere_b or is_capsule_b:                                                    <L 236>
            var_293 = var_249;
            if (!var_293) {
                var_293 = var_293 || var_253;
            }
            if (var_293) {
                // radius_eff_b = scale_b[0]                                                      <L 237>
                var_295 = wp::extract(var_67, var_294);
            }
            var_296 = wp::where(var_293, var_295, var_288);
            // contact_dist_0 = float(MAXVAL)                                                     <L 242>
            var_298 = wp::float(var_297);
            // contact_dist_1 = float(MAXVAL)                                                     <L 243>
            var_299 = wp::float(var_297);
            // contact_dist_2 = float(MAXVAL)                                                     <L 244>
            var_300 = wp::float(var_297);
            // contact_dist_3 = float(MAXVAL)                                                     <L 245>
            var_301 = wp::float(var_297);
            // contact_pos_0 = wp.vec3()                                                          <L 246>
            var_302 = wp::vec_t<3, wp::float32>();
            // contact_pos_1 = wp.vec3()                                                          <L 247>
            var_303 = wp::vec_t<3, wp::float32>();
            // contact_pos_2 = wp.vec3()                                                          <L 248>
            var_304 = wp::vec_t<3, wp::float32>();
            // contact_pos_3 = wp.vec3()                                                          <L 249>
            var_305 = wp::vec_t<3, wp::float32>();
            // contact_normal = wp.vec3()                                                         <L 250>
            var_306 = wp::vec_t<3, wp::float32>();
            // if is_plane_a and is_sphere_b:                                                     <L 255>
            var_307 = var_123;
            if (var_307) {
                var_307 = var_307 && var_249;
            }
            if (var_307) {
                // plane_normal = wp.quat_rotate(quat_a, wp.vec3(0.0, 0.0, 1.0))                  <L 256>
                var_311 = wp::vec_t<3, wp::float32>(var_308, var_309, var_310);
                var_312 = wp::quat_rotate(var_237, var_311);
                // sphere_radius = scale_b[0]                                                     <L 257>
                var_314 = wp::extract(var_67, var_313);
                // contact_dist_0, contact_pos_0 = collide_plane_sphere(plane_normal, pos_a, pos_b, sphere_radius)       <L 258>
                collide_plane_sphere_0(var_312, var_235, var_236, var_314, var_315, var_316);
                // contact_normal = plane_normal                                                  <L 259>
                var_317 = wp::copy(var_312);
            }
            if (!var_307) {
                // elif is_plane_a and is_ellipsoid_b:                                            <L 265>
                var_318 = var_123;
                if (var_318) {
                    var_318 = var_318 && var_255;
                }
                if (var_318) {
                    // plane_normal = wp.quat_rotate(quat_a, wp.vec3(0.0, 0.0, 1.0))              <L 266>
                    var_322 = wp::vec_t<3, wp::float32>(var_319, var_320, var_321);
                    var_323 = wp::quat_rotate(var_237, var_322);
                    // ellipsoid_rot = wp.quat_to_matrix(quat_b)                                  <L 267>
                    var_324 = wp::quat_to_matrix(var_238);
                    // ellipsoid_size = scale_b                                                   <L 268>
                    var_325 = wp::copy(var_67);
                    // contact_dist_0, contact_pos_0, contact_normal = collide_plane_ellipsoid(       <L 269>
                    // plane_normal, pos_a, pos_b, ellipsoid_rot, ellipsoid_size                  <L 270>
                    collide_plane_ellipsoid_0(var_323, var_235, var_236, var_324, var_325, var_326, var_327, var_328);
                }
                if (!var_318) {
                    // elif is_plane_a and is_box_b:                                              <L 277>
                    var_329 = var_123;
                    if (var_329) {
                        var_329 = var_329 && var_259;
                    }
                    if (var_329) {
                        // plane_normal = wp.quat_rotate(quat_a, wp.vec3(0.0, 0.0, 1.0))          <L 278>
                        var_333 = wp::vec_t<3, wp::float32>(var_330, var_331, var_332);
                        var_334 = wp::quat_rotate(var_237, var_333);
                        // box_rot = wp.quat_to_matrix(quat_b)                                    <L 279>
                        var_335 = wp::quat_to_matrix(var_238);
                        // box_size = scale_b                                                     <L 280>
                        var_336 = wp::copy(var_67);
                        // dists4_box, positions4_box, contact_normal = collide_plane_box(        <L 282>
                        // plane_normal,                                                          <L 283>
                        // pos_a,                                                                 <L 284>
                        // pos_b,                                                                 <L 285>
                        // box_rot,                                                               <L 286>
                        // box_size,                                                              <L 287>
                        // gap_sum + margin_offset_a + margin_offset_b,                           <L 288>
                        var_337 = wp::add(var_245, var_226);
                        var_338 = wp::add(var_337, var_228);
                        collide_plane_box_0(var_334, var_235, var_236, var_335, var_336, var_338, var_339, var_340, var_341);
                        // contact_dist_0 = dists4_box[0]                                         <L 291>
                        var_343 = wp::extract(var_339, var_342);
                        // contact_dist_1 = dists4_box[1]                                         <L 292>
                        var_345 = wp::extract(var_339, var_344);
                        // contact_dist_2 = dists4_box[2]                                         <L 293>
                        var_347 = wp::extract(var_339, var_346);
                        // contact_dist_3 = dists4_box[3]                                         <L 294>
                        var_349 = wp::extract(var_339, var_348);
                        // contact_pos_0 = wp.vec3(positions4_box[0, 0], positions4_box[0, 1], positions4_box[0, 2])       <L 295>
                        var_352 = wp::extract(var_340, var_350, var_351);
                        var_355 = wp::extract(var_340, var_353, var_354);
                        var_358 = wp::extract(var_340, var_356, var_357);
                        var_359 = wp::vec_t<3, wp::float32>(var_352, var_355, var_358);
                        // contact_pos_1 = wp.vec3(positions4_box[1, 0], positions4_box[1, 1], positions4_box[1, 2])       <L 296>
                        var_362 = wp::extract(var_340, var_360, var_361);
                        var_365 = wp::extract(var_340, var_363, var_364);
                        var_368 = wp::extract(var_340, var_366, var_367);
                        var_369 = wp::vec_t<3, wp::float32>(var_362, var_365, var_368);
                        // contact_pos_2 = wp.vec3(positions4_box[2, 0], positions4_box[2, 1], positions4_box[2, 2])       <L 297>
                        var_372 = wp::extract(var_340, var_370, var_371);
                        var_375 = wp::extract(var_340, var_373, var_374);
                        var_378 = wp::extract(var_340, var_376, var_377);
                        var_379 = wp::vec_t<3, wp::float32>(var_372, var_375, var_378);
                        // contact_pos_3 = wp.vec3(positions4_box[3, 0], positions4_box[3, 1], positions4_box[3, 2])       <L 298>
                        var_382 = wp::extract(var_340, var_380, var_381);
                        var_385 = wp::extract(var_340, var_383, var_384);
                        var_388 = wp::extract(var_340, var_386, var_387);
                        var_389 = wp::vec_t<3, wp::float32>(var_382, var_385, var_388);
                    }
                    if (!var_329) {
                        // elif is_sphere_a and is_sphere_b:                                      <L 303>
                        var_390 = var_247;
                        if (var_390) {
                            var_390 = var_390 && var_249;
                        }
                        if (var_390) {
                            // radius_a = scale_a[0]                                              <L 304>
                            var_392 = wp::extract(var_60, var_391);
                            // radius_b = scale_b[0]                                              <L 305>
                            var_394 = wp::extract(var_67, var_393);
                            // contact_dist_0, contact_pos_0, contact_normal = collide_sphere_sphere(pos_a, radius_a, pos_b, radius_b)       <L 306>
                            collide_sphere_sphere_0(var_235, var_392, var_236, var_394, var_395, var_396, var_397);
                        }
                        if (!var_390) {
                            // elif is_plane_a and is_capsule_b:                                  <L 312>
                            var_398 = var_123;
                            if (var_398) {
                                var_398 = var_398 && var_253;
                            }
                            if (var_398) {
                                // plane_normal = wp.quat_rotate(quat_a, wp.vec3(0.0, 0.0, 1.0))       <L 313>
                                var_402 = wp::vec_t<3, wp::float32>(var_399, var_400, var_401);
                                var_403 = wp::quat_rotate(var_237, var_402);
                                // capsule_axis = wp.quat_rotate(quat_b, wp.vec3(0.0, 0.0, 1.0))       <L 314>
                                var_407 = wp::vec_t<3, wp::float32>(var_404, var_405, var_406);
                                var_408 = wp::quat_rotate(var_238, var_407);
                                // capsule_radius = scale_b[0]                                    <L 315>
                                var_410 = wp::extract(var_67, var_409);
                                // capsule_half_length = scale_b[1]                               <L 316>
                                var_412 = wp::extract(var_67, var_411);
                                // contact_dist_0, contact_pos_0, contact_dist_1, contact_pos_1 = _collide_plane_capsule_contacts(       <L 318>
                                // plane_normal,                                                  <L 319>
                                // pos_a,                                                         <L 320>
                                // pos_b,                                                         <L 321>
                                // capsule_axis,                                                  <L 322>
                                // capsule_radius,                                                <L 323>
                                // capsule_half_length,                                           <L 324>
                                _collide_plane_capsule_contacts_0(var_403, var_235, var_236, var_408, var_410, var_412, var_413, var_414, var_415, var_416);
                                // contact_normal = plane_normal                                  <L 327>
                                var_417 = wp::copy(var_403);
                            }
                            if (!var_398) {
                                // elif use_plane_cylinder:                                       <L 333>
                                if (var_284) {
                                    // plane_normal = wp.quat_rotate(quat_a, wp.vec3(0.0, 0.0, 1.0))       <L 334>
                                    var_421 = wp::vec_t<3, wp::float32>(var_418, var_419, var_420);
                                    var_422 = wp::quat_rotate(var_237, var_421);
                                    // cylinder_axis = wp.quat_rotate(quat_b, wp.vec3(0.0, 0.0, 1.0))       <L 335>
                                    var_426 = wp::vec_t<3, wp::float32>(var_423, var_424, var_425);
                                    var_427 = wp::quat_rotate(var_238, var_426);
                                    // cylinder_radius = scale_b[0]                               <L 336>
                                    var_429 = wp::extract(var_67, var_428);
                                    // cylinder_half_height = scale_b[1]                          <L 337>
                                    var_431 = wp::extract(var_67, var_430);
                                    // dists4, positions4, contact_normal = collide_plane_cylinder(       <L 339>
                                    // plane_normal, pos_a, pos_b, cylinder_axis, cylinder_radius, cylinder_half_height       <L 340>
                                    collide_plane_cylinder_0(var_422, var_235, var_236, var_427, var_429, var_431, var_432, var_433, var_434);
                                    // contact_dist_0 = dists4[0]                                 <L 343>
                                    var_436 = wp::extract(var_432, var_435);
                                    // contact_dist_1 = dists4[1]                                 <L 344>
                                    var_438 = wp::extract(var_432, var_437);
                                    // contact_dist_2 = dists4[2]                                 <L 345>
                                    var_440 = wp::extract(var_432, var_439);
                                    // contact_dist_3 = dists4[3]                                 <L 346>
                                    var_442 = wp::extract(var_432, var_441);
                                    // contact_pos_0 = wp.vec3(positions4[0, 0], positions4[0, 1], positions4[0, 2])       <L 347>
                                    var_445 = wp::extract(var_433, var_443, var_444);
                                    var_448 = wp::extract(var_433, var_446, var_447);
                                    var_451 = wp::extract(var_433, var_449, var_450);
                                    var_452 = wp::vec_t<3, wp::float32>(var_445, var_448, var_451);
                                    // contact_pos_1 = wp.vec3(positions4[1, 0], positions4[1, 1], positions4[1, 2])       <L 348>
                                    var_455 = wp::extract(var_433, var_453, var_454);
                                    var_458 = wp::extract(var_433, var_456, var_457);
                                    var_461 = wp::extract(var_433, var_459, var_460);
                                    var_462 = wp::vec_t<3, wp::float32>(var_455, var_458, var_461);
                                    // contact_pos_2 = wp.vec3(positions4[2, 0], positions4[2, 1], positions4[2, 2])       <L 349>
                                    var_465 = wp::extract(var_433, var_463, var_464);
                                    var_468 = wp::extract(var_433, var_466, var_467);
                                    var_471 = wp::extract(var_433, var_469, var_470);
                                    var_472 = wp::vec_t<3, wp::float32>(var_465, var_468, var_471);
                                    // contact_pos_3 = wp.vec3(positions4[3, 0], positions4[3, 1], positions4[3, 2])       <L 350>
                                    var_475 = wp::extract(var_433, var_473, var_474);
                                    var_478 = wp::extract(var_433, var_476, var_477);
                                    var_481 = wp::extract(var_433, var_479, var_480);
                                    var_482 = wp::vec_t<3, wp::float32>(var_475, var_478, var_481);
                                }
                                if (!var_284) {
                                    // elif is_sphere_a and is_capsule_b:                         <L 355>
                                    var_483 = var_247;
                                    if (var_483) {
                                        var_483 = var_483 && var_253;
                                    }
                                    if (var_483) {
                                        // sphere_radius = scale_a[0]                             <L 356>
                                        var_485 = wp::extract(var_60, var_484);
                                        // capsule_axis = wp.quat_rotate(quat_b, wp.vec3(0.0, 0.0, 1.0))       <L 357>
                                        var_489 = wp::vec_t<3, wp::float32>(var_486, var_487, var_488);
                                        var_490 = wp::quat_rotate(var_238, var_489);
                                        // capsule_radius = scale_b[0]                            <L 358>
                                        var_492 = wp::extract(var_67, var_491);
                                        // capsule_half_length = scale_b[1]                       <L 359>
                                        var_494 = wp::extract(var_67, var_493);
                                        // contact_dist_0, contact_pos_0, contact_normal = collide_sphere_capsule(       <L 360>
                                        // pos_a, sphere_radius, pos_b, capsule_axis, capsule_radius, capsule_half_length       <L 361>
                                        collide_sphere_capsule_0(var_235, var_485, var_236, var_490, var_492, var_494, var_495, var_496, var_497);
                                    }
                                    if (!var_483) {
                                        // elif is_capsule_a and is_capsule_b:                    <L 368>
                                        var_498 = var_251;
                                        if (var_498) {
                                            var_498 = var_498 && var_253;
                                        }
                                        if (var_498) {
                                            // axis_a = wp.quat_rotate(quat_a, wp.vec3(0.0, 0.0, 1.0))       <L 369>
                                            var_502 = wp::vec_t<3, wp::float32>(var_499, var_500, var_501);
                                            var_503 = wp::quat_rotate(var_237, var_502);
                                            // axis_b = wp.quat_rotate(quat_b, wp.vec3(0.0, 0.0, 1.0))       <L 370>
                                            var_507 = wp::vec_t<3, wp::float32>(var_504, var_505, var_506);
                                            var_508 = wp::quat_rotate(var_238, var_507);
                                            // radius_a = scale_a[0]                              <L 371>
                                            var_510 = wp::extract(var_60, var_509);
                                            // half_length_a = scale_a[1]                         <L 372>
                                            var_512 = wp::extract(var_60, var_511);
                                            // radius_b = scale_b[0]                              <L 373>
                                            var_514 = wp::extract(var_67, var_513);
                                            // half_length_b = scale_b[1]                         <L 374>
                                            var_516 = wp::extract(var_67, var_515);
                                            // dists, positions, contact_normal = collide_capsule_capsule(       <L 376>
                                            // pos_a, axis_a, radius_a, half_length_a, pos_b, axis_b, radius_b, half_length_b       <L 377>
                                            collide_capsule_capsule_0(var_235, var_503, var_510, var_512, var_236, var_508, var_514, var_516, var_517, var_518, var_519);
                                            // contact_dist_0 = dists[0]                          <L 380>
                                            var_521 = wp::extract(var_517, var_520);
                                            // contact_pos_0 = wp.vec3(positions[0, 0], positions[0, 1], positions[0, 2])       <L 381>
                                            var_524 = wp::extract(var_518, var_522, var_523);
                                            var_527 = wp::extract(var_518, var_525, var_526);
                                            var_530 = wp::extract(var_518, var_528, var_529);
                                            var_531 = wp::vec_t<3, wp::float32>(var_524, var_527, var_530);
                                            // contact_dist_1 = dists[1]                          <L 382>
                                            var_533 = wp::extract(var_517, var_532);
                                            // contact_pos_1 = wp.vec3(positions[1, 0], positions[1, 1], positions[1, 2])       <L 383>
                                            var_536 = wp::extract(var_518, var_534, var_535);
                                            var_539 = wp::extract(var_518, var_537, var_538);
                                            var_542 = wp::extract(var_518, var_540, var_541);
                                            var_543 = wp::vec_t<3, wp::float32>(var_536, var_539, var_542);
                                        }
                                        if (!var_498) {
                                            // elif is_sphere_a and is_cylinder_b and scale_b[2] == 0.0:       <L 388>
                                            var_544 = var_247;
                                            if (var_544) {
                                                var_544 = var_544 && var_257;
                                            }
                                            if (var_544) {
                                                var_546 = wp::extract(var_67, var_545);
                                                var_548 = (var_546 == var_547);
                                                var_544 = var_544 && var_548;
                                            }
                                            if (var_544) {
                                                // sphere_radius = scale_a[0]                     <L 389>
                                                var_550 = wp::extract(var_60, var_549);
                                                // cylinder_axis = wp.quat_rotate(quat_b, wp.vec3(0.0, 0.0, 1.0))       <L 390>
                                                var_554 = wp::vec_t<3, wp::float32>(var_551, var_552, var_553);
                                                var_555 = wp::quat_rotate(var_238, var_554);
                                                // cylinder_radius = scale_b[0]                   <L 391>
                                                var_557 = wp::extract(var_67, var_556);
                                                // cylinder_half_height = scale_b[1]              <L 392>
                                                var_559 = wp::extract(var_67, var_558);
                                                // contact_dist_0, contact_pos_0, contact_normal = collide_sphere_cylinder(       <L 393>
                                                // pos_a, sphere_radius, pos_b, cylinder_axis, cylinder_radius, cylinder_half_height       <L 394>
                                                collide_sphere_cylinder_0(var_235, var_550, var_236, var_555, var_557, var_559, var_560, var_561, var_562);
                                            }
                                            if (!var_544) {
                                                // elif is_sphere_a and is_box_b:                 <L 400>
                                                var_563 = var_247;
                                                if (var_563) {
                                                    var_563 = var_563 && var_259;
                                                }
                                                if (var_563) {
                                                    // sphere_radius = scale_a[0]                 <L 401>
                                                    var_565 = wp::extract(var_60, var_564);
                                                    // box_rot = wp.quat_to_matrix(quat_b)        <L 402>
                                                    var_566 = wp::quat_to_matrix(var_238);
                                                    // box_size = scale_b                         <L 403>
                                                    var_567 = wp::copy(var_67);
                                                    // contact_dist_0, contact_pos_0, contact_normal = collide_sphere_box(       <L 404>
                                                    // pos_a, sphere_radius, pos_b, box_rot, box_size       <L 405>
                                                    collide_sphere_box_0(var_235, var_565, var_236, var_566, var_567, var_568, var_569, var_570);
                                                }
                                                var_571 = wp::where(var_563, var_568, var_298);
                                                var_572 = wp::where(var_563, var_569, var_302);
                                                var_573 = wp::where(var_563, var_570, var_306);
                                            }
                                            var_574 = wp::where(var_544, var_555, var_275);
                                            var_575 = wp::where(var_544, var_560, var_571);
                                            var_576 = wp::where(var_544, var_561, var_572);
                                            var_577 = wp::where(var_544, var_562, var_573);
                                            var_578 = wp::where(var_544, var_550, var_565);
                                        }
                                        var_579 = wp::where(var_498, var_275, var_574);
                                        var_580 = wp::where(var_498, var_521, var_575);
                                        var_581 = wp::where(var_498, var_533, var_299);
                                        var_582 = wp::where(var_498, var_531, var_576);
                                        var_583 = wp::where(var_498, var_543, var_303);
                                        var_584 = wp::where(var_498, var_519, var_577);
                                    }
                                    var_585 = wp::where(var_483, var_275, var_579);
                                    var_586 = wp::where(var_483, var_495, var_580);
                                    var_587 = wp::where(var_483, var_299, var_581);
                                    var_588 = wp::where(var_483, var_496, var_582);
                                    var_589 = wp::where(var_483, var_303, var_583);
                                    var_590 = wp::where(var_483, var_497, var_584);
                                    var_591 = wp::where(var_483, var_485, var_578);
                                }
                                var_592 = wp::where(var_284, var_422, var_270);
                                var_593 = wp::where(var_284, var_427, var_585);
                                var_594 = wp::where(var_284, var_436, var_586);
                                var_595 = wp::where(var_284, var_438, var_587);
                                var_596 = wp::where(var_284, var_440, var_300);
                                var_597 = wp::where(var_284, var_442, var_301);
                                var_598 = wp::where(var_284, var_452, var_588);
                                var_599 = wp::where(var_284, var_462, var_589);
                                var_600 = wp::where(var_284, var_472, var_304);
                                var_601 = wp::where(var_284, var_482, var_305);
                                var_602 = wp::where(var_284, var_434, var_590);
                                var_603 = wp::where(var_284, var_429, var_557);
                                var_604 = wp::where(var_284, var_431, var_559);
                            }
                            var_605 = wp::where(var_398, var_403, var_592);
                            var_606 = wp::where(var_398, var_275, var_593);
                            var_607 = wp::where(var_398, var_413, var_594);
                            var_608 = wp::where(var_398, var_415, var_595);
                            var_609 = wp::where(var_398, var_300, var_596);
                            var_610 = wp::where(var_398, var_301, var_597);
                            var_611 = wp::where(var_398, var_414, var_598);
                            var_612 = wp::where(var_398, var_416, var_599);
                            var_613 = wp::where(var_398, var_304, var_600);
                            var_614 = wp::where(var_398, var_305, var_601);
                            var_615 = wp::where(var_398, var_417, var_602);
                            var_616 = wp::where(var_398, var_408, var_490);
                            var_617 = wp::where(var_398, var_410, var_492);
                            var_618 = wp::where(var_398, var_412, var_494);
                        }
                        var_619 = wp::where(var_390, var_270, var_605);
                        var_620 = wp::where(var_390, var_275, var_606);
                        var_621 = wp::where(var_390, var_395, var_607);
                        var_622 = wp::where(var_390, var_299, var_608);
                        var_623 = wp::where(var_390, var_300, var_609);
                        var_624 = wp::where(var_390, var_301, var_610);
                        var_625 = wp::where(var_390, var_396, var_611);
                        var_626 = wp::where(var_390, var_303, var_612);
                        var_627 = wp::where(var_390, var_304, var_613);
                        var_628 = wp::where(var_390, var_305, var_614);
                        var_629 = wp::where(var_390, var_397, var_615);
                        var_630 = wp::where(var_390, var_392, var_510);
                        var_631 = wp::where(var_390, var_394, var_514);
                    }
                    var_632 = wp::where(var_329, var_334, var_619);
                    var_633 = wp::where(var_329, var_275, var_620);
                    var_634 = wp::where(var_329, var_343, var_621);
                    var_635 = wp::where(var_329, var_345, var_622);
                    var_636 = wp::where(var_329, var_347, var_623);
                    var_637 = wp::where(var_329, var_349, var_624);
                    var_638 = wp::where(var_329, var_359, var_625);
                    var_639 = wp::where(var_329, var_369, var_626);
                    var_640 = wp::where(var_329, var_379, var_627);
                    var_641 = wp::where(var_329, var_389, var_628);
                    var_642 = wp::where(var_329, var_341, var_629);
                    var_643 = wp::where(var_329, var_335, var_566);
                    var_644 = wp::where(var_329, var_336, var_567);
                }
                var_645 = wp::where(var_318, var_323, var_632);
                var_646 = wp::where(var_318, var_275, var_633);
                var_647 = wp::where(var_318, var_326, var_634);
                var_648 = wp::where(var_318, var_299, var_635);
                var_649 = wp::where(var_318, var_300, var_636);
                var_650 = wp::where(var_318, var_301, var_637);
                var_651 = wp::where(var_318, var_327, var_638);
                var_652 = wp::where(var_318, var_303, var_639);
                var_653 = wp::where(var_318, var_304, var_640);
                var_654 = wp::where(var_318, var_305, var_641);
                var_655 = wp::where(var_318, var_328, var_642);
            }
            var_656 = wp::where(var_307, var_312, var_645);
            var_657 = wp::where(var_307, var_275, var_646);
            var_658 = wp::where(var_307, var_315, var_647);
            var_659 = wp::where(var_307, var_299, var_648);
            var_660 = wp::where(var_307, var_300, var_649);
            var_661 = wp::where(var_307, var_301, var_650);
            var_662 = wp::where(var_307, var_316, var_651);
            var_663 = wp::where(var_307, var_303, var_652);
            var_664 = wp::where(var_307, var_304, var_653);
            var_665 = wp::where(var_307, var_305, var_654);
            var_666 = wp::where(var_307, var_317, var_655);
            var_667 = wp::where(var_307, var_314, var_591);
            // num_contacts = (                                                                   <L 411>
            // int(contact_dist_0 < MAXVAL)                                                       <L 412>
            var_668 = (var_658 < var_297);
            var_669 = wp::int(var_668);
            // + int(contact_dist_1 < MAXVAL)                                                     <L 413>
            var_670 = (var_659 < var_297);
            var_671 = wp::int(var_670);
            var_672 = wp::add(var_669, var_671);
            // + int(contact_dist_2 < MAXVAL)                                                     <L 414>
            var_673 = (var_660 < var_297);
            var_674 = wp::int(var_673);
            var_675 = wp::add(var_672, var_674);
            // + int(contact_dist_3 < MAXVAL)                                                     <L 415>
            var_676 = (var_661 < var_297);
            var_677 = wp::int(var_676);
            var_678 = wp::add(var_675, var_677);
            // if num_contacts > 0:                                                               <L 417>
            var_680 = (var_678 > var_679);
            if (var_680) {
                // contact_data = ContactData()                                                   <L 419>
                var_681 = ContactData_40360d7c();
                // contact_data.contact_normal_a_to_b = contact_normal                            <L 420>
                var_681.contact_normal_a_to_b = var_666;
                // contact_data.radius_eff_a = radius_eff_a                                       <L 421>
                var_681.radius_eff_a = var_292;
                // contact_data.radius_eff_b = radius_eff_b                                       <L 422>
                var_681.radius_eff_b = var_296;
                // contact_data.margin_a = margin_offset_a                                        <L 423>
                var_681.margin_a = var_226;
                // contact_data.margin_b = margin_offset_b                                        <L 424>
                var_681.margin_b = var_228;
                // contact_data.shape_a = shape_a                                                 <L 425>
                var_681.shape_a = var_43;
                // contact_data.shape_b = shape_b                                                 <L 426>
                var_681.shape_b = var_44;
                // if wp.static(speculative):                                                     <L 427>
                // contact_data.gap_sum = gap_sum                                                 <L 430>
                var_681.gap_sum = var_245;
                // total_separation_needed = radius_eff_a + radius_eff_b + margin_offset_a + margin_offset_b       <L 432>
                var_683 = wp::add(var_292, var_296);
                var_684 = wp::add(var_683, var_226);
                var_685 = wp::add(var_684, var_228);
                // normalized_contact_normal = wp.normalize(contact_normal)                       <L 433>
                var_686 = wp::normalize(var_666);
                // contact_0_valid = False                                                        <L 436>
                // if contact_dist_0 < MAXVAL:                                                    <L 437>
                var_688 = (var_658 < var_297);
                if (var_688) {
                    // contact_0_valid = _admit(                                                  <L 438>
                    // contact_data,                                                              <L 439>
                    // contact_pos_0,                                                             <L 440>
                    // contact_dist_0,                                                            <L 441>
                    // normalized_contact_normal,                                                 <L 442>
                    // total_separation_needed,                                                   <L 443>
                    // shape_transform,                                                           <L 444>
                    // shape_linear_velocity,                                                     <L 445>
                    // shape_angular_velocity,                                                    <L 446>
                    // collision_update_dt,                                                       <L 447>
                    // max_speculative_extension,                                                 <L 448>
                    var_689 = create_narrow_phase_primitive_kernel__locals___admit_0(var_681, var_662, var_658, var_686, var_685, var_shape_transform, var_shape_linear_velocity, var_shape_angular_velocity, var_collision_update_dt, var_max_speculative_extension);
                }
                var_690 = wp::where(var_688, var_689, var_687);
                // contact_1_valid = False                                                        <L 451>
                // if contact_dist_1 < MAXVAL:                                                    <L 452>
                var_692 = (var_659 < var_297);
                if (var_692) {
                    // contact_1_valid = _admit(                                                  <L 453>
                    // contact_data,                                                              <L 454>
                    // contact_pos_1,                                                             <L 455>
                    // contact_dist_1,                                                            <L 456>
                    // normalized_contact_normal,                                                 <L 457>
                    // total_separation_needed,                                                   <L 458>
                    // shape_transform,                                                           <L 459>
                    // shape_linear_velocity,                                                     <L 460>
                    // shape_angular_velocity,                                                    <L 461>
                    // collision_update_dt,                                                       <L 462>
                    // max_speculative_extension,                                                 <L 463>
                    var_693 = create_narrow_phase_primitive_kernel__locals___admit_0(var_681, var_663, var_659, var_686, var_685, var_shape_transform, var_shape_linear_velocity, var_shape_angular_velocity, var_collision_update_dt, var_max_speculative_extension);
                }
                var_694 = wp::where(var_692, var_693, var_691);
                // contact_2_valid = False                                                        <L 466>
                // if contact_dist_2 < MAXVAL:                                                    <L 467>
                var_696 = (var_660 < var_297);
                if (var_696) {
                    // contact_2_valid = _admit(                                                  <L 468>
                    // contact_data,                                                              <L 469>
                    // contact_pos_2,                                                             <L 470>
                    // contact_dist_2,                                                            <L 471>
                    // normalized_contact_normal,                                                 <L 472>
                    // total_separation_needed,                                                   <L 473>
                    // shape_transform,                                                           <L 474>
                    // shape_linear_velocity,                                                     <L 475>
                    // shape_angular_velocity,                                                    <L 476>
                    // collision_update_dt,                                                       <L 477>
                    // max_speculative_extension,                                                 <L 478>
                    var_697 = create_narrow_phase_primitive_kernel__locals___admit_0(var_681, var_664, var_660, var_686, var_685, var_shape_transform, var_shape_linear_velocity, var_shape_angular_velocity, var_collision_update_dt, var_max_speculative_extension);
                }
                var_698 = wp::where(var_696, var_697, var_695);
                // contact_3_valid = False                                                        <L 481>
                // if contact_dist_3 < MAXVAL:                                                    <L 482>
                var_700 = (var_661 < var_297);
                if (var_700) {
                    // contact_3_valid = _admit(                                                  <L 483>
                    // contact_data,                                                              <L 484>
                    // contact_pos_3,                                                             <L 485>
                    // contact_dist_3,                                                            <L 486>
                    // normalized_contact_normal,                                                 <L 487>
                    // total_separation_needed,                                                   <L 488>
                    // shape_transform,                                                           <L 489>
                    // shape_linear_velocity,                                                     <L 490>
                    // shape_angular_velocity,                                                    <L 491>
                    // collision_update_dt,                                                       <L 492>
                    // max_speculative_extension,                                                 <L 493>
                    var_701 = create_narrow_phase_primitive_kernel__locals___admit_0(var_681, var_665, var_661, var_686, var_685, var_shape_transform, var_shape_linear_velocity, var_shape_angular_velocity, var_collision_update_dt, var_max_speculative_extension);
                }
                var_702 = wp::where(var_700, var_701, var_699);
                // num_valid = int(contact_0_valid) + int(contact_1_valid) + int(contact_2_valid) + int(contact_3_valid)       <L 497>
                var_703 = wp::int(var_690);
                var_704 = wp::int(var_694);
                var_705 = wp::add(var_703, var_704);
                var_706 = wp::int(var_698);
                var_707 = wp::add(var_705, var_706);
                var_708 = wp::int(var_702);
                var_709 = wp::add(var_707, var_708);
                // if num_valid > 0:                                                              <L 498>
                var_711 = (var_709 > var_710);
                if (var_711) {
                    // base_index = wp.atomic_add(writer_data.contact_count, 0, num_valid)        <L 499>
                    var_712 = &((var_writer_data).contact_count);
                    var_715 = wp::load(var_712);
                    var_714 = wp::atomic_add(var_715, var_713, var_709);
                    // if base_index + num_valid > writer_data.contact_max:                       <L 503>
                    var_716 = wp::add(var_714, var_709);
                    var_717 = &((var_writer_data).contact_max);
                    var_719 = wp::load(var_717);
                    var_718 = (var_716 > var_719);
                    if (var_718) {
                        // continue                                                               <L 504>
                        goto start_for_1;
                    }
                    // if contact_0_valid:                                                        <L 507>
                    if (var_690) {
                        // contact_data.contact_point_center = contact_pos_0                      <L 508>
                        var_681.contact_point_center = var_662;
                        // contact_data.contact_distance = contact_dist_0                         <L 509>
                        var_681.contact_distance = var_658;
                        // contact_data.sort_sub_key = 0                                          <L 510>
                        var_681.sort_sub_key = var_720;
                        // writer_func(contact_data, writer_data, base_index)                     <L 511>
                        write_contact_0(var_681, var_writer_data, var_714);
                        // base_index += 1                                                        <L 512>
                        var_722 = wp::add(var_714, var_721);
                    }
                    var_723 = wp::where(var_690, var_722, var_714);
                    // if contact_1_valid:                                                        <L 515>
                    if (var_694) {
                        // contact_data.contact_point_center = contact_pos_1                      <L 516>
                        var_681.contact_point_center = var_663;
                        // contact_data.contact_distance = contact_dist_1                         <L 517>
                        var_681.contact_distance = var_659;
                        // contact_data.sort_sub_key = 1                                          <L 518>
                        var_681.sort_sub_key = var_724;
                        // writer_func(contact_data, writer_data, base_index)                     <L 519>
                        write_contact_0(var_681, var_writer_data, var_723);
                        // base_index += 1                                                        <L 520>
                        var_726 = wp::add(var_723, var_725);
                    }
                    var_727 = wp::where(var_694, var_726, var_723);
                    // if contact_2_valid:                                                        <L 523>
                    if (var_698) {
                        // contact_data.contact_point_center = contact_pos_2                      <L 524>
                        var_681.contact_point_center = var_664;
                        // contact_data.contact_distance = contact_dist_2                         <L 525>
                        var_681.contact_distance = var_660;
                        // contact_data.sort_sub_key = 2                                          <L 526>
                        var_681.sort_sub_key = var_728;
                        // writer_func(contact_data, writer_data, base_index)                     <L 527>
                        write_contact_0(var_681, var_writer_data, var_727);
                        // base_index += 1                                                        <L 528>
                        var_730 = wp::add(var_727, var_729);
                    }
                    var_731 = wp::where(var_698, var_730, var_727);
                    // if contact_3_valid:                                                        <L 531>
                    if (var_702) {
                        // contact_data.contact_point_center = contact_pos_3                      <L 532>
                        var_681.contact_point_center = var_665;
                        // contact_data.contact_distance = contact_dist_3                         <L 533>
                        var_681.contact_distance = var_661;
                        // contact_data.sort_sub_key = 3                                          <L 534>
                        var_681.sort_sub_key = var_732;
                        // writer_func(contact_data, writer_data, base_index)                     <L 535>
                        write_contact_0(var_681, var_writer_data, var_731);
                    }
                }
                // continue                                                                       <L 537>
                goto start_for_1;
            }
            // if (                                                                               <L 539>
            // (is_plane_a and (is_sphere_b or is_capsule_b or is_ellipsoid_b or use_plane_cylinder or is_box_b))       <L 540>
            var_734 = var_123;
            if (var_734) {
                var_735 = var_249;
                if (!var_735) {
                    var_735 = var_735 || var_253;
                }
                if (!var_735) {
                    var_735 = var_735 || var_255;
                }
                if (!var_735) {
                    var_735 = var_735 || var_284;
                }
                if (!var_735) {
                    var_735 = var_735 || var_259;
                }
                var_734 = var_734 && var_735;
            }
            var_733 = var_734;
            if (!var_733) {
                // or (is_sphere_a and (is_sphere_b or is_capsule_b or (is_cylinder_b and scale_b[2] == 0.0) or is_box_b))       <L 541>
                var_736 = var_247;
                if (var_736) {
                    var_737 = var_249;
                    if (!var_737) {
                        var_737 = var_737 || var_253;
                    }
                    if (!var_737) {
                        var_738 = var_257;
                        if (var_738) {
                            var_740 = wp::extract(var_67, var_739);
                            var_742 = (var_740 == var_741);
                            var_738 = var_738 && var_742;
                        }
                        var_737 = var_737 || var_738;
                    }
                    if (!var_737) {
                        var_737 = var_737 || var_259;
                    }
                    var_736 = var_736 && var_737;
                }
                var_733 = var_733 || var_736;
            }
            if (!var_733) {
                // or (is_capsule_a and is_capsule_b)                                             <L 542>
                var_743 = var_251;
                if (var_743) {
                    var_743 = var_743 && var_253;
                }
                var_733 = var_733 || var_743;
            }
            if (var_733) {
                // continue                                                                       <L 544>
                goto start_for_1;
            }
            // if wp.static(sparse_gjk_pairs):                                                    <L 549>
            // _append_pair_compacted(True, wp.vec2i(shape_a, shape_b), gjk_candidate_pairs, gjk_candidate_pairs_count)       <L 554>
            var_746 = wp::vec_t<2, wp::int32>(var_43, var_44);
            _append_pair_compacted_0(var_745, var_746, var_gjk_candidate_pairs, var_gjk_candidate_pairs_count);
            goto start_for_1;
        end_for_1:;
    }
}

